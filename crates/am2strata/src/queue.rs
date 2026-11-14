//! One queue for everything that needs the GPU's turn (the server runs one request at a time): chat requests, model
//! switches, media jobs. In arrival order, with one exception that saves a reload: when the next job needs a different
//! resident model (a text model, or the GPU for a media job with the text model stopped), the jobs that need the one
//! already loaded go first - up to BATCH of them in a row, so nothing waits behind them forever.
//!
//! A job waits as long as it must (no two-minute "busy"); a client that leaves takes its job out (the waiting future
//! is dropped); `queue_max_wait_s` in the run config turns an over-long wait into a refusal with the position.
//!
//! The rules, in this order (PRODUCTION-PLAN.md, step 7); the same queue always gives the same order:
//! 1. One job runs at a time. The next one comes from the best class waiting: high (a person at the web app), normal
//!    (API calls), low (media jobs a chat queued); `x-strata-priority` asks for one (a client key: normal or low).
//! 2. Every AGING of waiting raises a job one class, so a low job waits at most 2 x AGING behind newer high ones.
//! 3. Within the class, arrival order, with one exception that saves a model switch: when the first job needs another
//!    model, jobs for the model already loaded go first, at most BATCH in a row and none once the first has waited
//!    BATCH_WAIT.
//! 4. Of two clients with equal jobs, the one that did not just run goes first.
//! 5. While a job waits for a model that is not loaded, that model's files are read into the file cache (when there is
//!    memory for them), so the switch reads RAM, not the SSD.
//! The queue learns each kind of job's running time and each model's switch time (kept across restarts), and
//! gives each waiting job an estimate of its wait (/api/queue, and the stream's "queued" comment).

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

/// The most jobs taken out of order for the model already loaded, before the head of the queue gets its turn.
pub const BATCH: usize = 4;
/// ...and none at all once the job at the head has waited this long: long jobs (an agent's chat of 60,000 tokens takes
/// minutes) made "4 in a row" a 20-minute wait; past this, the queue goes strictly in order of arrival.
pub const BATCH_WAIT: Duration = Duration::from_secs(90);

/// The most jobs that may wait at once; one more is refused at once (a client cannot fill the memory with waiters).
pub const MAX_WAITING: usize = 64;

/// A job's class: High (a person waiting at the web app), Normal (API calls), Low (background work).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    High = 0,
    Normal = 1,
    Low = 2,
}

impl Priority {
    pub fn parse(s: &str) -> Option<Priority> {
        match s.trim().to_ascii_lowercase().as_str() {
            "high" | "interactive" => Some(Priority::High),
            "normal" | "default" => Some(Priority::Normal),
            "low" | "background" | "batch" => Some(Priority::Low),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self { Priority::High => "high", Priority::Normal => "normal", Priority::Low => "low" }
    }
}

tokio::task_local! {
    /// the priority and client of the request this task serves (set by the server's priority layer)
    pub static PRIORITY: Priority;
    pub static CLIENT: String;
}

/// `fut` with this task's priority and client (a spawned task starts without them).
pub fn carry<F: std::future::Future>(fut: F) -> impl std::future::Future<Output = F::Output> {
    PRIORITY.scope(current_priority(), CLIENT.scope(current_client(), fut))
}

fn current_priority() -> Priority {
    PRIORITY.try_with(|p| *p).unwrap_or(Priority::Normal)
}

pub fn current_client() -> String {
    CLIENT.try_with(|c| c.clone()).unwrap_or_default()
}

/// Waiting this long raises a job one class (a low job waits at most two of these behind high ones).
pub const AGING: Duration = Duration::from_secs(60);

/// What a job needs resident while it runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Need {
    /// a text model, by id
    Text(String),
    /// the GPU to itself: a media job, the text model stopped
    Gpu,
    /// nothing in particular (an unload, a job that runs beside the text model)
    Any,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Resident {
    None,
    Text(String),
    Gpu,
}

fn fits(need: &Need, r: &Resident) -> bool {
    match (need, r) {
        (Need::Any, _) => true,
        (Need::Text(a), Resident::Text(b)) => a == b,
        (Need::Gpu, Resident::Gpu) => true,
        _ => false,
    }
}

struct Ticket {
    id: u64,
    need: Need,
    kind: String,
    label: String,
    since: Instant,
    priority: Priority,
    client: String,
}

impl Ticket {
    /// its class now: the priority, raised one for every AGING it has waited
    fn rank(&self) -> u8 {
        (self.priority as u8).saturating_sub((self.since.elapsed().as_secs() / AGING.as_secs().max(1)) as u8)
    }
}

/// A job's kind and need, as the key of its learnt running time.
fn cost_key(kind: &str, need: &Need) -> String {
    match need {
        Need::Text(m) => format!("{kind}:{m}"),
        Need::Gpu => format!("{kind}:gpu"),
        Need::Any => format!("{kind}:any"),
    }
}

/// A running time to assume before one is learnt.
fn default_cost(kind: &str) -> f64 {
    match kind {
        "chat" => 20.0,
        "switch" => 30.0,
        "media" | "song" | "image" | "video" => 60.0,
        _ => 15.0,
    }
}

struct State {
    queue: VecDeque<Ticket>,
    running: Option<(u64, Need, String, String, Instant)>,
    resident: Resident,
    /// jobs taken out of order in a row (for the resident model)
    batched: usize,
    next: u64,
    /// learnt running times (seconds, a moving average) by cost_key, and each text model's switch time
    costs: std::collections::HashMap<String, f64>,
    switch_s: std::collections::HashMap<String, f64>,
    /// the client of the job that ran last (fairness between clients)
    last_client: String,
}

pub struct Queue {
    state: Mutex<State>,
    changed: Notify,
    pub max_wait: Option<Duration>,
    /// BATCH_WAIT (a test sets it to 0)
    batch_wait: Duration,
}

/// The turn: held while the job runs; dropped, the next job's.
pub struct Pass {
    q: Arc<Queue>,
    id: u64,
}

impl Drop for Pass {
    fn drop(&mut self) {
        let mut s = self.q.state.lock().unwrap();
        if s.running.as_ref().is_some_and(|r| r.0 == self.id) {
            let (_, need, kind, _, t) = s.running.take().unwrap();
            // the running time learnt (a moving average; a job of under a second is not one to learn from)
            let took = t.elapsed().as_secs_f64();
            if took >= 1.0 {
                let k = cost_key(&kind, &need);
                let v = s.costs.entry(k).or_insert(took);
                *v = *v * 0.7 + took * 0.3;
            }
        }
        drop(s);
        self.q.changed.notify_waiters();
        crate::live_channel::poke();
    }
}

/// A job still waiting: dropped (its client left), it leaves the queue.
struct Waiting<'a> {
    q: &'a Arc<Queue>,
    id: u64,
    done: bool,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.q.state.lock().unwrap().queue.retain(|t| t.id != self.id);
            self.q.changed.notify_waiters();
            crate::live_channel::poke();
        }
    }
}

#[derive(Debug)]
pub struct TooLong {
    pub position: usize,
    pub waited_s: u64,
    /// refused because MAX_WAITING jobs already wait
    pub full: bool,
}

impl std::fmt::Display for TooLong {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.full {
            write!(f, "{MAX_WAITING} jobs are already waiting for their turn; try again later")
        } else {
            write!(f, "waited {} s at place {} of the queue (queue_max_wait_s); try again later", self.waited_s, self.position)
        }
    }
}

impl Queue {
    pub fn new(max_wait: Option<Duration>) -> Arc<Queue> {
        Self::with_batch_wait(max_wait, BATCH_WAIT)
    }

    pub fn with_batch_wait(max_wait: Option<Duration>, batch_wait: Duration) -> Arc<Queue> {
        Arc::new(Queue {
            state: Mutex::new(State { queue: VecDeque::new(), running: None, resident: Resident::None, batched: 0, next: 1,
                                      costs: Default::default(), switch_s: Default::default(), last_client: String::new() }),
            changed: Notify::new(),
            max_wait,
            batch_wait,
        })
    }

    /// The job that goes next, if the turn is free.
    fn choose(s: &State, batch_wait: Duration) -> Option<u64> {
        if s.running.is_some() {
            return None;
        }
        // the best class now (priority raised by waiting), its jobs in arrival order
        let best = s.queue.iter().map(Ticket::rank).min()?;
        let class: Vec<&Ticket> = s.queue.iter().filter(|t| t.rank() == best).collect();
        let head = class[0];
        if !fits(&head.need, &s.resident) && s.batched < BATCH && head.since.elapsed() < batch_wait {
            if let Some(t) = class.iter().find(|t| fits(&t.need, &s.resident)) {
                return Some(t.id);
            }
        }
        // fairness: when the head's client just ran, another client's job that fits as well goes first
        if !head.client.is_empty() && head.client == s.last_client && head.since.elapsed() < batch_wait {
            let same_fit = fits(&head.need, &s.resident);
            if let Some(t) = class.iter().skip(1).find(|t| t.client != head.client && fits(&t.need, &s.resident) == same_fit) {
                return Some(t.id);
            }
        }
        Some(head.id)
    }

    /// The order the waiting jobs would run in if nothing changed (choose, repeated on a copy).
    fn planned(s: &State, batch_wait: Duration) -> Vec<(u64, Need, String)> {
        let mut sim = State { queue: s.queue.iter().map(|t| Ticket { id: t.id, need: t.need.clone(), kind: t.kind.clone(),
                                                                        label: String::new(), since: t.since, priority: t.priority,
                                                                        client: t.client.clone() }).collect(),
                              running: None, resident: s.resident.clone(), batched: s.batched, next: 0,
                              costs: Default::default(), switch_s: Default::default(), last_client: s.last_client.clone() };
        let mut out = Vec::new();
        while let Some(id) = Self::choose(&sim, batch_wait) {
            let pos = sim.queue.iter().position(|t| t.id == id).unwrap();
            let t = sim.queue.remove(pos).unwrap();
            if pos > 0 { sim.batched += 1 } else { sim.batched = 0 }
            sim.last_client = t.client.clone();
            if let Need::Text(m) = &t.need {
                sim.resident = Resident::Text(m.clone());
            } else if t.need == Need::Gpu {
                sim.resident = Resident::Gpu;
            }
            out.push((t.id, t.need, t.kind));
        }
        out
    }

    /// Seconds until each waiting job (by id) should start: what the running one has left, then the jobs before it
    /// with their learnt running times and the model switches between them.
    fn etas(s: &State, batch_wait: Duration) -> std::collections::HashMap<u64, f64> {
        let mut t = s.running.as_ref().map(|(_, n, k, _, at)| {
            let est = s.costs.get(&cost_key(k, n)).copied().unwrap_or_else(|| default_cost(k));
            (est - at.elapsed().as_secs_f64()).max(2.0)
        }).unwrap_or(0.0);
        let mut resident = s.resident.clone();
        let mut out = std::collections::HashMap::new();
        for (id, need, kind) in Self::planned(s, batch_wait) {
            if let Need::Text(m) = &need {
                if resident != Resident::Text(m.clone()) {
                    t += s.switch_s.get(m).copied().unwrap_or_else(|| default_cost("switch"));
                    resident = Resident::Text(m.clone());
                }
            }
            out.insert(id, t);
            t += s.costs.get(&cost_key(&kind, &need)).copied().unwrap_or_else(|| default_cost(&kind));
        }
        out
    }

    /// How long a model switch to `model` took (the manager says so after each load).
    pub fn set_switch_cost(&self, model: &str, secs: f64) {
        let mut s = self.state.lock().unwrap();
        let v = s.switch_s.entry(model.to_string()).or_insert(secs);
        *v = *v * 0.5 + secs * 0.5;
    }

    /// The estimated wait of the job at `position` (1 = next), in seconds.
    pub fn eta_at(&self, position: usize) -> Option<f64> {
        let s = self.state.lock().unwrap();
        let etas = Self::etas(&s, self.batch_wait);
        let order = Self::planned(&s, self.batch_wait);
        order.get(position.checked_sub(1)?).and_then(|(id, _, _)| etas.get(id).copied())
    }

    /// The text models waiting jobs need that are not loaded (for warming their files up).
    pub fn wanted_models(&self) -> Vec<String> {
        let s = self.state.lock().unwrap();
        let mut v: Vec<String> = s.queue.iter().filter_map(|t| match &t.need {
            Need::Text(m) if s.resident != Resident::Text(m.clone()) => Some(m.clone()),
            _ => None,
        }).collect();
        v.dedup();
        v
    }

    /// Waits for the turn. `on_position` hears the job's place (1 = next) each time it changes.
    pub async fn enter(self: &Arc<Self>, need: Need, kind: &str, label: &str, mut on_position: impl FnMut(usize))
                       -> Result<Pass, TooLong> {
        let id = {
            let mut s = self.state.lock().unwrap();
            if s.queue.len() >= MAX_WAITING {
                return Err(TooLong { position: s.queue.len() + 1, waited_s: 0, full: true });
            }
            let id = s.next;
            s.next += 1;
            s.queue.push_back(Ticket { id, need, kind: kind.to_string(), label: label.to_string(), since: Instant::now(),
                                       priority: current_priority(), client: current_client() });
            id
        };
        crate::live_channel::poke();
        let mut w = Waiting { q: self, id, done: false };
        let since = Instant::now();
        let mut told = 0usize;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut s = self.state.lock().unwrap();
                if Self::choose(&s, self.batch_wait) == Some(id) {
                    let pos = s.queue.iter().position(|t| t.id == id).unwrap();
                    let t = s.queue.remove(pos).unwrap();
                    s.last_client = t.client.clone();
                    // out of order for the resident model counts toward the bound; the head's turn resets it
                    if pos > 0 { s.batched += 1 } else { s.batched = 0 }
                    s.running = Some((id, t.need, t.kind, t.label, Instant::now()));
                    w.done = true;
                    crate::live_channel::poke();
                    return Ok(Pass { q: self.clone(), id });
                }
                // its place in the order the jobs will run (priorities included), told outside the lock (the
                // listener may ask the queue for an estimate)
                let position = Self::planned(&s, self.batch_wait).iter().position(|(t, _, _)| *t == id).map(|p| p + 1).unwrap_or(0);
                drop(s);
                if position != told {
                    told = position;
                    on_position(position);
                }
                if let Some(max) = self.max_wait {
                    if since.elapsed() >= max {
                        return Err(TooLong { position, waited_s: since.elapsed().as_secs(), full: false });
                    }
                }
            }
            match self.max_wait {
                Some(max) => {
                    let _ = tokio::time::timeout(max.saturating_sub(since.elapsed()) + Duration::from_millis(10), notified).await;
                }
                None => notified.await,
            }
        }
    }

    /// What the running job left resident (the manager says so after a load, an unload or a media job).
    pub fn set_resident(&self, r: Resident) {
        self.state.lock().unwrap().resident = r;
        self.changed.notify_waiters();
        crate::live_channel::poke();
    }

    pub fn resident(&self) -> Resident {
        self.state.lock().unwrap().resident.clone()
    }

    /// A job runs or waits: a new one would have to wait.
    pub fn busy(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.running.is_some() || !s.queue.is_empty()
    }

    pub fn waiting(&self) -> usize {
        self.state.lock().unwrap().queue.len()
    }

    /// GET /api/queue: the running job and the waiting ones, in the order they will run if nothing changes.
    pub fn snapshot(&self) -> Value {
        let s = self.state.lock().unwrap();
        let need = |n: &Need| match n {
            Need::Text(m) => json!({"text_model": m}),
            Need::Gpu => json!("gpu"),
            Need::Any => json!("any"),
        };
        let running = s.running.as_ref().map(|(id, n, k, l, t)| json!({"id": id, "kind": k, "label": l, "need": need(n),
                                                                         "running_s": t.elapsed().as_secs()}));
        // in the order they will run, each with its class, client and estimated wait
        let etas = Self::etas(&s, self.batch_wait);
        let order = Self::planned(&s, self.batch_wait);
        let waiting: Vec<Value> = order.iter().enumerate().filter_map(|(i, (id, _, _))| {
            let t = s.queue.iter().find(|t| t.id == *id)?;
            Some(json!({"position": i + 1, "id": t.id, "kind": t.kind, "label": t.label, "need": need(&t.need),
                        "waiting_s": t.since.elapsed().as_secs(), "priority": t.priority.name(), "client": t.client,
                        "eta_s": etas.get(id).map(|e| e.round())}))
        }).collect();
        let resident = match &s.resident {
            Resident::None => json!(null),
            Resident::Text(m) => json!({"text_model": m}),
            Resident::Gpu => json!("gpu"),
        };
        let mut costs: Vec<(&String, &f64)> = s.costs.iter().collect();
        costs.sort_by(|a, b| a.0.cmp(b.0));
        json!({"running": running, "waiting": waiting, "resident": resident, "batched": s.batched, "batch_bound": BATCH,
               "learnt_s": costs.into_iter().map(|(k, v)| (k.clone(), json!(v.round()))).collect::<serde_json::Map<_, _>>(),
               "switch_s": s.switch_s.iter().map(|(k, v)| (k.clone(), json!(v.round()))).collect::<serde_json::Map<_, _>>()})
    }
}

#[cfg(test)]
mod priority_tests {
    use super::*;

    async fn run_with(q: Arc<Queue>, jobs: Vec<(&'static str, Priority, &'static str)>, first: &str) -> Vec<&'static str> {
        let blocker = CLIENT.scope(first.to_string(), q.enter(Need::Any, "block", "", |_| {})).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for (name, p, client) in jobs {
            let (q2, out2) = (q.clone(), out.clone());
            hs.push(tokio::spawn(PRIORITY.scope(p, CLIENT.scope(client.to_string(), async move {
                let _pass = q2.enter(Need::Any, "chat", name, |_| {}).await.unwrap();
                out2.lock().unwrap().push(name);
            }))));
            tokio::time::sleep(Duration::from_millis(20)).await; // arrival order
        }
        drop(blocker);
        for h in hs {
            h.await.unwrap();
        }
        let v = out.lock().unwrap().clone();
        v
    }

    #[tokio::test]
    async fn high_goes_before_normal_and_low() {
        let q = Queue::new(None);
        let order = run_with(q, vec![("low", Priority::Low, ""), ("normal", Priority::Normal, ""), ("high", Priority::High, "")], "").await;
        assert_eq!(order, ["high", "normal", "low"]);
    }

    #[tokio::test]
    async fn clients_take_turns() {
        let q = Queue::new(None);
        let order = run_with(q, vec![("a1", Priority::Normal, "a"), ("a2", Priority::Normal, "a"), ("b1", Priority::Normal, "b")], "a").await;
        assert_eq!(order[0], "b1", "a just ran: b's job goes first: {order:?}");
    }

    #[test]
    fn waiting_raises_the_class() {
        let t = Ticket { id: 1, need: Need::Any, kind: "chat".into(), label: String::new(),
                         since: Instant::now() - AGING * 2, priority: Priority::Low, client: String::new() };
        assert_eq!(t.rank(), 0, "two AGINGs of waiting: a low job ranks as high");
    }

    #[test]
    fn estimates_add_the_jobs_before_and_the_switches() {
        let q = Queue::new(None);
        {
            let mut s = q.state.lock().unwrap();
            s.resident = Resident::Text("a".into());
            s.costs.insert("chat:a".into(), 10.0);
            s.costs.insert("chat:b".into(), 5.0);
            s.switch_s.insert("b".into(), 40.0);
            for (id, m) in [(1, "a"), (2, "b")] {
                s.queue.push_back(Ticket { id, need: Need::Text(m.into()), kind: "chat".into(), label: String::new(),
                                           since: Instant::now(), priority: Priority::Normal, client: String::new() });
            }
        }
        assert_eq!(q.eta_at(1), Some(0.0));
        assert_eq!(q.eta_at(2), Some(50.0), "a's 10 s, then the 40 s switch to b");
        assert_eq!(q.wanted_models(), vec!["b".to_string()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The order jobs run in: each takes the turn, notes its name, and ends.
    async fn order(q: Arc<Queue>, jobs: Vec<(&'static str, Need)>, resident_after: impl Fn(&Need) -> Resident + Send + Sync + 'static)
                   -> Vec<&'static str> {
        let blocker = q.enter(Need::Any, "block", "", |_| {}).await.unwrap();
        let out = Arc::new(Mutex::new(Vec::new()));
        let ra = Arc::new(resident_after);
        let mut hs = Vec::new();
        for (name, need) in jobs {
            let (q2, out2, ra2) = (q.clone(), out.clone(), ra.clone());
            hs.push(tokio::spawn(async move {
                let p = q2.enter(need.clone(), name, name, |_| {}).await.unwrap();
                out2.lock().unwrap().push(name);
                q2.set_resident(ra2(&need));
                drop(p);
            }));
            tokio::task::yield_now().await; // each one in the queue before the next arrives
            while q.waiting() < hs.len() {
                tokio::task::yield_now().await;
            }
        }
        drop(blocker);
        for h in hs {
            h.await.unwrap();
        }
        let v = out.lock().unwrap().clone();
        v
    }

    fn after(n: &Need) -> Resident {
        match n {
            Need::Text(m) => Resident::Text(m.clone()),
            Need::Gpu => Resident::Gpu,
            Need::Any => Resident::None,
        }
    }

    #[tokio::test]
    async fn jobs_for_the_loaded_model_go_first_within_the_bound() {
        let q = Queue::new(None);
        q.set_resident(Resident::Text("a".into()));
        let t = |m: &str| Need::Text(m.into());
        // image, chat a, image, chat a: the two chats for the loaded model first, then the two images together
        let o = order(q.clone(), vec![("img1", Need::Gpu), ("a1", t("a")), ("img2", Need::Gpu), ("a2", t("a"))], after).await;
        assert_eq!(o, ["a1", "a2", "img1", "img2"]);
        // more than BATCH chats for the loaded model: the image at the head gets its turn after BATCH of them
        q.set_resident(Resident::Text("a".into()));
        let mut jobs = vec![("img", Need::Gpu)];
        let names = ["c1", "c2", "c3", "c4", "c5", "c6"];
        jobs.extend(names.iter().map(|n| (*n, t("a"))));
        let o = order(q.clone(), jobs, after).await;
        assert_eq!(o, ["c1", "c2", "c3", "c4", "img", "c5", "c6"]);
    }

    #[tokio::test]
    async fn a_head_that_waited_long_enough_is_not_jumped() {
        // with the wait already past (0 s), nothing goes ahead of the image at the head: strictly in order of arrival
        let q = Queue::with_batch_wait(None, Duration::ZERO);
        q.set_resident(Resident::Text("a".into()));
        let t = |m: &str| Need::Text(m.into());
        let o = order(q.clone(), vec![("img", Need::Gpu), ("c1", t("a")), ("c2", t("a"))], after).await;
        assert_eq!(o, ["img", "c1", "c2"]);
    }

    #[tokio::test]
    async fn a_job_whose_client_left_leaves_the_queue_and_a_long_wait_is_refused() {
        let q = Queue::new(None);
        let p = q.enter(Need::Any, "run", "", |_| {}).await.unwrap();
        let q2 = q.clone();
        let h = tokio::spawn(async move { q2.enter(Need::Gpu, "img", "", |_| {}).await.map(|_| ()) });
        while q.waiting() == 0 {
            tokio::task::yield_now().await;
        }
        h.abort();
        let _ = h.await;
        assert_eq!(q.waiting(), 0, "the aborted job left");
        let q3 = Queue::new(Some(Duration::from_millis(50)));
        let _p3 = q3.enter(Need::Any, "run", "", |_| {}).await.unwrap();
        let mut seen = 0;
        let e = q3.enter(Need::Gpu, "img", "", |p| seen = p).await.err().unwrap();
        assert_eq!((e.position, seen), (1, 1));
        drop(p);
        // a full queue refuses at once
        let q4 = Queue::new(None);
        let _run = q4.enter(Need::Any, "run", "", |_| {}).await.unwrap();
        let mut hs = Vec::new();
        for _ in 0..MAX_WAITING {
            let q = q4.clone();
            hs.push(tokio::spawn(async move { q.enter(Need::Gpu, "img", "", |_| {}).await.map(|_| ()) }));
        }
        while q4.waiting() < MAX_WAITING {
            tokio::task::yield_now().await;
        }
        let e = q4.enter(Need::Gpu, "img", "", |_| {}).await.err().unwrap();
        assert!(e.full && e.to_string().contains("already waiting"), "{e}");
        hs.iter().for_each(|h| h.abort());
    }
}
