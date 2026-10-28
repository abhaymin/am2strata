//! One queue for everything that needs the GPU's turn (the server runs one request at a time): chat requests, model
//! switches, media jobs. In arrival order, with one exception that saves a reload: when the next job needs a different
//! resident model (a text model, or the GPU for a media job with the text model stopped), the jobs that need the one
//! already loaded go first - up to BATCH of them in a row, so nothing waits behind them forever.
//!
//! A job waits as long as it must (no two-minute "busy"); a client that leaves takes its job out (the waiting future
//! is dropped); `queue_max_wait_s` in the run config turns an over-long wait into a refusal with the position.

use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

/// The most jobs taken out of order for the model already loaded, before the head of the queue gets its turn.
pub const BATCH: usize = 4;

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
}

struct State {
    queue: VecDeque<Ticket>,
    running: Option<(u64, Need, String, String, Instant)>,
    resident: Resident,
    /// jobs taken out of order in a row (for the resident model)
    batched: usize,
    next: u64,
}

pub struct Queue {
    state: Mutex<State>,
    changed: Notify,
    pub max_wait: Option<Duration>,
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
            s.running = None;
        }
        drop(s);
        self.q.changed.notify_waiters();
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
        }
    }
}

#[derive(Debug)]
pub struct TooLong {
    pub position: usize,
    pub waited_s: u64,
}

impl Queue {
    pub fn new(max_wait: Option<Duration>) -> Arc<Queue> {
        Arc::new(Queue {
            state: Mutex::new(State { queue: VecDeque::new(), running: None, resident: Resident::None, batched: 0, next: 1 }),
            changed: Notify::new(),
            max_wait,
        })
    }

    /// The job that goes next, if the turn is free.
    fn choose(s: &State) -> Option<u64> {
        if s.running.is_some() {
            return None;
        }
        let head = s.queue.front()?;
        if !fits(&head.need, &s.resident) && s.batched < BATCH {
            if let Some(t) = s.queue.iter().find(|t| fits(&t.need, &s.resident)) {
                return Some(t.id);
            }
        }
        Some(head.id)
    }

    /// Waits for the turn. `on_position` hears the job's place (1 = next) each time it changes.
    pub async fn enter(self: &Arc<Self>, need: Need, kind: &str, label: &str, mut on_position: impl FnMut(usize))
                       -> Result<Pass, TooLong> {
        let id = {
            let mut s = self.state.lock().unwrap();
            let id = s.next;
            s.next += 1;
            s.queue.push_back(Ticket { id, need, kind: kind.to_string(), label: label.to_string(), since: Instant::now() });
            id
        };
        let mut w = Waiting { q: self, id, done: false };
        let since = Instant::now();
        let mut told = 0usize;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut s = self.state.lock().unwrap();
                if Self::choose(&s) == Some(id) {
                    let pos = s.queue.iter().position(|t| t.id == id).unwrap();
                    let t = s.queue.remove(pos).unwrap();
                    // out of order for the resident model counts toward the bound; the head's turn resets it
                    if pos > 0 { s.batched += 1 } else { s.batched = 0 }
                    s.running = Some((id, t.need, t.kind, t.label, Instant::now()));
                    w.done = true;
                    return Ok(Pass { q: self.clone(), id });
                }
                let position = s.queue.iter().position(|t| t.id == id).map(|p| p + 1).unwrap_or(0);
                if position != told {
                    told = position;
                    on_position(position);
                }
                if let Some(max) = self.max_wait {
                    if since.elapsed() >= max {
                        return Err(TooLong { position, waited_s: since.elapsed().as_secs() });
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
        let waiting: Vec<Value> = s.queue.iter().enumerate().map(|(i, t)| json!({"position": i + 1, "id": t.id, "kind": t.kind,
            "label": t.label, "need": need(&t.need), "waiting_s": t.since.elapsed().as_secs()})).collect();
        let resident = match &s.resident {
            Resident::None => json!(null),
            Resident::Text(m) => json!({"text_model": m}),
            Resident::Gpu => json!("gpu"),
        };
        json!({"running": running, "waiting": waiting, "resident": resident, "batched": s.batched, "batch_bound": BATCH})
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
    }
}
