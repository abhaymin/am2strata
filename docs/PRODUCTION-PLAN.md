# Production: no crash, no silent failure, overload told plainly

Owner's requirement (2026-11-10): am2Strata handles errors gracefully, manages memory and model preferences without
crashing, warns the user plainly when the PC is overloaded, and does not fail during delivery.

## What the code does today (audit, 2026-11-10)

| Area | Today | Gap |
| --- | --- | --- |
| A model that fails to load | the previous model is started again (`manager::load_locked`) | the error says what failed, not what to do |
| The chat engine dies mid-run (killed for memory, a crash) | the request gets "the engine stopped unexpectedly" | **the server keeps the dead engine: every later request fails until a manual reload** |
| A handler panics | the connection drops | no JSON error; a stream just ends |
| Memory before a load or a job | CPU speech waits up to 10 minutes for free RAM; media capabilities say what fits | a text model or a GPU job starts without checking free RAM, so the kernel may kill something |
| A full queue | jobs wait without limit (`queue_max_wait_s` is off by default); a waiting stream gets its position | no limit on how many wait; no `Retry-After`; the web app shows no overload warning |
| Model preferences | the request's `model` (with `switch_on_request`), else the loaded one | no fallback order when the asked model cannot load or does not fit |
| Delivery | media jobs a chat queues are kept and fetched by id | a long synchronous media request (song, video) is lost if the client drops; nothing is kept on disk across a restart |

## The work, in order

**State (2026-11-10):**
- **Steps 1 and 2 are built.**
  - A dead or out-of-step engine is dropped, and the model is loaded again for the next request. The failed request
    hears why and to send it again, and the alerts say so for 15 minutes. Tested: `a_dead_engine_is_replaced_for_the_next_request`.
  - A handler panic is answered as a JSON 500, and a stream that stops without its end sends an error event (the
    panic layer has no integration test: no handler can be made to panic from outside).
- **Steps 3 and 4 are built in part.**
  - What each model takes from free memory at its load is kept (`<data>/model-memory.json`).
  - A load that cannot fit is refused before the model in memory is stopped. A load that is short waits up to
    60 s for memory, then is refused with a plain message.
  - Every 503 and 429 gets `Retry-After`. API answers carry `x-strata-warning` while the PC is short of memory
    (under 3 GB or 5% free) or four or more requests wait, and the alerts say so.
  - Tested: `a_model_that_would_not_fit_is_refused_plainly` and `low_memory_is_told_with_every_answer`.
  - Still to do: GPU media jobs checked against free RAM (CPU speech already waits); a banner at the top of the
    web app (today the alerts show in the Monitor's hardware card).

1. **A dead engine is noticed and replaced.**
   - The manager checks the engine process before each request (and when a request fails with "stopped").
   - A dead engine is dropped and the model loaded again for the next request, with one log line and a Monitor note.
   - The request that was running gets an error event that says so and to try again.
2. **A panic is an error, never a dropped connection.** A panic layer answers JSON 500 (in the API's own error
   shape), and a streamed answer ends with an error event.
3. **Memory guard.**
   - Before a text model loads, or a GPU or CPU media job starts, free RAM is compared with what it needs (the
     model's size, measured job sizes).
   - Short of memory, the job waits for the memory (as CPU speech does), up to a limit, while other jobs run.
   - Past that limit it is refused with 503, `Retry-After`, and how much is free and how much is needed.
   - Never an out-of-memory kill.
4. **Overload told plainly.**
   - The queue gets limits: `queue_max_waiting` (default 32), and the wait limit as today.
   - Past a limit: 429 or 503 with `Retry-After` and the queue's length.
   - Every answer made under pressure carries `x-strata-warning` (low memory, a long queue, a fallback model).
   - The web app shows a banner from `/metrics` while the PC is short of memory or the queue is long.
5. **Model preferences.**
   - `"model_preferences": ["tiel", "orca", ...]` in the run config: the order to try when the asked model cannot
     load (missing, does not fit, failed).
   - The answer says which model answered (`x-strata-warning`, and the Monitor row).
6. **Delivery that survives.**
   - Synchronous song, video and image requests run as jobs; a client that drops can fetch the result by its id.
   - The job records and results are kept on disk (`<data>/media-jobs/`), so a restart keeps them for a day.
   - A streamed answer that the engine cuts ends with a clear error event, never silently.

## Tests

Each step gets a mock test:
- the mock engine exits mid-answer, and the next request is answered;
- a handler panic gives a JSON 500;
- a low-memory test hook (`AM2_TEST_MEM_GB`, as the speech wait uses) makes a load wait, then refuse with
  `Retry-After`;
- a full queue answers 429;
- an unloadable asked model falls back with the warning header;
- a dropped music request's result is fetched afterwards.

## 7. Request routing and quick switching (owner, 2026-11-10: strong, performance oriented, predictable, smart switching)

**Built (2026-11-10).** The rules are short and fixed, applied in this order; the same queue always runs in the same
order:

1. One job runs at a time, taken from the best class waiting:
   - high: a person at the web app;
   - normal: API calls;
   - low: media jobs a chat queued.

   `x-strata-priority` asks for a class; a client profile's key may ask normal or low only.
2. Every 60 s of waiting raises a job one class, so a low job waits at most 2 minutes behind newer high ones.
3. Within the class, jobs run in arrival order, with one exception that saves a model switch: when the first job
   needs another model, the jobs for the model already loaded go first.
   - at most 4 of them in a row;
   - none once the first job has waited 90 s.
4. Of two clients with equal jobs, the one that did not just run goes first.
5. Smart switching:
   - **Warm files:** while a job waits for a model that is not loaded, its GGUF files are read into the file cache,
     when free memory has room for them, so the switch loads from RAM.
   - **Learnt times:** each model's load time is learnt and kept (`<data>/model-load-s.json`), as is each kind of
     job's running time.
   - **Estimates:** `/api/queue` gives each waiting job its class, client and estimated wait, and a waiting stream
     hears "queued: N ahead, about S s".

Tested:
- the order of the classes, client turns, aging and the estimates (unit tests);
- a person at the web app going before a background API call (`a_person_at_the_web_app_goes_before_background_calls`).

Found and fixed on the way: the queue told a job its place while holding its own lock, so a listener that asked the
queue anything deadlocked it. The media path no longer holds the model's lock while the model stops.
