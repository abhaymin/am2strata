# The regression suite

What keeps am2Strata from breaking something that worked: a contract for every route, the shapes the
compatibility APIs answer in, one test for every class of bug fixed before, and a checklist to run against
the deployed server after an update.

**The rule: every bug fix adds a regression test.** A fix without one is not finished. Where the test goes
is in [Adding to it](#adding-to-it).

## What it covers

| Part | File | What fails it |
|---|---|---|
| Route contracts | `crates/am2strata/tests/regression_api.rs` | a route that answers a caller differently from its row; a new route with no row |
| Golden shapes | `crates/am2strata/tests/regression_shapes.rs`, `tests/golden/*.json` | a field of an OpenAI, Anthropic, Responses or Gemini answer renamed, dropped, added or retyped |
| Bug regressions | `crates/am2strata/tests/regression_bugs.rs` (and `regress_…` unit tests beside the code) | a bug fixed before coming back |
| Live runner | `am2strata-run regression` (`src/regression_run.rs`), tested by `tests/regression_runner.rs` | a running server that no longer answers as it should |

All of it runs without a GPU or a model: the tests start the real server binary with a stand-in engine (the
harness in `tests/common/`), each test its own server on its own port in its own temp folder.

### Route contracts

`ROUTES` in `regression_api.rs` has one row for every method of every route the server registers (138 rows
for 117 paths today). A row says who may call the route (no key / a client key / the owner's key only / its
own proof) and the status each caller gets:

1. no key
2. a wrong key of the right length (as `Authorization: Bearer` and as `x-api-key`)
3. a client profile's key
4. the owner's key with a minimal valid body
5. the owner's key with a broken body (never a 5xx; a 400 is JSON with `error.message`)
6. the owner's key from another site's page (a foreign `Origin`): **403 on every state-changing owner route**

Besides the walk through the table:

- **The guard** (`every_route_in_main_rs_has_a_contract_row`) reads `main.rs`'s `.route("…")` calls and the
  sub-routers it merges, and fails when a route has no row or a row has no route. A router built in a way the
  guard cannot read (a new `.merge(`, a `.nest(`, a fallback) fails it too, with what to do.
- **Wrong-typed JSON** (`wrong_typed_json_is_refused_never_a_500`): every route that takes a body is sent
  arrays, strings, numbers and objects with every usual field of the wrong type. A 5xx fails.
- **A stranger with a broken body** gets 401, not a parse error: the key is looked at before the body.
- Security headers on every answer; 404 and 405 for unknown paths and methods.

"Minimal valid body" never reaches outside the test: where a real call would download from Hugging Face,
start a program, delete a model or start the sandbox VM, the row names something that does not exist and
expects the refusal that proves the key was accepted and the body read. The row's note says so.

### Golden shapes

For each compatibility API the test records the SHAPE of the answer: its keys and the type of each value,
with a few naming fields (`object`, `type`, `role`, `finish_reason`, `stop_reason`, `finishReason`,
`status`) kept by value. Ids, timestamps, texts, token counts and timings are not part of a shape. A stream's
shape is, for each event name, the set of distinct shapes its data took.

Covered: `/v1/models`, `/v1/chat/completions` (whole and streamed, with tool calls, the usage block),
`/v1/messages` (whole, streamed, without thinking, tool use, `count_tokens`), `/v1/responses` (whole,
streamed, function calls), Gemini (`generateContent`, `streamGenerateContent`, `countTokens`, the model
list, function calls), the error bodies of each family (no key, wrong key, wrong request, broken JSON), and
`/health`. This server has no `/v1/completions` and no `/v1/embeddings`; a test pins that they are 404, so
adding one asks for its golden file.

### Bug regressions

One named test per class of bug, each with the pull request it came from:

| Bug class | From | Test |
|---|---|---|
| Workshop paths: `..`, absolute paths, outputs outside | PR #13 | `pr13_workshop_paths_cannot_traverse_out_of_their_folders` |
| Workshop symlinks out of its folders (inputs, outputs, datasets) | PR #13 | `pr13_workshop_symlinks_cannot_escape_their_folders` |
| Upload names placing a file elsewhere | PR #16 | `pr16_workshop_upload_names_cannot_place_a_file_elsewhere` |
| An id in the path read before the key (a stranger must get 401) | PR #13 | `pr13_an_id_in_the_path_is_read_only_after_the_key` |
| Processes orphaned by a cancelled job (the process group is killed) | PR #13 | `pr13_a_cancelled_job_leaves_no_orphaned_processes` |
| Open redirect through a forged `Host` | PR #12 | `pr12_the_https_redirect_is_never_an_open_redirect` |
| `strip_port` panicking on a malformed `Host` | PR #12 | `pr12_a_malformed_host_never_panics_the_server` |
| HSTS over plain HTTP; a pin while HTTP is kept | PR #12, #17 | `pr12_pr17_hsts_only_over_tls_and_only_pinned_when_http_redirects` |
| A refused certificate leaving a file behind | PR #12 | `pr12_a_refused_certificate_leaves_no_file_and_https_off` |
| The desktop WebSocket without the key (401, 426, 403, no frame) | PR #15 | `pr15_the_desktop_stream_needs_the_key_before_anything` |
| A client key on owner routes (Workshop, Computer, VM, keys, settings) | PR #13, #15 | `pr13_pr15_a_client_key_is_refused_on_every_owner_route` |
| A generated key acting as the owner; a revoked key still working | PR #19 | `pr19_a_generated_key_is_a_client_key_and_a_revoked_one_opens_nothing` |
| DNS rebinding (a `Host` this server does not answer to) | cf65b1a | `security_a_rebound_host_is_refused_before_anything_else` |
| Numbers from requests trusted instead of clamped | PR #16, dae5323 | `numbers_from_requests_are_clamped_not_trusted` |
| Tests changing the process environment | PR #14 | `pr14_no_test_changes_the_process_environment` |
| Tests writing into the repository | PR #15 | `pr15_the_golden_files_are_not_rewritten_by_a_test_run` |
| A JSON body that is no object panicking the handler (500) | this suite | `a_json_body_that_is_no_object_is_a_400_not_a_panic` |
| The body read before the key; refusals in plain text | this suite | `an_unreadable_body_is_401_for_a_stranger_and_a_json_400_for_the_owner` |
| Owner routes open to another site's page | this suite | `state_changing_owner_routes_refuse_another_sites_page` |

Regressions beside the code they guard (unit tests, since they test private functions):

| Bug class | From | Test |
|---|---|---|
| `chunk` looping forever (overlap ≥ `max_chars`; a boundary in the first chunk) | PR #13 | `strata-workshop` `dataset::tests::regress_pr13_chunk_never_loops_whatever_the_overlap`, `…_moves_on_after_a_boundary_in_the_first_chunk` |
| MinHash near-duplicates compared inexactly | PR #13 | `dataset::tests::regress_pr13_near_duplicates_compare_hashes_exactly` |
| A chunk longer than `max_chars` (whitespace counted as one character) | this suite | `dataset::tests::regress_chunk_counts_the_whitespace_it_keeps` |
| The VM's screen client accepting an oversized screen | PR #18 | `am2strata` `computer::vm::rfb::tests::regress_pr18_an_oversized_screen_is_refused_at_the_handshake`, `…_an_oversized_resize_or_rectangle_stops_the_reader` |
| Two servers of one account sharing the VM's control socket | this suite | `computer::vm::tests::regress_each_data_folder_has_its_own_socket_folder` |

The byte-for-byte gates against llama.cpp (quantize, convert) are not repeated: they are
`strata-workshop/tests/quantize_pure.rs`, `convert_reference.rs`, `convert_f16.rs` and
`strata-pack/tests/quantize.rs`. `the_fixture_gates_and_unit_regressions_still_exist` fails if one of them,
or one of the unit regressions above, is deleted or renamed.

### The live runner

`am2strata-run regression` checks a server that is RUNNING: the one you deployed, after an update.

```
am2strata-run regression [--url http://host:8000] [--api-key KEY] [--quick | --full] [--json]
```

- The address and key default to this PC's own server (from its run configs), or `AM2STRATA_URL` and
  `AM2STRATA_API_KEY`. A key from this PC's run configs is never sent to another address. **The key is never
  printed**: not in the table, not in `--json`, not in an error.
- `--quick` (the default) **changes nothing** on the server and makes **at most 3 short model calls**: a
  chat, a streamed chat and an Anthropic message. The other APIs are checked through calls that generate
  nothing (token counts, lists, the stateless-Responses 404).
- `--full` adds a real call to the Responses and Gemini APIs and a tool round trip (a forced tool call, then
  its result sent back): about 8 short model calls. It still changes no setting and starts no job.
- The sandbox VM is asked for its status only. It is never started, stopped or changed.
- Exit code 0 when nothing failed; non-zero on any failure. A row is `skip` when the server has no such
  feature (the VM on macOS, a route an older server lacks).

Rows: health; the API key is needed; models list; chat; chat, streamed; messages; count_tokens; Responses
(stateless, and with `--full` a call); Gemini (lists and countTokens, and with `--full` generateContent);
tool round trip (`--full`); errors; metrics; queue; status and system; web app; HTTPS and HSTS (when TLS is
on: the HTTPS port answers and carries HSTS; plain HTTP never does); Workshop files; Computer status; VM
status; usage; media, MCP and memory (`--full`).

## Running it

Locally (the PC may be running a model: each test starts a small server, so keep the thread count down):

```
deploy/regression.sh                       # the regression tests
cargo test -p am2strata --test regression_api --test regression_shapes \
           --test regression_bugs --test regression_runner -- --test-threads=4
cargo test -p am2strata --bin am2strata regress      # unit regressions in the server
cargo test -p strata-workshop --lib regress          # unit regressions in the Workshop
```

After a deployment, against the running server:

```
export AM2STRATA_API_KEY=...               # or let it read this PC's run configs
deploy/regression.sh --url http://127.0.0.1:8000 --live-only            # quick: 3 model calls, no changes
deploy/regression.sh --url http://127.0.0.1:8000 --live-only --full     # every API, tool round trip
am2strata-run regression --json            # this PC's server, as JSON (for a script)
```

Run `--quick` after every update; run `--full` when the update touched an API, the tools or the engine.
Both wait their turn in the queue like any client, so a busy server only makes them slower.

CI runs the suite as the `regression` job of `.github/workflows/ci.yml` on every pull request, and checks
that the tests left the repository as it was.

## Adding to it

### A route

Adding a route to `main.rs` fails the guard until `ROUTES` has its row. Put it where the route is in
`main.rs`'s order:

```rust
row("POST", "/api/things/{id}/rename", Owner).at("/api/things/nope/rename")
    .json(r#"{"name": "x"}"#).ok(404).note("no such thing: nothing is renamed"),
```

- `Public`, `Client` (the model APIs: owner or client key), `Owner` (the owner's key only), or `Special` with
  `.statuses(…)` and a note saying what its proof is.
- `.at(…)` fills in path parameters; `.json(…)` or `.raw(…)` is the minimal body; `.ok(status)` is what
  everyone the route lets in gets (200 if left out); `.client(…)` and `.malformed(…)` say otherwise for those
  two callers.
- A state-changing `Owner` route must refuse a foreign `Origin` (call `crate::foreign_origin` after the key
  check, as the others do); the table does not let a row say otherwise.
- Check the key before reading anything of the request. A handler may take `Json<Value>`: `main.rs`'s
  `body_rejections` makes sure a stranger with a broken body still gets 401.
- A sub-router merged into the app is listed in `MERGED` in `tests/common/regress.rs`, so its routes are read.

### A golden file

Add the call and `check_golden("name", &named_shape(&answer))` (or `stream_shape`) to
`regression_shapes.rs`, then make the file:

```
deploy/regression.sh --update-golden
# or: UPDATE_GOLDEN=1 cargo test -p am2strata --test regression_shapes -- --nocapture
```

This writes every shape as it is now into a **temp folder** (printed; `GOLDEN_OUT` names another) and never
into the repository: tests never write there, and a folder inside the repository is refused. Look at the
difference, then copy the files you mean into `crates/am2strata/tests/golden/` and commit them.

Refresh a golden file the same way when a shape changes on purpose. When a shape test fails and the change
was not meant, the server is wrong, not the file: a client somewhere parses that field.

Keep machine-dependent lists (this PC's GPUs) out of a shape, as the `/health` test does.

### A bug regression

When you fix a bug, add its test in the same pull request:

- a bug reachable over HTTP: a test in `regression_bugs.rs`, named for the pull request and the behaviour
  (`pr21_…`), with a comment saying what was wrong and the commit;
- a bug in a private function or a library crate: a unit test named `regress_…` beside the fix, and its name
  added to the list in `the_fixture_gates_and_unit_regressions_still_exist` and to the tables above.

The test must fail without the fix. Write it first, see it fail, then fix.

### A live check

Add a function to `CHECKS` in `src/regression_run.rs`. A check that makes the model write goes through
`Runner::model` (it is counted; `--quick` allows three, and the runner fails itself past that), and is
`--full` only unless it replaces one of the three. No check may change the server's state; the runner's test
compares the server's settings, config, memory, jobs, keys and loaded model before and after.

## The tests' own rules

- Servers only through the harness (`tests/common/`): it starts each in its own process group and stops the
  whole group, so nothing is left running.
- Never `std::env::set_var` in a test: pass environment to the server with the harness's `extra.env`.
- Tests never write into the repository: temp folders (`regress::TempDir`), removed afterwards.
- No fixed sleeps: poll with a deadline (`regress::wait_for`).
- Never against the installed server: a test's server has its own port, data folder and (for the sandbox VM)
  its own socket folder. Only `am2strata-run regression`, run by you, talks to the real one.
