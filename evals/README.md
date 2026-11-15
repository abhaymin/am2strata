# Agent evals

Real jobs a person asks of an agent, with checkers written and **locked before any run** (docs/GAPS-PLAN.md, A18). The
idea comes from a field test of small expert-model teams: the result counts only if the rules could not move after the
fact. A job passes only on what it leaves behind: a page that works in a browser, a dataset with the right rows, a
service that answers, an answer with the right facts. What the model says about its own success does not count.

```bash
am2strata-run eval list              # the jobs, and any that changed since the lock
am2strata-run eval selftest          # every checker passes on its known-good answer (run after changing a checker)
am2strata-run eval lock              # after reviewing the jobs: their checkers are fixed from here
am2strata-run eval run --tries 3     # the loaded model, or --model <id>; --jobs kanban,form-httpbin
```

| Job | Check | Needs |
|---|---|---|
| kanban | the answer's HTML in a browser: columns, add a card, no console errors | |
| threejs-pagoda | the answer's HTML in a browser: a visible canvas, no errors, nothing failed to load | |
| research-jakarta | the answer has the exact number from a Wikipedia table | `strata_web` |
| form-httpbin | the answer has what httpbin echoed after the form was sent | the browser |
| scrape-cities | a dataset `eval-cities` with 70+ rows and city and country columns | the browser |
| fix-nginx | nginx in the sandbox VM is running and serves its page | the VM and computer use |

- `jobs/*.json`: the jobs. `known-good/*.md`: correct answers that prove each checker. `LOCK.json`: the SHA-256 of each
  job's prompt, request, setup and checker. A changed job is refused until it is reviewed and locked again
  (`--unlocked` runs it as a trial, and the report says so).
- Reports go to `results/<time>-<model>.json`: per try, pass, score, seconds, tokens and tool calls.
- The runner allows the risky steps a job needs (it is the operator of a test), and prints each one.
