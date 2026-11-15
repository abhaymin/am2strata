# Browser use

am2Strata drives a real web browser for a chat, an API client, an MCP client or a coding agent. The server adds the
browser tools to a request and runs them itself, the same way it runs the image, video and audio tools. Nothing has to
be installed or configured on the caller's side, and the steps, page text and screenshots come back in the answer.

## What it does

| Job | Tools |
|---|---|
| Navigate | `browser_open`, `browser_click` (text, point or element number), `browser_back`, `browser_forward`, `browser_tabs`, `browser_wait`, `browser_scroll` |
| Read | `browser_read` (the page's text), `browser_elements` (links, buttons and fields, numbered `e1`, `e2` …) |
| Forms and surveys | `browser_fill` (many fields at once, by label, name, placeholder, number or CSS: text, selects, checkboxes, radio groups, dates), `browser_type`, `browser_select`, `browser_upload` |
| Scraping and cataloging | `browser_extract` (tables, repeated items, links, page metadata and schema.org JSON-LD; follows "next" for up to 50 pages), `browser_save` |
| Maps and canvases | `browser_mouse` (drag to pan, wheel to zoom, hover, double and right click), `browser_key` (keys and combinations) |
| Testing sites | `browser_check` (text, elements, title, address, HTTP status, console errors, failed requests, load time, broken links), `browser_console` |
| Pictures | `browser_screenshot` (the whole page, or one element) |
| Scripts | `browser_script` (off unless `"browser": {"allow_script": true}`; it asks every time) |

`browser_fill` never sends a form. Sending it is a click on its submit button, and that asks first.

## Which browser

Pick one per request with `"strata_browser"`:

| Value | The browser | Logins |
|---|---|---|
| `"server"` | a headless Chrome on the server, with a fresh profile; public addresses only | none |
| `"mine"` | your own Chrome, started with `--remote-debugging-port=9222 --user-data-dir=<a folder>` | yours |
| `"sandbox"` | Chromium on the sandbox VM's desktop (computer use). Watch it and take over on the Computer tab | the VM's own |
| `true` | the run config's choice (`"browser": {"connect": …}` set means `mine`, otherwise `server`) | |

## Risky steps ask first

These steps wait for your yes:
- typing into a password field;
- sending a form;
- a click whose control says publish, post, send, submit, upload, save, delete, share, buy, pay or a similar word;
- an upload or a download;
- a script.

The web app shows Allow / Deny in the chat. For an MCP or API job, they show on the Browser tab under "Waiting for you"
for 3 minutes. `"browser": {"approve": "job"}` asks once per job and site. `"auto_approve": true` never asks; use it only
for a browser you trust with everything.

## Datasets

`browser_extract {…, "dataset": "books", "key": ["url"]}` and `browser_save` keep records in `<data>/datasets/`. A key
makes a dataset a catalog: a record found again replaces the old one instead of adding a copy. View, download (CSV,
JSON, JSONL) and delete them on the Browser tab, with `am2strata-run browser datasets` / `dataset <name>`, or over
`GET /api/datasets/<name>?format=csv`. CSV cells that start with `=`, `+`, `-` or `@` are written so a spreadsheet does
not run them as formulas.

## Site tests

A test is a list of steps and checks that runs the same way every time, with no model:

```json
{"base": "https://shop.example", "steps": [
  {"open": "/"}, {"expect": [{"status": 200}, {"text": "Welcome"}, {"console_errors": 0}, {"broken_links": 30}]},
  {"fill": [{"field": "Search", "value": "lamp"}]}, {"key": "Enter"}, {"wait": "results"},
  {"extract": {"kind": "items", "selector": ".product", "fields": {"name": "h3", "price": ".price"}}, "rows": {"min": 1}},
  {"click": "Next"}, {"expect": [{"url": "page=2"}]}]}
```

Run tests on the Browser tab (Tests), or with `am2strata-run browser test site.json --junit report.xml` in CI, or with
`POST /api/browser/test` (`?format=junit`). A failed step keeps a screenshot of the page.

## Connect an agent (MCP)

The browser tools are an MCP server at `/v1/mcp/browser` (Streamable HTTP; `?target=server|mine|sandbox`):

```bash
claude mcp add --transport http am2strata-browser http://<server>:8000/v1/mcp/browser --header "Authorization: Bearer <key>"
```

Use the owner's key, or a client key whose profile includes the `browser` tools.

## From a terminal

```bash
am2strata-run browser open https://example.com
am2strata-run browser elements
am2strata-run browser click e3
am2strata-run browser fill '[{"field": "Email", "value": "me@example.com"}]'
am2strata-run browser extract --kind table --dataset prices --key name
am2strata-run browser check '[{"status": 200}, {"broken_links": 30}]'
am2strata-run browser shot page.jpg --full
am2strata-run browser dataset prices --format csv --out prices.csv
```

## How it is tested

- `browser_live_tests.rs` drives a real headless Chrome over local test pages in every CI run: forms, merged-header
  tables across pages, keyed datasets, map drag, wheel and keys, the site checks, the console log, broken links,
  screenshots, scripts, and a whole site test with its JUnit report.
- `tests/regression_api.rs` holds a contract row for every browser route: who may call it, and what each caller gets,
  including a key from another site's page.
