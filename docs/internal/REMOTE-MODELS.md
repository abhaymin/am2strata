# Remote models (internal)

Not linked from the README, DEPLOYMENT.md, `--help`, the web app or the release notes. The code is in the public
repository (`crates/am2strata/src/remote/`), so it can be found by reading it; it is not advertised.

am2Strata can send a request to an OpenAI-compatible or Anthropic-compatible provider instead of a local model, and
can pick between local and remote models per request (the "mix"). It is **off** unless the owner's run config has a
`"remote"` block. With no block:

- no model list changes (`/v1/models`, `/api/models`), no request is routed differently (a name like `openai/x` is
  just a name, answered by the loaded model as before), the web app shows nothing new;
- the only thing present is the owner's page and its API (below), so the owner can turn the feature on.

## The page: `/rtenai`

`GET /rtenai` (http and https alike) is the settings page. It is linked from nowhere. It loads without a key, asks
for the server's own API key (the same one the web app keeps in this browser), and shows nothing about providers
until `/api/rtenai` accepts it. Client profile keys and generated keys get 401. From it the owner can:

- turn remote models on (writes an empty `"remote"` block) or off (removes the block);
- add, edit and remove providers (kind, base URL, models, `max_concurrent`, kinds, a key reference);
- set or forget a provider's key (kept in `<data>/remote-keys.json`, mode 0600; never in the run config);
- Test a provider (the server asks it for its model list, now);
- edit the mix rules as an ordered list, and named mixes;
- read the tokens per day per remote model, from the usage counts (no prices).

Saving writes the `"remote"` block of the default model's run config (the earlier file is kept as `.bak`) and is
used at once; no restart.

Its API, all owner-only (writes refused from another site's page):

| Route | What |
| --- | --- |
| `GET /api/rtenai` | on or off; providers with `key: {set, source, last4, error, ref}`; the mix; the block |
| `POST /api/rtenai` | `{"remote": {...}}` writes the block (turns it on); `{"remote": null}` removes it |
| `POST /api/rtenai/keys/{provider}` | `{"key": "..."}` keeps the key in remote-keys.json |
| `DELETE /api/rtenai/keys/{provider}` | forgets the key set there |
| `POST /api/rtenai/test/{provider}` | lists the provider's models now (calls the provider) |
| `GET /api/rtenai/models/{provider}` | the models offered (config list, or the provider's list kept 10 min) |
| `GET /api/rtenai/usage?days=30` | per provider: totals and tokens per day per model |

These routes are in the route contract table (`tests/regression_api.rs`), marked "intentionally unlinked".

## Config reference

In the default model's run config (the one am2Strata reads its keys, clients and MCP servers from):

```json
"remote": {
  "providers": {
    "openai":    {"kind": "openai",    "base_url": "https://api.openai.com/v1",  "key": "env:OPENAI_API_KEY",    "models": ["gpt-5.5", "gpt-5.5-mini"]},
    "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com",  "key": "env:ANTHROPIC_API_KEY", "models": ["claude-sonnet-5-5"]},
    "groq":      {"kind": "openai",    "base_url": "https://api.groq.com/openai/v1", "key": "file:~/.config/am2strata/groq.key", "models": []}
  },
  "mix": {"default": "local", "routes": [
    {"when": {"tokens_over": 24000}, "use": "anthropic/claude-sonnet-5-5"},
    {"when": {"has_images": true},   "use": "openai/gpt-5.5"},
    {"when": {"local_busy": true},   "use": "groq/llama-4-70b"}
  ]},
  "mixes": {"code": {"default": "openai/gpt-5.5", "routes": []}}
}
```

Provider fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `kind` | (required) | `openai`: any OpenAI Chat Completions endpoint (OpenAI, Groq, Together, OpenRouter, Mistral, DeepSeek, vLLM, Ollama...). `anthropic`: the Messages API |
| `base_url` | (required) | `openai`: up to and with `/v1` (`{base}/chat/completions`, `{base}/models`, `{base}/images/generations`). `anthropic`: the host (`{base}/v1/messages`, `{base}/v1/models`) |
| `key` | none | `env:NAME`, `file:PATH` (refused when group- or world-readable: `chmod 600`), or set from the page. A key set from the page wins. A plain key written by hand into the config works but is never shown by the API |
| `models` | `[]` | the models offered; `[]` asks the provider (`GET /models`), kept 10 minutes (a failure 1 minute). With a list, other names are refused (404) |
| `max_concurrent` | 4 | requests at once to this provider; more wait their turn |
| `kinds` | `["chat"]` | `chat` and/or `image` (OpenAI kind only: `/v1/images/generations` is sent on) |
| `context` | 128000 | the context the server's tool loop keeps a conversation within (older tool results shortened) |
| `max_tokens` | 8192 | Anthropic kind: max_tokens when the client sends none (the API needs one) |
| `timeout_s` | 600 | the longest one call may take |
| `max_tokens_field` | `max_completion_tokens` for api.openai.com, else `max_tokens` | the name max tokens is sent under |
| `reasoning_effort` | true for api.openai.com, else false | send the client's reasoning effort as `reasoning_effort` |

Provider names: letters, digits, `-`, `_`; not `mix`, `local`, `remote`, `default`.

A model is named `<provider>/<model>` (`openai/gpt-5.5`) in requests, `/v1/models` (`"remote": true, "provider"`),
the picker and the usage counts.

## Who may use them

- The server's own key: every remote model and mix.
- A generated key (`/api/keys`, `am2strata-run keys`): only when its `models` list names the model, `<provider>/*` or
  `remote/*`; a mix when it names `mix`, `mix/<name>` or `mix/*`. A key with no list ("every model") does **not** get
  remote models: they cost the owner money.
- A run-config client profile: the same, from a `"models"` list in its entry (`"clients": {"laptop": {"key": ...,
  "models": ["remote/*"]}}`). This list is read for remote models and mixes only; local models stay open to profiles.
- A cluster master's key: never (403).

Anything else does not see remote models in `/v1/models` and gets 403 when it names one. The key's day limits
(requests, tokens) apply as for local models; usage is counted per key and per model (`openai/gpt-5.5`).

The web app shows remote models in the model picker (grouped by provider, after the local models) only to the owner.
Choosing one sends the chat to it; the local model stays loaded.

## The mix

A client asks for the model `mix` (or `mix/<name>`). Rules are checked in order; the first whose conditions all hold
gives the model, else `default`. Conditions:

| Condition | Holds when |
| --- | --- |
| `tokens_over: N` | the request is longer than N tokens (estimated: 4 characters a token, pictures not counted, tool schemas counted) |
| `has_images: true/false` | the request has a picture (any API's shape) |
| `tools: true/false` | the request has tools of its own, or asks for the server's (`strata_mcp`, `strata_web`, `strata_media`, `strata_browser`) |
| `local_busy: true/false` | requests wait in the local queue, or no text model is resident (loading, switching, a media job) |
| `key: "..."` | the caller is that generated key (id or name), client profile (name), or `owner` |

There is no `kind` condition (code vs chat): the server has no request classifier, and one was not invented.

A target is `local` (the loaded / default local model), a local model id (loaded as a request naming it would load
it), or `<provider>/<model>`.

**Fallback.** When a remote target fails before answering (not reachable, HTTP 5xx, 429), the next rule that holds
(or the default) is tried, once. A model named directly (not through a mix) gets no fallback: its error goes back.

**Headers.** Every request for a remote model or a mix gets `x-strata-routed: <the model that answered>` (a local
answer names the local model's id). `x-strata-note` says what was changed: a fallback ("openai/x did not answer
(HTTP 500): anthropic/y was tried instead"), or a parameter not sent (top_k to an OpenAI API, thinking with a forced
tool call, temperature with Anthropic thinking, a reasoning effort to a provider without `reasoning_effort`).

Remote requests never take the local GPU queue's turn: they run beside local work, limited only by each provider's
`max_concurrent` and the key's limits.

## What happens to a request

1. The API layer (`remote::layer`, inside the API monitor and the usage counts) reads the model; for a remote model or
   a mix it checks the caller, picks the mix's targets, and puts the route in a task-local.
2. The API's handler (OpenAI chat, Anthropic messages, Responses; Gemini through OpenAI chat) normalizes the request
   exactly as for a local model (the same front end), adds the server's tools the request opted into (memory, web,
   MCP, media, browser, computer, code), and starts it on a `Target::Remote`.
3. `remote/translate.rs` writes the provider's own API from the normalized request: messages (system, user with
   pictures as base64 data, assistant text and tool calls, tool results), tools, tool_choice, max tokens,
   temperature, top_p (top_k for Anthropic), seed and penalties (OpenAI), stop strings, reasoning effort or a
   thinking budget where both sides take it. The call is always streamed from the provider.
4. The provider's stream (or a whole JSON answer, if it sends one) becomes the engine's events (text, reasoning,
   tool calls, the end with the provider's usage), which each client API writes in its own shape, streamed or not.
5. Calls to the server's own tools are run here by the tool loop (`run::start_loop`, the same loop as for local
   models), and the conversation goes back to the provider for the next round. Anthropic's signed thinking blocks
   are kept for the next round of the same request.
6. Usage (prompt, completion, cached tokens; every round of a tool loop is billed by the provider, so the prompt
   tokens are summed) goes back in the client's shape and is counted per key and model. The API monitor records the
   call like any other.

Errors: a provider's refusal goes back in the client's error shape with the provider's status; 429 keeps the
provider's `Retry-After`; 401 and 403 from the provider become **502** ("the provider refused the server's key"): the
server's problem, not the client's. A client that disconnects ends the provider's call (the connection is closed;
the server sends keep-alives while a provider is silent, so a departure is noticed within a few seconds).

## Images

A provider with `"kinds": ["chat", "image"]` (OpenAI kind) answers `POST /v1/images/generations` with
`"model": "<provider>/<model>"`: the body is sent on with the bare model name, the provider's JSON comes back. Only
direct use: the mix routes chats, not images, and the chat's own `generate_image` tool still uses this PC's media
engine. Video, speech and music have no remote route (not clean enough to add here).

## Security and privacy

- **Keys** are read when used: from the page's store (`<data>/remote-keys.json`, 0600), an environment variable, or a
  file that only its owner can read. They are never written to the run config by the page, never logged, never in
  an API answer (only set / not set and the last 4 characters of a key of 12 or more), never sent to the web page.
  A provider's error text is scrubbed of anything that looks like a key before it is passed on.
- **What a provider receives**: the whole conversation of the request, as the client sent it after the server's own
  additions: the system prompt, every message, pictures (as base64), the tool definitions (the client's and the
  server's own: MCP, memory, web, media, browser, computer), **tool results including the server's own tools'
  output** (web pages read, MCP results, memory recalled, browser and computer screenshots when the model reads
  pictures), and the memory block of facts when the request uses memory. It also receives the server's key for it
  and the request's parameters. It does not receive the client's API key, the client's IP address or headers.
- Anything a local model would keep on this PC leaves it when a remote model answers. Use the key permissions and
  the mix rules (e.g. `{"when": {"tools": true}, "use": "local"}`) to keep private work local.
- A cluster master's key cannot use this server's remote models.

## Command line

Not in `--help`:

```
am2strata-run remote status            # providers, key set or not (last 4), mixes
am2strata-run remote test <provider>   # the server lists the provider's models now
am2strata-run remote models <provider> # the models offered
am2strata-run remote set-key <provider> < keyfile   # the key from stdin, never the command line
```

With the feature off each says so.

## Known limits

- Thinking from earlier turns of a conversation is not sent back to Anthropic (no signature); a request that
  continues a client's own tool call with thinking on therefore runs without thinking (said in `x-strata-note`).
- `/v1/messages/count_tokens` and Gemini `countTokens` count with the local model's tokenizer.
- The Gemini model list (`/v1beta/models`) does not list remote models; naming one works
  (`/v1beta/models/openai%2Fgpt-5.5:generateContent`).
- Structured output (`response_format` with a schema) is enforced the server's way (instructions and a check), not
  with the provider's native structured output.
- The tool loop's turns carry no ids of their own; ids are made from the positions when sent to a provider.
