# Clients: keys, profiles and the tools a key gets

A client is an app, a person or a machine that calls am2Strata with its own key. The key picks a **profile**, and the
profile picks the server's tools that the client's requests get, with no flag in the request. That way an agent that
can send only an address and a key gets the tools: Claude Code, Codex, opencode, and anything OpenAI- or
Anthropic-compatible.

## Keys

- **The owner's key** (`api_key` in the run config) opens everything, including the server's settings.
- **Client keys** open the model APIs only (chat, messages, responses, Gemini, media, `/v1/memory`, `/v1/mcp/browser`
  when the profile has `browser`). They never open the settings.
- **Making keys at run time:** Settings › Clients or `am2strata-run keys create --name laptop --tools web,memory
  --expires 30d`. Each has usage counts per day and per model (`am2strata-run keys usage`, `GET /v1/usage` for a
  client's own). A key can carry limits: `--requests-per-day`, `--tokens-per-day`, and models.

## Profiles in the run config

```json
"clients": {
  "laptop":  {"key": "…", "tools": ["memory", "web", "image", "speech", "browser", "mcp:*"], "memory": "laptop",
              "limits": {"media_jobs_per_hour": 20}},
  "default": {"tools": ["web"]}
}
```

| `tools` entry | What the client's requests get |
|---|---|
| `memory` | `remember`, `recall`, `forget`, and relevant facts added to the prompt |
| `web` | `web_search`, `fetch_url` (and agents' own web search requests answered by the server) |
| `image`, `video`, `speech`, `music` | The media tools for that kind (MEDIA.md) |
| `browser` | The browser tools (BROWSER-USE.md) |
| `code` | `run_python` in the sandbox |
| `mcp:<name>` / `mcp:*` | An MCP server by name, or every address server. A program server runs on this PC, so a profile only ever gets it by name. |

- **`memory`** names the profile's memory space; it defaults to the profile's name. Two profiles naming the same space
  share it.
- **`default`** (with no key) is what the owner's key gets without flags.
- **Narrowing one request:** `x-strata-tools: web,image`. Naming a set the profile doesn't have is refused.

## Memory for a client

| Call | What it does |
|---|---|
| `GET /v1/memory` | The facts this client sees: its own space's, and those kept for everyone |
| `POST /v1/memory {"fact": "…"}` | Saves a fact in this client's space |
| `DELETE /v1/memory/{id}` | Removes a fact from its own space |

Memory is on by default for API calls in the `api` space; `"strata_memory": false` turns it off for one request.

## Not done yet

Profiles are edited in the run config; Settings › Clients shows them but doesn't edit them yet (GAPS-PLAN A8).
