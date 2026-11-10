# Server tools for every client: memory, web, media and MCP, run by am2Strata

**State (2026-11-10):**
- **Built:**
  - part 1, client profiles: `clients.rs`; the key picks the profile; `x-strata-tools` narrows it and never widens it; each profile has its own memory space; a client key opens the model APIs, not the settings;
  - part 2, Anthropic's `web_search_*` and `web_fetch_*` server tools: `server_tool_use`, `web_search_tool_result` and `web_fetch_tool_result` blocks, `usage.server_tool_use`;
  - part 2, the Responses `web_search` tool: `web_search_call` items;
  - part 2, Chat Completions' `web_search_options`;
  - part 4, `agentrun --tools`: the header through Claude Code's `ANTHROPIC_CUSTOM_HEADERS`, Codex's `http_headers` and OpenCode's `headers`.
- **Tested:** `a_client_key_gets_its_profiles_tools_without_flags`, `anthropic_web_search_is_answered_by_the_server`, `anthropic_web_search_needs_a_profile_with_web` and `responses_web_search_gives_a_web_search_call_item`.
- **Next:**
  - the Responses `image_generation` and `mcp` hosted tools, and Gemini's `google_search`;
  - part 3, media inside the turn (speech on the CPU; `defer` / `swap`);
  - the per-profile limits;
  - Settings › Clients, and Settings › Connect saying which built-in tools am2Strata answers.

The USP (owner, 2026-11-09): a client points at am2Strata, over the OpenAI, Anthropic or Responses API, locally or
from another PC. It gets memory, web search, image, video and speech generation and the configured MCP tools,
without setting up any of them. **The server runs the tools inside the call.** The client's configuration lives on
the server.

## What exists (server_tools.rs, run::start_mcp)

- Tools run in the server's tool loop, for all three APIs:
  - memory (remember, recall, forget);
  - media (`generate_image`, `generate_video`, `speak`, queued as jobs);
  - web (`web_search`, `fetch_url`, through SearXNG or DuckDuckGo);
  - browser and MCP.
- Calls to these tools never reach the client. The client's own tools still go back to it, and they win a name
  clash.
- **The gap:** a request turns each set on with a body flag (`strata_media`, `strata_web`, `strata_mcp`, ...), and
  MCP is offered only on the server's own page. A coding agent cannot send these flags. And a GPU media job cannot
  run in the middle of a turn while the text model holds the cards.

## 1. Client profiles: the client's configuration, kept on the server

- **Profiles in the run config** (and Settings › Clients):
  `"clients": {"laptop": {"key": "...", "tools": ["memory", "web", "image", "video", "speech", "mcp:*"], "memory": "laptop"}}`
- **The key picks the profile:** a request with that profile's key gets its tools with no flag at all. The main key
  is the "default" profile.
- **Per profile:**
  - which tool sets it gets;
  - which MCP servers (`mcp:search`, `mcp:*`);
  - its memory space: its own, or shared with others;
  - limits: media jobs an hour, the largest image or video size, web queries a turn.
- **For one request:** a header `x-strata-tools: web,image` narrows the set, or widens it within what the profile
  allows. Claude Code sends extra headers through `ANTHROPIC_CUSTOM_HEADERS`, and Codex through `http_headers`.
  `am2strata agentrun` writes the header from `--tools`.
- The body flags keep working, as today.
- **MCP for remote clients:** only through a profile that lists the servers. The own-page rule stays for requests
  without a profile. A command (stdio) MCP server runs on this PC, so a profile gets it only by name, never with
  `mcp:*` (see MCP-UI-PLAN.md).

## 2. The APIs' own built-in tools, answered by am2Strata

So an agent's own web-search button works against am2Strata as it does against the cloud:
- **Anthropic Messages:** the server tools `web_search_20250305` and `web_fetch_20250910` (Claude Code's WebSearch
  and WebFetch) run the server's search and fetch. They answer with `server_tool_use` and `web_search_tool_result` /
  `web_fetch_tool_result` blocks, and `usage.server_tool_use.web_search_requests`.
- **OpenAI Responses:**
  - `{"type": "web_search"}` gives `web_search_call` output items (Codex's web search);
  - `{"type": "image_generation"}` gives `image_generation_call` with the base64 result;
  - `{"type": "mcp", "server_label": ...}` runs a configured MCP server by its label.
- **OpenAI Chat Completions:** `web_search_options` turns on web search for that request.
- **Gemini:** `tools: [{"google_search": {}}]` runs web search, with `groundingMetadata` in the answer.

Each of these is allowed only when the profile has that tool set. Otherwise the request is answered as the API does
for a tool it does not offer.

## 3. Changing context in a turn: from chat to image, video and speech

When the model calls a media tool, the scheduler decides where it runs (see the lanes in CHATTERBOX-PLAN.md):
- **Speech** (Chatterbox, Whisper) runs on the CPU **inside the turn**. The result goes back to the model, and the
  answer goes on.
- **Image or video with a free GPU** (a second card, when the text model fits on one) runs **inside the turn**, on
  that card.
- **Image or video when the text model holds every card:** the profile's `media_in_turn` decides.
  - `"defer"` (the default): the job is queued for when the turn ends. The tool result gives its id, an estimate and
    the URL (`/v1/media/jobs/{id}/content`), and the answer says so. The client fetches the result, or a later turn
    shows it.
  - `"swap"`: the turn pauses.
    1. The conversation cache keeps the prompt's state.
    2. The text model stops and the media job runs.
    3. The text model loads again (Tiel: about 20 s; Orca: about 83 s) and the turn goes on with the result.
    Slow, but one call gives the finished answer.
- **Results:**
  - images and audio inline where the API has a place for them (Responses' `image_generation_call`, an Anthropic
    image block);
  - otherwise a link with the job's id, and the Monitor row with its player.

## 4. Every agent, local or remote

The aim (owner, 2026-11-09): Codex, or any agent, started on another PC with `am2strata agentrun codex --url ...`
or with `unsloth start codex`, reaches the same memory, web, media and MCP tools as the server's own app.
- `unsloth start` passes only `UNSLOTH_STUDIO_URL` and `UNSLOTH_API_KEY` (no extra headers), so **the key alone
  picks the profile and its tools**. The header is optional, never needed.

- `am2strata agentrun <agent> --url http://<host>:8000 --tools web,image` points the agent at the server with the
  profile's key and header. Claude Code's WebSearch then goes to am2Strata's search, and Codex's `web_search` too.
- Settings › Connect shows, per agent, which of its built-in tools am2Strata answers.

## Tests (mock engine, mock search, mock media)

- A profile's key gets its tools with no flag; another key does not get them.
- `x-strata-tools` narrows the set; widening past the profile is refused.
- Claude Code's request shape with `web_search_20250305`: the answer carries `server_tool_use` and
  `web_search_tool_result`, and the usage counts the requests.
- Codex's Responses shape with `web_search` gives `web_search_call` items. `image_generation` gives an
  `image_generation_call` with the PNG.
- A speech tool call runs inside the turn on the CPU.
- An image call with the text model on every card: `defer` queues it and the turn ends with the job's URL; `swap`
  answers with the image after a reload, which the mock counts.
- A remote request without a profile gets no MCP tools.
- Memory spaces: what one profile remembers, another does not recall.
