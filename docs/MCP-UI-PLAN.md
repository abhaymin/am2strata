# MCP tool servers from the app (Settings › Tools)

Today the tool servers come only from the run config (`"mcp_servers"`, or Claude Desktop's `"mcpServers"` pasted as
is). Settings shows them read-only. With none, it shows a JSON snippet to paste into the file by hand: nothing can be
added, edited, removed or tested from the app (owner's report, 2026-11-09).

## What the app gets

- **Add a server:**
  - a **URL** (Streamable HTTP or SSE), with headers such as `Authorization` and a timeout;
  - or a **command** (stdio): the program, its arguments, environment variables and a working folder.
- **Paste a config:** a Claude Desktop / Cursor / VS Code `mcpServers` block. Each server in it becomes a row.
- **Each server row:**
  - its state (connected, starting, failed with the error and the tail of its log);
  - its tools, each with its description and a switch;
  - **Test**, which connects and lists the tools without saving;
  - edit, turn off, remove.
- **Where they are kept:** `<data>/mcp-servers.json` (server-wide), next to the run config's own servers. The run
  config's servers show as "from the run config" and stay read-only in the app.
  - Header values and environment values are stored as given. The app shows them masked (`••••`) and never sends
    them back in full.
- **Changes apply at once:** a saved change starts, restarts or stops that one server, with no restart of am2Strata.
  The tools a chat may call update with it.

## Safety: a command server runs a program on this PC

- **Adding or changing a command server is allowed only:**
  - from this PC (the request comes from 127.0.0.1 / ::1, on the server's own page), or
  - when the run config sets `"mcp_remote_commands": true`.
- **Otherwise the app explains why,** and offers a URL server instead.
- **URL servers** can be added from any device with the API key, on the server's own page only (the
  `foreign_origin` check, as for memory).
- **Before saving a command server,** the dialog shows the exact command line that will run and asks to confirm.
- **Every change is logged:** who (address), what, when. It also appears in API calls.

## API

| Route | What |
| --- | --- |
| `GET /api/mcp/servers` | all servers: source (app or run config), state, tools, masked secrets |
| `POST /api/mcp/servers` | add one (`{name, url, headers}` or `{name, command, args, env, cwd}`), or import a pasted `mcpServers` block |
| `PUT /api/mcp/servers/{name}` | change one (masked values left as `••••` keep their stored values) |
| `DELETE /api/mcp/servers/{name}` | remove one (app servers only) |
| `POST /api/mcp/servers/test` | connect and list the tools, nothing saved |
| `POST /api/mcp/servers/{name}/tools` | `{tool: on/off}` |

## Tests (mock MCP server, as the current ones)

- adding, testing, editing and removing a URL server and a command server;
- a pasted Claude Desktop block becomes rows;
- a command server added from another address is refused (403), and allowed with `mcp_remote_commands`;
- secrets come back masked, and a PUT with `••••` keeps them;
- a turned-off tool is not offered to the model;
- run config servers cannot be removed from the app.
