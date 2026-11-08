# Coding agents

`am2strata agent` points a coding agent at am2Strata, on this PC or another one, with the model you choose. It does
what `unsloth start` and OmniRoute's `run` / `setup-<tool>` do, for this server. Settings › Connect › Coding agents
in the web app shows the same list, with what to enter for each agent.

```bash
am2strata agent list                                   # every agent, what it supports, which are installed
am2strata agent start claude                           # one run on this PC's server; nothing of yours changes
am2strata agent start codex --model tiel -- exec "fix the failing test"   # after --: the agent's own arguments
am2strata agent setup opencode                         # opencode's default from now on (its config backed up)
am2strata agent reset opencode                         # the backup back
am2strata agent show omp                               # what to enter, for agents set up in their own settings
```

Another am2Strata: `--url http://192.168.1.20:8000 --api-key ...` (or `AM2STRATA_URL` and `AM2STRATA_API_KEY`).
Without `--model`, the agent gets the model the server has loaded; `--small-model` sets the small one (Claude Code's
haiku and background work, opencode's small_model).

The key is never written into an agent's config: the config names `AM2STRATA_API_KEY`, which your shell sets.
`start --dry-run` and `setup --dry-run` print what would run or be written, with the key masked.

| Agent | Command | API | start | setup | Format from |
| --- | --- | --- | --- | --- | --- |
| Claude Code | `claude` | Anthropic | env (`ANTHROPIC_BASE_URL`, `ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU_MODEL`, ...) | `~/.claude/settings.json` (`env`, `apiKeyHelper`) | Claude Code's settings docs |
| Codex CLI | `codex` | Responses | `-c model_providers.am2strata=...` | a block at the top of `~/.codex/config.toml` | Codex's config docs |
| opencode | `opencode` | OpenAI | `OPENCODE_CONFIG_CONTENT` | `~/.config/opencode/opencode.json` | run here with opencode 1.19.5 |
| Pi | `pi` | OpenAI | a temporary `PI_CODING_AGENT_DIR` | `~/.pi/agent/models.json` | Pi's models and configuration docs |
| Factory Droid | `droid` | Anthropic | - | `~/.factory/settings.json` `customModels` | Factory's BYOK docs |
| Qwen Code | `qwen` | OpenAI | `OPENAI_BASE_URL`, `OPENAI_MODEL` | (shell profile, `agent show`) | Qwen Code's variables |
| Aider | `aider` | OpenAI | `OPENAI_API_BASE`, `--model openai/<id>` | (shell profile) | Aider's docs |
| Goose | `goose` | OpenAI | `GOOSE_PROVIDER`, `OPENAI_HOST`, `GOOSE_MODEL` | (shell profile) | Goose's provider variables |
| Oh My Pi, Crush, Hermes, OpenClaw, Continue, Cline, Kilo Code, Zed, VS Code | | OpenAI | - | `agent show` gives the block or fields | each agent's docs |

Not supported: **agy** (Antigravity CLI) takes no custom model endpoint; OmniRoute reaches it only by intercepting
its traffic, which am2Strata does not do. **cursor-agent** (Cursor CLI) documents no custom base URL. **gemini** speaks
Google's own Gemini API, which am2Strata does not serve yet.

Only opencode was run here; the others follow their documentation. A setup that does not work for your agent's
version: `agent reset <agent>` puts its config back, and an issue with the agent's version helps.
