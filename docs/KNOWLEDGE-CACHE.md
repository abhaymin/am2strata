# Knowledge and caching above the KV cache — plan

Status: **memory built** (2026-10-24, crates/am2strata/src/memory.rs, Settings › Memory): shared facts with scopes, the
`remember` / `recall` / `forget` tools run by the server, matching facts added to the prompt, and conversations that
continue past the model's context (the oldest turns moved to the conversation's archive and recalled when they
match). Search is BM25 over words; embeddings (phase C), collections (B), the answer cache (A) and prompt snapshots (E)
are still to do. UI mockups: `design/Knowledge.dc.html`.

Today the only reuse is the engine's conversation cache: the KV state of the last conversation stays in memory, so a
follow-up reads only what is new. It is gone when the model unloads, it holds one conversation (more with
`--conversation-cache-mib`), and nothing is shared between clients. This plan adds three layers in front of it, in
the order a request meets them.

```
request ─▶ 1 answer cache ─▶ 2 memory + knowledge (added to the prompt) ─▶ 3 prompt snapshot (KV from disk)
        ─▶ conversation cache (KV in RAM/VRAM, today) ─▶ engine
```

## 1. Answer cache (exact repeats)

- Key: SHA-256 of (model id, model file hash, template, the normalized request: messages, tools, response format,
  sampling, max tokens, seed). Only requests whose answer is determined: temperature 0, or a seed.
- Value: the final response (both API shapes can be rebuilt from it), usage and timings, with a hit counter.
- Store: one file per entry under `Strata-data/cache/answers/<model>/`, an in-memory index, LRU by size (default
  256 MB) and age (7 days). Streaming hits replay the text in chunks.
- Off per request with `"strata_cache": false`; never across models, templates or settings. A hit is marked in the
  API monitor and in `usage` (`cached: true`).
- Cost: small, server only. **Phase A.**

A similar-question cache (embedding distance) is the risky version: it can return a wrong answer. It stays off by
default, with a high threshold (0.97), and is offered only after Phase C has embeddings.

## 2. Knowledge: documents and shared memory, added to the prompt

**Collections** are documents the user adds: files, folders, PDFs (the existing pdftotext/pdftoppm path), web pages,
pasted text.
- They are split into passages of about 300–500 tokens, on headings and paragraphs, with code kept whole.
- Each passage gets an embedding plus a BM25 index for keywords; the two scores are fused.
- Storage: `Strata-data/knowledge/<collection>/` holds the passages (JSONL), a vector file and a manifest with file
  hashes, so an update re-embeds only what changed.

**Embeddings, in Rust:**
- Option A, preferred: a small GGUF embedding model (bge-small / nomic-embed, ~100 MB) run by the engine with an
  `EMBED` command. This needs an engine change: a pooled embedding output.
- Option B: candle on the CPU, pure Rust, a little slower. It needs no engine change, so B goes first and A
  replaces it.

**Shared memory** is a list of short facts with who wrote it, when, and its scope: everyone, the web app, or one API
key. Facts come from three places:
- the user, in Settings;
- the model, through an internal `remember` / `recall` / `forget` tool, offered like the MCP tools and only where
  memory writes are allowed;
- the API: `POST /api/memory`.

**Injection** is the policy: off, when asked (`"strata_knowledge": {"collections": [...], "top_k": 6}`), or
automatic per client.
- Retrieval takes the last user turn plus a short summary of the conversation, then keeps the top-k passages above a
  relevance threshold, within a token budget (default 3,000).
- The result goes in as one system block, `<knowledge>`, after the system prompt, with source labels. This keeps
  the system prompt's KV prefix stable, which layer 3 depends on.
- Sources are cited in the answer when asked. What was added shows in API calls › Input.
- Clients are opted in per API key (Settings › Knowledge › How it is added).

**Phases:** collections and keyword search first (**Phase B**), then embeddings and fusion (**Phase C**), then the
memory tool (**Phase D**).

## 3. Prompt snapshots: KV prefixes kept on disk

Long, stable prompt starts are worth reading once:
- an agent's system prompt and tool list (omp, Claude Code: 10–30K tokens);
- a large document pinned to a collection.

The engine already has `SAVE` / `RESTORE` of slot state to a file (docs/PROTOCOL.md), so most of this is server
work:

1. **Detect a prefix** that repeats across requests: the hash of the token ids up to the last system/tool block, seen
   twice.
2. **Save** after the request that read it (`SAVE <file>`):
   - path: `Strata-data/cache/kv/<model>/<ctx>-<kv type>/<prefix hash>.kv`;
   - a manifest records the token count, the model file hash and the engine version.
3. **Restore** on a later request with the same prefix, when the conversation cache misses: `RESTORE`, then
   read only the rest.
   - Measured to beat re-reading by 50–100× on long prompts: at the ~880 tok/s prefill measured here, 97K tokens
     take ~110 s; a ~10 GB snapshot from NVMe takes ~1–3 s.
4. **Budget:** an SSD size limit (default 40 GB), LRU, and "pin" for collections the user wants instant.

Snapshots are only valid for the same model file, context size, KV type and engine version; any mismatch is a miss,
never a wrong answer. Streamed-KV models (Orca, 262K) save only the positions the engine holds. **Phase E**,
measured on Orca and MiniCPM first.

## Safety and privacy

- Everything stays on this PC under `Strata-data/`. Nothing is sent anywhere.
- Scopes keep one API key's memory out of another's answers. The web app shows every fact and can forget it.
- Injected text is data: it goes in a clearly labelled block, and the model is told it may contain instructions it
  must not follow.
- Management endpoints (`/api/knowledge`, `/api/memory`, cache clear) need the key and this server's own page,
  like settings and MCP.

## Order of work

| Phase | What | Needs |
|---|---|---|
| A | Answer cache + Settings › Knowledge › Caches | server only |
| B | Collections, keyword search, injection when asked | server only |
| C | Embeddings (candle, CPU) + automatic injection | a Rust embedding crate |
| D | Shared memory + remember/recall tool - **done** (keyword search), plus chats past the context | B |
| E | Prompt snapshots on disk (SAVE/RESTORE) - **built** (2026-10-29, crates/am2strata/src/snapshots.rs): saved before the model stops and after a 16K+ prompt, restored when one gives 2,048+ more tokens than the engine holds; `strata` (Orca) only. Live (2026-10-29): a 4,529-token Orca conversation saved on a model switch (292 MB); after switching back, RESTORE took 0.28 s and the follow-up's prompt read 1.1 s instead of 8.1 s (4,483 tokens reused) | engine SAVE/RESTORE (exists), measurements |
| F | Similar-question cache (opt-in), engine `EMBED` | C, engine change |

Each phase ships with Rust tests on the mock engine (the mock gets SAVE/RESTORE and EMBED stubs) and a live check on
the GPU, as before.
