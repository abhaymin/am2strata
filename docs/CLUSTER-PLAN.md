# Clusters: several machines as one am2Strata

Owner's request (2026-11-10): machines on the same network, or remote ones the user pairs with keys (peer to peer),
work together. A model's layers are shared across them, for quicker requests and longer context, as exo and other
cluster tools do, old phones included.

**State: phase 1 (request routing), part 1 (nodes and joining) and encryption with the paired keys built and tested
on one machine (2026-11-11); measurements on a second machine are next.** This page is the design and the
order of the work. Each step is measured on real machines before the next starts.

### Part 1 as built: nodes and joining (`crates/am2strata/src/cluster.rs`)

- **A worker:** `"cluster": {"role": "worker"}` in its run config (and optionally `"name"`).
  - It has an ed25519 identity in `<data>/node-key` (0600, made on first start). Its node id is the first 16 bytes
    of the SHA-256 of the public key.
  - It announces itself on the LAN by mDNS as `_am2strata._tcp` (TXT: name, version, node_id), using the
    `mdns-sd` crate (Apache-2.0 OR MIT).
  - It shows a 6-digit pairing code that changes every 10 minutes and after each use. The code appears in its log
    and its own Settings › Cluster, and only to its own main key.
- **The master:**
  - It browses for `_am2strata._tcp`. `GET /api/cluster` lists the servers found and not paired as "found";
    Settings › Cluster shows them as "wants to join", with a code field and a Pair button. A worker reached through
    a VPN can be paired by its address.
  - The address used is a private LAN address first, then a VPN's, then IPv6.
- **Pairing:**
  1. The master's `POST /api/cluster/pair {"url" or "node_id", "code"}` (main key, own page) sends the code to the
     worker's `POST /api/cluster/join`.
  2. The worker compares the code in constant time. Five wrong codes lock pairing for 10 minutes, even for the
     right code.
  3. For the right code it mints a random key for that master alone, keeps only the key's SHA-256 in
     `<data>/cluster-masters.json` (0600), and answers the key once.
  4. The master keeps the worker and its key in `<data>/cluster-workers.json` (0600), merged with any run-config
     workers at start, and polls it at once (no restart).
- **What a master's key opens on the worker:** the model APIs, plus the model list and queue the master polls.
  Never the worker's settings or its own pairing code. The worker's main key never leaves it.
- **Removing:** `DELETE /api/cluster/workers/{name}` on the master stops using the worker, and tells it to forget
  that master (`POST /api/cluster/forget` with the master's key). After that the key is refused.
- `AM2STRATA_NO_MDNS=1` turns mDNS off (the tests set it).
- **Tested:**
  - `a_worker_pairs_by_its_code_and_is_removed_cleanly`, on two servers on one machine: a wrong code is refused;
    the right code pairs; the key files are 0600; the worker keeps only the hash; the minted key opens
    `/v1/models`, `/api/models` and `/api/queue` but not `/api/cluster` or `/api/system`; a chat for the worker's
    model is answered by it through the master; removal makes the key refused; five wrong codes lock pairing.
  - Unit tests: the node identity, the code and the lock, the address preference.
  - `security_every_route_needs_the_key` names `/api/cluster/join` as the one route proven by a code (refused on a
    server that is not a worker).
- **Checked by hand on this PC (2026-11-11):** a worker and a master as two processes, mDNS on. The master found the
  worker (name, node id, version) with no configuration. It took the worker's Tailscale address, the first IPv4 the
  worker announced, which is why private LAN addresses now go first.
- **Security:** pairing needs physical or remote access to the worker's code. Each master has its own revocable
  key. Traffic is still plain HTTP with that key, so across a network you don't trust, use your own VPN.
  **Encryption with the paired keys (Noise or TLS pinned to the node ids) is the next step.**

### Phase 1 as built (`crates/am2strata/src/cluster.rs`)

- **Configuration:** the master's run config has
  `"cluster": {"workers": {"pc2": {"url": "http://host:8000", "key": "env:PC2_KEY"}}}`. A worker is an ordinary
  am2Strata with its own API key, which only the master is given.
- **Polling:** the master asks each worker for its models and queue every 5 s. A worker that has not answered for
  15 s gets no new work.
- **Routing:** a chat (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/v1/completions`) goes to a worker
  in two cases:
  - its `model` is not installed on the master but is on a worker. The worker with the model loaded goes first,
    then the one with the shortest queue.
  - the master has work waiting and a worker has the same model loaded and idle.
- **The answer:** streamed back through the master, with `x-strata-worker: <name>`, and kept in the master's API
  call records.
- **Failure:** a worker that does not take a request leaves it to the master when the master has the model;
  otherwise the request is refused plainly (503, naming the worker).
- **Authentication:** only a request with the master's own key is sent on, never the client's key. The worker gets
  the worker's key.
- **The status:** `GET /api/cluster` shows each worker's models, its loaded model, its queue, when it was last seen,
  and how many requests it has taken.
- **Security for now:** plain HTTP with the key, so across a network you don't trust, use your own VPN
  (Tailscale, WireGuard). Paired ed25519 keys, encryption and mDNS discovery are the next steps (parts 1 and 2
  below).
- **Tested:** `a_master_routes_a_chat_to_the_worker_that_has_the_model` (two servers on one machine: a worker's
  model answered through the master, streamed too; the master's own model answered locally; no key, not sent on),
  and the routing rules in unit tests.

## What a cluster can and cannot buy (the physics first)

- **A model too big for one machine runs at all.** Its layers are split across machines (pipeline parallelism): each
  machine holds some layers, and the activations go from one to the next. This is what exo and llama.cpp's RPC
  backend do.
- **More requests at once.** Different machines answer different requests, each with a model it holds (request
  routing). This is the quickest win: no traffic between machines inside a request.
- **Longer context.** The KV cache is split across machines, by layers (each machine keeps the KV of its own layers)
  or by sequence (ring attention). More memory in all means more context.
- **Not a faster single answer over plain Ethernet.** A token must pass every machine in turn, so pipeline splitting
  adds one network hop per machine per token. About 0.2-1 ms on gigabit Ethernet; Wi-Fi is worse. Tensor parallelism
  (splitting each layer) needs far more bandwidth than home networks have. It is left out except over Thunderbolt /
  USB4 links, measured first.
- **Old phones** hold little RAM (2-8 GB) and slow memory. They are useful for small jobs:
  - a small model of their own;
  - a draft model for speculative decoding;
  - embeddings;
  - speech to text;
  - a few layers of a big model, where the network allows.

## The parts

1. **Nodes and joining.**
   - Every am2Strata can be a node: `"cluster": {"role": "node"}`, or the Settings › Cluster page.
   - **On the same network**, nodes find each other by mDNS (`_am2strata._tcp`). A new node shows as "wants to join"
     until the user accepts it on the main machine. Nothing joins by itself.
   - **Remote machines** pair with keys: each node has an ed25519 key pair; pairing exchanges public keys once, by a
     code shown on one machine and typed on the other, or a QR code for a phone.
   - Traffic is end-to-end encrypted and mutually authenticated (Noise or TLS with the pinned keys).
   - Across the internet: a direct connection when the routers allow it (hole punching); otherwise through the
     user's own VPN (Tailscale, WireGuard), which the plan supports first because it is simplest and safe.
   - No third-party relay by default.
2. **What each node reports:**
   - its CPU, GPUs, RAM, VRAM and free disk;
   - the network's speed to each other node, measured with a short test;
   - the models it holds;
   - its battery and temperature (phones);
   - whether it is busy.
3. **Request routing (phase 1, the first to build).**
   - The main machine's queue (queue.rs) sees every node's models and queue.
   - A request goes to the node that holds its model and will answer soonest, by the same estimates the queue
     already makes. Media jobs go to a node with a free GPU.
   - The client still talks to the main machine only, and the answer is streamed back through it.
4. **Layers across machines (phase 2).**
   - A model too big for one node is split by layers, in proportion to each node's free memory and speed, as exo
     plans its partitions.
   - The engines speak a small protocol: activations in, activations out, KV kept where its layers are.
   - The first version uses llama.cpp's RPC backend as the measured reference: our engines are built on llama.cpp,
     which has `rpc-server`.
   - Strata's own engines get the same split after that, gated token for token against one-machine answers, as every
     engine change is.
   - MoE models (Tiel, Orca, the Qwen3 MoEs) can also split their experts: each node holds a share of the experts,
     and a token's router sends it to the nodes holding the experts it picked. This suits Strata's expert placement
     and is measured against plain layer splitting.
5. **Long context across machines (phase 3).** The KV of each node's layers stays on that node (it comes free with
   phase 2). Sequence splitting (ring attention) for contexts beyond what layer splitting gives is kept for later,
   measured first.
6. **Phones (phase 4).**
   - A small node app: the same Rust engine built for Android (aarch64, by Termux first, an app later). It runs
     what fits, and phase 2 layers when the network allows.
   - Battery and temperature limits: a phone leaves the cluster when it is low on battery or hot, and its work goes
     back to the others.
7. **Failure.**
   - A node that stops answering is dropped from the routing at once.
   - A request it was running is told and retried on another node when one holds the model; for a split model the
     split is rebuilt without it, or the request is refused plainly when the model no longer fits.
   - The PRODUCTION-PLAN rules hold across the cluster: no crash, and overload told plainly.

## Master and workers (owner, 2026-11-11)

The cluster has one **master** and any number of **workers**:
- **The master:**
  - is the am2Strata people and apps talk to; it holds the API keys, client profiles, memory, MCP servers and the
    queue;
  - routes each request to a worker (phase 1) or splits a model across workers (phase 2);
  - streams the answer back.
- **A worker:**
  - runs engines and media models only;
  - takes work from its master and from no one else; its own API is closed by default (`"cluster": {"role":
    "worker", "master": "<master's key fingerprint>"}`);
  - reports its models, free memory, queue and health every few seconds.
- **Joining:** a worker joins by pairing (a code typed on the master, or a QR code for a phone). The master can
  remove it at any time.
- **Failover:**
  - a worker that stops reporting is drained (no new work) and, after a timeout, dropped; its running requests
    are retried on another worker that holds the model, or refused plainly;
  - when the master itself goes down, workers wait; they do not take over, so there is never a second master.
    A standby master is a later step.
- **Peer to peer:**
  - on the same network, workers find the master by mDNS;
  - across the internet, through the user's own VPN (Tailscale, WireGuard) first; direct connections by hole
    punching later;
  - traffic is always end-to-end encrypted with the paired keys.

## Security

- **Encryption with the paired keys (built 2026-11-11, `crates/am2strata/src/cluster_tls.rs`):** mutual TLS 1.3.
  - Every node has a self-signed certificate in `<data>/node-tls.json` (0600). Its SHA-256 is its pin.
  - Pairing swaps the pins over the same request the 6-digit code proves: the master sends its pin, and the worker
    answers with its own and its TLS port. Each side keeps the other's pin (`cluster-workers.json` /
    `cluster-masters.json`, 0600).
  - A worker serves a second listener for its masters (`"cluster": {"role": "worker", "tls_port": N}`, default the
    API port + 443): TLS 1.3 only, a client certificate required, and only a paired master's certificate is let in.
  - A master's key works only on a connection showing that same master's certificate. Plain HTTP with it is refused
    unless the worker sets `"allow_plain": true` (masters paired before TLS keep working over plain HTTP).
  - The master reaches the worker only over that listener, checking the worker's certificate against its pin (no
    certificate authority: the pin is the trust), and shows its own certificate.
  - Why TLS and not Noise: rustls is already in the build (reqwest), audited and fast, and HTTP and streaming run over
    it unchanged.
  - Tested: `a_paired_worker_is_reached_only_over_pinned_tls` (two servers on one machine), plus a unit test with real
    handshakes (`only_pinned_certificates_talk`):
    - the master's key over plain HTTP: 401;
    - the TLS listener without a client certificate: refused at the handshake, as is an unpaired certificate;
    - a worker showing another certificate is refused by the master, which reports "the worker's certificate is not
      the one paired";
    - chats, streamed too, go through.
  - Measured on this PC (mock engines, 40 short chats each): the median chat through the master took 1.0 ms over
    plain HTTP and 1.0-1.1 ms over TLS. Connections are reused, so TLS adds about 0.1 ms.
  - **Limits:**
    - the pairing request itself is plain HTTP, so pair on a network you trust or through a VPN (someone in the middle
      of that one request could swap the pins);
    - workers listed in the run config by url and key stay plain HTTP, shown as "Plain HTTP: use a VPN";
    - hole punching across the internet is not built.
- **Joining:** only machines the user accepted or paired, never by being on the network alone.
- **Keys:** a node's keys never leave it. The main machine's API keys and client profiles stay on the main machine.
- **Prompts and answers:** they leave the main machine only to the nodes doing the work, encrypted. The node list
  shows what was sent where (in API calls).
- **No programs:** a node never runs MCP servers, programs or browser sessions for another node. Only model work.

## Measurements before any claim

On this PC (2 x RTX 5060 Ti) and a second machine over gigabit Ethernet, then Wi-Fi, then a phone, for each phase:
- tokens a second;
- the time to the first token;
- the largest model and context that run;
- each against one machine alone.

The numbers go in this file before the feature is called done.

## Related work (to read, not to copy without checking each licence)

- exo (exo labs): automatic partitioning by memory, phones as nodes.
- llama.cpp's RPC backend (`rpc-server`): layers on remote machines; MIT licence, like the rest of llama.cpp.
- Petals: layers across volunteers' machines over the internet.
- Prior art on ring attention for long context.
