# Clusters: several machines as one am2Strata

Owner's request (2026-11-10): machines on the same network, or remote ones the user pairs with keys (peer to peer),
work together. A model's layers are shared across them, for quicker requests and longer context, as exo and other
cluster tools do, old phones included.

**State: planned, nothing built.** This page is the design and the order of the work. Each step is measured on real
machines before the next starts.

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
