"""Draws the site's infographics as SVG, each in a dark (Catppuccin Mocha) and a light (Catppuccin Latte) version.

    python3 tools/site/infographics.py        # writes site/assets/<name>-{dark,light}.svg and docs/media copies

Text colours are the palette's AAA ones (7:1 or more on every background they sit on, checked here with
tools/site/contrast.py); the bright Catppuccin hues are used only for decoration (bars, outlines), where 3:1 counts.
"""
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from contrast import ratio  # noqa: E402

PALETTES = {
    "dark": dict(  # Catppuccin Mocha
        bg="#1e1e2e", card="#28283b", line="#45475a", ink="#cdd6f4", muted="#b1b7cf",
        teal="#94e2d5", blue="#90b9fa", mauve="#cca8f7", pink="#f5c2e7", green="#a6e3a1",
        ui=dict(teal="#94e2d5", blue="#89b4fa", mauve="#cba6f7", pink="#f5c2e7", green="#a6e3a1", track="#45475a"),
        on_ui="#11111b", disc="ui", on_disc="#11111b"),
    "light": dict(  # Catppuccin Latte, text deepened to 7:1
        bg="#eff1f5", card="#ffffff", line="#bcc0cc", ink="#303246", muted="#42445b",
        teal="#0c4d50", blue="#073ca6", mauve="#590fbb", pink="#801463", green="#1f4e15",
        ui=dict(teal="#179299", blue="#1e66f5", mauve="#8839ef", pink="#ea76cb", green="#40a02b", track="#ccd0da"),
        on_ui="#11111b", disc="text", on_disc="#ffffff"),
}
FONT = "font-family=\"Outfit, Inter, 'Segoe UI', Helvetica, Arial, sans-serif\""
e = html.escape


def head(w, h, title, desc, P):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" {FONT} role="img" aria-labelledby="t d">',
            f'<title id="t">{e(title)}</title>', f'<desc id="d">{e(desc)}</desc>',
            f'<rect width="{w}" height="{h}" rx="28" fill="{P["bg"]}"/>']


def T(x, y, s, P, color="muted", size=19, weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{P[color]}"{extra}>'
            f'{e(s)}</text>')


def overview(P):
    tiles = [
        ("Compatible APIs", "any OS", "teal", ["OpenAI, Anthropic, Responses", "and Gemini endpoints", "Tools, images, MCP servers"]),
        ("Bigger than RAM", "Rust engine", "teal", ["Hot experts in RAM, cold on SSD", "2.5x llama.cpp's generation", "on a 16 GB Mac (17.5 GB model)"]),
        ("Cluster", "mutual TLS", "blue", ["Pair a Mac or PC by 6-digit code", "Small models on workers", "answer as expert 'brains'"]),
        ("Browser use", "any OS", "blue", ["Scrape tables into datasets", "Fill forms, ask before sending", "From chat, CLI and MCP"]),
        ("Web app", "any OS", "mauve", ["Chat with canvas, Python,", "whiteboard and flows", "Monitor: speed, GPUs, queue"]),
        ("Workshop", "any OS", "mauve", ["Hugging Face model to GGUF", "Quantize, merge LoRA, install", "Training data from documents"]),
        ("Sandbox VM", "Linux · KVM", "pink", ["Computer use in its own VM", "The agent is root inside;", "your PC is untouched"]),
        ("Computer tab", "Linux", "pink", ["Watch the agent's desktop", "Take over with mouse and keys", "Rules decide what it may do"]),
        ("Coding agents", "any OS", "teal", ["Connect Claude Code, Codex,", "opencode and more", "to your own models, one command"]),
        ("Media", "build from source", "blue", ["Songs in 50+ languages", "Voices, images, video", "Speech to text"]),
        ("Secure by default", "any OS", "mauve", ["HTTPS and wss; Tailscale certs", "API keys with limits and expiry", "Usage per key and per model"]),
        ("Install and verify", "six platforms", "pink", ["One line: Linux, macOS, Windows", "x64 and Arm, checksum verified", "doctor shows what each needs"]),
    ]
    W, H, cw, ch, gx, gy, x0, y0 = 1600, 1060, 352, 226, 24, 24, 64, 190
    o = head(W, H, "am2Strata: what is in it", "; ".join(t[0] + ": " + " ".join(t[3]) for t in tiles), P)
    o += [T(64, 100, "am2Strata", P, "ink", 60, 700), T(64, 144, "A local AI server in Rust: what is in it (November 2026)", P, size=24)]
    for i, (t, tag, c, lines) in enumerate(tiles):
        x, y = x0 + (i % 4) * (cw + gx), y0 + (i // 4) * (ch + gy)
        tw = len(tag) * 8.6 + 26
        o.append(f'<g transform="translate({x} {y})"><rect width="{cw}" height="{ch}" rx="18" fill="{P["card"]}" stroke="{P["line"]}" stroke-width="2"/>'
                 f'<rect x="0" y="24" width="6" height="40" rx="3" fill="{P["ui"][c]}"/>' + T(28, 56, t, P, "ink", 27, 700)
                 + f'<rect x="28" y="76" width="{tw:.0f}" height="26" rx="13" fill="none" stroke="{P["ui"][c]}"/>'
                 + T(round(28 + tw / 2), 94, tag, P, c, 14, 600, "middle")
                 + "".join(T(28, 138 + k * 30, l, P) for k, l in enumerate(lines)) + '</g>')
    o += [T(64, H - 44, "Free and open source (MIT)", P, "teal", 22, 600), T(64, H - 16, "github.com/abhaymin/am2strata · am2strata.am2.ai", P, size=18), '</svg>']
    return "\n".join(o)


def tiers(P):
    W, H = 1600, 960
    o = head(W, H, "Bigger than your RAM: how am2Strata runs a model that does not fit",
             "Three tiers. The weights every token uses sit in GPU memory or RAM; the experts a learned profile ranks hot are pinned in "
             "RAM; the cold experts stay on the SSD and are read in parallel only when the router picks them. Measured on a Mac mini M4 "
             "with 16 GB running GLM-4.7-Flash (17.5 GB): am2Strata generates 8.9 tokens per second without swapping, llama.cpp 3.6 with "
             "plain mmap and 2.05 by default while writing about 98 GB of swap. Prompt: am2Strata 16.9, llama.cpp 17.3 (mmap) and 11.4 "
             "(default).", P)
    o += [T(64, 92, "Bigger than your RAM", P, "ink", 46, 700),
          T(64, 134, "A mixture-of-experts model uses only a few experts per token. am2Strata keeps those close and leaves the rest on the SSD.", P, size=22),
          '<g transform="translate(64 190)">']
    slabs = [("teal", "GPU memory and RAM", ["Attention, shared experts, the output head:", "the weights every token uses, always resident"]),
             ("blue", "Hot experts in RAM", ["A learned profile ranks them; they are pinned so", "the OS cannot evict them (98% of routing hits here)"]),
             ("mauve", "Cold experts on the SSD", ["Read only when the router picks them: all of a", "step's experts at once, in parallel, megabytes each"])]
    for k, (c, t, lines) in enumerate(slabs):
        u = P["ui"][c]
        o.append(f'<g transform="translate(0 {k * 176})"><path d="M40 0 H600 L640 34 H80 Z" fill="{u}" opacity=".25"/>'
                 f'<rect x="40" y="34" width="600" height="118" rx="14" fill="{P["card"]}" stroke="{u}" stroke-width="2.5"/>'
                 f'<rect x="40" y="34" width="10" height="118" rx="4" fill="{u}"/>' + T(72, 74, t, P, "ink", 26, 700)
                 + "".join(T(72, 106 + j * 26, l, P) for j, l in enumerate(lines)) + '</g>')
    u = P["ui"]["teal"]
    o += [f'<path d="M668 480 C 712 420, 712 250, 668 190" fill="none" stroke="{u}" stroke-width="3" stroke-dasharray="7 7"/>',
          f'<path d="M660 198 L668 186 L678 196" fill="none" stroke="{u}" stroke-width="3"/>',
          T(736, 342, "promoted when they heat up", P, "teal", 17, 400, "middle", ' transform="rotate(-90 736 342)"'),
          T(40, 556, "Prompts go in chunks: each chunk's tokens are grouped by expert,", P),
          T(40, 584, "so an expert is read once per chunk, not once per token.", P), '</g>',
          '<g transform="translate(860 190)">',
          f'<rect width="676" height="660" rx="22" fill="{P["card"]}" stroke="{P["line"]}" stroke-width="2"/>',
          T(36, 54, "Measured: a 17.5 GB model on a 16 GB Mac", P, "ink", 26, 700),
          T(36, 86, "Mac mini M4, GLM-4.7-Flash (Q4_K_XL), 512-token prompt, 64 tokens", P, size=18)]

    def bars(y, title, rows, mx):
        g = [T(36, y, title, P, "ink", 21, 600)]
        for i, (name, v, label, us) in enumerate(rows):
            yy = y + 44 + i * 50
            w = 300 * v / mx
            g += [T(36, yy, name, P, "ink" if us else "muted", 18, 700 if us else 400),
                  f'<rect x="230" y="{yy - 20}" width="300" height="30" rx="8" fill="{P["ui"]["track"]}"/>',
                  f'<rect x="230" y="{yy - 20}" width="{w:.0f}" height="30" rx="8" fill="{P["ui"]["teal"] if us else P["muted"]}"/>',
                  T(546, yy + 2, label, P, "teal" if us else "muted", 18, 700 if us else 400)]
        return g
    o += bars(146, "Generation, tokens per second", [("am2Strata", 8.9, "8.9", True), ("llama.cpp, mmap", 3.58, "3.6", False),
                                                     ("llama.cpp, default", 2.05, "2.05", False)], 9)
    o += bars(364, "Prompt, tokens per second", [("am2Strata", 16.85, "16.9", True), ("llama.cpp, mmap", 17.31, "17.3", False),
                                                 ("llama.cpp, default", 11.39, "11.4", False)], 18)
    o += [f'<line x1="36" y1="550" x2="640" y2="550" stroke="{P["line"]}" stroke-width="2"/>',
          f'<text x="36" y="586" font-size="20" fill="{P["ink"]}"><tspan font-weight="700" fill="{P["teal"]}">2.5x</tspan> llama.cpp\'s generation, no swapping</text>',
          T(36, 614, "llama.cpp's default swapped ~98 GB; Metal ran out of memory.", P, "ink", 20),
          T(36, 642, "am2Strata at 8-10 threads; llama.cpp measured at 4. Tokens checked against llama.cpp.", P, size=15),
          '</g>', T(64, 920, "am2Strata · free and open source (MIT) · github.com/abhaymin/am2strata · numbers in docs/HYBRID-SPEED.md", P, size=16),
          '</svg>']
    return "\n".join(o)


def cluster(P):
    W, H = 1200, 800
    o = head(W, H, "am2Strata cluster: pairing a worker with a master",
             "Two machines. The master holds the main API key and talks to clients; it keeps node-tls.json and cluster-workers.json. "
             "The worker keeps its own key, node-key and node-tls.json, and cluster-masters.json with only the hash of the key it gave. "
             "Steps: 1 install on both with a main api_key each; 2 make a worker with cluster role worker, started on 0.0.0.0, which shows "
             "a 6-digit code new every 10 minutes; 3 pair on the master in Settings, Cluster by typing the code (5 wrong codes lock pairing "
             "for 10 minutes); 4 use it: the master answers with the worker's model and hands it overflow. After pairing, every connection "
             "is mutual TLS 1.3 with pinned certificates on port 8443. Remove a worker in Settings, Cluster, Remove.", P)
    o += [T(48, 64, "A cluster in four steps", P, "ink", 30, 700),
          T(48, 96, "One master that people and apps talk to; workers that answer for it. Keys are named here, never shown.", P, size=17)]

    def box(x, title, sub, keeps, foot_t, foot):
        return ([f'<g transform="translate({x} 128)"><rect width="430" height="296" rx="16" fill="{P["card"]}" stroke="{P["line"]}" stroke-width="2"/>',
                 T(24, 40, title, P, "ink", 24, 700), T(24, 66, sub, P, size=15), T(24, 104, "It keeps", P, "ink", 16, 700)]
                + [T(24, 130 + i * 24, k, P, size=15) for i, k in enumerate(keeps)]
                + [T(24, 240, foot_t, P, "ink", 16, 700), T(24, 264, foot, P, size=15), '</g>'])
    o += box(48, "Master", "e.g. the Linux PC with the big GPUs",
             ["• its main API key (run config \"api_key\")", "• node-tls.json (its certificate = its pin)",
              "• cluster-workers.json: each worker's url,", "   pin and the key it was given (0600)"],
             "Clients use only the master", "their keys are never sent on to a worker")
    o += box(722, "Worker", "e.g. a Mac mini; \"cluster\": {\"role\": \"worker\"}",
             ["• its own main API key (never leaves it)", "• node-key (ed25519) and node-tls.json",
              "• cluster-masters.json: each master's pin", "   and only the HASH of its key (0600)"],
             "Ports", "8000 API · 8443 cluster (TLS) · 8001 HTTPS")
    a, b, c = P["ui"]["blue"], P["ui"]["green"], P["ui"]["teal"]
    o += [f'<line x1="488" y1="196" x2="704" y2="196" stroke="{a}" stroke-width="3"/><path d="M700 188 L714 196 L700 204 z" fill="{a}"/>',
          T(600, 184, "pair: code + master's pin", P, "blue", 14, 600, "middle"),
          f'<line x1="712" y1="262" x2="496" y2="262" stroke="{b}" stroke-width="3"/><path d="M500 254 L486 262 L500 270 z" fill="{b}"/>',
          T(600, 250, "a new key + worker's pin", P, "green", 14, 600, "middle"),
          f'<line x1="496" y1="330" x2="704" y2="330" stroke="{c}" stroke-width="5"/><path d="M700 320 L716 330 L700 340 z" fill="{c}"/><path d="M500 320 L484 330 L500 340 z" fill="{c}"/>',
          T(600, 316, "mutual TLS 1.3, pins checked", P, "teal", 14, 700, "middle"),
          T(600, 358, "chats, model list, queue", P, size=14, anchor="middle")]
    steps = [("Install on both", ["The one-line installer,", "a model on each, and a", "main \"api_key\" in each", "run config (your own", "long random value)."]),
             ("Make a worker", ["Add \"cluster\":", "{\"role\": \"worker\",", "\"name\": \"macm4\"}; start", "it with --host 0.0.0.0.", "It shows a 6-digit code", "(new every 10 minutes)."]),
             ("Pair on the master", ["Settings › Cluster: the", "worker \"wants to join\";", "type its code, Pair.", "Over a VPN: pair by its", "address. 5 wrong codes", "lock pairing 10 minutes."]),
             ("Use it", ["Ask the master for the", "worker's model: the answer", "carries x-strata-worker.", "The master also hands a", "worker overflow when busy", "and it has the model idle."])]
    cols = ["teal", "blue", "mauve", "pink"]
    for i, (t, lines) in enumerate(steps):
        x = 48 + i * 278
        o.append(f'<g transform="translate({x} 452)"><rect width="262" height="268" rx="14" fill="{P["card"]}" stroke="{P["line"]}" stroke-width="2"/>'
                 f'<circle cx="34" cy="38" r="17" fill="{disc(P, cols[i])}"/>'
                 f'<text x="34" y="44" font-size="17" font-weight="700" text-anchor="middle" fill="{P["on_disc"]}">{i + 1}</text>'
                 + T(62, 44, t, P, "ink", 19, 700) + "".join(T(20, 84 + j * 26, l, P, size=16) for j, l in enumerate(lines)) + '</g>')
    o += [T(48, 764, "Remove: Settings › Cluster › Remove (the worker forgets that master; its key stops working at once).", P, size=15), '</svg>']
    return "\n".join(o)


def disc(P, c):
    """A step number's disc: the bright hue in dark mode, the deep text hue in light mode (white digits on it)."""
    return P["ui"][c] if P["disc"] == "ui" else P[c]


def check(svg, P, name):
    """Every text colour reaches 7:1 on both backgrounds it can sit on; the step numbers on their coloured discs too."""
    bad = [(c, round(min(ratio(c, P["card"]), ratio(c, P["bg"])), 2))
           for c in set(re.findall(r'fill="(#[0-9a-f]{6})">', svg)) - {P["on_disc"]}
           if min(ratio(c, P["card"]), ratio(c, P["bg"])) < 7]
    bad += [(f'{P["on_disc"]} on {k}', round(ratio(P["on_disc"], disc(P, k)), 2)) for k in ("teal", "blue", "mauve", "pink")
            if ratio(P["on_disc"], disc(P, k)) < 7]
    if bad:
        raise SystemExit(f"{name}: below the AAA contrast: {bad}")


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    for theme, P in PALETTES.items():
        for name, f in (("info-overview", overview), ("info-tiers", tiers), ("cluster-setup", cluster)):
            svg = f(P)
            check(svg, P, f"{name}-{theme}")
            for d in ("site/assets", "docs/media"):
                with open(os.path.join(root, d, f"{name}-{theme}.svg"), "w", encoding="utf-8") as fh:
                    fh.write(svg + "\n")
            print(f"{name}-{theme}.svg  ok")
