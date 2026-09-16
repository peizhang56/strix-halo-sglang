# SGLang on Strix Halo (gfx1151) — local agentic Claude Code

Run a 27B model on an **AMD Ryzen AI Max / Radeon 8060S** iGPU under Windows 11 + WSL2,
serve it with SGLang, and point **Claude Code** at it. The whole agent loop stays on the
box — no external API calls.

This repo is self-contained: everything below runs from the scripts in here.

| script | where it runs | what it does |
|---|---|---|
| `launch_docker.sh` | WSL host | starts the ROCm/SGLang container, mounts this repo at `/workspace` |
| `patches/apply.sh` | container | applies the int4 + GEMM-tuning patches to the image's SGLang |
| `sglang_server.sh` | container | starts the model server |
| `stop.sh` | container | stops it (**always use this**, not `pkill`) |
| `claude_local.sh` | container | starts Claude Code against that server |
| `chat_template_qwen3_agentic.jinja` | — | patched chat template, load-bearing (see [Traps](#traps)) |

**Validated on:** HP ZBook Ultra G1a · Windows 11 build 26200 · Radeon 8060S driver
32.0.22018.5 · ~96 GiB RAM · Ubuntu 24.04 on WSL2 · ROCm 7.2.1 host / 7.2.4 in-image ·
ROCDXG 1.2.0 · image `rocm/sgl-dev:v0.5.19-rocm724-gfx1151-20260914`.

---

## Quick path

If the machine is already set up, the whole demo is five commands:

```bash
# host (WSL)
./launch_docker.sh && docker exec -it sglang-dev bash

# container
./patches/apply.sh                                  # after every launch_docker.sh
SGLANG_SPEC=dflash ./sglang_server.sh 2>&1 | tee server.log   # terminal 1
./claude_local.sh                                   # terminal 2, once /health is 200
```

Everything else on this page is the one-time setup behind those five lines.

---

## 1. Windows host

### 1.1 AMD WSL driver

Install [AMD Software: Adrenalin Edition 26.2.2 for WSL2](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html)
and restart Windows.

Do not assume a newer generic Adrenalin driver carries the same WSL compute support —
check AMD's current [ROCm WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html) first.

### 1.2 WSL2 + Ubuntu 24.04

**PowerShell as Administrator:**

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --list --verbose          # Ubuntu must show VERSION 2
```

If it shows version 1: `wsl --set-version Ubuntu-24.04 2`.

If `wsl --install` only reports that WSL is not installed, enable the features explicitly,
restart, and retry:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Open Ubuntu once and create your Linux user. If it was installed with `--no-launch` it can
default to `root` without ever showing the account-creation screen:

```powershell
wsl -d Ubuntu-24.04 -u root -- useradd --create-home --shell /bin/bash --groups sudo <user>
wsl -d Ubuntu-24.04 -u root passwd <user>
```

then add to `/etc/wsl.conf` inside Ubuntu:

```ini
[user]
default=<user>
```

and `wsl --terminate Ubuntu-24.04`.

---

## 2. ROCm and ROCDXG inside WSL

AMD supports Strix Halo in WSL through **ROCDXG**: there is no `/dev/kfd` here, so HSA
reaches the iGPU through `/dev/dxg` via `librocdxg.so`. Everything in this section runs in
Ubuntu.

```bash
sudo apt update
sudo apt install -y wget gnupg ca-certificates
sudo install -d -m 0755 /etc/apt/keyrings

wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key |
  gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null

sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null <<'EOF'
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.1 noble main
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.2.1/ubuntu noble main
EOF

sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null <<'EOF'
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF

sudo apt update && sudo apt install -y rocm
```

Then the ROCDXG release matched to ROCm 7.2.x
([quickstart](https://github.com/ROCm/librocdxg#quickstart)):

```bash
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb
echo "3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e  rocdxg-roct_1.2.0_amd64.deb" |
  sha256sum --check
sudo dpkg -i rocdxg-roct_1.2.0_amd64.deb
sudo usermod -aG render,video "$USER"

echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc && source ~/.bashrc
```

Verify:

```bash
ls -l /dev/dxg
rocminfo | grep -E 'gfx|Wavefront|Compute Unit'
```

Expected:

```text
Name:                    gfx1151
Marketing Name:          AMD Radeon(TM) 8060S Graphics
Compute Unit:            40
Wavefront Size:          32
```

`rocminfo`, not `rocm-smi`, is the installation check. AMD-SMI has limited functionality
under WSL2 — see [Traps](#traps).

---

## 3. Docker in WSL

```bash
sudo apt update
sudo apt install -y docker.io docker-buildx
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker
```

Then `wsl --terminate Ubuntu-24.04` from PowerShell once so the `docker` group applies.

Pull the image (~30 GB; it already contains ROCm 7.2.4, torch 2.9.1 and an SGLang
checkout at `/sgl-workspace/sglang`):

```bash
docker pull rocm/sgl-dev:v0.5.19-rocm724-gfx1151-20260914
```

> No Dockerfile, no PR stack, no source build. Earlier revisions of this guide built an
> image from SGLang PR #33939 — that is obsolete. Use the published image.

---

## 4. Models

Models live **on the WSL host** and are bind-mounted into the container, so they survive
`--recreate` and are shared between containers. Never download inside a container to a
path that isn't the mount.

```bash
hf download amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16   # 17.8 GB — the default
hf download incoai/Qwen3.8-27B-DFlash2             # 3.85 GB — the DFLASH drafter
hf download Qwen/Qwen3.8-27B                       # 51.2 GB — bf16, optional
```

They land in `~/.cache/huggingface`, which `launch_docker.sh` mounts at
`/root/.cache/huggingface`. Do not delete that directory when rebuilding containers.

---

## 5. Launch the container

On the **host**, from this repo:

```bash
./launch_docker.sh
docker exec -it sglang-dev bash
```

That is the whole command. It starts the container detached on `sleep infinity`, mounts
this repo at `/workspace`, mounts the HF cache, publishes port 30000, and passes the iGPU
through. Options: `--name`, `--image`, `--hf-cache`, `--host-dir`, `--port`, `--recreate`
(`--help` for details).

Only two libraries are mounted from the host — `libdxcore.so` and `librocdxg.so` — because
there is no `/dev/kfd` under WSL and the container's HSA runtime dlopens `librocdxg.so` to
reach `/dev/dxg`. ROCm userspace otherwise comes from the image. **Do not also mount the
host `libhsa-runtime64`**: it is byte-identical to the image's copy, and the path the old
guide used does not exist, so Docker silently creates an empty directory and mounts it over
the real library.

Sanity check inside the container:

```bash
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_properties(0).gcnArchName)'
# True gfx1151
```

### 5.1 One-time setup inside the container

**Re-run after every `launch_docker.sh`.** These edits live in `/sgl-workspace/sglang`,
which is inside the image, so a recreated container loses them.

```bash
cd /workspace
./patches/apply.sh
```

That applies two patches, both idempotent:

- `quark-int4-w4a16.patch` — the W4A16 scheme the int4 checkpoint needs. Without it the
  default model will not load.
- `0001-gfx1151-w4a16-qwen38-configs.patch` — 14 missing Triton GEMM configs for
  Qwen3.8-27B's shapes. Worth **+16% decode**, bit-identical output. SGLang's shipped
  gfx1151 table was tuned for a different model and this one's `qkv_proj` was missing at
  every M bucket, falling through to a generic heuristic.

Install Claude Code (also lost on recreate):

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"     # add to ~/.bashrc
claude --version                          # verified with 2.1.270
```

You do **not** need to configure `~/.claude.json`, and you should **not** copy one from
another machine: it carries that machine's API tokens and pins `ANTHROPIC_BASE_URL` to the
corporate gateway, which is [trap 1](#traps). `claude_local.sh` passes `--settings`, which
outranks user config, so a clean install needs nothing.

---

## 6. Run the server

Terminal 1, inside the container:

```bash
cd /workspace
SGLANG_SPEC=dflash ./sglang_server.sh 2>&1 | tee server.log
```

Keep the `tee` — the script logs to stdout only, and `server.log` is where you read cache
hits and throughput.

Terminal 2, wait for ready. Poll the **status code**: `/health` returns 503 for the entire
warmup, so `curl -s .../health && echo ok` reports success while the model is still loading.

```bash
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:30000/health)" = "200" ]; do
  printf '.'; sleep 10
done; echo " READY"
```

**Expect 3–9 minutes** — dominated by weight load, and it swings with the host page cache
(104 s warm vs 413 s cold). A first launch on a fresh image adds more while aiter JIT-builds
and Triton compiles; both then cache under `/workspace/.cache/`.

**Do not send traffic before you see 200.** The server runs its own warmup request
concurrently and kills the whole tree if that request times out.

Stop with `./stop.sh` — see [Traps](#traps) for why never `pkill`.

### Which decoder

`SGLANG_SPEC=eagle|dflash|none`. **Prefer `dflash`** for agentic work — it is faster on
every end-to-end measurement:

| | EAGLE/MTP | **DFLASH** |
|---|---|---|
| GSM8K 10 q, `--parallel 4`, warm | 46.9 s / 31.1 tok/s | **37.2 s / 39.0 tok/s** |
| single-stream decode | 4.84 tok/s, ITL 207 ms | **6.89 tok/s, ITL 145 ms** |
| accept length | 3.1 / 4 | **5.6 / 8** |
| free mem after load | 12.96 GB | **15.44 GB** |
| GSM8K accuracy | 1.000 | 1.000 |

DFlash2 is a block-diffusion drafter: it proposes 8 tokens in one pass and a learned
selector traces a path through them, where EAGLE walks 3 sequential steps for 4 tokens.
Decoding is lossless. It works against the int4 target — even though its card lists bf16
`Qwen/Qwen3.8-27B` as the base — because it taps the target `lm_head`, which the Quark
checkpoint leaves dense BF16.

### The resolved command

For reference; `sglang_server.sh` builds this:

```bash
python3 -m sglang.launch_server \
    --model-path amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 \
    --attention-backend triton \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --context-length 65536 \
    --chunked-prefill-size 1024 \
    --max-running-requests 4 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template /workspace/chat_template_qwen3_agentic.jinja \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 \
    --speculative-draft-attention-backend triton \
    --speculative-num-draft-tokens 8
```

Five of those should not be changed casually:

- **`--attention-backend triton`** — AITER targets CDNA, not gfx1151.
- **`--mem-fraction-static 0.85`** — **do not raise to 0.93.** It hung the whole host, hard
  restart required. The ~103 GB of GTT is carved out of the same physical RAM the host is
  using, so 0.93 asks for ~96 GB of a pool that mostly isn't there and WSL2 wedges. The trap
  is that it doesn't fail every time — one 0.93 server ran clean and scored GSM8K 1.000 six
  times. A single clean run is not evidence that it is safe.
- **`--chunked-prefill-size 1024`** — worth **3.7× on prefill** (43.9 → 163.1 tok/s at 8k).
  Above 256 rows the W4A16 path has no ROCm AWQ GEMM and dequantizes the whole weight to
  bf16 per layer per forward; a 1024-row chunk keeps that tile resident, an 8192-row chunk
  does not. The optimum is sharp (512 is worse) and specific to this quantization.
- **`--speculative-num-draft-tokens 8`** — must equal the drafter's `block_size`. SGLang
  errors if they disagree, and defaults to 16 if it cannot read the config.
- **`--chat-template`** — carries two required patches. See traps 3 and 5.

Useful overrides: `SGLANG_CONTEXT_LEN` (65536), `SGLANG_MEM_FRACTION` (0.85),
`SGLANG_MAX_RUNNING` (4), `SGLANG_CHUNKED_PREFILL` (1024), `SGLANG_NO_SPEC=1` for the
conservative no-graph path, and `./sglang_server.sh Qwen/Qwen3.8-27B` for bf16 weights.

### Smoke test

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"What is 2 + 2? Reply with only the number."}],
       "temperature":0,"max_tokens":8,"chat_template_kwargs":{"enable_thinking":false}}'
```

---

## 7. Claude Code against the local server

SGLang serves a native Anthropic-compatible `/v1/messages`, so no LiteLLM or
claude-code-router shim is needed. The route is always on; there is no flag for it.

Terminal 2, once `/health` is 200:

```bash
cd /workspace
./claude_local.sh                        # interactive
./claude_local.sh -p 'Reply: LOCAL OK'   # one-shot smoke test
```

Run it from a **normal shell, not from inside another Claude Code session** (trap 2).

The script gates on the server being ready, writes a settings file that redirects the
endpoint, trims the tool schemas, and sets `--effort low`.

| variable | default | what it does |
|---|---|---|
| `CLAUDE_LOCAL_TOOLS=all` | `lean` | send all tool schemas (+~10.9k tokens per cold turn) |
| `CLAUDE_LOCAL_DROP_TOOLS` | agent/cron/worktree set | override which tools are dropped |
| `CLAUDE_LOCAL_EFFORT` | `low` | `low`/`medium`/`xhigh`; `high` now maps to `low` — see trap 3 |
| `SGLANG_HOST` | `127.0.0.1` | use `localhost` from the host; port 30000 is published |

Dropping the tool schemas is the single biggest lever. Measured on a real one-word `hi`
turn: 21 tool schemas were 13,253 of 21,094 tokens (63%). `--disallowed-tools` removes them
from the request *body*, not just the permission list — 21 tools → 6 takes the whole request
to 8,511. The dropped set is agent/cron/worktree machinery, and `Agent`/`Workflow`/
`SendMessage` are worse than useless here: they spawn further model calls, each another
multi-minute prefill.

### Verify it is actually local

This matters more than it sounds — two separate failure modes answer you correctly *while
never touching the local server* (traps 1 and 2). Watch the prefill count move:

```bash
before=$(grep -c 'Prefill batch' server.log)
./claude_local.sh -p 'Reply: LOCAL OK'
grep -c 'Prefill batch' server.log        # must be higher than $before
```

If the count does not move, you are still talking to the gateway.

### What to expect

Prefill dominates. Decode is ~7 tok/s single-stream under DFLASH, so ask for short answers.

| | |
|---|---|
| first turn, cold cache (lean tools, ~8.5k tokens) | ~1 min |
| follow-up turn, warm prefix cache | seconds |
| GSM8K 10 q accuracy | **1.000** |
| aggregate throughput, `--parallel 4` | ~39 tok/s |

Read the cache behaviour live — this one line tells you whether a turn was cheap:

```bash
tail -f server.log | grep --line-buffered -oE '#new-token: [0-9]+, #cached-token: [0-9]+'
```

A healthy follow-up looks like `#new-token: 34, #cached-token: 6656`. A turn showing
`#cached-token: 0` re-prefilled the whole conversation — see trap 5.

This buys independence from the gateway, not speed.

---

## Demo runbook

Rough timings for a hackathon slot. Do steps 1–2 **before** the audience arrives.

| | step | time |
|---|---|---|
| 1 | `./launch_docker.sh && docker exec -it sglang-dev bash` | ~10 s |
| 2 | `./patches/apply.sh` then `SGLANG_SPEC=dflash ./sglang_server.sh 2>&1 \| tee server.log` | 3–9 min |
| 3 | poll `/health` until 200 | — |
| 4 | `./claude_local.sh -p 'Reply: LOCAL OK'` + prefill-count check | ~1 min |
| 5 | `./claude_local.sh`, ask it to read and edit a file in `/workspace` | ~1 min/turn |

Have `tail -f server.log | grep -oE '#new-token: [0-9]+, #cached-token: [0-9]+'` on a
second pane — watching `#cached-token` climb is the most legible proof it is running here.

**Run one session at a time.** With `--max-running-requests 4` a second session competes
for the same batch, and a queued turn silent past 300 s makes the client give up with
"check your network" mid-demo. That is contention, not a network fault.

---

## Traps

Every one of these looks like something it is not.

**1. `export ANTHROPIC_BASE_URL` does nothing.** `~/.claude.json` has an `env` block
pinning the endpoint to the gateway, and a settings-file `env` block **overrides the
process environment**. Your export is silently ignored and every request still goes to the
gateway. The override must enter at higher precedence — `claude --settings <file>`, which
is exactly why `claude_local.sh` writes a settings file instead of exporting.

**2. A nested `claude` never calls the API.** Started from inside another Claude Code
session it inherits `CLAUDE_CODE_MESSAGING_SOCKET` / `CLAUDE_CODE_CHILD_SESSION` and
delegates to the parent — answering correctly and fast while the local server sees zero
traffic. `claude_local.sh` scrubs those. Always confirm with the prefill count.

**3. Effort `high` returns HTTP 500 on the stock template.** Claude Code sends one of six
effort levels in `output_config`; the model's template accepts only `xhigh`/`medium`/`low`,
so the default `high` raises a Jinja error and the server answers 500 *before any inference
runs*. Reported as "500 … usually temporary", which is misleading — it is deterministic.
`claude_local.sh` passes `--effort low`, and the patched chat template collapses all six
levels onto the three the model knows.

Two details in that remap. It maps **`high` → `low`**, not to `xhigh`, because Claude Code's
*internal* summarise call hardcodes `{"effort": "high"}` and ignores the session setting —
and that internal call is the expensive one: 88% of one 17-minute turn was a single decode
of it. Replaying the captured request: `high` 1281 output tokens (56% thinking) vs `low` 502.
The consequence is that **you can no longer ask for `high` on the main turn** — it now means
`low`. Use `medium` or `xhigh` explicitly; those pass through untouched. The template also
defaults to `low` when no effort is sent at all, which covers the GSM8K harness, plain curl
to `/v1/messages`, and Claude Code's internal `WebFetch`/summarise calls — all of which were
silently getting the most expensive setting.

**4. A busy server looks exactly like a dead one.** `/health` is not a cheap probe — it
pushes a real generate request through the scheduler, so a large turn blocks it for
*minutes* while the log says `Health check failed...` and the server is happily prefilling.
`curl` reports `%{http_code}` as `000` for a timeout exactly as for a refused connection,
so the status code alone cannot distinguish them; the **exit code** can (7 refused, 28
timeout). Probe `/model_info` first — answered in the HTTP layer without touching the
scheduler — then use `/health` for readiness. `claude_local.sh` already does this.

**5. A trailing `system` message destroys the prefix cache.** Claude Code appends a fresh
`<system-reminder>` as a `role: "system"` message at the *end* of `messages` on most turns.
SGLang probes the chat template for inline-system support; if the probe fails it **hoists
every mid-conversation system message into the leading system block**, rewriting the prompt
at token 0 so the radix tree matches nothing and the whole conversation re-prefills — every
turn. The stock template failed the probe because it raised on a non-leading system message.
`chat_template_qwen3_agentic.jinja` renders it in place instead. Measured on an identical
conversation: `#cached-token` 0 → 1408, 15.2 s → 1.4 s. This is worth more than every other
prefill optimisation combined.

**If you ever replace the chat template, re-check traps 3 and 5 — both fixes live in it,
and it only takes effect on a server restart.**

**6. Never stop the server with `pkill`.** A running server is ~37 processes, not 3: the
launcher, `sglang::scheduler`, `sglang::detokenizer`, a `resource_tracker` and ~32 Torch
inductor compile workers. Only the first three have "sglang" in their name, so
`pkill -f sglang` misses the rest and leaves one holding port 30000 — the usual cause of a
failed relaunch. `./stop.sh` walks the real process tree, shuts down parent-first,
escalates to SIGKILL only if the tree won't drain, and waits for the port. Idempotent,
~9 s. It reports leftover zombie entries; those are already-dead table entries holding no
memory, no GPU handle and no port. The residual one is `resource_tracker`, which by design
exits only after the launcher has closed its pipe — i.e. after the process that would reap
it is gone. No shutdown order fixes that; the permanent fix is host-side `--init` on the
`docker run` in `launch_docker.sh` so tini reaps orphans (PID 1 here is `sleep infinity`,
which never does). Not currently applied.

**7. There is no GPU telemetry here.** `rocm-smi` is installed but every call prints
`Driver not initialized (amdgpu not found in modules)` and does nothing — WSL2 has no
`amdgpu` module and no `/dev/kfd`. Worse, it **exits 0 while printing that error**, so
never gate a script on it. For memory numbers, grep `server.log` for `avail mem`.
Also ignore `free`: WSL exposes ~103 GB unified GTT carved out of the same pool as host
RAM, so host "used" and GPU "used" are the same bytes counted once.

**8. Corporate TLS interception (Zscaler).** On an AMD-managed machine, fresh Ubuntu
rejects intercepted certificates:

```text
SSL certificate problem: unable to get local issuer certificate
```

Export your organization's root certificate as Base-64 X.509 and install it in WSL:

```bash
sudo cp /mnt/c/Users/<you>/Desktop/zscaler-root.cer \
  /usr/local/share/ca-certificates/zscaler-root.crt
sudo update-ca-certificates
curl -I https://pypi.org && git ls-remote https://github.com/sgl-project/sglang.git HEAD
```

`launch_docker.sh` then mounts `/usr/local/share/ca-certificates` into the container and
rebuilds the bundle there automatically, and sets `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`
because Python's `certifi` ships its own bundle and ignores the system one. Do not use
`--no-check-certificate`, `git config http.sslVerify false`, or a global pip `trusted-host`.
Keep the certificates out of git.

---

## Known-untested paths

These are upstream `/v1/messages` issues — check them before blaming local config:

- multi-tool turns can crash the SDK on the second tool call
  ([#24293](https://github.com/sgl-project/sglang/issues/24293))
- `WebSearch` / `WebFetch` may be rejected
  ([#22655](https://github.com/sgl-project/sglang/issues/22655))
- streaming reports `input_tokens: 0`, so the context meter reads 0% and auto-compaction
  never fires ([#20678](https://github.com/sgl-project/sglang/issues/20678))

## Also known

- **CUDA graphs are only safe in combination.** Bounded graph capture *on its own* produces
  corrupted first output and a stalled second request. It runs clean together with a
  speculative decoder and `SGLANG_MAMBA_SSM_DTYPE=bfloat16`, which is what
  `sglang_server.sh` sets. Do not enable it by itself.
- **Memory is not the constraint.** At 65536 context the KV pool is 667,516 tokens with
  15.4 GB still free. The pool is sized from `--mem-fraction-static`, not from the context
  length, so raising context costs nothing until a request actually uses it. Model max is
  262144.
- **`launch_server_temp_dflash.sh`-style configs are wrong for this box.** If you find one
  floating around with `--mem-fraction-static 0.93`, `--chunked-prefill-size 4096`, no
  `--context-length` and no `--chat-template`: all four are wrong here, in that order of
  severity. Use `SGLANG_SPEC=dflash ./sglang_server.sh`.
