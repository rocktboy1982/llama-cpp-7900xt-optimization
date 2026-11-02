# Avalon on an RX 7900 XT: Qwen3.8-Flash-Next (125B MoE) at ~100 t/s

Goal: serve **Qwen3.8-Flash-Next** (125B mixture-of-experts, 512 experts, 48 layers) from a single RX 7900 XT (20 GB) with [**Avalon**](https://github.com/freriadoc/Avalon), a llama.cpp fork that keeps the dense weights, the KV cache and a cache of the hottest experts on the GPU and the rest of the experts in system RAM. With its MTP draft head plus `ngram-simple` speculation it decodes at about **100 t/s** on new text and **~170-180 t/s** on code edits, and reads prompts at about **1,750 t/s**, with a 262,144-token context.

Measured 2026-11-02 on the same box as the rest of this repository: Ryzen 7 5700G (PCIe Gen3), 57 GB RAM, RX 7900 XT, Ubuntu 26.04, Linux 7.0, **ROCm 10.0.0**, Avalon `a91372e`, model `freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF` (IQ3_KS, 33 shards, ~89 GiB on disk).

| | |
|---|---|
| Status | Everything below was run on that machine. The scripts were syntax-checked; `build-avalon.sh` was not run end to end in a fresh clone (the build it performs was) |
| Engine | Avalon `a91372e`, built with ROCm 10 (Avalon's installer only accepts ROCm 7.x; see 2.2) |
| Port | 11437 via a systemd user unit (started at boot with lingering), so opencode / Copilot CLI need no change |
| Speculation | `--spec-type ngram-simple,draft-mtp` (Avalon's default command uses `draft-mtp` only; see 1.4) |

## 1. Results

### 1.1 Short prompts (Avalon's own `bench.py`: 5 prompts x 2 passes, 512 tokens, temperature 0, thinking off)

| Run | 1st pass median | 2nd pass median | Best | Draft acceptance |
|---|---:|---:|---:|---|
| Linux, ROCm 10 (this guide) | 94.6 t/s | **100.3 t/s** | 115.2 t/s | 55-65 % prose, 88-93 % code |
| Same model and flags on Windows (HIP SDK 7.2 / ROCm 10 runtime, earlier run) | 62-72 t/s | | ~75 t/s | 60-92 % |

Code and lists run at 104-115 t/s, prose at 86-100 t/s: the MTP head guesses structured text better. The first request after a start is slow (6.7 t/s here) while the experts are read from disk.

### 1.2 Strata's benchmark suite (`tools/strata_bench.py --long`, 3 runs each, medians)

The same suite this repository's Strata numbers come from, pointed at Avalon over HTTP (Strata's script only refuses non-Strata servers by name; a three-line wrapper replaced that check, nothing else changed).

| Test | Result | Range |
|---|---:|---|
| Short chat decode | **96.4 t/s** | 65.6-106.3 |
| Short chat time to first token | 0.36 s | 0.31-0.70 |
| 4K prompt, fresh: prefill / first token | 1,180 t/s / 3.4 s | 707-1,183 |
| 16K prompt, fresh: prefill / first token | **1,771 t/s** / 8.9 s | 1,566-1,813 |
| 32K prompt, fresh: prefill / first token | **1,805 t/s** / 17.5 s | 1,471-1,818 |
| Agent session (system prompt + 4 tools, 5 follow-up turns of ~1,500 words each, ends at ~14.5k tokens), warm runs 2-3 | first token 4.1-4.3 s per turn (max 9.2 s), decode 66-72 t/s, 69 % of each prompt reused from the cache | |
| Same, run 1 (first session after the server started) | first token up to 89 s, decode 52 t/s | |

Run 1 of the agent session is the cold start: 89 GiB of experts do not fit in 57 GB of RAM, so the first session pulls them from disk. Raw data: [`../data/results-avalon-strata-bench.json`](../data/results-avalon-strata-bench.json). The suite's own README misreports two fields for a non-Strata server (GPU "CPU only", a cache hit of 3,028,500 %), so the cache figure above was computed from the raw per-turn data.

### 1.3 Real use

Over 17 requests from opencode and Copilot CLI before a reboot: decode median ~78 t/s (up to 129), prefill median ~1,400 t/s on prompts over 2k tokens, largest prompt 76,102 tokens with no error. Agent prompts are long, so real decode sits below the short-prompt benchmark.

### 1.4 `ngram-simple` next to MTP (2026-11-03)

Taken from this repository's llama.cpp setup, where MTP + `ngram-simple` raised code edits from 98 to 175 t/s on the 27B. Avalon accepts the same list: `--spec-type ngram-simple,draft-mtp`. Two workloads, back to back, same server flags otherwise: Avalon's `bench.py` (new text) and a code-edit test (return a 180-line Python file with a rename, or with docstrings added; 1,500 tokens, 2 rounds).

| Workload | `draft-mtp` (Avalon default) | `ngram-simple,draft-mtp` |
|---|---:|---:|
| Code edits, median (2nd round) | 118.0 t/s (118.3) | **168.1 t/s (180.9)**, range 125-194 |
| Draft tokens accepted on edits | 998-999 of ~1,000 | 1,442-1,462 of 1,510-1,561 |
| New text, `bench.py` 2nd pass median | 95.2 t/s | 92.3 t/s |

* On edits MTP alone is already right almost every time; what limits it is the draft length (`--spec-draft-n-max 2`). `ngram-simple` drafts long runs copied from the prompt (up to 48 tokens), so a correct guess moves the output much further per step.
* New text is unchanged within run-to-run noise (about ±3-5 t/s). The first `bench.py` pass of the ngram run (74.3 t/s) was right after a server restart and is not counted.
* The edit test flatters ngram speculation (the answer is mostly a copy of the prompt), but agents rewriting files do exactly that. The launcher in this repository uses `ngram-simple,draft-mtp`.

Raw data: [`../data/results-avalon-bench.txt`](../data/results-avalon-bench.txt).

### 1.5 What did not change anything

| Tried | Result | Why |
|---|---|---|
| `-ctk q4_0` instead of `q8_0` (frees ~0.75 GiB) | 96.0 / 98.8 t/s (default 94.6 / 100.3) | `--fit` hands the freed VRAM to the expert cache, which then runs no faster: VRAM stays at 19.74 GiB in every case |
| `-c 131072` instead of 262144 (frees ~1.2 GiB) | 91.5 / 97.8 t/s | same |
| TurboQuant KV cache ([`turboquant.md`](turboquant.md)) | not ported | Only 12 of 48 layers keep a KV cache (2 KV heads, 256 dims), so the whole 256k cache is ~2.4 GiB; `turbo3`/`turbo4` would save ~1 GiB, which the two rows above show is worth nothing here. The patch also collides with Avalon's type IDs 43/44 (IQ3_K/IQ3_KS) and 5 files conflict |
| ROCm 7.2.4 instead of 10 | not needed | [JashicTM/qwen38-flash-next-7900xt](https://github.com/JashicTM/qwen38-flash-next-7900xt) measured 1,741-1,754 t/s prefill with 7.2.4 against 946-975 with 7.10a. ROCm 10 gives 1,771-1,805 t/s here: the gain they saw was leaving 7.10a, not reaching 7.2.4 |

## 2. Setup (Linux, ROCm 10)

### 2.1 ROCm 10

Use [`../scripts/linux/install-rocm10.sh`](../scripts/linux/install-rocm10.sh) as for the llama.cpp guide. Avalon needs the HIP, hipBLAS and rocBLAS development files (`$ROCM/lib/cmake/hip-lang`, `hipblas`, `rocblas`).

### 2.2 Build

```bash
scripts/linux/build-avalon.sh            # clones freriadoc/Avalon @ a91372e into ~/Avalon (AVALON_DIR=... to change), builds llama-server for gfx1100
```

Two things break on ROCm 10 with Avalon's own `install.sh`, which is why the script exists:

1. `install.sh` stops with `ROCm 10.0.0 found; Avalon is tested with ROCm 7.2.4 only`. `FORCE=1` gets past it but also turns every later failure into a warning, and it then writes a launcher for a binary that was never built.
2. CMake fails with `Failed to find ROCm root directory` (from `enable_language(HIP)`), because ROCm 10 keeps clang under `lib/llvm/bin` and the installer only sets the C/C++ compilers. Setting `HIPCXX=$ROCM/lib/llvm/bin/clang++` fixes it.

If the tree was checked out on Windows, every file has CRLF line endings and `install.sh` dies with `set: pipefail: invalid option name`. The script restores the committed LF files (`git checkout -- .` alone does nothing there, because the index's cached file stats still match).

### 2.3 Model

```bash
pip install -U "huggingface_hub>=0.34"
hf download freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF --local-dir ~/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF --exclude '*.jpg' --exclude '*.md'
```

About 89 GiB: 33 IQ3_KS shards, the MTP draft head in `MTP/` and `moe-cache-ranking.bin` (the expert ranking Avalon preloads). Avalon's `install.sh` (with `SKIP_MODEL=0`) also verifies every file against its SHA-256 on Hugging Face.

### 2.4 Chat template fix

The model's template has two problems for agent clients:

| Symptom | Cause | Fix in [`../patches/avalon-chat-template.patch`](../patches/avalon-chat-template.patch) |
|---|---|---|
| HTTP 500 `Unexpected reasoning effort high` from Copilot CLI (it offers low / medium / high / xhigh / max) | the template accepts only `low`, `medium`, `xhigh` (its default) | `high` and `max` map to `xhigh` |
| HTTP 500 `System message must be at the beginning` from opencode | opencode sends extra system messages mid-conversation | a later system message is rendered as a normal system turn |

```bash
# with the server running once without --chat-template-file:
curl -s 127.0.0.1:8080/props | python3 -c "import json,sys; print(json.load(sys.stdin)['chat_template'], end='')" > ~/Avalon-models/avalon-chat-template.jinja
patch ~/Avalon-models/avalon-chat-template.jinja patches/avalon-chat-template.patch
```

The launcher picks the file up automatically when it exists.

### 2.5 Launcher and service

```bash
install -m 755 configs/linux/avalon-server ~/.local/bin/avalon-server
avalon-server                                   # foreground, port 8080; PORT=..., HOST=..., extra flags go to llama-server
# or as a service on 11437:
cp configs/linux/avalon.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now avalon.service
```

The launcher runs Avalon's own command line unchanged and adds:

* `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`: without it, `libhipblas.so.3` and `librocblas.so.5` are not found and `libamdhip64` resolves to a different copy in `/usr/lib`.
* `HIP_VISIBLE_DEVICES=0`: ROCm 10 also lists the 5700G's Vega iGPU as `ROCm1` (29 GB of shared memory). The build has no kernels for it, and `--fit` would place layers on it.
* `--spec-type ngram-simple,draft-mtp` instead of `draft-mtp` (1.4).

Do not rerun Avalon's `install.sh` over an installed launcher: it rewrites `~/.local/bin/avalon-server` without these lines.

**Start at boot.** A user unit normally starts only when you log in. To load Avalon at power-on, enable lingering once: `loginctl enable-linger $USER`. The unit's `ExecStartPre` waits up to 3 minutes for the model files, so a late-mounted model disk does not fail the start. To switch back to another server permanently: `systemctl --user disable --now avalon.service && systemctl --user enable --now <other>.service`.

Loading takes about 80 s (`listening on` in the log). VRAM use is 19.7 GiB of 20. Only one server can hold the card: stop Strata, TabbyAPI or llama.cpp first (the unit's `Conflicts=` does that for Strata).

### 2.6 Clients

* **opencode**: point the provider's `baseURL` at `http://127.0.0.1:11437/v1`; any model name works. Set `reasoningEffort` to `xhigh` for the highest effort (`high` also works with the patched template).
* **Keep opencode's `small_model` (session titles) off Avalon.** `--parallel 1` gives one slot, so the title request waits for the main answer and can hit opencode's 300 s header timeout on a long reply. Point `small_model` at another provider. `--parallel 2` would also work, but Avalon's README advises one slot so the KV pool is not split.
* **Copilot CLI**: an OpenAI-compatible provider on the same URL; pick `high` or `xhigh` in the reasoning picker.

## 3. Why prefill is fast and decode is slower

Before the first output token the server reads the whole prompt (prefill); then it produces one token at a time (decode). With the experts in RAM, every expert a token needs has to cross PCIe to the GPU.

* **Prefill** runs 9,536 tokens per batch (`-ub 9536`), so each expert copied over PCIe is used for thousands of tokens. That is why it is ~17x faster than decode. JashicTM found that 13,312 and above fail with `failed to allocate compute pp buffers`; 9,536 is the safe maximum.
* **Decode** uses each copied expert for one token. Avalon's GPU expert cache, its expert prefetch and the MTP draft head (about 2.2-2.8 tokens per step) are what get it to ~100 t/s on PCIe Gen3. When the answer repeats text from the prompt (code edits), `ngram-simple` drafts whole runs of it, so each expensive step yields more tokens (1.4).
* **The prompt cache matters most for agents.** With `--parallel 1` the processed conversation stays in the slot, so each turn reads only the new part: in the agent session above, 69 % of every follow-up prompt came from the cache and the first token took about 4 s instead of the 8-9 s a fresh 14k prompt would need.
* **Toolchain**: the matrix work runs in rocBLAS/hipBLASLt, so the ROCm version can halve prefill (7.10a did). ROCm 10 is fine.
* **Power**: JashicTM measured +3 % prefill at 274 W against the 246 W default (junction 79 °C). Not tried here.

## 4. Limits

* One machine, one run per configuration; decode noise is about ±3-5 t/s between runs.
* 57 GB of RAM holds about two thirds of the 89 GiB model, so the first session after a start is slow (1.1, 1.2). More RAM removes that.
* Speed only: answer quality, tool-calling accuracy and long-context recall were not measured for Avalon. The llama.cpp/TabbyAPI numbers in the rest of this repository use a different (27B dense) model, so do not compare quality across them.
* `ngram-simple` was measured with its default sizes (`--spec-ngram-simple-size-n 12`, `-size-m 48`); other sizes and the other ngram types (`ngram-mod`, `ngram-map-k`) were not tried.
* Not tried: ROCm 7.2.4 on this box, `--parallel 2`, the 274 W power limit, decode at depth (30k-100k tokens), a TurboQuant port into Avalon.
