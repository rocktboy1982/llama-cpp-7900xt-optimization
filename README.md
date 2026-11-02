# Local LLMs on one AMD RX 7900 XT (20 GB)

How to run large Qwen3.8 models on a single **Radeon RX 7900 XT (20 GB, RDNA3 / gfx1100)** with Linux and ROCm 10, and what every setting is worth, measured on one desktop.

The work went through two model sizes:

1. **October 2026: Qwen3.8-27B (dense, ~12 GB quantized).** The whole model fits in VRAM. Two engines were tuned and compared: upstream **llama.cpp** and **TabbyAPI** (ExLlamaV3).
2. **November 2026: Qwen3.8-Flash-Next (125B mixture-of-experts, ~89 GiB quantized).** The model is 4-5x larger than the card. **Avalon**, a llama.cpp fork, keeps the dense part and the most-used experts on the GPU and the rest in system RAM.

The surprising result: **the 125B model is faster than the 27B** on the same card (about 100 t/s against 62-74 t/s on new text, and 1.8-2.9x the prefill speed, measured at shorter prompts), for reasons explained [below](#why-the-125b-model-is-faster-than-the-27b).

## Which setup to use

| | [llama.cpp, 27B](docs/llama-cpp-guide.md) | [TabbyAPI, 27B](docs/tabbyapi-guide.md) | [**Avalon, 125B MoE**](docs/avalon-guide.md) |
|---|---|---|---|
| Model | Swift 1.5 Qwen3.8-27B, GGUF IQ3_S + MTP head (12.1 GB) | Qwen3.8-27B EXL3 3.5 bpw + DFlash2 drafter | Qwen3.8-Flash-Next, Avalon IQ3_KS (~89 GiB) + MTP head |
| Where the weights live | all in VRAM | all in VRAM | dense weights, KV cache and hottest experts in VRAM; other experts in RAM |
| Speculative decoding | MTP + `ngram-simple` | DFlash2 draft model | MTP + `ngram-simple` |
| New text, short context | ~62 t/s | ~74 t/s | **~95-100 t/s** |
| Code edits (reply copies the prompt) | ~175 t/s | 77-89 t/s | **168-181 t/s** |
| Prefill | 630-725 t/s (40-60k prompt) | 930-990 t/s (40-60k) | **1,771-1,805 t/s** (16-32k) |
| Context | 131k (262k with 4-bit KV) | 98k | 262k |
| RAM needed | little | little | the model is memory-mapped: 16 GB works, 64 GB+ avoids disk reads |
| Disk | ~12 GB | model + drafter (not recorded) | ~90 GiB |
| Status | measured 2026-10-05 | measured 2026-10-05 | measured 2026-11-02/03, **current setup** |

The three columns use different models and slightly different prompts (see [the comparison caveats](#how-to-read-the-comparison)): compare speed, not answer quality. Only one engine can use the 20 GB card at a time.

**Recommendation:** with 64 GB of RAM and ~90 GiB of free disk, use **Avalon**: it is the largest model and the fastest on new text and prefill, and it ties llama.cpp on code edits. Without that RAM or disk, use the **27B**: llama.cpp for code edits and contexts above 98k, TabbyAPI for long agent prompts below 98k (it keeps 60-70 t/s at 12-57k tokens where llama.cpp falls to 36-51).

## Results

### 125B MoE with Avalon (2026-11-02/03)

Avalon `a91372e`, ROCm 10.0.0, 262,144-token context, `q8_0` K / `q4_0` V cache, MTP + `ngram-simple`. Full tables and method: [docs/avalon-guide.md](docs/avalon-guide.md); raw data: [`data/results-avalon-*`](data).

| Test | Result |
|---|---:|
| New text, short prompts (Avalon `bench.py`, 2nd-pass median / best) | 95-100 / 115 t/s |
| Short chat (Strata's `strata_bench.py`, median of 9) | 96.4 t/s, first token 0.36 s |
| Code edits, MTP only / MTP + `ngram-simple` (median) | 118 / **168-181 t/s** |
| Prefill, fresh 4K / 16K / 32K prompt | 1,180 / 1,771 / 1,805 t/s |
| Time to first token, fresh 4K / 16K / 32K prompt | 3.4 / 8.9 / 17.5 s |
| Agent session (system prompt + 4 tools, 5 turns growing to ~14.5k tokens), warm | first token ~4.2 s per turn, decode 66-72 t/s, 69 % of each prompt reused from cache |
| Same session right after the server started | first token up to 89 s (the model is larger than RAM and is read from disk once) |
| Real opencode / Copilot CLI use (17 requests, prompts up to 76k tokens) | decode median ~78 t/s, prefill median ~1,400 t/s |
| Same model and flags on Windows (earlier) | 62-72 t/s |

What did and did not help on Avalon:

| Change | Effect |
|---|---|
| `ngram-simple` next to MTP | **code edits +45-55 %** (118 -> 168-181 t/s); new text unchanged within noise |
| Linux instead of Windows, same flags | about +40 % decode (62-72 -> 95-100 t/s) |
| 4-bit K cache, or 131k instead of 262k context (frees 0.75-1.2 GiB VRAM) | no change: Avalon gives freed VRAM to its expert cache, which did not get faster |
| TurboQuant KV cache | not ported: the KV cache is only ~2.4 GiB at 262k, so the ~1 GiB it would save is worth nothing (row above) |
| ROCm 7.2.4 instead of 10 | not needed: ROCm 10 already gives the 1,750+ t/s prefill another 7900 XT setup reached with 7.2.4 |

### 27B dense with llama.cpp and TabbyAPI (2026-10-05)

llama.cpp `2ca15f540`, TabbyAPI `f07131c` with `bsvinay/exllamav3-rocm` `360c902`. Full tables, ranges and method: [docs/benchmarks.md](docs/benchmarks.md).

| Workload | llama.cpp (Swift 1.5 27B IQ3_S, MTP + `ngram-simple`) | TabbyAPI (27B EXL3 3.5 bpw + DFlash2) |
|---|---:|---:|
| New code, short context | ~62 t/s (56-68) | ~74 t/s (64-92) |
| Code edits | **~175 t/s** | 77-89 t/s |
| Decode at ~12-14k tokens of context | 51 t/s | 66-70 t/s |
| Decode at ~25-28k | 38 t/s | 68-69 t/s |
| Decode at ~49-57k | 36 t/s | 60-63 t/s |
| Decode at ~103k | 29 t/s | not tested (98k limit) |
| Prefill at 40-60k tokens | 630-725 t/s | 930-990 t/s |
| Tool calling (7-case suite) | 7/7 | 7/7 |
| Context tested | 131k (262k with 4-bit KV) | 98k (131k at about half the speed) |

What moved llama.cpp on the 27B: **no speculation 37 t/s -> MTP 62-64 t/s on new code; edits 36 -> 85 (MTP) -> 175 t/s (MTP + `ngram-simple`)**. What did not: HIP graphs, an unroll compiler flag, the COMPUTE power profile, draft-probability thresholds, checkpoint/cache tuning, Vulkan, the AMD-Ecosystem fork, and a TurboQuant `turbo4` KV cache (about -20 % decode at 57-103k tokens; [docs/turboquant.md](docs/turboquant.md)).

## What we learned

1. **Speculative decoding is the biggest lever on both models.** MTP made new text 1.7x faster on the 27B (37 -> 62-64 t/s; Avalon's author reports 1.5x on the 125B). `ngram-simple` on top of MTP makes replies that copy the prompt (code edits) 1.4-2.1x faster (27B 85 -> 175, 125B 118 -> 168-181 t/s), with no measurable cost on new text.
2. **A large MoE can beat a small dense model on a 20 GB card.** Per token, Flash-Next uses 10 of its 512 experts, so only a small fraction of its 125B weights is read; most of what the GPU needs stays in its cache, and the CPU computes the rest.
3. **Freeing VRAM stopped mattering once the model was fitted.** On the 27B, a smaller KV cache only bought context. On the 125B, Avalon spends spare VRAM on its expert cache automatically, and more cache did not make it faster. That is why the TurboQuant KV cache was not worth it in either case.
4. **The toolchain matters more than the flags.** Same weights and flags: Linux was about 40 % faster than Windows for Avalon, and another 7900 XT setup measured 1.8x the prefill after moving off ROCm 7.10a. ROCm 10 is at least as fast as ROCm 7.2.4 here.
5. **Long agent conversations depend on the prompt cache.** With one server slot (`--parallel 1`) each turn only reads its new tokens: about 4 s to the first token at 14k tokens on the 125B, against ~9 s for the same prompt from scratch.
6. **Decode slows with context depth.** On the 27B with llama.cpp it fell from 62 to 29 t/s at 103k tokens. Decode at depth was not measured for Avalon here (its author reports 88 t/s at 131k on a similar machine).

## Why the 125B model is faster than the 27B

| | Qwen3.8-27B (dense) | Qwen3.8-Flash-Next (MoE) |
|---|---|---|
| Weights read per token | all ~12 GB | the dense part (~4.6 GiB, always on the GPU) + 10 of 512 experts per layer |
| Attention layers with a KV cache | 16 of 64 | 12 of 48 (2 KV heads); the others are linear attention |
| Where the work runs | GPU only | GPU for the dense part and cached experts; **CPU** (AVX2) for experts missing from the GPU, in parallel |
| Prefill | batch of 1,024 tokens | batch of 9,536 tokens; experts are prefetched over PCIe while the GPU works |

Decode on a dense model is limited by reading all its weights for every token. The MoE reads far fewer bytes per token, and the GPU expert cache (9-14 GiB on a 7900 XT, per Avalon's author) holds the experts most tokens use. Prefill runs very large batches, so each expert is used by thousands of tokens once it is on the GPU. The cost: RAM bandwidth and the CPU set the speed for cache misses, and the first session after a start reads the model from disk when RAM is smaller than the model (57 GB against ~89 GiB here).

## How to read the comparison

* **Different models.** The 27B tables use a 3-bit GGUF of the Swift fine-tune (llama.cpp) and a 3.5-bit EXL3 of the base model (TabbyAPI); the MoE uses Avalon's IQ3_KS. Speed and tool calling were measured; **answer quality was not compared**.
* **Different harnesses.** 27B: this repository's `bench/bench.py` (512-token new code, a 150-line edit, reasoning `low`, temperature 0.2). 125B: Avalon's `bench.py` (5 prompts x 2 passes, 512 tokens, thinking off, temperature 0), a 180-line edit test, and Strata's `strata_bench.py` for prefill and the agent session. Edits are comparable in kind, not to the token.
* **Prefill depth differs.** 27B prefill was measured at 40-60k tokens, 125B prefill at 4-32k fresh prompts.
* **Noise.** Decode varies about ±3-5 t/s between runs; most rows are medians of 3-10 requests, single runs are marked in the detailed documents.

## Test machine

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XT, 20 GB, gfx1100 / RDNA3, PCIe 3.0 x16 |
| CPU | AMD Ryzen 7 5700G, 8 cores / 16 threads (its Vega iGPU is visible to ROCm 10 as a second device and must be hidden) |
| RAM | DDR4; the OS reported 45 GB usable in October and 57 GB in November |
| OS | Ubuntu 26.04, kernel 7.0 |
| ROCm | 10.0.0 (`/opt/rocm/core-10.0`); PyTorch rocm7.2 wheels inside the TabbyAPI venv |

## Quick start

```bash
git clone https://github.com/rocktboy1982/llama-cpp-7900xt-optimization && cd llama-cpp-7900xt-optimization
scripts/linux/install-rocm10.sh --dry-run --with-vulkan    # review, then run without --dry-run
```

**125B MoE with Avalon** (details: [docs/avalon-guide.md](docs/avalon-guide.md)):

```bash
scripts/linux/build-avalon.sh                                   # freriadoc/Avalon @ a91372e, gfx1100, ROCm 10
hf download freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF --local-dir ~/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF --exclude '*.jpg' --exclude '*.md'
install -m 755 configs/linux/avalon-server ~/.local/bin/ && avalon-server   # port 8080
# as a service on port 11437, started at boot:
cp configs/linux/avalon.service ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now avalon.service
loginctl enable-linger $USER
```

Then apply the chat-template fix (guide section 2.4) if you use opencode or Copilot CLI.

**27B with llama.cpp** (details: [docs/llama-cpp-guide.md](docs/llama-cpp-guide.md)):

```bash
scripts/linux/build-llama.sh                              # llama.cpp @ 2ca15f540 (HIP + Vulkan)
scripts/linux/download-model.sh all                       # Swift 27B + the two small models
scripts/linux/install-launchers.sh && ~/run-llama.sh      # API key is generated into ~/.llama-api-key on first run
```

**27B with TabbyAPI:** see [docs/tabbyapi-guide.md](docs/tabbyapi-guide.md) (clone `bsvinay/exllamav3-rocm`, run its `setup_env.sh`, `build.sh`, `download_models.sh`, `install_tabbyapi.sh`, then `run_tabbyapi.sh`).

## Repository layout

```
docs/
  avalon-guide.md           125B MoE with Avalon: results, ROCm 10 build, model, template fix, launcher, service, how it works
  llama-cpp-guide.md        27B with llama.cpp: Linux setup (Windows adapted, untested)
  tabbyapi-guide.md         27B with TabbyAPI / ExLlamaV3-ROCm: setup, config, opencode
  benchmarks.md             every 27B test, method, result and caveat
  turboquant.md             TurboQuant KV-cache port to upstream llama.cpp: changes, tests, results
  windows-original-guide.md the earlier Windows guide (fork + patches, ROCm 7.2), kept for reference
scripts/linux/              install-rocm10.sh, build-llama.sh, build-avalon.sh, download-model.sh, install-launchers.sh
configs/
  linux/                    avalon-server, avalon.service; run-llama.sh, start-llama.sh, start-tabbyapi.sh
  windows/                  start-llama-server.bat (untested)
  tabbyapi/config.yml       TabbyAPI config (98,304 tokens, 8-bit cache)
  opencode/opencode.json    opencode providers (no secrets; the key is read from a file)
bench/                      the 27B benchmark harness (Python, standard library only)
  runners/                  the batch scripts that produced the 27B tables
data/                       raw results: results-avalon-* for the 125B, the other files for the 27B
patches/
  avalon-chat-template.patch               Avalon model template: reasoning effort high/max, later system messages
  turboquant-on-upstream-2ca15f540.patch   TurboQuant on upstream llama.cpp
```

## Reproducing the numbers

* **27B:** run `bench/runners/run-ngram.sh` and friends after copying `runners/common.env.example` to `runners/common.env`, or call the scripts directly: `bench/bench.py` (3 workloads), `bench-pp.py` (prefill), `bench-deep.py` / `bench-api-deep.py` (decode at depth), `bench-tools.py` (tool calling), `bench-quality.py` (needle recall, greedy diff), `tabby-ctx.py` (TabbyAPI context sweep). Paths assume llama.cpp in `~/llama.cpp` and models in `~/models`.
* **125B:** Avalon's own `bench.py` (in the Avalon tree), the edit test described in the Avalon guide section 1.4, and Strata's `tools/strata_bench.py --long` pointed at the server with `--url` (it refuses non-Strata servers by name; the guide explains the three-line wrapper used).

## Limits of this work

* One machine, one GPU, mostly short runs.
* **No answer-quality comparison** between the 27B and the 125B, and no tool-calling suite or long-context recall test on the 125B yet.
* Prompts are synthetic or repository source code; the edit workloads flatter ngram speculation. Real agent use was only observed (125B: 17 requests), not benchmarked.
* The Windows instructions for the 27B are an unverified adaptation of the earlier guide.
* Not tried: decode at depth on the 125B, Strata on this machine through the same suite, ROCm 7.2.4, the 274 W power limit, other ngram sizes and types, the ROCmFPX fork, other 27B quants, VRAM overclocking.

## Credits and licenses

* [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, Copyright the ggml authors.
* [Avalon](https://github.com/freriadoc/Avalon) by freriadoc (llama.cpp fork, MIT) and its model `freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF`. The ROCm 7.2.4 comparison and the `-ub` limit come from [JashicTM/qwen38-flash-next-7900xt](https://github.com/JashicTM/qwen38-flash-next-7900xt); the agent-session suite is Strata's `tools/strata_bench.py` ([Niko1221/Strata](https://github.com/Niko1221/Strata)).
* TurboQuant implementation by Pascal Wachowski ([`Pascal-SAPUI5/llama.cpp-turboquant`](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant), MIT), based on the TurboQuant paper (Zandieh et al., ICLR 2026). The port is described in [docs/turboquant.md](docs/turboquant.md).
* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the ROCm port [`bsvinay/exllamav3-rocm`](https://github.com/bsvinay/exllamav3-rocm); [TabbyAPI](https://github.com/theroyallab/tabbyAPI).
* Models: Swift 1.5 GGUFs by `ukisai` (GSQ-RCO quantization by ISTA-DASLab), EXL3 quants by `Mia-AiLab`, Qwen3.8 and Qwen3.8-Flash-Next by the Qwen team. Check each model's license on its Hugging Face page before use.
