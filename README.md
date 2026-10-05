# Local 27B inference on an AMD RX 7900 XT

Two tested, benchmarked recipes for running **Qwen3.8-27B** on a single **Radeon RX 7900 XT (20 GB, RDNA3 / gfx1100)** on Linux with ROCm 10, plus a TurboQuant KV-cache port and the full set of measurements behind every setting.

| Guide | Engine | Best at |
|---|---|---|
| [**llama.cpp guide**](docs/llama-cpp-guide.md) (Linux tested, Windows adapted) | upstream llama.cpp, ROCm/HIP, MTP + ngram speculative decoding | code edits (~175 t/s), contexts up to 131k-262k |
| [**TabbyAPI guide**](docs/tabbyapi-guide.md) | ExLlamaV3-ROCm + TabbyAPI, DFlash2 speculative decoding | agent loops with long prompts (60-70 t/s at 12-50k tokens) |

All numbers, methods and caveats: [**docs/benchmarks.md**](docs/benchmarks.md). The TurboQuant port: [**docs/turboquant.md**](docs/turboquant.md).

## Results at a glance

Measured 2026-10-05 on a Ryzen 7 5700G / 45 GB RAM / RX 7900 XT, Ubuntu 26.04, ROCm 10.0.0, llama.cpp `2ca15f540`. Same prompts for both engines; decode in tokens/s.

| Workload | llama.cpp (Swift 1.5 27B IQ3_S, MTP + `ngram-simple`) | TabbyAPI (Qwen3.8-27B EXL3 3.5 bpw + DFlash2) |
|---|---:|---:|
| New code, short context | ~62 | ~74 |
| Code edits (reply copies the prompt) | **~175** | 77-89 |
| Decode at ~12-14k tokens of context | 51 | 66-70 |
| Decode at ~25-28k | 38 | 68-69 |
| Decode at ~49-57k | 36 | 60-63 |
| Prefill at 40-60k tokens | 630-725 | 930-990 |
| Tool calling (7-case suite) | 7/7 | 7/7 |
| Context tested | 131k (262k with 4-bit KV) | 98k |

What moved the needle on llama.cpp (single model, same GPU): **no speculation 37 t/s, +MTP 64 t/s on new code, +`ngram-simple` 175 t/s on edits**. What did not: HIP graphs, an unroll compiler flag, the COMPUTE power profile, draft-probability thresholds, checkpoint/cache tuning, Vulkan for the 27B, an AMD-Ecosystem fork, and a TurboQuant `turbo4` KV cache (-20% decode at depth). Full tables, ranges and the list of tests that were *not* run are in the benchmark document.

## Quick start

```bash
# llama.cpp on Ubuntu 26.04 (details and manual steps: docs/llama-cpp-guide.md)
git clone https://github.com/rocktboy1982/llama-cpp-7900xt-optimization && cd llama-cpp-7900xt-optimization
scripts/linux/install-rocm10.sh --dry-run --with-vulkan   # review, then run without --dry-run
scripts/linux/build-llama.sh                              # llama.cpp @ 2ca15f540 (HIP + Vulkan)
scripts/linux/download-model.sh all                       # Swift 27B + the two small models
scripts/linux/install-launchers.sh && ~/run-llama.sh      # API key is generated into ~/.llama-api-key on first run
```

TabbyAPI: see [docs/tabbyapi-guide.md](docs/tabbyapi-guide.md) (clone `bsvinay/exllamav3-rocm`, run its `setup_env.sh`, `build.sh`, `download_models.sh`, `install_tabbyapi.sh`, then `run_tabbyapi.sh`). Only one engine can use the 20 GB card at a time.

## Repository layout

```
docs/
  llama-cpp-guide.md        Linux + Windows setup for llama.cpp
  tabbyapi-guide.md         TabbyAPI / ExLlamaV3-ROCm setup, config, opencode
  benchmarks.md             every test, method, result and caveat
  turboquant.md             TurboQuant KV-cache port: what changed, tests, results
  windows-original-guide.md the earlier Windows guide (fork + patches, ROCm 7.2), kept for reference
scripts/linux/              install-rocm10.sh, build-llama.sh, download-model.sh, install-launchers.sh
configs/
  linux/                    run-llama.sh, start-llama.sh, start-tabbyapi.sh (desktop launchers)
  windows/                  start-llama-server.bat (untested)
  tabbyapi/config.yml       active TabbyAPI config (98,304 tokens, 8-bit cache)
  opencode/opencode.json    opencode providers (no secrets; the key is read from a file)
bench/                      the benchmark harness (Python, standard library only)
  runners/                  the batch scripts that produced the tables (copy common.env.example to common.env)
data/                       raw result files (JSON lines) for every table
patches/
  turboquant-on-upstream-2ca15f540.patch   TurboQuant on current upstream llama.cpp
```

## Reproducing the numbers

Run `bench/runners/run-ngram.sh` and friends (after copying `runners/common.env.example` to `runners/common.env`), or call the scripts directly. `BENCH_DIR` (default: the `bench/` folder) holds `logs/`, `quality/` and the result files. `bench/bench.py` (3 workloads), `bench-pp.py` (warm-up + 3 runs), `bench-deep.py` and `bench-api-deep.py` (decode at depth), `bench-tools.py` (7-case tool-calling suite), `bench-quality.py` (needle recall and greedy diff), `bench-api.py` (any OpenAI-compatible server) and `tabby-ctx.py` (TabbyAPI context sweep). They start and stop the server themselves where noted in each file's docstring and write JSON lines like those in `data/`. Paths assume llama.cpp in `~/llama.cpp` and models in `~/models`; edit the constants at the top.

## Limits of this work

* One machine, one GPU; run-to-run decode noise is about ±5 t/s. Single-run tables are marked as such.
* Prompts are synthetic or repository source code. **No real agent traces** were used, and the edit workload (output copies the prompt) flatters ngram speculation.
* The two engines serve different models (3-bit GGUF of the Swift fine-tune vs 3.5-bit EXL3 of the base model). Speed and tool calling were compared; **answer quality was not**.
* The Windows instructions are an unverified adaptation of the earlier guide.
* Not tried: Strata and MoE offload, the ROCmFPX fork and quants, ThinkingCap, other quants of the 27B (disk space), VRAM overclocking, real InfraAgent traces. See the end of the benchmark document.

## Credits and licenses

* [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, Copyright the ggml authors.
* TurboQuant implementation by Pascal Wachowski ([`Pascal-SAPUI5/llama.cpp-turboquant`](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant), MIT), based on the TurboQuant paper (Zandieh et al., ICLR 2026). The port to current upstream is described in [docs/turboquant.md](docs/turboquant.md).
* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the ROCm port [`bsvinay/exllamav3-rocm`](https://github.com/bsvinay/exllamav3-rocm); [TabbyAPI](https://github.com/theroyallab/tabbyAPI).
* Models: Swift 1.5 GGUFs by `ukisai` (the GSQ-RCO quantization method is by ISTA-DASLab), EXL3 quants by `Mia-AiLab`, Qwen3.8 by the Qwen team; check each model's license on its Hugging Face page before use.
* The Windows guide in `docs/windows-original-guide.md` is the repository's earlier README.
