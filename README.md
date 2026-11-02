# Local 27B inference on an AMD RX 7900 XT

Tested, benchmarked recipes for running **Qwen3.8-27B** and the **125B Qwen3.8-Flash-Next MoE** on a single **Radeon RX 7900 XT (20 GB, RDNA3 / gfx1100)** on Linux with ROCm 10, plus a TurboQuant KV-cache port and the full set of measurements behind every setting.

| Guide | Engine | Best at |
|---|---|---|
| [**llama.cpp guide**](docs/llama-cpp-guide.md) (Linux tested, Windows adapted) | upstream llama.cpp, ROCm/HIP, MTP + ngram speculative decoding | code edits (~175 t/s), contexts up to 131k-262k |
| [**TabbyAPI guide**](docs/tabbyapi-guide.md) | ExLlamaV3-ROCm + TabbyAPI, DFlash2 speculative decoding | agent loops with long prompts (60-70 t/s at 12-50k tokens) |
| [**Avalon guide**](docs/avalon-guide.md) | [Avalon](https://github.com/freriadoc/Avalon) (llama.cpp fork), experts in system RAM, MTP | a 125B MoE on 20 GB: ~100 t/s decode (~170-180 on edits), ~1,800 t/s prefill, 262k context |

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

### Avalon: Qwen3.8-Flash-Next 125B MoE (2026-11-02)

Same machine (now 57 GB RAM), ROCm 10.0.0, Avalon `a91372e`, IQ3_KS (~89 GiB, experts in RAM), 262k context, MTP draft head. Details: [docs/avalon-guide.md](docs/avalon-guide.md).

| Test | Result |
|---|---:|
| Decode, short prompts (Avalon `bench.py`, median of 2nd pass / best) | **100.3** / 115.2 t/s |
| Decode, short chat (Strata's `strata_bench.py`, median) | 96.4 t/s |
| Decode, code edits: MTP only / **MTP + `ngram-simple`** (median) | 118 / **168-181 t/s** |
| Prefill, fresh 16K / 32K prompt | **1,771 / 1,805 t/s** (first token 8.9 / 17.5 s) |
| Agent session (4 tools, 5 growing turns to ~14.5k tokens), warm | first token ~4.2 s per turn, decode 66-72 t/s |
| Same flags on Windows earlier | 62-72 t/s decode |

Freeing VRAM (4-bit K cache, 131k context) changed nothing, so the TurboQuant KV cache was not ported to Avalon; ROCm 10 already matches the ROCm 7.2.4 prefill reported elsewhere.

## Quick start

```bash
# llama.cpp on Ubuntu 26.04 (details and manual steps: docs/llama-cpp-guide.md)
git clone https://github.com/rocktboy1982/llama-cpp-7900xt-optimization && cd llama-cpp-7900xt-optimization
scripts/linux/install-rocm10.sh --dry-run --with-vulkan   # review, then run without --dry-run
scripts/linux/build-llama.sh                              # llama.cpp @ 2ca15f540 (HIP + Vulkan)
scripts/linux/download-model.sh all                       # Swift 27B + the two small models
scripts/linux/install-launchers.sh && ~/run-llama.sh      # API key is generated into ~/.llama-api-key on first run
```

Avalon (needs ~90 GiB of disk and, ideally, 64 GB+ of RAM; details in [docs/avalon-guide.md](docs/avalon-guide.md)):

```bash
scripts/linux/build-avalon.sh                                   # freriadoc/Avalon @ a91372e, gfx1100, ROCm 10
hf download freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF --local-dir ~/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF --exclude '*.jpg' --exclude '*.md'
install -m 755 configs/linux/avalon-server ~/.local/bin/ && avalon-server   # or configs/linux/avalon.service on port 11437
```

TabbyAPI: see [docs/tabbyapi-guide.md](docs/tabbyapi-guide.md) (clone `bsvinay/exllamav3-rocm`, run its `setup_env.sh`, `build.sh`, `download_models.sh`, `install_tabbyapi.sh`, then `run_tabbyapi.sh`). Only one engine can use the 20 GB card at a time.

## Repository layout

```
docs/
  llama-cpp-guide.md        Linux + Windows setup for llama.cpp
  tabbyapi-guide.md         TabbyAPI / ExLlamaV3-ROCm setup, config, opencode
  avalon-guide.md           Avalon (125B MoE, experts in RAM): build on ROCm 10, launcher, service, template fix, results
  benchmarks.md             every test, method, result and caveat
  turboquant.md             TurboQuant KV-cache port: what changed, tests, results
  windows-original-guide.md the earlier Windows guide (fork + patches, ROCm 7.2), kept for reference
scripts/linux/              install-rocm10.sh, build-llama.sh, build-avalon.sh, download-model.sh, install-launchers.sh
configs/
  linux/                    run-llama.sh, start-llama.sh, start-tabbyapi.sh (desktop launchers), avalon-server, avalon.service
  windows/                  start-llama-server.bat (untested)
  tabbyapi/config.yml       active TabbyAPI config (98,304 tokens, 8-bit cache)
  opencode/opencode.json    opencode providers (no secrets; the key is read from a file)
bench/                      the benchmark harness (Python, standard library only)
  runners/                  the batch scripts that produced the tables (copy common.env.example to common.env)
data/                       raw result files (JSON lines) for every table
patches/
  turboquant-on-upstream-2ca15f540.patch   TurboQuant on current upstream llama.cpp
  avalon-chat-template.patch               Avalon model template: accept reasoning effort high/max, allow later system messages
```

## Reproducing the numbers

Run `bench/runners/run-ngram.sh` and friends (after copying `runners/common.env.example` to `runners/common.env`), or call the scripts directly. `BENCH_DIR` (default: the `bench/` folder) holds `logs/`, `quality/` and the result files. `bench/bench.py` (3 workloads), `bench-pp.py` (warm-up + 3 runs), `bench-deep.py` and `bench-api-deep.py` (decode at depth), `bench-tools.py` (7-case tool-calling suite), `bench-quality.py` (needle recall and greedy diff), `bench-api.py` (any OpenAI-compatible server) and `tabby-ctx.py` (TabbyAPI context sweep). They start and stop the server themselves where noted in each file's docstring and write JSON lines like those in `data/`. Paths assume llama.cpp in `~/llama.cpp` and models in `~/models`; edit the constants at the top.

## Limits of this work

* One machine, one GPU; run-to-run decode noise is about ±5 t/s. Single-run tables are marked as such.
* Prompts are synthetic or repository source code. **No real agent traces** were used, and the edit workload (output copies the prompt) flatters ngram speculation.
* The two engines serve different models (3-bit GGUF of the Swift fine-tune vs 3.5-bit EXL3 of the base model). Speed and tool calling were compared; **answer quality was not**.
* The Windows instructions are an unverified adaptation of the earlier guide.
* The Avalon results use a different model (125B MoE, experts in RAM) from the 27B tables; compare speed only, and only within a section.
* Not tried: the ROCmFPX fork and quants, ThinkingCap, other quants of the 27B (disk space), VRAM overclocking, real InfraAgent traces. See the end of the benchmark document.

## Credits and licenses

* [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, Copyright the ggml authors.
* TurboQuant implementation by Pascal Wachowski ([`Pascal-SAPUI5/llama.cpp-turboquant`](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant), MIT), based on the TurboQuant paper (Zandieh et al., ICLR 2026). The port to current upstream is described in [docs/turboquant.md](docs/turboquant.md).
* [Avalon](https://github.com/freriadoc/Avalon) by freriadoc (llama.cpp fork, MIT) and its model `freriadoc/Qwen3.8-Flash-Next-Avalon-GGUF`; the ROCm 7.2.4 comparison and `-ub` limit come from [JashicTM/qwen38-flash-next-7900xt](https://github.com/JashicTM/qwen38-flash-next-7900xt); the agent-session suite is Strata's `tools/strata_bench.py` ([Niko1221/Strata](https://github.com/Niko1221/Strata)).
* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the ROCm port [`bsvinay/exllamav3-rocm`](https://github.com/bsvinay/exllamav3-rocm); [TabbyAPI](https://github.com/theroyallab/tabbyAPI).
* Models: Swift 1.5 GGUFs by `ukisai` (the GSQ-RCO quantization method is by ISTA-DASLab), EXL3 quants by `Mia-AiLab`, Qwen3.8 by the Qwen team; check each model's license on its Hugging Face page before use.
* The Windows guide in `docs/windows-original-guide.md` is the repository's earlier README.
