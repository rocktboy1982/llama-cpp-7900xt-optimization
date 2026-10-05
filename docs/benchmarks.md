# Benchmarks and test results

Every number here was measured on one machine between 2026-10-05 01:00 and 05:00 (EEST). Raw result files are in [`../data`](../data) and the harness that produced them is in [`../bench`](../bench). Nothing is estimated unless it is marked *estimate*.

## 1. Test machine

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 7900 XT, 20 GB (19.98 GiB usable), gfx1100 / RDNA3 |
| iGPU | Ryzen 7 5700G Vega 8 (gfx90c), 16 GB shared, Vulkan only |
| CPU / RAM | Ryzen 7 5700G, 8C/16T; 45 GB usable DDR4 |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38 |
| ROCm | 10.0.0 (`/opt/rocm/core-10.0`) for llama.cpp; PyTorch 2.13.0+rocm7.2 wheels inside the TabbyAPI venv |
| Vulkan | Mesa RADV, Vulkan 1.4.335 |
| llama.cpp | upstream commit `2ca15f540` (2026-10-04), built with HIP (`gfx1100`) and Vulkan |
| TabbyAPI | `theroyallab/tabbyAPI` @ `f07131c` with the `bsvinay/exllamav3-rocm` fork @ `360c902` |

## 2. Methodology

* **Three synthetic workloads** (`bench/bench.py`): *new code* (write a 512-token Python module), *edit* (return a 150-line file with one rename, so the reply copies the prompt) and *prefill* (an ~9k-token prompt, 16 output tokens). Reasoning effort `low`, temperature 0.2 unless stated.
* **Depth tests** (`bench-deep.py`, `bench-api-deep.py`): real source code truncated to a target length, 300-token summary requested. Prompts have distinct prefixes so the prompt cache cannot help.
* **Repeats.** The batch-size, unroll-flag, power-profile, AMD-fork and ngram tests used a warm-up plus 3 runs and report the median and range. Earlier tests are single runs; run-to-run noise on decode is about **±5 t/s**, so differences smaller than that are not significant. Single-run tables are marked.
* **Not measured:** real agent traces. All prompts are synthetic or repository source code, which favours speculative decoding on the *edit* workload.

---

## 3. llama.cpp results

Model unless stated: **Swift 1.5 Qwen3.8-27B, GSQ-RCO IQ3_S with MTP head** (12.12 GB), ROCm 10, 65,536-token context, `q8_0` KV cache.

### 3.1 Speculative decoding (single runs)

| Config | New code t/s | Edit t/s | Prefill t/s | Draft accepted (new code) |
|---|---:|---:|---:|---:|
| No speculation (baseline) | 37.4 | 36.4 | 826 | n/a |
| MTP, draft 3 | 58.4 | 79.1 | 791 | 61% |
| MTP, draft 4 | 63.8 | 84.9 | 789 | 64% |
| MTP, draft 5 | 63.2 | 91.3 | 789 | 59% |
| MTP, draft 6 | 57.8 | 92.7 | 787 | 49% |
| `ngram-mod` 12/16 + MTP, draft 4 | 59.6 | 97.9 | 789 | 57% |
| MTP, draft 8 (with ngram-mod) | 51.6 | 101.6 | 798 | 38% |

MTP is the largest single gain (about 1.7x on new code, 2.3-2.6x on edits).

### 3.2 ngram variants (3 runs each, medians; range in brackets)

| Speculation type | New code t/s | Edit t/s |
|---|---:|---:|
| `ngram-mod` 12/16 + MTP | 60.8 (60.5-61.6) | 97.9 (97.3-98.5) |
| `ngram-mod` 24/48 + MTP | 60.6 (58.8-64.1) | 151.6 (151.3-153.1) |
| **`ngram-simple` + MTP** | **62.3 (56.3-63.9)** | **175.5 (175.3-176.9)** |
| `ngram-simple` + `ngram-mod` + MTP | 60.0 (59.0-66.5) | 170.4 (153.8-175.6) |
| `ngram-mod` 16/16 + MTP | 61.6 (60.1-62.2) | 107.2 (107.2-107.4) |

Single runs of other types: `ngram-map-k4v` + MTP 59.1 / 123.8 t/s; ngram-mod 8/8 gave 61.3 / 61.8 t/s (worse on edits). **Caveat:** the edit workload copies its prompt, the best case for ngram drafting. Real agent loops that merely echo tool output will gain less.

### 3.3 Settings that did not help (all at the same config)

| Test | Result | Verdict |
|---|---|---|
| `-DGGML_HIP_GRAPHS=ON` rebuild (2 runs each) | edit 98.3 vs 98.1 t/s; prefill 792 vs 790 | no gain |
| `-mllvm --amdgpu-unroll-threshold-local=600` (3 runs x2) | prefill 784 vs 787 t/s; decode within noise | no gain |
| Power profile `COMPUTE` (restored afterwards) | prefill 781 t/s unchanged; decode within noise | no gain |
| Temperature 0.2 / 0.6 / 1.0 | new code 59.4 / 61.3 / 57.0 t/s; acceptance 57 / 60 / 54% | speed-neutral |
| `--spec-draft-p-min` 0.5 / 0.6 / 0.7 (draft 6) | new code 46.5 / 51.6 / 54.3 t/s (vs ~60) | slower |
| f16 instead of `q8_0` KV | 60.1 / 98.6 t/s, +1.7 GiB VRAM | no speed gain |
| Server CPU threads 4 / 6 / 8 / 12 | 54.3 / 64.0 / 60.5 / 56.1 t/s | within noise |
| `--checkpoint-min-step` 8192 / 4096 / 2048 / 512 (+`--ctx-checkpoints 64`) | steps 2-5 of an agent loop re-process only the new ~700 tokens (~1.5 s) at every setting | no effect |
| `--cache-ram 16384 --cache-reuse 256` | same as default | no effect |
| Vulkan instead of ROCm (2 runs) | new code 44.9 / 51.5, edit 53.4 / 85.7, prefill 685 t/s | slower (VRAM -1.6 GiB) |
| AMD-Ecosystem/llama.cpp fork @ `450e1f6` | prefill 744 vs 785 t/s; new code 53.8; edit 92.5 | ~5% slower; fork = older upstream + one CMake option |
| rocWMMA FlashAttention | option removed upstream (PR #26046) | n/a |

### 3.4 Batch / micro-batch (warm-up + 3 runs, ~16k-token prompts)

| `-ub` / `-b` | Prefill t/s (median) | Decode t/s (median, range) | VRAM GiB |
|---|---:|---|---:|
| 512 / 2048 | 781.1 | 43.5 (39.9-46.4) | 16.12 |
| 1024 / 2048 | 783.9 | 50.2 (48.2-52.1) | 16.43 |
| 2048 / 4096 | 789.3 | 50.3 (46.5-52.2) | 17.09 |

Prefill is identical. The decode difference may be noise; `1024/2048` was kept for a small, cheap margin.

### 3.5 Context size vs VRAM (`ngram-mod` + MTP, `q8_0` unless stated)

| Context | KV cache | VRAM peak | Result |
|---:|---|---:|---|
| 131,072 | `q8_0` | 18.87 GiB | OK |
| 163,840 | `q8_0` | 19.76 GiB | OK, almost no headroom |
| 196,608 / 229,376 | `q8_0` | n/a | out of memory at load |
| 262,144 | `q4_0` / `q4_0` | 19.89 GiB | OK at the same speed (new 58.3, edit 94.1 t/s) |
| 262,144 | `q8_0` K, `q4_0` V | n/a | out of memory: that K/V pairing has no fast kernel and falls back to f16 conversion |

K and V must use the same cache type, or the fast flash-attention path is lost.

### 3.6 Decode and prefill at depth (131k context, `ngram-mod` + MTP)

| Prompt depth | `q8_0` KV: prefill / decode t/s | `q4_0` KV | `turbo4` KV (TurboQuant port) |
|---:|---|---|---|
| ~14k | 759 / 51.4 | 768 / 54.9 | 757 / 48.6 |
| ~28k | 725 / 38.1 | 736 / 45.1 | 728 / 37.8 |
| ~57k | 632 / 36.4 | 650 / 32.0 | 639 / 27.1 |
| ~103k | 519 / 29.1 | 542 / 29.7 | 532 / 23.3 |

Decode roughly halves from short context to 100k. A 4-bit KV cache is not consistently faster. `turbo4` is about 20% slower at depth.

### 3.7 Quality gates

| Check | Result |
|---|---|
| Needle recall (code hidden at 10/50/90% of ~28k and ~57k-token prompts) | **6/6** for `q8_0` without speculation, `q8_0` with speculation, and `q4_0` with speculation |
| Greedy output, speculation off vs `ngram-mod` + MTP, ~21k-token prompt | **identical** (MTP correct past 20k tokens) |
| Same, two short prompts | identical for the first 588 and 769 characters, then drift (numeric noise) |
| `q4_0` vs `q8_0` KV, 21k-token prompt | first difference at character 89 (expected 4-bit drift) |
| KL divergence of KV cache vs f16, 10 chunks of 512 tokens | `q8_0`: mean KLD 0.00057, top-token match 98.5%. `q4_0`: mean KLD 0.0036, 97.1% |
| Tool calling, 7 cases (single, enum, integer, no-tool, multi-step, number+enum, parallel) | **7/7** at `low` (632 completion tokens) and `medium` (652) effort |

The KL test used short windows; drift at long context was not measured.

---

## 4. Model and quantization comparison

### 4.1 Swift 1.5 quantization quality (published by the model author, KL vs BF16 at 32k)

| Quant | Size | Mean KLD | 99th-pct KLD | Top-token match |
|---|---:|---:|---:|---:|
| Q8_0 | 29.0 GB | 0.0006 | 0.005 | 98.9% |
| Q4_K_M | 17.4 GB | 0.0134 | 0.163 | 95.0% |
| IQ4_XS | 15.5 GB | 0.0173 | 0.187 | 95.0% |
| IQ3_M | 14.9 GB | 0.0380 | 0.409 | 91.8% |
| IQ3_XS | 12.8 GB | 0.0885 | 1.130 | 88.5% |

Source: the model card for `ukisai/Swift-1.5-Qwen3.8-27B-GGUF`. These were not re-measured locally. For the base Qwen3.8-27B the perplexity cost of IQ3_XS vs Q8_0 is only +3.7% (bartowski), so the loss looks small by perplexity and larger by KLD.

### 4.2 Small models for the iGPU and CPU (`llama-bench`, 2 repeats, Q4_K_M)

| Model | iGPU (Vulkan) pp512 / tg128 | CPU (12 threads) pp512 / tg128 | Tool calling (7 cases) |
|---|---|---|---|
| Qwen3.8-2B-Distill (1.31 GB) | 371 / 24.1 t/s | 226 / 22.5 t/s | **3/7** |
| Qwen3.8-4B-Distill (2.78 GB) | 149 / 11.1 t/s | 90 / 10.5 t/s | **7/7** (788 tokens) |
| Qwen3.5-4B, official (2.74 GB) | 159 / 11.6 t/s | 94 / 10.2 t/s | 7/7 (1149 tokens) |
| Qwen2.5-7B-Instruct (earlier) | 91 / 7.2 t/s | 60 / 7.1 t/s | not run |

Both devices are limited by the same DDR4 bandwidth, so running both at once slows each. ROCm 10 does not support the Vega iGPU (`gfx90c`): the HIP backend crashes with `invalid kernel file`, so the iGPU runs on Vulkan.

### 4.3 Earlier dense-model reference (Qwen3-14B Q4_K_M, 7900 XT)

| Backend | pp512 t/s | tg128 t/s |
|---|---:|---:|
| ROCm 10 | 1706 | 65.2 |
| Vulkan | 1627 | 66.1 |

At 16k depth with an f16 KV cache: prefill 939 t/s, decode 48.4 t/s.

---

## 5. TabbyAPI (ExLlamaV3) results

Model: `Qwen3.8-27B-EXL3-3.5bpw` (15 GB) with the `Qwen3.8-27B-DFlash2-EXL3-5.0bpw` draft model (1.4 GB), 8-bit KV cache, draft KV 4-bit. Streaming client (`bench-api.py`); decode = tokens / (end - first token).

### 5.1 Short context, same three workloads (65k-token config)

| Run | New code t/s | Edit t/s |
|---|---:|---:|
| 1 | 91.6 | 81.4 |
| 2 | 70.6 | 77.3 |
| 3 | 73.7 | 84.9 |
| 4 | 64.3 | 89.5 |
| 5 | 85.0 | 89.2 |
| over LAN | 86.3 | 88.0 |
| **median** | **~74** | **~85** |

Prefill of the ~9k-token prompt: 800 t/s (first run; later runs hit the prompt cache and are not valid measurements).

### 5.2 Decode at depth (65k-token config)

| Prompt tokens | Prefill t/s | Decode t/s |
|---:|---:|---:|
| 12,358 | 854 | 65.6 |
| 24,588 | 1,052 | 68.4 |
| 49,223 | 966 | 60.3 |

### 5.3 Context size and cache type

Each row: restart TabbyAPI with that config, measure VRAM, then a ~41k-token request, a request at ~60% of the context, and a request at ~80-86% of the context with a code hidden at 10% depth (reasoning effort `low`). The needle check searches the reasoning and the answer.

| Config | VRAM loaded / peak (GiB) | Prefill t/s (41k / mid / long) | Decode t/s (41k / mid / long) | Needle |
|---|---|---|---|---|
| **98,304 tokens, 8-bit cache** | 18.60 / 19.69 | 983 / 927 / 857 | **80 / 102 / 60** (41k / 56k / 78k) | found at 78k |
| 131,072, 4-bit cache | 17.84 / 18.83 | 982 / 893 / 785 | 40 / 54 / 28 (41k / 68k / 105k) | found at 105k |
| 131,072, 6-bit cache | 18.83 / 19.82 | 786 / 890 / 781 | 30 / 20 / 15 (41k / 68k / 105k) | found at 105k |
| 196,608, 4-bit cache | n/a / 19.97 | n/a | n/a | loads, then CUDA out of memory on a long prompt |

The 8-bit cache is roughly twice as fast as the 4-bit and 6-bit caches; 98k is the largest 8-bit context that fits. Decode figures are noisy because speculative-decoding gains depend on the answer. A first batch of this test had an invalid needle check (the reasoning used the whole 350-token budget) and was discarded; its numbers are not used.

### 5.4 Tool calling and agent use

* 7/7 on the tool-calling suite at `low` and `medium` effort.
* **opencode** (`opencode run ... < /dev/null`): fixed a seeded off-by-one bug end to end (read, edit, run tests) in 59 s. opencode plus the `oh-my-opencode` plugin sends a ~24k-token system prompt. Requests after the first hit 94-100% prompt cache, decoded at 62-120 t/s, and 33-65% of drafted tokens were accepted.

### 5.5 Launcher script vs hand-started process

The ExLlamaV3-ROCm fork ships `rocm/scripts/run_tabbyapi.sh`, which sets `PYTHONPATH` to the fork and `EXL3_NOGRAPH=mlp,gdn` (eager MLP and DeltaNet decode; the fork measures about 0.5% gain). The first tests started TabbyAPI directly without those variables. The same config was then re-tested as a separate process through the launcher:

| | Hand-started | Via `run_tabbyapi.sh` |
|---|---:|---:|
| New code, t/s | ~74 median (64-92) | ~67 median (65-88), 4 runs |
| Edit, t/s | 77-89 | 86-90 |
| Decode at ~12k / ~25k / ~49k tokens, t/s | 65.6 / 68.4 / 60.3 | 70.3 / 69.0 / 63.3 |
| Prefill at those depths, t/s | 854 / 1052 / 966 | 1085 / 1061 / 958 |
| Tool calling | 7/7 | 7/7 |
| VRAM used | 19.7 GiB | 19.5 GiB |

The launcher is at most a few percent better at depth, within noise elsewhere. Use it for convenience; it does not change the picture.

### 5.6 Vendor claims vs these results

The ExLlamaV3-ROCm README reports 147-157 t/s decode with DFlash2 on an RX 7900 XTX. Its benchmark script (`rocm_tests/bench_long.py`) uses greedy sampling and highly repetitive synthetic filler code, which is the best case for speculative decoding, on a 24 GB card. The numbers above use real source code, temperature 0.2, and a 20 GB 7900 XT. They are not comparable; the vendor script was not run here.

---

## 6. llama.cpp vs TabbyAPI, side by side

| Workload | llama.cpp (Swift 27B IQ3_S, MTP + `ngram-simple`) | TabbyAPI (Qwen3.8-27B EXL3 3.5 bpw + DFlash2) |
|---|---|---|
| New code, short context | ~62 t/s (56-68) | ~74 t/s (64-92) |
| Code edits | **~175 t/s** | 77-89 t/s |
| Decode at ~12-14k depth | 51 t/s | 66 t/s |
| Decode at ~25-28k | 38 t/s | 68 t/s |
| Decode at ~49-57k | 36 t/s | 60 t/s |
| Prefill at 40-60k depth | 630-725 t/s | 930-990 t/s |
| Tool calling (7 cases) | 7/7 | 7/7 |
| Maximum context tested | 131k (262k with 4-bit KV) | 98k (131k at about half the speed) |

The models differ (a 3-bit GGUF quant of the Swift fine-tune vs a 3.5-bit EXL3 quant of the base model). Speed and tool-calling correctness were compared; answer quality was not.
