# TurboQuant KV-cache port to current llama.cpp

TurboQuant (Zandieh et al., ICLR 2026) compresses the KV cache to 3-4 bits per element with a random Hadamard rotation plus a Lloyd-Max codebook. Upstream llama.cpp does not have it (feature request in discussion #20969; PR #21089 unmerged at the time of writing). This directory's patch re-bases a community implementation onto current upstream so it builds with ROCm 10 on an RX 7900 XT.

## Credits

The implementation is by **Pascal Wachowski** ([`Pascal-SAPUI5/llama.cpp-turboquant`](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant), MIT, forked from `ggml-org/llama.cpp`), which builds on the TurboQuant paper. Only the port described below is new here. The patch contains the original authors' commits; llama.cpp is MIT licensed (Copyright the ggml authors).

## What the port is

[`../patches/turboquant-on-upstream-2ca15f540.patch`](../patches/turboquant-on-upstream-2ca15f540.patch) is a diff against upstream **`2ca15f540`** (2026-10-04, 26 files, +2535/-11). Applies with `git apply --check` on a clean checkout.

New KV cache types: `turbo3` (3.5 bits/element) and `turbo4` (4.5 bits/element). Used as `--cache-type-k turbo4 --cache-type-v turbo4` with flash attention on.

Changes made while porting (March 2026 base to October 2026 upstream):

| Conflict | Resolution |
|---|---|
| Upstream took type IDs 41/42 for `Q1_0` / `Q2_0` | `GGML_TYPE_TURBO3_0 = 43`, `GGML_TYPE_TURBO4_0 = 44`, `GGML_TYPE_COUNT = 45`. KV cache types are not stored in GGUF files, so renumbering is safe |
| Upstream rewrote flash-attention dispatch (runtime kernel lookup, `GGML_CUDA_FA_QUANTS`) | Re-applied only the type handling: turbo K/V are accepted and converted to f16 (inverse Hadamard transform) inside `launch_fattn`, and the "no vector kernel compiled" warning is silenced for them |
| `set_rows` support check restructured upstream | Turbo types added to the new condition |
| `dequantize.cuh` got the k-quant dequantizers and has no include guard | Rebuilt from upstream plus the turbo block; added `#pragma once` |
| Fused in-kernel decode paths (`fattn-vec-turbo.cuh`, turbo tile loaders) | **Not ported.** They target upstream's old tile kernel. The fork's own notes show the fused vector kernel was slower than bulk conversion and is off by default |

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 2ca15f540
git apply ../patches/turboquant-on-upstream-2ca15f540.patch
R=/opt/rocm/core-10.0
HIPCXX=$R/llvm/bin/clang HIP_PATH=$R CMAKE_PREFIX_PATH=$R \
  cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
cmake --build build -j16 --target llama-server llama-bench test-turboquant test-backend-ops
```

HIP only: there are no Vulkan or Metal kernels for the turbo types. The CPU backend has reference implementations.

## Verification

| Test | Result |
|---|---|
| `test-turboquant` (CPU reference: Hadamard transform, MSE, bit packing, chunk independence) | 23/23 passed |
| `test-backend-ops -o SET_ROWS -b ROCm0` | 159/159 passed |
| `test-backend-ops -o FLASH_ATTN_EXT -b ROCm0` (1312 of the cases use turbo types) | 5298/5298 passed |

## Results on an RX 7900 XT (Swift 1.5 27B IQ3_S + MTP + ngram speculation)

| Measurement | `q8_0` KV | `turbo4` KV |
|---|---:|---:|
| 65k context: new code / edit / prefill | 60.5 / 99.3 / 801 t/s | 63.0 / 95.2 / 792 t/s |
| 65k context VRAM peak | 16.43 GiB | 15.43 GiB |
| Decode at ~14k depth (131k context) | 51.4 t/s | 48.6 t/s |
| ~28k | 38.1 t/s | 37.8 t/s |
| ~57k | 36.4 t/s | 27.1 t/s |
| ~103k | 29.1 t/s | 23.3 t/s |

**Conclusion.** `turbo4` saves about 1 GiB of VRAM at 65k context with no speed loss at short context, but decode is about 20% slower at 57k-103k tokens because the whole cache is dequantized to f16 on every attention call (the fused kernels that would avoid that were not ported). For this model only 16 of 64 layers keep a KV cache, so `q8_0` already reaches 131k-163k context in 20 GB and a 4-bit `q4_0` cache loads at 262k and ran at the same short-context speed (decode at depth with 262k was not tested). TurboQuant is therefore not part of the recommended configuration. Quality (KL divergence, needle recall) was not measured for `turbo3`/`turbo4`.
