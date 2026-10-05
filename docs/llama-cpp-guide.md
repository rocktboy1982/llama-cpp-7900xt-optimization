# llama.cpp on an RX 7900 XT: setup guide (Linux and Windows)

Goal: serve **Swift 1.5 Qwen3.8-27B** (GSQ-RCO IQ3_S with MTP head, 12.1 GB) from a single RX 7900 XT (20 GB) at about 62 t/s on new code and about 175 t/s on code edits, with a 131k-token context. Benchmarks behind every choice are in [`benchmarks.md`](benchmarks.md).

| | Linux (tested) | Windows (**not tested in this work**) |
|---|---|---|
| Status | Everything in section 2 was run and measured | Adapted from [the earlier Windows guide](windows-original-guide.md); commands unverified here |
| ROCm | 10.0.0 | HIP SDK 7.2 (the earlier guide found no benefit from 10.0 on Windows) |
| llama.cpp | upstream `2ca15f540` | upstream, same flags |

## 1. Why this configuration

| Choice | Reason (measured) |
|---|---|
| **Upstream llama.cpp, no fork** | MTP (`draft-mtp`), `ngram-simple` and the RDNA3 macros are already upstream. The AMD-Ecosystem fork was ~5% slower; a TurboQuant port was ~20% slower at depth ([`turboquant.md`](turboquant.md)) |
| **ROCm (HIP) backend** | Equal to Vulkan on a 14B model; on this 27B about 15-30% faster on new code and ~15% faster on prefill |
| **MTP + `ngram-simple` speculation** | 37 to 62 t/s on new code, 36 to 175 t/s on edits |
| **`q8_0` KV cache, same type for K and V** | Same speed as f16, saves 1.7 GiB. Mixed K/V types fall off the fast attention path |
| **`-ub 1024 -b 2048`** | Prefill identical to 512; slightly higher decode median |
| **Context 131,072** | Peaks at ~19 of 19.98 GiB. 163,840 fits with no headroom; 262,144 needs a 4-bit KV cache |

What did **not** help (do not retry): HIP graphs, the `-mllvm --amdgpu-unroll-threshold-local` flag, the COMPUTE power profile, `--spec-draft-p-min`, draft length above 5, `--checkpoint-min-step`/`--cache-ram` tuning, server thread count, temperature, Vulkan for the 27B. Details in [`benchmarks.md` section 3.3](benchmarks.md#33-settings-that-did-not-help-all-at-the-same-config).

## 2. Linux (Ubuntu 26.04)

### Scripted setup (recommended)

Four scripts in [`../scripts/linux`](../scripts/linux) automate sections 2.1-2.4. Run them in order from a clone of this repository:

```bash
scripts/linux/install-rocm10.sh --with-vulkan     # ROCm 10 (gfx1100 packages only) + Vulkan; add --dry-run to preview, --purge-old to remove ROCm 7.x
scripts/linux/build-llama.sh                      # llama.cpp @ 2ca15f540, HIP + Vulkan; add --turboquant for the patched HIP-only build
scripts/linux/download-model.sh all               # Swift 27B (12.1 GB), 4B (2.8 GB), 2B (1.3 GB): resumable, size-checked, MTP head verified
scripts/linux/install-launchers.sh                # ~/run-llama.sh, ~/start-llama.sh, ~/start-tabbyapi.sh + desktop shortcuts
~/run-llama.sh
```

What was verified for these scripts (2026-10-05, on the machine the benchmarks came from, where ROCm and the models were already installed): `build-llama.sh` ran end to end in a fresh clone and produced working binaries; `build-llama.sh --turboquant --check-only` confirmed the patch applies; `download-model.sh` skip/size/MTP-head checks ran against the real files (and caught and fixed a bug); `install-launchers.sh` ran in a throwaway home directory; `install-rocm10.sh` was run with `--dry-run` and its package names were checked with `apt install -s` (all valid, nothing further needed on that machine). **`install-rocm10.sh` was not run for real on a clean system**, so treat the first run on a fresh install as untested and read its `--dry-run` output first.

### 2.1 Install ROCm 10.0.0

Follow AMD's [install page](https://rocm.docs.amd.com/en/latest/install/rocm.html). Remove ROCm 7.x first (`apt remove --purge` the old `rocm-*` and `amdgpu-install` packages). AMD's page also lists downloading an `amdgpu-install` package; that step was **not** run here, the kernel's own `amdgpu` driver was used and only AMD's package repository was registered:

```bash
sudo mkdir -p -m0755 /etc/apt/keyrings
wget -qO- https://stable.repo.amd.com/rocm/gpg/packages.gpg | gpg --dearmor | sudo tee /etc/apt/keyrings/amdrocm.gpg >/dev/null
sudo tee /etc/apt/sources.list.d/amdrocm-stable.sources <<'EOF'
X-Repo-Id: amdrocm-stable
Types: deb
URIs: https://stable.repo.amd.com/rocm/core/packages/ubuntu2604/
Suites: stable
Components: main
Architectures: amd64
Signed-By: /etc/apt/keyrings/amdrocm.gpg
Enabled: yes
EOF
sudo apt update
sudo apt install amdrocm10.0 amdrocm10.0-gfx1100
# development packages needed to compile llama.cpp (not pulled in by the runtime package):
sudo apt install amdrocm-core-dev10.0 amdrocm-runtime-dev10.0 amdrocm-llvm-dev10.0 amdrocm-blas-dev10.0 amdrocm-hipblas-common-dev10.0
```

Notes from this machine:

* ROCm 10 installs under `/opt/rocm/core-10.0`, not `/opt/rocm`. The CMake packages (`lib/cmake/hip-lang`) appear only after the `-dev` packages are installed.
* The `amdrocm10.0` meta-package installs libraries for **every** GPU generation (about 11 GB extra). After installing, the other-architecture packages (`amdrocm*-gfx*` except `gfx1100`) were removed with `apt remove` to free disk and the build and runtime still worked. That is a space optimisation, not a requirement.
* The Vega iGPU (`gfx90c`) is **not supported** by ROCm 10: HIP aborts with `invalid kernel file`. Use Vulkan for the iGPU.
* Runtime libraries: `export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`. There is no `/opt/rocm/bin` symlink after the cleanup; call `/opt/rocm/core-10.0/bin/rocm-smi` directly.

Vulkan packages (for the iGPU or a Vulkan build): `sudo apt install glslc libvulkan-dev vulkan-tools mesa-vulkan-drivers spirv-headers`.

### 2.2 Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 2ca15f540          # the commit all results were measured on
R=/opt/rocm/core-10.0
HIPCXX=$R/llvm/bin/clang HIP_PATH=$R CMAKE_PREFIX_PATH=$R \
  cmake -S . -B build -DGGML_HIP=ON -DGGML_VULKAN=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16 --target llama-server llama-bench
LD_LIBRARY_PATH=$R/lib build/bin/llama-server --list-devices
```

Expected devices: `ROCm0` (RX 7900 XT), `Vulkan0` (iGPU), `Vulkan1` (RX 7900 XT). The optional web-UI download during configure can fail; it does not affect the server. Compile time is about 10 minutes.

### 2.3 Get the model

```bash
mkdir -p ~/models && cd ~/models
curl -L -C - -O https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf   # 12.12 GB
head -c 20000000 Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf | grep -a -o 'blk\.64\.nextn\.[a-z_.]*' | sort -u   # MTP tensors present?
```

Use the `-mtp` file: without the `blk.64.nextn.*` tensors `draft-mtp` has nothing to draft with.

### 2.4 Run it

[`../configs/linux/run-llama.sh`](../configs/linux/run-llama.sh) starts three servers. The core of the dGPU server:

```bash
LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib llama-server \
  --model ~/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --device ROCm0 \
  --host 0.0.0.0 --port 11437 --api-key "$(cat ~/.llama-api-key)" \
  --n-gpu-layers 999 --ctx-size 131072 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 1024 \
  --spec-type ngram-simple,draft-mtp --spec-draft-n-max 4 \
  --threads 8 --parallel 1 --jinja --reasoning-format deepseek \
  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
```

Environment overrides: `DGPU_CTX`, `DGPU_KV`, `DGPU_UBATCH`, `DGPU_EFFORT`, `DGPU_HOST`, `DGPU_PORT`, `LLAMA_API_KEY`, `LLAMA_NO_AUTH=1`. For the maximum context use `DGPU_KV=q4_0 DGPU_CTX=262144 DGPU_UBATCH=512` (peak VRAM 19.9 of 19.98 GiB, no headroom).

The script generates `~/.llama-api-key` on first run (mode 600). The dGPU server is skipped, with a message, if more than 2 GiB of VRAM is already in use (for example TabbyAPI), instead of failing with an out-of-memory error. `reasoning_effort` accepts `low`, `medium` and `xhigh`; the server defaults (`--temp 0.6`, `--chat-template-kwargs`) are untested choices that clients can override per request.

The iGPU (port 8081) runs **Qwen3.8-4B-Distill Q4_K_M** (tool calling 7/7, ~11 t/s) on `--device Vulkan0`. The CPU server (port 8082) runs **Qwen3.8-2B-Distill Q4_K_M** (~22 t/s; only 3/7 tool calls pass, so use it for light chat only). Both bind to `127.0.0.1`. They share DDR4 bandwidth.

Desktop shortcuts: [`start-llama.sh`](../configs/linux/start-llama.sh) asks before stopping TabbyAPI and then runs `run-llama.sh`; see [`tabbyapi-guide.md`](tabbyapi-guide.md#6-desktop-shortcuts).

### 2.5 Verify

```bash
curl -s http://127.0.0.1:11437/health
curl -s http://127.0.0.1:11437/v1/chat/completions -H "Authorization: Bearer $(cat ~/.llama-api-key)" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists. Code only."}],"max_tokens":400,"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  | python3 -c "import json,sys; t=json.load(sys.stdin)['timings']; print(t['predicted_per_second'], t['draft_n_accepted'], t['draft_n'])"
```

`draft_n` above 0 means speculation is active; expect 60-80 t/s on this prompt. The harness in [`../bench`](../bench) reproduces the published numbers (`bench.py`, `bench-pp.py`, `bench-deep.py`, `bench-tools.py`, `bench-quality.py`).

### 2.6 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `libhipblas.so.3: cannot open shared object file` | `export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib` |
| CMake: "ROCm root does not contain the HIP runtime CMake package" | install the `amdrocm-*-dev10.0` packages and point `HIP_PATH`/`CMAKE_PREFIX_PATH` at `/opt/rocm/core-10.0` |
| Server out of memory at load | lower `--ctx-size`; K and V must use the same cache type; close other GPU users |
| `invalid kernel file` on the iGPU | ROCm cannot run `gfx90c`; use `--device Vulkan0` |
| Speed far below ~60 t/s on new code | check `draft_n` is above 0 and that the file is the `-mtp` variant |
| Another process holds the GPU | `/opt/rocm/core-10.0/bin/rocm-smi --showpids` |

## 3. Windows 11 (untested adaptation)

Nothing in this section was run in this work. It restates the earlier Windows write-up ([`windows-original-guide.md`](windows-original-guide.md), measured by its author on the same GPU with ROCm 7.2: 55-56 t/s new code, ~160 t/s edits, ~620 t/s prefill, 122,880-token context while the desktop shares the GPU) with the changes this work found for current upstream.

1. **Driver and SDK.** AMD Adrenalin 32.0.31021.6002 or newer; **HIP SDK 7.2** in its default folder. That guide found ROCm 10.0 no better on Windows (new code +4.5%, code edits -10%, prefill -4%, more VRAM). Do not install 10.0 for this setup.
2. **Build tools.** Visual Studio 2022 Build Tools (Desktop C++ workload: MSVC v143, Windows 11 SDK, CMake, Ninja), Git, Python 3.11+.
3. **Source: upstream, no fork or patches.** The earlier guide used the `TheTom/llama-cpp-turboquant` fork plus three patches. On upstream `2ca15f540` the MTP CUDA-graph change (patch 0002, upstream PR #28549) is already merged and `hip.h` already defines `RDNA3` (patch 0001). Patch 0003 (`srv->set_tcp_nodelay(true)`) is a one-line latency tweak that was not tested here. TurboQuant (`turbo4`) is not needed.
4. **Build** (from a `cmd` prompt; adjust paths): HIP, `gfx1100`, `-j 6` (the earlier guide reports `-j 8` runs out of RAM):

   ```bat
   call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
   set HIP_PATH=C:\Program Files\AMD\ROCm\7.2
   set PATH=%HIP_PATH%\bin;%PATH%
   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 2ca15f540
   cmake -S . -B build-hip -G Ninja -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang.exe" -DGPU_TARGETS=gfx1100 ^
         -DGGML_HIP=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF
   cmake --build build-hip --config Release -j 6 --target llama-server llama-bench
   ```
5. **Model and launcher.** Download the same `-mtp` GGUF (section 2.3) and start it with [`../configs/windows/start-llama-server.bat`](../configs/windows/start-llama-server.bat), which uses the flags from section 2.4. Start `llama-server.exe` from `cmd`/PowerShell with the ROCm `bin` folder on `PATH`, not from Git Bash.
6. **GPU memory.** If the monitor is plugged into the 7900 XT, Windows desktop apps take 3-4.5 GiB and the context must be reduced (the earlier guide settled on 122,880); plugging the monitor into the iGPU output frees the card. Spilling into system RAM collapses prefill speed.

Everything else (flags, speculation types, KV cache, batch sizes) is platform independent; if you validate this section on Windows, please send the numbers.
