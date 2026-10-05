# llama-cpp-7900xt-optimization



Guide for a LLM to setup AMD RX 7900 XT Llama.cpp optimization Turboquant MTP+ngram Qwen3.8-27B-GSQ-RCO-GGUF 


# Local LLM server on AMD RX 7900 XT: step-by-step setup (llama.cpp TurboQuant fork)

This document reproduces the production setup of 2026-09-28 from scratch. All numbers were measured on this machine.

**For the LLM/agent following it:** do the steps in order. Every step ends with a **Check**; do not continue until the check passes. Run commands in **PowerShell** unless a step says `cmd`. Never use Git Bash to *start* `llama-server.exe` (see Troubleshooting).

Paths used throughout (change them if the user name differs):

| Name | Path |
|---|---|
| `ROOT` | `C:\Users\.GAMING-PC\turboquant` |
| `SRC` | `C:\Users\.GAMING-PC\turboquant\hip-tq-new` (llama.cpp source) |
| `MODELS` | `C:\Users\.GAMING-PC\.lmstudio\models` |
| `ROCM` | `C:\Program Files\AMD\ROCm\7.2` |

---

## 1. Model and why

**Qwen3.8-27B, GSQ-RCO IQ3_S quant with built-in MTP head** (by ISTA-DASLab).
Hugging Face repo `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`, file `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, 12,120,016,960 bytes.

| Why | Detail |
|---|---|
| Best coding model that fits 20 GB with room for context | 27B quality, whole model on the GPU |
| Reliable tool calling | 7/7 on the tool-calling test suite |
| Controllable thinking | `reasoning_effort` = `xhigh` / `medium` / `low` via the chat template; the server uses `medium` (`xhigh` overthinks) |
| Built-in MTP draft head (`blk.64.nextn.*`) | Speculative decoding with no second model: 32 -> 55 tok/s |
| GSQ-RCO IQ3_S instead of a 4-bit quant | Reported task-lossless by ISTA-DASLab, ~2 GB smaller than UD-IQ4_XS; the saved VRAM becomes context |
| Hybrid architecture | Only 16 of 64 layers use attention (48 are Gated DeltaNet), so the KV cache is small (~30-35 KB/token) |

Facts: architecture `qwen35`; 64 layers + 1 MTP layer; 24 heads / 4 KV heads; head dim 256; trained context 262144; multimodal with M-RoPE, so `--cache-reuse` and context shift cannot work (the server silently disables them).

---

## 2. Hardware and base software

| Part | This machine | Needed for |
|---|---|---|
| dGPU | AMD Radeon RX 7900 XT, 20 GB, RDNA3 **gfx1100** | Runs the whole model via HIP/ROCm |
| iGPU | Ryzen 5700G Radeon Vega 8 | Not used by this server |
| CPU | Ryzen 7 5700G, 8 cores / 16 threads | Sampling, HTTP, ngram draft lookup (8 threads) |
| RAM | 48 GB | RAM prompt cache (8 GiB) + memory-mapped model |
| Disk | >= 30 GB free | Model 12.1 GB, source + build ~6 GB |
| OS | Windows 11 Pro 25H2 (build 26200) | |
| GPU driver | AMD Adrenalin 32.0.31021.6002 | HIP runtime (`System32\amdhip64_7.dll`) |
| ROCm | HIP SDK for Windows 7.2 (7.2.60201) | Compiler + rocBLAS/hipBLAS |
| Build tools | Visual Studio 2022 Build Tools (MSVC 14.44, Windows SDK 10.0.26100, CMake, Ninja), Git, Python 3.11+ | Build + model download |

---

## 3. Step-by-step setup

### Step 1. GPU driver

Install AMD Software: Adrenalin Edition 32.0.31021.6002 or newer.

**Check:**
```powershell
Get-CimInstance Win32_VideoController | Where-Object Name -match '7900' | Select-Object Name, DriverVersion
```
Expect `AMD Radeon RX 7900 XT` and driver `32.0.31021.6002` or higher.

### Step 2. HIP SDK 7.2

Install "AMD HIP SDK for Windows" version 7.2 into its default folder `C:\Program Files\AMD\ROCm\7.2`. Do not install ROCm 10 for this setup (tested slower, see Findings).

**Check:**
```powershell
& "C:\Program Files\AMD\ROCm\7.2\bin\hipconfig.exe" --version
```
Expect `7.2.60201-...`.

### Step 3. Build tools

1. Install Visual Studio 2022 Build Tools with the workload **Desktop development with C++** (includes MSVC v143, Windows 11 SDK, "C++ CMake tools for Windows" with CMake and Ninja).
2. Install Git for Windows and Python 3.11 or newer, then run `python -m pip install huggingface_hub`.

**Check:**
```powershell
Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
git --version; python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```
Expect `True`, `True`, a git version, and a huggingface_hub version.

### Step 4. Get the llama.cpp source (TurboQuant fork + 3 patches)

Why a fork: upstream llama.cpp (latest v0.5.0) has no TurboQuant compressed KV cache (`turbo4`). The fork `TheTom/llama-cpp-turboquant` has it, plus MTP support for this model. The fork tip used here is `bcb85fc` (2026-09-28). It last synced with upstream on 2026-08-06; the one later upstream change that matters (PR #28549, CUDA graphs for MTP drafting, which also applies to HIP) is added as a patch.

```powershell
cd C:\Users\.GAMING-PC\turboquant
git clone https://github.com/TheTom/llama-cpp-turboquant.git hip-tq-new
cd hip-tq-new
git checkout -b tq-latest-tuned bcb85fc3ae85efa0f5f392c6c880dfc524923860
git -c user.name=local -c user.email=local@localhost am ..\patches\0001-hip-define-RDNA2-3-4-arch-family-macros-from-GPU_TAR.patch ..\patches\0002-Enable-CUDA-graph-for-MTP-draft-28549.patch ..\patches\0003-server-enable-TCP_NODELAY-on-accepted-connections-st.patch
```

The 3 patch files are in `ROOT\patches` on this machine. What they do:

1. `0001` RDNA3 patch: appends to `ggml/src/ggml-hip/CMakeLists.txt`, just before the final `target_link_libraries(ggml-hip ...)` line:
   ```cmake
   # Define architecture family macros based on GPU_TARGETS
   if (GPU_TARGETS MATCHES "gfx11")
       target_compile_definitions(ggml-hip PRIVATE RDNA3)
       message(STATUS "Defined RDNA3 for GPU_TARGETS=${GPU_TARGETS}")
   elseif (GPU_TARGETS MATCHES "gfx10")
       target_compile_definitions(ggml-hip PRIVATE RDNA2)
       message(STATUS "Defined RDNA2 for GPU_TARGETS=${GPU_TARGETS}")
   elseif (GPU_TARGETS MATCHES "gfx12")
       target_compile_definitions(ggml-hip PRIVATE RDNA4)
       message(STATUS "Defined RDNA4 for GPU_TARGETS=${GPU_TARGETS}")
   endif()
   ```
2. `0002` is upstream commit `2f3fd02526682adbd3ba771d929d271e477a35c5` (PR #28549) with conflicts in `src/llama-context.cpp` resolved.
3. `0003` adds one line in `tools/server/server-http.cpp`, right after `srv->set_write_timeout(params.timeout_write);`:
   ```cpp
   srv->set_tcp_nodelay(true);
   ```

If the patch files are missing, recreate 0001 and 0003 by hand as shown, commit each, then:
```powershell
git fetch https://github.com/ggml-org/llama.cpp 2f3fd02526682adbd3ba771d929d271e477a35c5
git cherry-pick -x 2f3fd0252
```
and resolve the conflicts in `src/llama-context.cpp` (the file uses CRLF line endings):
- In `process_ubatch()`: keep the fork's KV-stream phase block, then use `auto * res = get_gf_res_prev();`.
- Before `graph_reserve()`: keep the fork's `graph_reserve` signature; add only the new `get_gf_res_prev()` function from the upstream side; drop upstream's `ubatch_prepare_reserve()` (unrelated context, unused in the fork).
- Also replace the remaining `gf_res_prev->reset();` (in the phase-arena repartition code, after `sched.reset();`) with a loop that resets both `gf_res_prev` entries, followed by `gf_res_prev_active = nullptr;`.

**Check:**
```powershell
git log --oneline -4
git diff --stat bcb85fc3a HEAD
```
Expect the 3 patch commits on top of `bcb85fc`, and `4 files changed, 62 insertions(+), 12 deletions(-)`.

### Step 5. Build llama-server (HIP, gfx1100)

Create `ROOT\build-hip-new.bat` and run it from `cmd`:

```bat
@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set HIP_PATH=C:\Program Files\AMD\ROCm\7.2
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2
set HIP_VISIBLE_DEVICES=0
set PATH=%HIP_PATH%\bin;%PATH%
set CMAKE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set NINJA="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
cd /d C:\Users\.GAMING-PC\turboquant\hip-tq-new
%CMAKE% -S . -B build-hip -G Ninja -DCMAKE_MAKE_PROGRAM=%NINJA% ^
    -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang.exe" -DGPU_TARGETS="gfx1100" ^
    -DGGML_HIP=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release ^
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF
%CMAKE% --build build-hip --config Release -j 6 --target llama-server llama-bench
echo CMAKE_BUILD_EXIT=%errorlevel%
```

```powershell
cmd /c "C:\Users\.GAMING-PC\turboquant\build-hip-new.bat" *> C:\Users\.GAMING-PC\turboquant\build-hip-new.log
```

Takes 20-40 minutes. Rules: use `-j 6` (`-j 8` runs out of RAM on heavy files and fails without a clear error); if you rebuild after a large source update, delete `build-hip` first.

**Check:**
```powershell
Select-String C:\Users\.GAMING-PC\turboquant\build-hip-new.log -Pattern 'Defined RDNA3|CMAKE_BUILD_EXIT'
$env:PATH = "C:\Program Files\AMD\ROCm\7.2\bin;$env:PATH"
& C:\Users\.GAMING-PC\turboquant\hip-tq-new\build-hip\bin\llama-server.exe --list-devices
```
Expect `Defined RDNA3 for GPU_TARGETS=gfx1100`, `CMAKE_BUILD_EXIT=0`, and `ROCm0: AMD Radeon RX 7900 XT (20464 MiB, ...)`.

### Step 6. Download the model

```powershell
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF', filename='Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf', local_dir=r'C:\Users\.GAMING-PC\.lmstudio\models\ISTA-DASLab\Qwen3.8-27B-GSQ-RCO-GGUF'))"
```

**Check:**
```powershell
(Get-Item "C:\Users\.GAMING-PC\.lmstudio\models\ISTA-DASLab\Qwen3.8-27B-GSQ-RCO-GGUF\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf").Length
```
Expect `12120016960`.

### Step 7. Create an API key and the launcher

Generate a key once and keep it:
```powershell
-join ((1..48) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) })
```

Create `ROOT\start-dgpu.bat` (replace `<API_KEY>` with the generated key and `<CTX>` with the value from Step 9; start with `122880`, the value used in production):

```bat
@echo off
setlocal
set HIP_VISIBLE_DEVICES=0
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2
set PATH=%ROCM_PATH%\bin;%PATH%
set SERVER=C:\Users\.GAMING-PC\turboquant\hip-tq-new\build-hip\bin\llama-server.exe
set MODEL=C:\Users\.GAMING-PC\.lmstudio\models\ISTA-DASLab\Qwen3.8-27B-GSQ-RCO-GGUF\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
set CTX=<CTX>

"%SERVER%" ^
    -m "%MODEL%" ^
    -c %CTX% ^
    -ngl 99 ^
    --parallel 1 ^
    --cache-type-k q8_0 ^
    --cache-type-v turbo4 ^
    --reasoning-format deepseek ^
    --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" ^
    --spec-type ngram-mod,draft-mtp-adaptive ^
    --spec-draft-n-max 4 ^
    --spec-ngram-mod-n-match 12 ^
    --spec-ngram-mod-n-min 16 ^
    --slot-save-path "C:\Users\.GAMING-PC\turboquant\kv-slots" ^
    --temp 0.2 ^
    --top-p 0.95 ^
    --top-k 20 ^
    --min-p 0.0 ^
    --port 11437 ^
    --host 0.0.0.0 ^
    --api-key "<API_KEY>" ^
    -a qwen3.8-27b-tq ^
    --log-prefix
```

Also create the folder `ROOT\kv-slots`.

**Check:** the file exists and contains no `<` placeholders:
```powershell
Select-String C:\Users\.GAMING-PC\turboquant\start-dgpu.bat -Pattern '<API_KEY>|<CTX>'
```
Expect no output.

### Step 8. Start and smoke-test the server

```powershell
Start-Process cmd.exe -ArgumentList '/c "C:\Users\.GAMING-PC\turboquant\start-dgpu.bat"' -WindowStyle Minimized
```

Wait ~20 s, then:
```powershell
$h = @{ Authorization = "Bearer <API_KEY>" }
Invoke-RestMethod http://127.0.0.1:11437/health
(Invoke-RestMethod http://127.0.0.1:11437/v1/models -Headers $h).data.id
$body = @{ messages = @(@{ role = "user"; content = "Write a Python function that merges two sorted lists. Code only." }); max_tokens = 512; chat_template_kwargs = @{ reasoning_effort = "low" } } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod http://127.0.0.1:11437/v1/chat/completions -Method Post -Headers $h -ContentType 'application/json' -Body $body
"{0:N1} tok/s, draft accepted {1}/{2}" -f $r.timings.predicted_per_second, $r.timings.draft_n_accepted, $r.timings.draft_n
```

**Check:** `status: ok`; model id `qwen3.8-27b-tq`; about 50-60 tok/s with `draft_n` above 0 (MTP is active). Below ~40 tok/s means a VRAM problem (Step 9).

### Step 9. Choose the context size (VRAM budget)

The 7900 XT is shared with everything that draws on screen when the monitor is plugged into it. On this machine the Windows desktop (dwm, 4K) uses ~1.6 GiB, Chrome ~1.1 GiB, VS Code and Edge ~0.4 GiB each: **3-4.5 GiB in total, changing through the day.** If the server does not fit next to them, Windows moves part of it into system RAM. Decode then drops a little, but **prompt processing (prefill) collapses** (630 -> 110-430 tok/s measured).

Server GPU memory needed ≈ 12.3 GiB + 0.036 GiB per 1000 tokens of context (measured: ~14.6 GiB at 65536, ~16.7 GiB at 122880, ~17.8 GiB at 153600).

1. With the server stopped, measure what the other apps use:
   ```powershell
   $c = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage'; $c.CounterSamples | Where-Object CookedValue -gt 1GB | ForEach-Object { '{0:N2} GiB used' -f ($_.CookedValue/1GB) }
   ```
2. Pick the largest CTX (multiple of 8192) where `server VRAM + other apps + 0.5 GiB margin <= 20`.
3. Verify with a long prompt (8K+ tokens): prefill should be ~600 tok/s. If it is far lower, reduce CTX by 8192 and retry.

Measured on 2026-09-28 with ~3.3 GiB used by the desktop and browsers (prefill of an 8.4K-token prompt; decode stayed ~55 tok/s in all cases):

| CTX | Prefill (tok/s) | Verdict |
|---|---|---|
| 153600 | 425 | spills |
| 147456 | 135 | spills |
| 139264 | 107 | spills |
| 131072 | 432, 492 | spills |
| **122880** | **692, 691, 634** | **production value** |
| 114688 | 560 | fits |
| 65536 | 615-624 | fits |

Results near the limit jump around between runs because Windows moves memory in and out of the GPU dynamically. Repeat a borderline test 2-3 times before trusting it.

Best fix: plug the monitor into the motherboard (iGPU) output. The desktop and browsers then render on the iGPU, the 7900 XT's VRAM is free for the model, and CTX can go back to 153600 or higher.

### Step 10. Start automatically at logon

```powershell
schtasks /create /tn "TurboQuant dGPU Server" /tr "C:\Windows\System32\cmd.exe /c \"C:\Users\.GAMING-PC\turboquant\start-dgpu.bat\"" /sc onlogon /f
```

**Check:** `schtasks /query /tn "TurboQuant dGPU Server"` shows the task as `Ready`. To restart the server later: stop `llama-server.exe`, then `schtasks /run /tn "TurboQuant dGPU Server"`.

Clients use the OpenAI-compatible API at `http://127.0.0.1:11437/v1`, model `qwen3.8-27b-tq`, header `Authorization: Bearer <API_KEY>`.

---

## 4. Configuration reference (every setting)

| Area | Setting | Value | Why |
|---|---|---|---|
| GPU | `HIP_VISIBLE_DEVICES` | `0` | HIP sees only the 7900 XT |
| GPU | `-ngl` | `99` | All layers on the GPU |
| GPU / VRAM | `-c` | see Step 9 | Largest context that fits next to the desktop without spilling |
| KV cache | `--cache-type-k` / `-v` | `q8_0` / `turbo4` | Most context per GB; f16 and q8_0 variants are not faster on this model |
| Batch | `-b` / `-ub` | defaults 2048 / 512 | Larger micro-batches overflow VRAM |
| Attention | flash attention | auto (on) | Required by quantized V, so always on |
| Speculative | `--spec-type` | `ngram-mod,draft-mtp-adaptive` | Per token, the first drafter with a draft wins. ngram-mod (CPU) drafts long runs when the reply copies text already in the prompt (code edits). The MTP head covers the rest |
| Speculative | `--spec-draft-n-max` | `4` | Best balance (adaptive depth 3..4) |
| Speculative | `--spec-ngram-mod-n-match` / `-n-min` | `12` / `16` | Beats the default 24/48 and 8 on code edits |
| CPU | threads / priority | defaults (8 / normal) | CPU is not the bottleneck |
| RAM | `--cache-ram` | default 8192 MiB | Saves idle conversations; switching back re-processes ~30 tokens instead of the whole prompt |
| Slots | `--parallel` | `1` | One user; follow-ups reuse the same slot's KV |
| Reasoning | `--reasoning-format`, effort | `deepseek`, `medium` | Thinking goes to `reasoning_content`; template accepts only `xhigh`/`medium`/`low` |
| Sampling | temp / top-p / top-k / min-p | 0.2 / 0.95 / 20 / 0 | Low temperature raises MTP acceptance; requests can override |
| Files | `--slot-save-path` | `ROOT\kv-slots` | Enables `/slots/N/action/save` and `restore` |
| Network | `--host` / `--port` | `0.0.0.0` / `11437` | Use `127.0.0.1` if no other machine needs access |
| Network | `--api-key` | generated | Required on every request |
| Network | TCP_NODELAY | on (patch 0003) | Streamed tokens are sent immediately |
| Network | timeout / HTTP threads | defaults 3600 s / 15 | Enough for long prefills |

---

## 5. Backends and ROCm versions

| Option | Result on this machine | Use |
|---|---|---|
| **HIP + ROCm 7.2, dGPU only** | 55-56 tok/s new code, ~160 code edit, ~620 prefill | **Production** |
| HIP + ROCm 10.0.0 (TheRock tarball, side-by-side in `ROOT\rocm-10.0.0`) | new code +4.5%, code edit -10%, prefill -4%, needs more VRAM | Not used |
| Vulkan, dGPU only (`--device Vulkan0`) | -28% new code, -16% code edit | Not used |
| Vulkan, dGPU + iGPU pooled (`--device Vulkan0,Vulkan1 -sm layer --tensor-split 75,25`) | Much slower (27B: 4.6 tok/s); only useful when model **weights** exceed 20 GB | Not used |
| ROCm 7.1 -> 7.2 | +5% decode | Done |

ROCm 10 notes, in case it is retested: the Windows developer kit is the tarball `therock-dist-windows-gfx110X-all-10.0.0.tar.gz` (2.27 GB, `https://stable.repo.amd.com/rocm/core/tarball/`). Extract it to its own folder and set its variables only inside a build script; do not uninstall HIP SDK 7.2 or set machine-wide variables. The pip method is for PyTorch only. Windows loads `amdhip64_7.dll` from the exe folder first, then System32 (the driver's copy), and only then PATH.

---

## 6. Findings (tuning round 2026-09-28)

Workloads: **new code** = a 512-token Python module; **code edit** = return a 150-line file with a rename (the reply copies the prompt); **prefill** = an 8.4K-token prompt. Units: tok/s. Noise about +-5%.

| Change | Result | Decision |
|---|---|---|
| Newer fork + MTP CUDA-graph patch (vs old build `df7f547`) | new code 52.1 -> 54.1, code edit 66.0 -> 68.7 | **Keep** |
| **ngram-mod before MTP, match 12** | code edit 71 -> **159-169**; new code unchanged | **Keep** (2.3x on edits) |
| ngram-mod match 24 / 8, ngram-simple | code edit 148 / 126 / 153 | Match 12 is best |
| KV f16 or q8_0 for K and V | speed within noise | Rejected (costs 20-50% context) |
| MTP fixed depth 2 / 3 | code edit 56.1 / 65.7 | Rejected |
| MTP adaptive 2..6 | code edit +5%, new code -3.5%, +0.3 GiB | Rejected |
| `--spec-chain`, `--gdn-replay` | slower | Rejected |
| `--prio 2`, `--poll 100` | no change | Rejected |
| ubatch 1024 / 2048 | prefill collapses at large context (VRAM) | Rejected |
| ROCm 10.0.0 | see section 5 | Rejected |
| RAM prompt cache | back to an earlier chat: 31 tokens re-processed (0.5 s) vs 11,396 (18.6 s) without it | Keep default |
| Context vs desktop VRAM | with ~3.3 GiB used by the desktop and browsers, 131072-153600 spill and prefill drops to 107-492; 122880 holds 634-692 | **CTX lowered 153600 -> 122880** (Step 9) |
| `--reasoning-budget N` (seen in a Qwen3.8-Flash-Next config) | caps thinking length: faster answers, can cut reasoning short | Not used; optional |
| Stability | 20/20 stress requests on the final config; one ROCm "unspecified launch failure" seen once in ~60 early requests, not reproduced | Watch |

---

## 7. Performance gain

| Workload | Before (old build, MTP only) | After | Gain |
|---|---|---|---|
| New code / explanations | 52.1 tok/s | 56.1 tok/s | **+8%** |
| Whole-file code edit | 66.0 tok/s | 163.5 tok/s | **2.5x** |
| Same edit asked again | 66.0 tok/s | 174.6 tok/s | **2.6x** |
| Back to an earlier chat | 0.5 s | 0.5 s | already optimal |
| Tool calling | 7/7 | 7/7 | unchanged |

Against the original setup without MTP (32.4 tok/s), new-code speed is **+73%**.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `llama-server.exe` exits instantly, no output, code `0xC0000135` | ROCm `bin` not on PATH, or started from Git Bash | Start from cmd/PowerShell with `PATH=%ROCM_PATH%\bin;%PATH%` (the launcher does this) |
| HTTP 401 | Missing or wrong key | Send `Authorization: Bearer <API_KEY>` |
| HTTP 500 "Unexpected reasoning effort" | `reasoning_effort` not in `xhigh`/`medium`/`low` | Use one of those |
| Decode fine but prefill slow (<450 tok/s on long prompts) | VRAM spilled to system RAM | Lower CTX (Step 9), close GPU-heavy apps, or move the monitor to the iGPU |
| Speed well below 50 tok/s | `draft_n` = 0 (MTP off) or VRAM spill | Check the `--spec-type` line; see Step 9 |
| Server killed but port still busy | `Start-Process` on a .bat returns cmd.exe's PID | `Get-CimInstance Win32_Process -Filter "name='llama-server.exe'"` and stop that PID |
| `ROCm error: unspecified launch failure` | Rare GPU fault (seen once) | Restart the server; if it repeats, use the old build (`ROOT\hip-tq\build-hip`) |
| Build fails with no clear error | Out of RAM | Rebuild with `-j 4` |
| `--help` output cannot be redirected | Windows console behavior | Read flags in `SRC\common\arg.cpp` |

---

## 9. Benchmark tools and rollback

- Benchmark harness on this machine: `ROOT\bench-tune.py` (one config) and `ROOT\bench-plan.py` (phases); results in `bench-results.jsonl`, logs in `bench-logs\`. Stop the production server first; only one model fits in VRAM. When comparing runs, use the first run of each config: ngram-mod remembers earlier replies and speeds up identical repeats.
- Tool-calling test: `set LLAMA_API_KEY=<API_KEY>` then `python ROOT\test-tooling.py qwen3.8-27b-tq http://127.0.0.1:11437/v1` (expect 7/7).
- Rollback: `ROOT\start-dgpu.pre-tune-2026-09-28.bat` is the previous launcher (old build, MTP only). Copy it over `start-dgpu.bat` and restart the task.
