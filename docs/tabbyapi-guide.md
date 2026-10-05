# TabbyAPI (ExLlamaV3-ROCm) on an RX 7900 XT: setup guide

Goal: serve **Qwen3.8-27B (EXL3 3.5 bpw) with the DFlash2 speculative draft model** through an OpenAI-compatible API from one RX 7900 XT (20 GB): about 65-90 t/s decode on code, 60-70 t/s at 12-50k tokens of context, 98,304-token context, tool calling 7/7, usable from opencode. Measurements are in [`benchmarks.md`](benchmarks.md#5-tabbyapi-exllamav3-results).

## 1. When to choose TabbyAPI over llama.cpp

Measured on the same machine and prompts ([side-by-side table](benchmarks.md#6-llamacpp-vs-tabbyapi-side-by-side)):

| | Prefer TabbyAPI | Prefer llama.cpp ([guide](llama-cpp-guide.md)) |
|---|---|---|
| Agent loops with long, growing prompts and mostly new text | decode 60-70 t/s at 12-50k tokens vs 36-51 t/s; prefill 850-1090 vs 630-790 t/s | |
| Edits that copy the prompt | 77-90 t/s | ~175 t/s |
| Context | up to 98k at full speed | 131k (262k with a 4-bit KV cache) |
| Tool calling (7-case suite) | 7/7 | 7/7 |

Only one engine can use the 7900 XT at a time (TabbyAPI peaks at 19.7 GiB). The two models differ (3.5-bit EXL3 of the base Qwen3.8-27B vs a 3-bit GGUF of the Swift fine-tune), so these figures compare speed and tool calling, not answer quality.

## 2. Components

| Part | Source | Version on this machine |
|---|---|---|
| ExLlamaV3 ROCm port | [`bsvinay/exllamav3-rocm`](https://github.com/bsvinay/exllamav3-rocm) (RDNA3 kernels for the EXL3 matmul and decode attention) | `360c902` |
| TabbyAPI | [`theroyallab/tabbyAPI`](https://github.com/theroyallab/tabbyAPI) (patched by the fork's `install_tabbyapi.sh`) | `f07131c` |
| PyTorch | installed by the fork's `setup_env.sh` | 2.13.0+rocm7.2 (bundled ROCm 7.2 runtime) |
| Main model | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (Apache-2.0, base `Qwen/Qwen3.8-27B`), 15 GB | EXL3 v1.4.2, 3.5 bpw, 6-bit head |
| Draft model | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` (base `incoai/Qwen3.8-27B-DFlash2`), 1.4 GB | EXL3 v1.4.2, 5.0 bpw |

The venv's own ROCm 7.2 wheels ran fine next to the system ROCm 10.0.0 used for llama.cpp. Requirements from the fork: Linux, an RDNA3 GPU, about 20 GB of disk for models, 32 GB+ RAM.

## 3. Install

These steps are the fork's documented procedure (its README, "Quick start"). The stack on this machine was installed that way; the commands were **not re-run** while writing this guide.

```bash
git clone https://github.com/bsvinay/exllamav3-rocm && cd exllamav3-rocm
rocm/scripts/setup_env.sh .venv-rocm            # Python 3.12 + torch 2.13.0 (ROCm 7.2 wheels) + dependencies
source .venv-rocm/bin/activate
ROCM_HOME=/opt/rocm rocm/scripts/build.sh        # builds the exllamav3_ext kernels for gfx1100 (~10 min)
rocm/scripts/download_models.sh models           # main + DFlash2 draft model from Hugging Face
rocm/scripts/install_tabbyapi.sh ../tabbyAPI models
```

Notes:

* `ROCM_HOME=/opt/rocm` is the fork's documented value. On this machine ROCm 10 lives under `/opt/rocm/core-10.0` and the build/run worked with the venv's bundled runtime.
* `~/tabbyAPI/models` is a symlink to `~/exllamav3-rocm/models`.
* The disk cost is large: the venv is about 15 GB, the models 16 GB, plus the uv package cache.

## 4. Configuration

[`../configs/tabbyapi/config.yml`](../configs/tabbyapi/config.yml) (the active config, copied to `~/tabbyAPI/config.yml`):

| Setting | Value | Meaning |
|---|---|---|
| `network.host` / `port` | `0.0.0.0` / `11437` | reachable from the LAN; use `127.0.0.1` for local-only |
| `network.disable_auth` | `false` | API key required (see section 5) |
| `model.max_seq_len` / `cache_size` | `98304` | largest 8-bit-cache context that fits (table below) |
| `model.cache_mode` | `Q8` | 8-bit KV cache; `Q4` and `Q6` are slower here |
| `model.chunk_size` | `2048` | prefill chunk |
| `model.max_batch_size` | `1` | single user |
| `model.vision` / `vision_offload` | `true` / `true` | vision weights live in system RAM, not VRAM |
| `model.reasoning` | `true` (`<think>` / `</think>`) | thinking goes to `reasoning_content` |
| `model.tool_format` | `qwen3_coder` | tool-call parser |
| `draft_model.draft_model_name` / `draft_cache_mode` | DFlash2 EXL3 5.0 bpw / `Q4` | speculative decoding; a 4-bit draft cache gave the same acceptance as FP16 in the fork's greedy tests |
| `memory.sysmem_recurrent_cache` | `4096` | MiB of system RAM for the recurrent (DeltaNet) cache |

### Context size and cache type (measured)

| `max_seq_len` | `cache_mode` | VRAM peak | Decode t/s at ~41k / mid / long | Needle recall |
|---:|---|---:|---|---|
| **98,304** | **Q8** | **19.69 GiB** | **80 / 102 / 60** | found at 78k tokens |
| 131,072 | Q4 | 18.83 GiB | 40 / 54 / 28 | found at 105k |
| 131,072 | Q6 | 19.82 GiB | 30 / 20 / 15 | found at 105k |
| 196,608 | Q4 | 19.97 GiB | | loads, then out of memory on a long prompt |

98k with an 8-bit cache is the largest fast setting; use 131k with `Q4` only if you need the length and accept about half the decode speed. Tests reached 80-86% of each context, not 100%. The `Qwen3.8-27B` config comment about a 196,608-token context describes a 24 GB card.

## 5. Authentication and network access

TabbyAPI writes its keys to `~/tabbyAPI/api_tokens.yml` on first start (`api_key` for inference, `admin_key` for admin routes). To use your own inference key, replace the `api_key` value in that file and restart. Never commit this file.

* Send `Authorization: Bearer <api_key>`. Without it the API returns **401**; `/health` needs no key.
* Linux firewall: if `ufw` is active, other machines are blocked until you allow the port from your LAN only, e.g. `sudo ufw allow from 192.168.0.0/16 to any port 11437 proto tcp`.
* Do not expose the port to the internet. If you only need local clients, set `host: 127.0.0.1`.

## 6. Running it

Always launch through the fork's script, which sets `PYTHONPATH` to the fork and `EXL3_NOGRAPH=mlp,gdn`:

```bash
cd ~/exllamav3-rocm && source .venv-rocm/bin/activate
rocm/scripts/run_tabbyapi.sh config.yml ~/tabbyAPI     # runs: python main.py --config config.yml in ~/tabbyAPI
```

A hand-started `python main.py` also worked once the venv python was invoked through the venv path (the venv's `python` is a symlink to the base interpreter and loses the venv's packages if you call the base interpreter directly: `ModuleNotFoundError: No module named 'loguru'`). Measured difference between the two: a few percent at depth, within noise elsewhere ([benchmarks section 5.5](benchmarks.md#55-launcher-script-vs-hand-started-process)).

### Desktop shortcuts

[`../configs/linux/start-tabbyapi.sh`](../configs/linux/start-tabbyapi.sh) and [`start-llama.sh`](../configs/linux/start-llama.sh) are terminal launchers. Each checks whether the *other* engine holds the GPU and asks before stopping it. On GNOME, create `~/Desktop/<name>.desktop` with `Exec=/path/to/script`, `Terminal=true`, make it executable, and mark it trusted with `gio set <file> metadata::trusted true`.

## 7. Using it

```bash
curl -s http://127.0.0.1:11437/health
curl -s http://127.0.0.1:11437/v1/models -H "Authorization: Bearer $KEY"
curl -s http://127.0.0.1:11437/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B-EXL3-3.5bpw","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":300}'
```

The model id is `Qwen3.8-27B-EXL3-3.5bpw`. `reasoning_effort` (`low` / `medium` / `xhigh`) can be passed in `chat_template_kwargs`. TabbyAPI returns `"usage": null` on non-streaming responses; for speed use `stream: true` with `stream_options: {"include_usage": true}` (as `bench/bench-api.py` does) or read the server log, which prints tokens per second, cache hit and draft acceptance for every request.

### opencode

[`../configs/opencode/opencode.json`](../configs/opencode/opencode.json) adds a `tabby` provider (merge it into `~/.config/opencode/opencode.json`). It reads the key with `"apiKey": "{file:~/.llama-api-key}"`, so the key is not duplicated; put the key in that file or change the path. Run headless as:

```bash
opencode run -m tabby/Qwen3.8-27B-EXL3-3.5bpw "your task" < /dev/null
```

**`< /dev/null` is required** in a non-interactive shell: opencode otherwise waits on stdin and never sends a request (it looks like the known Qwen tool-calling hang). Test result: a seeded off-by-one bug was found, fixed and verified in 59 s using the read, edit and bash tools. With the `oh-my-opencode` plugin, opencode sends a ~24k-token system prompt: the first request takes ~25 s, later steps are 94-100% prompt-cache hits and decode at 62-120 t/s with 33-65% of drafted tokens accepted. Set `limit.context` to the server's `max_seq_len` (98304).

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'loguru'` | launched with the base interpreter; use `run_tabbyapi.sh` or `.venv-rocm/bin/python` |
| Out of memory at load or on a long prompt | lower `max_seq_len`; another process holds the GPU (`/opt/rocm/core-10.0/bin/rocm-smi --showpids`); at `Q4` 196k the long request OOMs |
| HTTP 401 | missing or wrong key, or `api_tokens.yml` was regenerated |
| Port 11437 already in use | llama.cpp's `run-llama.sh` also defaults to 11437; run one engine at a time |
| Needle or long-answer tests return empty text | reasoning used the token budget; raise `max_tokens` and/or set `reasoning_effort: low` |
| Speeds far below the vendor README (147-157 t/s) | the README's script uses repetitive synthetic filler code with greedy sampling on a 24 GB 7900 XTX; see [benchmarks 5.6](benchmarks.md#56-vendor-claims-vs-these-results) |
