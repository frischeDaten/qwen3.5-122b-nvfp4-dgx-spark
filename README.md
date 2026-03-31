# vLLM – Qwen3.5-122B-A10B-NVFP4 on DGX Spark

> **This repository has been archived and consolidated into [spark_vllm_docker](https://github.com/JungkwanBan/spark_vllm_docker).**
> All future updates will be made in the unified repository.

**English** | [한국어](README.ko.md)

Run [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) with vLLM on **NVIDIA DGX Spark (GB10 / SM121)**.

Self-contained multi-stage Docker build that compiles FlashInfer from source for SM121, installs vLLM nightly, and applies all NVFP4-specific patches required to serve the Qwen3.5 VL MoE architecture. No external pre-built base image required.

---

## Model Overview

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.5-122B-A10B-Instruct |
| Quantization | NVFP4 (W4A4, block-size 16) via llm-compressor |
| Architecture | 48 hybrid layers: 36 GDN (Gated Delta Net / linear-attn) + 12 full-attention, all-MoE FFN |
| Experts | 256 experts, top-8, 1 shared expert per layer |
| Max context | 262 144 tokens |
| KV cache | FP8 |
| MTP weights | Extracted from [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) (BF16, 785 keys, 4.7 GB) |

> **Note on MTP weights:** The NVFP4 quantized checkpoint (`txn545/Qwen3.5-122B-A10B-NVFP4`) does not include `mtp.*` weights — they are stripped during quantization. To enable MTP speculative decoding, BF16 MTP weights were extracted from the original [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) and saved as `mtp_weights.safetensors` in the NVFP4 checkpoint directory. A Dockerfile patch (`mtp_quant_exclusion_fix`) ensures these layers remain in BF16 rather than being incorrectly processed through the NVFP4 path.
>
> [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) added MTP weights to their checkpoint as of 2026-03-02, but this has not been tested with the setup in this repository.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA DGX Spark (GB10) | SM12x GPU required |
| NVIDIA Container Toolkit | `nvidia-ctk`, `docker` with GPU support |
| Docker Buildx | Required for multi-stage build with cache mounts |
| Docker Compose v2 | `docker compose` (not `docker-compose`) |
| External Docker network `monitoring` | `docker network create monitoring` |
| Model weights | Download from [Hugging Face](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) |

---

## Quick Start

### 1. Clone this repository

```bash
git clone git@github.com:JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4.git
cd SPARK_Qwen3.5-122B-A10B-NVFP4
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — at minimum set MODEL_HOST_PATH
```

Key variables in `.env`:

| Variable | Default (example) | Description |
|---|---|---|
| `MODEL_HOST_PATH` | `/path/to/Qwen3.5-122B-A10B-NVFP4` | Host path to the downloaded model |
| `HOST_PORT` | `8000` | Port exposed on the host |
| `MAX_MODEL_LEN` | `131072` | Max sequence length (model max: 262144) |
| `MAX_NUM_SEQS` | `4` | Max concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU VRAM for vLLM |
| `SWAP_SPACE` | `16` | CPU swap space in GiB |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | Max tokens per chunked-prefill batch |

### 3. Build the image

```bash
docker compose build
```

### 4. Start the service

```bash
docker compose up -d
docker compose logs -f   # watch startup (~5-10 min for weight loading)
```

### 5. Test inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "txn545_Qwen3.5-122B-A10B-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## Architecture & Key Fixes

### Why a custom model class?

vLLM does not yet include a built-in class for `Qwen3_5MoeForConditionalGeneration` (the VL MoE variant). `qwen3_5_vl_moe.py` provides this class and is registered into vLLM's model registry at image build time.

### Bug fixes applied

#### 1. `tile_tokens_dim` – FlashInfer 0.6.1 compatibility
FlashInfer 0.6.1 removed the `tile_tokens_dim` parameter from `trtllm_fp4_block_scale_moe()`.
Fix: `sed` patch in `Dockerfile` removes the argument from vLLM's call site.

#### 2. SM12x MoE backend selection
On DGX Spark (SM121), the two FlashInfer MoE paths both fail:

| Backend | Why it fails on SM12x |
|---|---|
| `latency` → TRT-LLM JIT | JIT only compiles for `major=10` (SM100), not SM12x |
| `throughput` → SM120 MXFP4_MINIMAL | Kernel requires FP8 activations; vLLM passes BF16 |

Fix: `VLLM_USE_FLASHINFER_MOE_FP4` is left unset (defaults to `0`) so vLLM falls back to the **native `cutlass_moe_fp4`** path, which works correctly on SM12x.

#### 3. GDN `in_proj` weights loaded as zeros (→ `!!!!` output)
The checkpoint's NVFP4 quantization `ignore` list contains the original HuggingFace split names (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) as unquantized BF16 tensors. vLLM's `Qwen3NextGatedDeltaNet` fuses these into `in_proj_qkvz` and `in_proj_ba` — names **not** in the ignore list — so vLLM incorrectly applies NVFP4 quantization and creates `weight_packed` parameters. The BF16 weight loader then cannot find a `weight` parameter, loads nothing, and all 36 GDN layers produce zero output.

Fix: `qwen3_5_vl_moe.py` appends two regex patterns to `quant_config.ignore` before the language model is instantiated:
```python
"re:.*linear_attn\\.in_proj_qkvz"
"re:.*linear_attn\\.in_proj_ba"
```
This forces these layers to use `UnquantizedLinearMethod` (BF16), matching the actual checkpoint data.

---

## Benchmarks (2026-03-01)

Hardware: NVIDIA DGX Spark (GB10, SM121), single GPU, NVFP4 W4A4
Tool: [llama-benchy](https://github.com/menloresearch/llama-benchy) v0.3.3, concurrency=1, 3 runs per config

### llama-benchy: MTP OFF vs MTP ON

#### Prefill (prompt processing, tok/s)

| Prompt tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| pp128 | 441 | 286 | -35% |
| pp256 | 732 | 582 | -21% |
| pp512 | 1,146 | 938 | -18% |
| pp1024 | 1,602 | 1,401 | -13% |

#### Decode (token generation, tok/s)

| Gen tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| tg128 | 15.1 | 12.7 | -16% |
| tg256 | 15.1 | 12.6 | -17% |
| tg512 | 15.1 | 12.6 | -17% |
| **Peak** | **16.0** | **14.0** | -13% |

#### Time to First Token (e2e_ttft, ms)

| Prompt tokens | MTP OFF | MTP ON | Change |
|---|---|---|---|
| pp128 | 295 | 505 | +71% |
| pp256 | 354 | 445 | +26% |
| pp512 | 450 | 550 | +22% |
| pp1024 | 643 | 735 | +14% |

> **Note:** llama-benchy measures raw prefill/decode throughput without reasoning tokens.
> MTP adds overhead per step (draft model forward pass) that is not recouped at concurrency=1
> with short, non-reasoning completions. The benefit of MTP appears in end-to-end inference
> with reasoning/thinking mode where the speculative tokens offset the draft overhead.

### End-to-End Chat Completions (reasoning mode)

Test: code generation (binary search), `max_tokens=512`, `temperature=0.0`, 5 warm runs
Metric: total completion_tokens (thinking + content) / wall time

| | MTP OFF (tok/s) | MTP ON (tok/s) | Change |
|---|---|---|---|
| Run 1 | 7.8* | 24.5 | — |
| Run 2 | 15.2 | 24.6 | +62% |
| Run 3 | 15.1 | 24.4 | +62% |
| Run 4 | 15.1 | 24.4 | +62% |
| Run 5 | 15.2 | 24.4 | +61% |
| **Avg (warm)** | **15.15** | **24.46** | **+61.5%** |

> \*Run 1 is cold start (torch.compile + CUDA graph first capture).
> MTP weights (785 keys, 4.7 GB BF16) extracted from original [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) and merged into the NVFP4 checkpoint.

### Other A/B Tests

#### MoE Backend: CUTLASS vs Marlin

| Backend | Avg tok/s | Notes |
|---|---|---|
| **CUTLASS W4A4** (native) | **15.2** | SM121 native FP4 tensor cores |
| Marlin W4A16 | 15.3 | No benefit on SM121; "Not enough SMs for max_autotune_gemm" |

#### KV Cache: BF16 vs FP8

| KV dtype | Available KV memory | 262K concurrent capacity |
|---|---|---|
| BF16 (auto) | ~12 GiB | ~3.7x |
| **FP8** | **24.59 GiB** | **7.45x** |

### Final Configuration

| Setting | Value |
|---|---|
| MoE backend | CUTLASS W4A4 (native) |
| KV cache dtype | FP8 |
| MTP spec decode | Enabled (`num_speculative_tokens=1`) |
| Chunked prefill | Enabled |
| torch.compile | Enabled (CUDA graph) |
| **Throughput (reasoning)** | **24.5 tok/s** |
| 262K concurrent | 5.58x |

---

## Memory Management on DGX Spark GB10

This is the most commonly overlooked problem when running this model on a DGX Spark. **Without the steps below the container will be OOM-killed during weight loading every time.**

### Why it OOMs

The DGX Spark GB10 has 128 GB of unified memory, but Linux sees only ~119 GiB as usable. The model's steady-state memory footprint is manageable:

| Component | Size |
|---|---|
| Model weights on disk (NVFP4 + FP8 scales) | ~71 GiB |
| CUDA runtime + OS + Docker overhead | ~15 GiB |
| **Steady-state total** | **~86 GiB** |

That fits comfortably. The problem is **during loading**, when the file I/O creates a second allocation on top of the model tensors:

| Weight loader | Loading peak | Result |
|---|---|---|
| `safetensors` (mmap) | 86 GiB model + 47 GiB shard-1 page cache = **133 GiB** | OOM mid-load |
| `fastsafetensors` | 86 GiB model + 47 GiB I/O staging buffer = **133 GiB** | OOM immediately |

The root cause is **page cache double-counting**: on unified memory, the kernel page cache for the weight files and the CUDA tensor allocations are separate physical DRAM allocations. There is no separation between CPU and GPU memory — they share the same pool — so a 71 GiB weight file that is mmap-read while 71 GiB of CUDA tensors are being allocated can push the total to 133–158 GiB.

### Why `safetensors` and not `fastsafetensors`

`fastsafetensors` pre-allocates a full per-shard I/O staging buffer upfront (47 GiB for shard 1) before a single tensor is loaded. This immediately exhausts memory before any weights reach the GPU.

`safetensors` uses `mmap`, which lets the kernel evict stale page cache pages to swap under memory pressure. With sufficient swap space, loading completes successfully, and at steady state the model is entirely in RAM.

### Required: add swap space

You need at least 32 GiB of additional swap beyond what ships with the system. The default Ubuntu install on DGX Spark includes a 16 GiB swap file, leaving ~12 GiB free — not enough to absorb the ~24 GiB loading overage.

#### One-time setup

```bash
# Create and activate a 32 GiB swap file
sudo fallocate -l 32G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2

# Make it permanent across reboots
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab
```

After this, loading consistently completes in ~10–15 minutes. At steady state, all model weights are in DRAM and swap is unused.

#### Memory budget after swap

| | Value |
|---|---|
| Usable RAM | ~119 GiB |
| Default swap (`/swap.img`) | ~15 GiB |
| Added swap (`/swapfile2`) | 32 GiB |
| **Total addressable** | **~166 GiB** |
| Loading peak | ~133 GiB |
| **Headroom** | **~33 GiB** |

### Load format

Always use `--load-format safetensors` on DGX Spark GB10. Do **not** use `fastsafetensors`.

---

## HuggingFace Model Path — Symlink Caveat

When mounting a HuggingFace Hub cache directory into Docker, mount the **repository root** (`models--<org>--<name>/`), not the snapshot subdirectory.

The snapshot directory (`snapshots/<sha>/`) contains only symlinks pointing to `../../blobs/`. If you mount only the snapshot directory, the symlink targets resolve outside the container mount and the model cannot be read.

**Correct:**
```yaml
volumes:
  - ~/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4:/models/qwen3.5-122b-hf:ro
```

Then pass the snapshot path to `vllm serve`:
```
vllm serve /models/qwen3.5-122b-hf/snapshots/<sha>
```

Find the current snapshot SHA with:
```bash
ls ~/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/
```

---

## Model Variants

Three quantized checkpoints of this model are publicly available. All are compatible with this Dockerfile:

| Checkpoint | MTP weights | Notes |
|---|---|---|
| [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) | Separately extracted (see Note below) | Original, used in benchmarks above |
| [RedHatAI/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) | Not included | Red Hat quantization; text-only serving without MTP |
| [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) | Included (as of 2026-03-02) | Untested with this Dockerfile |

For text-only serving without speculative decoding, `RedHatAI/Qwen3.5-122B-A10B-NVFP4` works out of the box with no MTP weight extraction step. Remove `--speculative-config` from the vllm serve command.

---

## Auto-start After Reboot

The `docker-compose.yml` already sets `restart: unless-stopped`. For this to take effect after a reboot, the container must be **started at least once** after the compose file is written, so Docker registers the restart policy:

```bash
docker compose up -d
```

Ensure Docker itself is enabled to start on boot:

```bash
sudo systemctl enable docker
```

With both in place and `/swapfile2` in `/etc/fstab`, the model comes up automatically after every reboot with no manual intervention. Weight loading takes ~10–15 minutes at cold start.

---

## References

- **Model weights** – [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) on Hugging Face
- **Base model** – [Qwen/Qwen3.5-122B-A10B-Instruct](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-Instruct) — Qwen Team, Alibaba Cloud
- **Quantization tool** – [llm-compressor](https://github.com/vllm-project/llm-compressor) (SparseML / Neural Magic)
- **vLLM** – [vllm-project/vllm](https://github.com/vllm-project/vllm)
- **Base Docker image** – `nvcr.io/nvidia/pytorch:26.01-py3` (NVIDIA NGC PyTorch)
- **FlashInfer** – [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **Qwen3Next / GDN architecture** – [`vllm/model_executor/models/qwen3_next.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py) in vLLM
- **compressed-tensors** – [neuralmagic/compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
- **NVIDIA DGX Spark** – [NVIDIA DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
