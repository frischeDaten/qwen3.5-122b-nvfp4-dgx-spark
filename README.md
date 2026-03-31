# Qwen3.5-122B-A10B-NVFP4 on DGX Spark — GB10 Memory & Setup Notes

This repository documents the additional steps required to run `RedHatAI/Qwen3.5-122B-A10B-NVFP4` with vLLM on an **NVIDIA DGX Spark (GB10, 128 GB unified memory)**.

**The Dockerfile, patches, model class, and benchmarks in this repo are the work of the original authors.** See the original repository for full documentation on those:

> **[bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4](https://github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4)** — original Dockerfile, all vLLM patches, MTP setup, benchmarks
> (later consolidated into [JungkwanBan/spark_vllm_docker](https://github.com/JungkwanBan/spark_vllm_docker))

My contribution is the sections below: solving the OOM problem during weight loading, the HuggingFace symlink caveat, making the service start automatically after reboot, and running MTP speculative decoding with thinking enabled.

---

## 1. Memory Management — Solving OOM During Weight Loading

This is the main problem not documented elsewhere. **Without the steps below, the container will be OOM-killed during weight loading every time, even though the model fits comfortably at steady state.**

### Why it OOMs

The DGX Spark GB10 has 128 GB unified memory; Linux sees ~119 GiB usable. The model's steady-state footprint is fine:

| Component | Size |
|---|---|
| Model weights (NVFP4 + FP8 scales) | ~71 GiB |
| CUDA runtime + OS + Docker overhead | ~15 GiB |
| **Steady-state total** | **~86 GiB** |

The problem is **during loading**. The weight file I/O creates a second allocation on top of the model tensors:

| Loader | Loading peak | Result |
|---|---|---|
| `safetensors` | 86 GiB model + 47 GiB shard-1 page cache = **133 GiB** | OOM mid-load |
| `fastsafetensors` | 86 GiB model + 47 GiB I/O staging buffer = **133 GiB** | OOM immediately |

The root cause is **page cache double-counting on unified memory**: the kernel page cache for the weight files and the CUDA tensor allocations are both physical DRAM. On unified memory there is no CPU/GPU separation — they share the same pool — so reading a 71 GiB shard while allocating 71 GiB of CUDA tensors pushes the total well above 119 GiB.

### Why `safetensors` and not `fastsafetensors`

`fastsafetensors` pre-allocates a full per-shard I/O staging buffer (47 GiB for shard 1) before any tensor is loaded. This exhausts memory before loading even begins.

`safetensors` uses `mmap`. The kernel can evict stale page cache pages to swap under memory pressure. With enough swap, loading completes and at steady state the model is entirely in DRAM.

Always use `--load-format safetensors` on DGX Spark GB10.

### Fix: add swap space

A fresh DGX Spark Ubuntu install includes a 16 GiB swap file (`/swap.img`), leaving ~12 GiB free — not enough to absorb the ~24 GiB loading overage. Add 32 GiB more:

```bash
sudo fallocate -l 32G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2

# Persist across reboots
echo '/swapfile2 none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Memory budget after swap:**

| | Value |
|---|---|
| Usable RAM | ~119 GiB |
| Default swap (`/swap.img`) | ~15 GiB |
| Added swap (`/swapfile2`) | 32 GiB |
| **Total addressable** | **~166 GiB** |
| Loading peak | ~133 GiB |
| **Headroom** | **~33 GiB** |

Loading takes ~10–15 minutes. After startup the model is entirely in DRAM and swap is not used.

---

## 2. HuggingFace Model Path — Symlink Caveat

When mounting a HuggingFace Hub cache directory into Docker, mount the **repository root** (`models--<org>--<name>/`), not the snapshot subdirectory.

The snapshot directory (`snapshots/<sha>/`) contains only symlinks pointing to `../../blobs/`. If you mount only the snapshot directory, the symlink targets resolve outside the container mount and the model cannot be read.

**Correct volume mount:**
```yaml
volumes:
  - ~/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4:/models/qwen3.5-122b-hf:ro
```

Then pass the snapshot path to `vllm serve`:
```
vllm serve /models/qwen3.5-122b-hf/snapshots/<sha>
```

Find the SHA with:
```bash
ls ~/.cache/huggingface/hub/models--RedHatAI--Qwen3.5-122B-A10B-NVFP4/snapshots/
```

---

## 3. Auto-start After Reboot

The `docker-compose.yml` uses `restart: unless-stopped`. For this to take effect after a reboot, the container must be **started at least once** after that line is written — Docker registers the restart policy when the container is first created, not when the compose file is edited.

```bash
docker compose up -d
```

Also ensure Docker itself starts on boot:

```bash
sudo systemctl enable docker
```

With `/swapfile2` in `/etc/fstab`, Docker enabled, and the container started once, the model comes up automatically after every reboot with no manual steps.

---

## 4. MTP Speculative Decoding + Thinking Mode

MTP (Multi-Token Prediction) speculative decoding significantly improves throughput for reasoning/thinking workloads — the original repo benchmarks show **+61% tok/s** in reasoning mode. For short non-reasoning replies it is slightly slower, so it is most useful when thinking is enabled.

### Model variant

Use [`txn545/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) — this checkpoint includes the extracted MTP weights (`mtp_weights.safetensors`, 4.7 GB BF16). The `RedHatAI` variant does not include MTP weights and cannot use speculative decoding.

```bash
huggingface-cli download txn545/Qwen3.5-122B-A10B-NVFP4
```

### vllm serve flags

Add to the base configuration:

```
--enable-chunked-prefill
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

### Important: max-num-batched-tokens

Do **not** use a large `--max-num-batched-tokens` (e.g. 131072) with the MTP model on GB10. The txn545 checkpoint loads 77.26 GiB (vs ~71 GiB for RedHatAI), and the MTP draft model adds compilation overhead on top. During CUDA graph capture the system will run out of memory and hard-crash.

Use `--max-num-batched-tokens 32768` — this is sufficient for all normal workloads and keeps compilation memory within bounds:

```
--max-num-batched-tokens 32768
--gpu-memory-utilization 0.82
```

### Enabling thinking mode

`--chat-template-kwargs` is not supported as a server flag in current vLLM nightly builds. Enable thinking per-request instead by injecting `chat_template_kwargs` into the request body:

```json
{
  "model": "qwen3.5-122b-mtp",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": true}
}
```

If you use a proxy in front of vLLM (e.g. for model name normalization), the cleanest approach is to inject this field automatically for all requests to the MTP endpoint so callers don't need to set it explicitly.

---

## Model Variant Used

Start with [`RedHatAI/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-NVFP4) for a straightforward text-only setup without MTP. Switch to [`txn545/Qwen3.5-122B-A10B-NVFP4`](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4) when you want speculative decoding with thinking enabled. For information on other variants see the original repository linked above.
