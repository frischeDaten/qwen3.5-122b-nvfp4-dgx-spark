# =========================================================
# vLLM for Sehyo/Qwen3.5-122B-A10B-NVFP4 (DGX Spark / SM121)
#
# Base: vllm-mxfp4-spark:latest
#   - SM121/GB10 최적화 flashinfer-cutlass + NVFP4 지원
#   - 최신 nightly 베이스 이미지 빌드:
#     cd ../spark-vllm-docker && ./build-and-copy.sh --exp-mxfp4 --rebuild-vllm -t vllm-mxfp4-spark:latest
# =========================================================
FROM vllm-mxfp4-spark:latest

# Qwen3.5 VL MoE 모델 지원을 위한 transformers 버전 고정
RUN uv pip install transformers==5.2.0 --upgrade --no-deps --system
RUN uv pip install huggingface-hub --upgrade --no-deps --system

# Qwen3.5 VL MoE 네이티브 vLLM 모델 클래스 추가
COPY qwen3_5_vl_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_vl_moe.py

# flashinfer 0.6.1 호환 패치: tile_tokens_dim 파라미터 미지원 → 제거
RUN sed -i '/tile_tokens_dim=None,/d' \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py

# Qwen3_5MoeForConditionalGeneration을 vLLM 모델 레지스트리에 등록
RUN python3 - <<'EOF'
import re, sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py"
with open(path) as f:
    src = f.read()

entry = '''    "Qwen3_5MoeForConditionalGeneration": (
        "qwen3_5_vl_moe",
        "Qwen3_5MoeForConditionalGeneration",
    ),\n'''

anchor = '"Qwen3VLMoeForConditionalGeneration": ('
idx = src.find(anchor)
if idx == -1:
    print("ERROR: anchor not found in registry.py", file=sys.stderr)
    sys.exit(1)

end = src.find("),", idx) + len("),")
if entry in src:
    print("Qwen3_5MoeForConditionalGeneration already registered, skipping.")
else:
    src = src[:end] + "\n" + entry + src[end:]
    with open(path, "w") as f:
        f.write(src)
    print("Registered Qwen3_5MoeForConditionalGeneration in registry.py")
EOF

# Fix: SpeculativeConfig auto-detection falls through to "draft_model" for qwen3_5_moe.
# Two-part patch:
#   (1) Extend the pass-through list to include "mtp" so the NotImplementedError is avoided.
#   (2) For qwen3_5_moe targets, redirect the draft model's hf_config to text_config with
#       model_type="qwen3_next_mtp" so get_model() resolves to Qwen3NextMTP (the native
#       lightweight MTP drafter that only loads the MTP head, not the full VL model).
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py"
with open(path) as f:
    src = f.read()

MARKER = "# [qwen3_5_moe MTP drafter patch]"
if MARKER in src:
    print("speculative.py: already patched, skipping.")
    sys.exit(0)

# The two lines we're replacing (16-space indent for if, 20-space for pass)
old = ('                if self.method in ("eagle", "eagle3"):\n'
       '                    pass\n')

# Replacement keeps the same 16-space indent level as the surrounding code.
# Do NOT use textwrap.dedent() here — the leading spaces are intentional.
new = (
    '                {marker}\n'
    '                if self.method in ("eagle", "eagle3", "mtp"):\n'
    '                    if (\n'
    '                        self.method == "mtp"\n'
    '                        and hasattr(self, "draft_model_config")\n'
    '                        and self.draft_model_config is not None\n'
    '                        and getattr(\n'
    '                            self.draft_model_config.hf_config, "model_type", ""\n'
    '                        ) == "qwen3_5_moe"\n'
    '                        and hasattr(self.draft_model_config.hf_config, "text_config")\n'
    '                    ):\n'
    '                        import copy as _copy\n'
    '                        _tc = _copy.copy(\n'
    '                            self.draft_model_config.hf_config.text_config\n'
    '                        )\n'
    '                        _tc.model_type = "qwen3_next_mtp"\n'
    '                        _tc.architectures = ["Qwen3NextMTP"]\n'
    '                        if not hasattr(_tc, "num_nextn_predict_layers"):\n'
    '                            _tc.num_nextn_predict_layers = getattr(\n'
    '                                _tc, "mtp_num_hidden_layers", 1\n'
    '                            )\n'
    '                        if not hasattr(_tc, "decoder_sparse_step"):\n'
    '                            _tc.decoder_sparse_step = 1\n'
    '                        if not hasattr(_tc, "intermediate_size"):\n'
    '                            _tc.intermediate_size = _tc.moe_intermediate_size\n'
    '                        if not hasattr(_tc, "norm_topk_prob"):\n'
    '                            _tc.norm_topk_prob = True\n'
    '                        if not hasattr(_tc, "mlp_only_layers"):\n'
    '                            _tc.mlp_only_layers = []\n'
    '                        if not hasattr(_tc, "layer_scale"):\n'
    '                            _tc.layer_scale = False\n'
    '                        if not hasattr(_tc, "dtype"):\n'
    '                            _tc.dtype = None\n'
    '                        # Strip VL MRoPE fields — MTP drafter is text-only,\n'
    '                        # needs standard RoPE not MRotaryEmbedding.\n'
    '                        for _rope_attr in ("rope_parameters", "rope_scaling"):\n'
    '                            _rd = getattr(_tc, _rope_attr, None)\n'
    '                            if isinstance(_rd, dict):\n'
    '                                _rd = dict(_rd)\n'
    '                                _rd.pop("mrope_section", None)\n'
    '                                _rd.pop("mrope_interleaved", None)\n'
    '                                setattr(_tc, _rope_attr, _rd)\n'
    '                        self.draft_model_config.hf_config = _tc\n'
    '                        print(\n'
    '                            "[speculative patch] Redirected Qwen3_5Moe draft model "\n'
    '                            "hf_config to qwen3_next_mtp (loads Qwen3NextMTP drafter)."\n'
    '                        )\n'
).format(marker=MARKER)

if old not in src:
    print("ERROR: anchor not found in speculative.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched speculative.py: mtp pass-through + qwen3_5_moe draft redirect.")
PYEOF

# Fix: Qwen3NextMTP.remap_weight_names() — strip "language_model." prefix from VL model weights
# so that embed_tokens / lm_head are found correctly during weight-sharing.
# Native Qwen3Next: "model.embed_tokens.weight"
# VL checkpoint:    "model.language_model.embed_tokens.weight"  ← needs stripping
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [vl_mtp_remap patch]"
if MARKER in src:
    print("qwen3_next_mtp.py: remap patch already applied, skipping.")
    sys.exit(0)

old = (
    "        def remap_weight_names(weights):\n"
    "            for name, weight in weights:\n"
    "                if name.startswith(\"mtp.\"):\n"
    "                    name = name.replace(\"mtp.\", \"model.\")\n"
    "                elif not any(key in name for key in shared_weight_names):\n"
    "                    continue\n"
    "                yield name, weight\n"
)
new = (
    "        def remap_weight_names(weights):\n"
    "            " + MARKER + "\n"
    "            for name, weight in weights:\n"
    "                if name.startswith(\"mtp.\"):\n"
    "                    name = name.replace(\"mtp.\", \"model.\")\n"
    "                elif not any(key in name for key in shared_weight_names):\n"
    "                    continue\n"
    "                # VL model: model.language_model.<X> -> model.<X>\n"
    "                name = name.replace(\"model.language_model.\", \"model.\")\n"
    "                yield name, weight\n"
)

if old not in src:
    print("ERROR: anchor not found in qwen3_next_mtp.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: VL language_model prefix stripped in remap_weight_names.")
PYEOF

# Fix: EagleProposer.load_model multimodal handling — add Qwen3_5MoeForConditionalGeneration
# to the Qwen3VL image_token_id branch so it uses config.image_token_id (not image_token_index).
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
with open(path) as f:
    src = f.read()

old = '"Qwen3VLForConditionalGeneration",'
new = '"Qwen3VLForConditionalGeneration",\n                "Qwen3_5MoeForConditionalGeneration",'
if new in src:
    print("eagle.py: Qwen3_5MoeForConditionalGeneration already in VL list, skipping.")
elif old not in src:
    print("ERROR: Qwen3VLForConditionalGeneration anchor not found in eagle.py", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("Patched eagle.py: Qwen3_5MoeForConditionalGeneration uses image_token_id.")
PYEOF

# Fix: mamba/abstract.py allows Mamba+spec-decoding only for qwen3_next.
# Qwen3.5 MoE uses the same GDN layers → add qwen3_5_moe to the allow-list.
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/abstract.py"
with open(path) as f:
    src = f.read()

old = '    and vllm_config.model_config.hf_config.model_type not in ["qwen3_next"]'
new = '    and vllm_config.model_config.hf_config.model_type not in ["qwen3_next", "qwen3_5_moe"]'
if new in src:
    print("mamba/abstract.py: qwen3_5_moe already in allow-list, skipping.")
elif old not in src:
    print("ERROR: anchor not found in mamba/abstract.py", file=sys.stderr)
    sys.exit(1)
else:
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print("Patched mamba/abstract.py: added qwen3_5_moe to Mamba+spec-decoding allow-list.")
PYEOF

# Fix: eagle.py _get_positions() returns mrope_positions[:, :N] where N varies
# with the CUDAGraph capture size (1, 2, 4, 8, 16). At compile time N=32768
# (max_num_tokens), so the compiled eagle_head graph bakes in assert_size_stride
# for positions (s80, 32768). CUDAGraph capture then fails when N=16:
#   AssertionError: expected size 16==32768 at dim=1
#
# Fix: always return mrope_positions[:, :max_num_tokens] (fixed-size non-contiguous
# slice, shape always (3, 32768)). The compiled assert_size_stride always passes.
# mrope.py forward_native narrows to query.shape[0] (dynamic = actual N) via:
#   positions = positions[:, :num_tokens]  (num_tokens = query.shape[0])
# so cos/sin are computed only for the actual N tokens, making shapes dynamic.
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_full_buffer_fix]"
if MARKER in src:
    print("eagle.py: _get_positions max-size fix already applied, skipping.")
    sys.exit(0)

old = (
    "    def _get_positions(self, num_tokens: int):\n"
    "        if self.uses_mrope:\n"
    "            return self.mrope_positions[:, :num_tokens]\n"
    "        return self.positions[:num_tokens]\n"
)
new = (
    "    def _get_positions(self, num_tokens: int):\n"
    "        " + MARKER + "\n"
    "        # For MRoPE, always return the max-size slice mrope_positions[:, :max_num_tokens]\n"
    "        # (non-contiguous, shape (3, max_num_tokens)) regardless of num_tokens.\n"
    "        # The compiled eagle_head's assert_size_stride(positions, (s80, max_num_tokens), ...)\n"
    "        # always passes. mrope.py narrows positions[:, :query.shape[0]] (dynamic)\n"
    "        # so cos/sin are computed only for the actual N tokens in the batch.\n"
    "        if self.uses_mrope:\n"
    "            return self.mrope_positions[:, :self.max_num_tokens]\n"
    "        return self.positions[:num_tokens]\n"
)

if old not in src:
    print("ERROR: _get_positions anchor not found in eagle.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched eagle.py: _get_positions() returns mrope_positions[:, :max_num_tokens] for MRoPE.")
PYEOF

# Fix: @support_torch_compile on Qwen3NextMultiTokenPredictor infers dynamic_arg_dims
# by marking dim=0 on every Tensor argument. For 1D positions (N,) this is correct.
# For 2D MRoPE positions (3, N), dim=0 = 3 is STATIC (always 3), dim=-1 = N is the
# token count. The default marking misses dim=-1.
#
# Fix: explicitly specify dynamic_arg_dims with positions=-1.
# Note: with the full-buffer fix above, positions.shape[-1] = max_tokens+1 (constant).
# Marking it dynamic allows the compiled guard to accept any shape[1], giving
# flexibility. shape_invariants ties hidden_states to input_ids token count but
# does NOT require positions.shape[-1] == n (positions is a larger fixed buffer).
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_dynamic_positions_fix]"
if MARKER in src:
    print("qwen3_next_mtp.py: dynamic positions fix already applied, skipping.")
    sys.exit(0)

old = "@support_torch_compile\nclass Qwen3NextMultiTokenPredictor(nn.Module):"
new = (
    MARKER + "\n"
    "# shape_invariants: links hidden_states token count to input_ids.\n"
    "# positions.shape[-1] is NOT linked to n because the full mrope buffer\n"
    "# (shape (3, max_tokens+1)) is passed — larger than the actual token count.\n"
    "def _qwen3_next_mtp_shape_invariants(\n"
    "    input_ids, positions, hidden_states,\n"
    "    intermediate_tensors=None, inputs_embeds=None, **kwargs\n"
    "):\n"
    "    n = input_ids.size()[0] if input_ids is not None else inputs_embeds.size()[0]\n"
    "    torch._check(n == hidden_states.size()[0])\n"
    "\n"
    "@support_torch_compile(\n"
    "    dynamic_arg_dims={\n"
    '        "input_ids": 0,\n'
    '        "positions": -1,\n'
    '        "hidden_states": 0,\n'
    '        "intermediate_tensors": 0,\n'
    '        "inputs_embeds": 0,\n'
    "    },\n"
    "    shape_invariants=_qwen3_next_mtp_shape_invariants,\n"
    ")\n"
    "class Qwen3NextMultiTokenPredictor(nn.Module):"
)

if old not in src:
    print("ERROR: '@support_torch_compile\\nclass Qwen3NextMultiTokenPredictor' not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: positions marked dynamic on dim=-1, shape_invariants updated (no n==p check).")
PYEOF

# [Experimental] Allow forcing Marlin MoE backend via VLLM_NVFP4_MOE_FORCE_MARLIN=1
# NOTE: On SM121 with native CUTLASS FP4, Marlin (W4A16 dequant) may be slower.
#       Enable only for benchmarking comparison with CUTLASS.
RUN python3 - <<'EOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/nvfp4_moe_support.py"
with open(path) as f:
    src = f.read()

force_marlin_check = '''
    # [Experimental patch] Force Marlin if VLLM_NVFP4_MOE_FORCE_MARLIN=1
    import os as _os
    if _os.environ.get("VLLM_NVFP4_MOE_FORCE_MARLIN", "0") == "1" and is_fp4_marlin_supported():
        _logger.warning(
            "VLLM_NVFP4_MOE_FORCE_MARLIN=1: overriding CUTLASS with Marlin FP4 MoE kernel (experimental)."
        )
        return NvFp4Support(cutlass_supported=cutlass_supported, allow_flashinfer=False, use_marlin=True)
'''

anchor = "    cutlass_supported = cutlass_fp4_supported()"
if "VLLM_NVFP4_MOE_FORCE_MARLIN" not in src:
    if anchor not in src:
        print("ERROR: anchor not found in nvfp4_moe_support.py", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor, force_marlin_check + "\n" + anchor)
    with open(path, "w") as f:
        f.write(src)
    print("Patched nvfp4_moe_support.py: VLLM_NVFP4_MOE_FORCE_MARLIN override added.")
else:
    print("Already patched, skipping.")
EOF

# Fix: MRotaryEmbedding.forward_native() causes ConstraintViolationError in eagle_head
# torch.compile. Two related issues:
#
# (1) num_tokens = positions.shape[-1]
#     EagleProposer passes mrope_positions[:, :max_num_tokens] — a concrete (3, 32768) slice.
#     Using positions.shape[-1] as num_tokens specializes query.shape[0] to 32768.
#
# (2) cos_sin = self.cos_sin_cache[positions]  (positions still concrete (3, 32768))
#     → cos_sin.shape = (3, 32768, rotary_dim) — concrete
#     → after split/cat: cos.shape = (32768, rotary_dim//2) — concrete
#     → apply_rotary_emb broadcasting with dynamic query_rot requires N_q == 32768
#     → specializes input_ids.shape[0] == 32768 → ConstraintViolationError
#
# Fix: narrow 2D positions to query.shape[0] BEFORE the cache lookup.
# This makes cos/sin shapes depend on the dynamic query dim, breaking the concrete chain.
# The narrow is a runtime no-op (positions.shape[-1] == query.shape[0] at runtime).
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/rotary_embedding/mrope.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mrope_positions_narrow_fix]"
if MARKER in src:
    print("mrope.py: already patched, skipping.")
    sys.exit(0)

old = (
    "        num_tokens = positions.shape[-1]\n"
    "        cos_sin = self.cos_sin_cache[positions]\n"
)
new = (
    "        " + MARKER + "\n"
    "        # Use query.shape[0] for num_tokens, and narrow 2D positions BEFORE the\n"
    "        # cache lookup to avoid concrete shape propagation. EagleProposer passes\n"
    "        # mrope_positions[:, :max_num_tokens] (shape (3, 32768) concrete). Without\n"
    "        # narrowing, cos/sin inherit the concrete 32768 shape and broadcasting with\n"
    "        # dynamic query_rot creates constraint N_q==32768 → ConstraintViolationError.\n"
    "        # The narrow is a runtime no-op (positions.shape[-1] == query.shape[0]).\n"
    "        num_tokens = query.shape[0]\n"
    "        if positions.ndim == 2:\n"
    "            positions = positions[:, :num_tokens]\n"
    "        cos_sin = self.cos_sin_cache[positions]\n"
)

if old not in src:
    print("ERROR: anchor not found in mrope.py forward_native", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched mrope.py: positions narrowed to query.shape[0] before cache lookup.")
PYEOF

# Fix: SpecDecodingProm.observe() can receive negative num_accepted_tokens when
# len(generated_token_ids) == 0 (request abortion, EOS before spec token, etc.).
# scheduler.py computes: num_accepted = len(generated_token_ids) - 1
# If 0 tokens generated → num_accepted = -1 → prometheus Counter ValueError.
#
# Fix: guard all counter increments with max(0, value) in SpecDecodingProm.observe().
# The logging path (SpecDecodingLogging) still records raw values for diagnostics.
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/metrics.py"
with open(path) as f:
    src = f.read()

MARKER = "# [negative_counter_guard]"
if MARKER in src:
    print("spec_decode/metrics.py: negative counter guard already applied, skipping.")
    sys.exit(0)

old = (
    "    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):\n"
    "        if not self.spec_decoding_enabled:\n"
    "            return\n"
    "        self.counter_spec_decode_num_drafts[engine_idx].inc(\n"
    "            spec_decoding_stats.num_drafts\n"
    "        )\n"
    "        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(\n"
    "            spec_decoding_stats.num_draft_tokens\n"
    "        )\n"
    "        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(\n"
    "            spec_decoding_stats.num_accepted_tokens\n"
    "        )\n"
    "        for pos, counter in enumerate(\n"
    "            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]\n"
    "        ):\n"
    "            counter.inc(spec_decoding_stats.num_accepted_tokens_per_pos[pos])\n"
)
new = (
    "    def observe(self, spec_decoding_stats: SpecDecodingStats, engine_idx: int = 0):\n"
    "        " + MARKER + "\n"
    "        # Guard all counter increments with max(0, value) to prevent\n"
    "        # ValueError when num_accepted_tokens is negative (e.g. when\n"
    "        # len(generated_token_ids)==0 due to request abort or early EOS).\n"
    "        if not self.spec_decoding_enabled:\n"
    "            return\n"
    "        self.counter_spec_decode_num_drafts[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_drafts)\n"
    "        )\n"
    "        self.counter_spec_decode_num_draft_tokens[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_draft_tokens)\n"
    "        )\n"
    "        self.counter_spec_decode_num_accepted_tokens[engine_idx].inc(\n"
    "            max(0, spec_decoding_stats.num_accepted_tokens)\n"
    "        )\n"
    "        for pos, counter in enumerate(\n"
    "            self.counter_spec_decode_num_accepted_tokens_per_pos[engine_idx]\n"
    "        ):\n"
    "            counter.inc(max(0, spec_decoding_stats.num_accepted_tokens_per_pos[pos]))\n"
)

if old not in src:
    print("ERROR: observe() anchor not found in spec_decode/metrics.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched spec_decode/metrics.py: negative counter guard added to SpecDecodingProm.observe().")
PYEOF

# Fix: GDN layers use Triton FLA kernels (fused_recurrent_gated_delta_rule,
# chunk_gated_delta_rule, fused_gdn_gating) that require a runtime Triton
# memory allocator for scratch buffers.  vLLM only sets this in matmul_ogs.py
# for MoE matmul; GDN linear-attention layers need it too.  Without it, eager
# mode crashes with:
#   RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
#
# Fix: register a simple torch-backed CUDA allocator at module level and call
# triton.set_allocator() at the beginning of GDNAttention._forward_core().
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_triton_allocator_fix]"
if MARKER in src:
    print("qwen3_next.py: GDN Triton allocator fix already applied, skipping.")
    sys.exit(0)

# 1. Add allocator class after logger definition
old_logger = 'logger = init_logger(__name__)\n\nKVCache = tuple[torch.Tensor, torch.Tensor]'
new_logger = (
    'logger = init_logger(__name__)\n\n'
    + MARKER + '\n'
    '# Triton FLA kernels need a runtime memory allocator for global scratch space.\n'
    '# vLLM sets this in matmul_ogs.py for MoE, but GDN layers also need it.\n'
    'class _GDNTorchCudaAllocator:\n'
    '    """Torch-backed CUDA allocator for Triton FLA kernel scratch buffers."""\n'
    '    def __call__(self, size: int, alignment: int, stream=None) -> torch.Tensor:\n'
    '        return torch.empty(size, dtype=torch.uint8, device="cuda")\n'
    '\n'
    '_gdn_triton_allocator = _GDNTorchCudaAllocator()\n'
    '\n'
    'KVCache = tuple[torch.Tensor, torch.Tensor]'
)

if old_logger not in src:
    print("ERROR: logger anchor not found in qwen3_next.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old_logger, new_logger, 1)

# 2. Call triton.set_allocator() at start of _forward_core
old_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        forward_context = get_forward_context()'
)
new_fcore = (
    '        """\n'
    '        Core attention computation (called by custom op).\n'
    '        """\n'
    '        # Set Triton allocator for FLA kernels that need global scratch memory.\n'
    '        triton.set_allocator(_gdn_triton_allocator)\n'
    '        forward_context = get_forward_context()'
)

if old_fcore not in src:
    print("ERROR: _forward_core docstring anchor not found in qwen3_next.py", file=sys.stderr)
    idx = src.find("def _forward_core")
    if idx >= 0:
        print("Context:", repr(src[idx:idx+300]))
    sys.exit(1)

src = src.replace(old_fcore, new_fcore, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied GDN Triton allocator fix to qwen3_next.py.")
PYEOF

# Fix: NVFP4 CUTLASS MoE kernel occasionally produces NaN values during prefill
# when processing GDN-derived activations (specifically at layer 8 in the
# Qwen3.5-122B-A10B-NVFP4 model).  Without a guard these NaN values propagate
# through all subsequent layers, causing argmax(logits) == 0 ("!") for every
# output token.
#
# Fix: add a torch.nan_to_num guard on the MoE FFN output inside
# Qwen3NextDecoderLayer.forward().  The guard is cheap at inference time
# (isnan() on BF16 tensor is fast) and prevents silent corruption.
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    src = f.read()

MARKER = "# [gdn_nan_guard]"
if MARKER in src:
    print("qwen3_next.py: NaN guard already applied, skipping.")
    sys.exit(0)

old_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        if self.layer_scale:'
)
new_mlp = (
    '        hidden_states = self.mlp(hidden_states)\n'
    '\n'
    '        ' + MARKER + '\n'
    '        # NVFP4 CUTLASS MoE kernel can produce NaN during prefill with\n'
    '        # GDN-derived activations.  Replace unconditionally (no data-dependent\n'
    '        # branch) so this is fully compatible with torch.compile / dynamo.\n'
    '        hidden_states = hidden_states.nan_to_num(nan=0.0)\n'
    '\n'
    '        if self.layer_scale:'
)

if old_mlp not in src:
    print("ERROR: MLP output anchor not found in Qwen3NextDecoderLayer.forward()", file=sys.stderr)
    idx = src.find("hidden_states = self.mlp(hidden_states)")
    if idx >= 0:
        print("Context:", repr(src[idx:idx+200]))
    sys.exit(1)

src = src.replace(old_mlp, new_mlp, 1)

with open(path, "w") as f:
    f.write(src)
print("Applied NaN guard to Qwen3NextDecoderLayer.forward() in qwen3_next.py.")
PYEOF

# Fix: Qwen3NextMultiTokenPredictor receives the shared NVFP4 quant_config, which
# would apply ModelOptNvFp4LinearMethod to mtp.fc and all MTP sub-layers.
# BUT the MTP checkpoint weights are plain BF16 (no pre-quantized NVFP4 tensors in
# the checkpoint), so the NVFP4 linear forward produces zeros/NaN.
#
# Fix: add "mtp." to quant_config.exclude_modules BEFORE any sub-layer is created,
# so get_quant_method() returns UnquantizedLinearMethod / None (BF16) for all MTP layers.
# This is a no-op when MTP is not used (PPMissingLayer absorbs mtp.* weights).
RUN python3 - <<'PYEOF'
import sys

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py"
with open(path) as f:
    src = f.read()

MARKER = "# [mtp_quant_exclusion_fix]"
if MARKER in src:
    print("qwen3_next_mtp.py: MTP quant exclusion fix already applied, skipping.")
    sys.exit(0)

old = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        config: Qwen3NextConfig = model_config.hf_config\n"
)
new = (
    "        model_config = vllm_config.model_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "\n"
    "        " + MARKER + "\n"
    "        # MTP checkpoint weights are plain BF16 (no pre-quantized NVFP4 tensors).\n"
    "        # Exclude all mtp.* layers from NVFP4 quantization so they run in BF16.\n"
    "        # This must happen BEFORE any sub-layer is constructed so that\n"
    "        # get_quant_method() sees the updated exclude_modules list.\n"
    "        if quant_config is not None and hasattr(quant_config, 'exclude_modules'):\n"
    "            if 'mtp.' not in quant_config.exclude_modules:\n"
    "                quant_config.exclude_modules.append('mtp.')\n"
    "                logger.info(\n"
    "                    'MTP: added mtp. to quant_config.exclude_modules '\n"
    "                    '→ all MTP sub-layers will use unquantized BF16.')\n"
    "\n"
    "        config: Qwen3NextConfig = model_config.hf_config\n"
)

if old not in src:
    print("ERROR: anchor not found in qwen3_next_mtp.py", file=sys.stderr)
    sys.exit(1)

src = src.replace(old, new, 1)
with open(path, "w") as f:
    f.write(src)
print("Patched qwen3_next_mtp.py: MTP layers excluded from NVFP4 quantization (BF16 path).")
PYEOF
