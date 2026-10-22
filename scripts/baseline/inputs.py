"""Deterministic input builder for tasks cases.

input_shape can be:
  depth-2: [[d0, d1, ...], [d0, d1, ...], ...]  -> N tensors
  depth-3: [[[d, d], [d, d]], [[d, d], [d, d]]]  -> N tensor-LISTS

dtype is a list of strings; when shorter than the input count, the LAST
entry is broadcast (most ops use a single dtype across all inputs).

value_range is one of:
  - [low, high]         -> applies to ALL inputs
  - [[lo,hi], [lo,hi]]  -> per-input
  - [-inf, inf]         -> uniform with random sign + occasional ±inf
  - [nan, nan]          -> all NaN
  - [0, 0]              -> zeros
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


_DTYPE_MAP = {
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "int8": torch.int8, "uint8": torch.uint8,
    "int16": torch.int16, "int32": torch.int32, "int64": torch.int64,
    "bool": torch.bool,
}

InputT = Union[torch.Tensor, List[torch.Tensor]]


def parse_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype: {name}")
    return _DTYPE_MAP[name]


def _is_shape(x: Any) -> bool:
    """A 'shape' is a flat list of ints."""
    return isinstance(x, list) and all(isinstance(v, int) for v in x)


def _is_shape_list(x: Any) -> bool:
    """A list of shapes."""
    return isinstance(x, list) and all(_is_shape(s) for s in x)


def _coerce_range(r: Any) -> Tuple[float, float, str]:
    """Parse a [low, high] entry into (lo, hi, mode).

    mode is one of: 'normal', 'inf', 'nan', 'zero'.
    """
    if not (isinstance(r, list) and len(r) == 2):
        raise ValueError(f"bad value_range: {r}")
    lo, hi = r
    # nan markers
    def _is_nan(v):
        return (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.lower() == "nan")
    def _is_inf(v):
        if isinstance(v, str):
            s = v.lower()
            return s in ("inf", "+inf", "-inf")
        return isinstance(v, float) and math.isinf(v)
    if _is_nan(lo) and _is_nan(hi):
        return float("nan"), float("nan"), "nan"
    if _is_inf(lo) and _is_inf(hi):
        return float("-inf"), float("inf"), "inf"
    if lo == 0 and hi == 0:
        return 0.0, 0.0, "zero"
    return float(lo), float(hi), "normal"


def _resolve_value_ranges(value_range: Any, num_inputs: int):
    """Returns a list of length num_inputs. Each entry is either:
      - a (lo, hi, mode) tuple (single range, for Tensor or shared by tensor-list)
      - a list of (lo, hi, mode) tuples (per-sub-tensor in a tensor-list input)
      - None (for an OPTIONAL input that this case omits)
    """
    if value_range is None:
        return [(-1.0, 1.0, "normal")] * num_inputs

    def _resolve_one(r):
        # Optional input placeholder
        if r is None:
            return None
        # 2-element list whose entries are NOT lists → flat range
        if (isinstance(r, list) and len(r) == 2
                and not any(isinstance(v, list) for v in r)):
            return _coerce_range(r)
        # Nested list of ranges → per-sub-tensor
        if isinstance(r, list):
            return [_coerce_range(rr) if rr is not None else None for rr in r]
        raise ValueError(f"bad value_range entry: {r}")

    # already per-input list?
    if isinstance(value_range, list) and len(value_range) > 0 and (
        isinstance(value_range[0], list) or value_range[0] is None
    ):
        out = [_resolve_one(r) for r in value_range]
        while len(out) < num_inputs:
            out.append(out[-1])
        return out[:num_inputs]
    # single range applied to all
    r = _coerce_range(value_range)
    return [r] * num_inputs


def _resolve_dtypes(dtype: Any, num_inputs: int):
    """Returns a list of length num_inputs. Each entry is either:
      - a single torch.dtype (Tensor input)
      - a list[torch.dtype] (tensor-list input with per-sub-tensor dtype,
        e.g. grouped_matmul whose dtype is nested-list)
      - None (an OPTIONAL input that this case omits, e.g. gru's initial
        hidden state)
    """
    if isinstance(dtype, str):
        dtype = [dtype]
    out = []
    for d in dtype:
        if d is None or (isinstance(d, str) and d.lower() in ("none", "null")):
            out.append(None)
        elif isinstance(d, list):
            out.append([parse_dtype(s) if s is not None and not (isinstance(s, str) and s.lower() in ("none", "null")) else None for s in d])
        else:
            out.append(parse_dtype(d))
    while len(out) < num_inputs:
        out.append(out[-1])
    return out[:num_inputs]


def _make_one(shape: List[int], dt: torch.dtype, lo: float, hi: float, mode: str,
              gen: torch.Generator) -> torch.Tensor:
    if mode == "nan":
        out = torch.full(shape, float("nan"), dtype=torch.float32)
    elif mode == "inf":
        # uniform in [-1e3, 1e3] with ~5% set to ±inf so we exercise both paths
        out = (torch.rand(shape, generator=gen) * 2.0 - 1.0) * 1e3
        mask = torch.rand(shape, generator=gen) < 0.05
        signs = torch.where(torch.rand(shape, generator=gen) < 0.5, 1.0, -1.0)
        out = torch.where(mask, float("inf") * signs, out)
    elif mode == "zero":
        out = torch.zeros(shape, dtype=torch.float32)
    else:
        if dt in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool):
            lo_i = int(math.floor(lo))
            hi_i = int(math.ceil(hi))
            if hi_i <= lo_i:
                hi_i = lo_i + 1
            out = torch.randint(lo_i, hi_i, shape, generator=gen, dtype=torch.int64)
        else:
            out = torch.rand(shape, generator=gen) * (hi - lo) + lo
    # cast to target dtype
    if dt == torch.bool:
        return out.to(torch.bool)
    return out.to(dt)


def _apply_op_aliases(out: List[InputT], op_key: str) -> List[InputT]:
    """Enforce per-op semantic aliasing constraints (documented in desc.md).

    Some attention ops conceptually share KV cache memory between input slots
    (latent KV cache). The bench harness must reflect this so accuracy checks
    compare against inputs that match production semantics, not the unrelated
    random tensors that build_inputs would otherwise produce.

    - level4/mla: v == k_nope                  (v lives in slot 4, k_nope in slot 2)
    - level4/sparse_flash_attention:
        value == key[..., :Dv]                 (value in slot 2, key in slot 1)
        - Dk == Dv → full alias (value == key)
        - Dk >  Dv → prefix slice (typical MLA Dk=576 Dv=512)

    We `.clone()` so the aliased tensor has its own storage (just identical
    values); independent storage prevents accidental in-place pollution and
    keeps memory layout uniform.
    """
    if op_key == "level4/mla" and len(out) >= 5:
        # v = k_nope (slot 2). Both are [B, S_kv, N_kv, d_nope] in BSND or
        # [B, N_kv, S_kv, d_nope] in BNSD.
        if isinstance(out[2], torch.Tensor) and isinstance(out[4], torch.Tensor):
            out = list(out)
            out[4] = out[2].clone()
    elif op_key == "level4/sparse_flash_attention" and len(out) >= 3:
        key = out[1]
        val = out[2]
        if isinstance(key, torch.Tensor) and isinstance(val, torch.Tensor):
            Dv = val.shape[-1]
            # Take the first Dv dims of key (Dk >= Dv guaranteed by API contract).
            out = list(out)
            out[2] = key[..., :Dv].contiguous().to(val.dtype).clone()
    elif op_key == "level3/roi_align" and len(out) >= 2:
        # Cases.yaml gives boxes as (N, 5) — col 0 is batch_idx ∈ [0, B-1] but
        # uniform float sampling in [-1, 1] would produce out-of-bounds indices
        # that crash the kernel. Override col 0 with cyclic arange(N)%B so every
        # batch is covered and indices are valid. This is INPUT DATA shaping
        # (not kernel wrapping) — third-party authors writing a kernel against
        # this contract must accept the same {batch_idx as col 0 of (N,5)} convention.
        features = out[0]
        boxes = out[1]
        if isinstance(features, torch.Tensor) and isinstance(boxes, torch.Tensor) \
                and features.dim() >= 1 and boxes.dim() == 2 and boxes.shape[1] >= 1:
            B = features.shape[0]
            N = boxes.shape[0]
            new_boxes = boxes.clone()
            new_boxes[:, 0] = (torch.arange(N) % B).to(boxes.dtype)
            out = list(out)
            out[1] = new_boxes
    elif op_key == "level3/moe_re_routing" and len(out) >= 2:
        # Kernel requires Sum(expert_token_num_per_rank) == A = tokens.shape[0]
        # and each cell > 0 (per desc.md). Replace the random tensor with a
        # near-uniform partition of A across the N*E cells so both invariants
        # hold deterministically.
        tokens = out[0]
        ept = out[1]
        if isinstance(tokens, torch.Tensor) and isinstance(ept, torch.Tensor):
            A = tokens.shape[0]
            N, E = ept.shape
            total_cells = N * E
            base = A // total_cells
            remainder = A - base * total_cells
            new = torch.full((total_cells,), base, dtype=ept.dtype)
            if remainder > 0:
                new[:remainder] += 1
            out = list(out)
            out[1] = new.reshape(N, E)
    return out


def build_inputs(input_shape: List[Any], dtype: List[str], value_range: Any,
                 case_id: int, op_key: str = "") -> List[InputT]:
    """Build a deterministic list of inputs from the case spec.

    Returns a list. Each entry is either a torch.Tensor (CPU) or a List[torch.Tensor]
    if that input position is a tensor-list.

    `op_key` (e.g. "level4/mla") enables op-specific input aliasing — see
    `_apply_op_aliases` for the constraints (MLA: v==k_nope, SFA: value==key[:Dv]).
    """
    # detect depth-3 (tensor-list per position) vs depth-2 (tensor per position)
    is_list_input = [_is_shape_list(s) for s in input_shape]
    num_inputs = len(input_shape)
    dtypes = _resolve_dtypes(dtype, num_inputs)
    ranges = _resolve_value_ranges(value_range, num_inputs)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0xC0FFEE + case_id * 31337)

    out: List[InputT] = []
    for i, sp in enumerate(input_shape):
        # Preserve None positions for optional inputs that this case omits.
        if sp is None:
            out.append(None)
            continue
        dt = dtypes[i]
        rng = ranges[i]
        if is_list_input[i]:
            # Each sub-tensor may have its own dtype / range, or share a
            # single dtype / range for the whole list.
            if isinstance(dt, list):
                sub_dts = list(dt)
                while len(sub_dts) < len(sp):
                    sub_dts.append(sub_dts[-1])
            else:
                sub_dts = [dt] * len(sp)
            if isinstance(rng, list):
                sub_rngs = list(rng)
                while len(sub_rngs) < len(sp):
                    sub_rngs.append(sub_rngs[-1])
            else:
                sub_rngs = [rng] * len(sp)
            tensors = [_make_one(s, sub_dts[j], *sub_rngs[j], gen)
                       for j, s in enumerate(sp)]
            out.append(tensors)
        else:
            # Defensive: if a flat-tensor input position somehow got a
            # list-of-dtypes / list-of-ranges, take the first.
            if isinstance(dt, list):
                dt = dt[0]
            if isinstance(rng, list):
                rng = rng[0]
            lo, hi, mode = rng
            out.append(_make_one(sp, dt, lo, hi, mode, gen))
    if op_key:
        out = _apply_op_aliases(out, op_key)
    return out


def apply_npu_op_aliases(inputs: List[InputT], op_key: str,
                         attrs: Optional[Dict[str, Any]] = None) -> List[InputT]:
    """Per-op NPU-side transforms applied AFTER `to_device`, OUTSIDE the timed
    window. By default a no-op; ops that need NPU-side input shaping (format
    conversion, in-place fix-up that requires NPU runtime) can opt in here.

    NOTE: this hook is intentionally narrow — anything done here is hidden
    from `baseline_perf_us`, so it must not mask kernels that production
    callers will actually pay. Used only when the harness needs to set up
    state that `to_device` cannot express (which is rare).
    """
    if op_key == "level3/quant_matmul":
        inputs = _quant_matmul_prepack_scale(inputs, attrs or {})
    return inputs


def _quant_matmul_prepack_scale(inputs: List[InputT], attrs: Dict[str, Any]) -> List[InputT]:
    """Pre-pack fp32 scale into int64 via `npu_trans_quant_param` so the
    measured `npu_quant_matmul` doesn't include the `TransQuantParamV2`
    kernel. This mirrors production deployment where scale is packed once
    at model-load time, not per inference.

    Applied only when output is fp16, scale is fp32, and pertoken_scale is
    absent — other configs already fuse to a single `QuantBatchMatmulV3`.
    """
    if len(inputs) < 3 or inputs[2] is None:
        return inputs
    if attrs.get("output_dtype") != "float16":
        return inputs
    scale = inputs[2]
    if not isinstance(scale, torch.Tensor) or scale.dtype != torch.float32:
        return inputs
    pertoken = inputs[4] if len(inputs) > 4 else None
    if pertoken is not None:
        return inputs
    bias = inputs[5] if len(inputs) > 5 else None
    if isinstance(bias, torch.Tensor) and bias.is_floating_point():
        return inputs
    import torch_npu  # local import to keep CPU-only paths import-free
    packed = torch_npu.npu_trans_quant_param(scale)
    return [*inputs[:2], packed, *inputs[3:]]


def to_device(inputs: List[InputT], device: str) -> List[InputT]:
    out = []
    for x in inputs:
        if x is None:
            out.append(None)
        elif isinstance(x, list):
            out.append([t.to(device) if t is not None else None for t in x])
        else:
            out.append(x.to(device))
    return out


def to_fp32(inputs: List[InputT]) -> List[InputT]:
    """Cast all FLOATING tensors to fp32. Integer / bool tensors are kept as-is."""
    floats = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(torch.float32) if t.dtype in floats else t
    out = []
    for x in inputs:
        if isinstance(x, list):
            out.append([_cast(t) for t in x])
        else:
            out.append(_cast(x))
    return out


def cast_outputs_to(outputs, ref_dtypes):
    """Cast a (possibly nested) output structure back to a list of target dtypes,
    matching by position. Integer / bool outputs are left alone.

    ref_dtypes is a list of torch.dtype values. If the output is a single tensor,
    only the first dtype is consulted. If output is tuple/list, applied per-position
    (broadcasting last entry if too few).
    """
    floats = (torch.float16, torch.float32, torch.float64, torch.bfloat16)
    def _cast(t: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            return t
        if t.dtype in floats and dt in floats:
            return t.to(dt)
        return t
    if isinstance(outputs, torch.Tensor):
        return _cast(outputs, ref_dtypes[0])
    if isinstance(outputs, (list, tuple)):
        out = []
        for i, o in enumerate(outputs):
            dt = ref_dtypes[min(i, len(ref_dtypes) - 1)]
            if isinstance(o, (list, tuple)):
                out.append(type(o)(_cast(x, dt) for x in o))
            else:
                out.append(_cast(o, dt))
        return type(outputs)(out)
    return outputs
