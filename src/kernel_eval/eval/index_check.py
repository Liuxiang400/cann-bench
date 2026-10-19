#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""索引类输出（TopK / ArgSort / ArgMax 等）的"指向值"校验。

背景（issue #40）：TopK 的 golden 用 torch.topk，相等元素的索引顺序非确定，
NPU 候选算子对并列元素可能选不同下标，导致索引输出**逐元素**比对必然不过。
此前的做法是把索引输出标 `compare: false` 直接跳过——但这等于放弃验证索引，
留下"返回正确数值却乱填索引也能过"的防作弊缺口。

本模块提供 tie 顺序无关的索引校验：对在 proto.yaml 中声明了 `index_gather` 的
输出，校验"候选返回的索引确实指向候选返回的值"——
    x.gather(dim, idx_candidate) == values_candidate
配合框架已有的"值输出 vs golden"比对（候选值 ≈ golden 值），即可在不依赖并列
顺序的前提下完整验证索引正确性：
- 值不对   → 值输出比对（关系 golden）判失败；
- 索引乱填 → 本校验（gather 自洽）判失败；
- 仅并列顺序不同 → 两者都通过（正确接受）。

proto.yaml 声明示例（TopK 的 idx 输出）::

    outputs:
      - name: y            # 值输出（正常参与比对）
        ...
      - name: idx
        dtype: [int64]
        compare: false                     # 不做逐元素比对
        index_gather:
          input: x                         # 从哪个输入张量按索引取值
          dim_attr: dim                    # 取 dim 的 attr 名（值见 case attrs）
          value_output: y                  # 与哪个值输出做自洽校验
"""

from typing import Any, Dict, List, Optional, Tuple

import torch


def _as_list(outputs: Any) -> List[Any]:
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def validate_index_output(
    x: torch.Tensor,
    dim: int,
    idx_candidate: torch.Tensor,
    values_candidate: torch.Tensor,
    *,
    cumulative: bool = False,
    reference_indices: Optional[torch.Tensor] = None,
) -> Tuple[bool, str, Dict[str, Dict[str, Any]]]:
    """校验"候选索引指向候选值"：x.gather(dim, idx) 须逐元素等于 values。

    与并列元素顺序无关：只要索引指向的元素值正确即通过。
    返回 (ok, msg, checks)。Cummin 还校验前缀范围及 golden 的索引语义。
    """
    checks = {name: {"status": "not-run"} for name in ("structure", "range", "gather")}
    if cumulative:
        checks.update(prefix={"status": "not-run"}, reference={"status": "not-run"})

    def fail(check, message):
        checks[check] = {"status": "failed", "diagnostic": message}
        return False, message, checks

    if not isinstance(x, torch.Tensor) or not isinstance(idx_candidate, torch.Tensor) \
            or not isinstance(values_candidate, torch.Tensor):
        return fail("structure", "index_gather: 输入/索引/值必须均为 Tensor")
    if idx_candidate.dtype not in (torch.int32, torch.int64) or (cumulative and idx_candidate.dtype != torch.int64):
        return fail("structure", "index_gather: 索引 dtype 必须为整数；Cummin 要求 int64")

    x_c = x.detach().cpu()
    idx_c = idx_candidate.detach().cpu().to(torch.int64)
    val_c = values_candidate.detach().cpu()

    if idx_c.dim() != x_c.dim():
        return fail("structure", f"index_gather: 索引维度 {idx_c.dim()} 与输入维度 {x_c.dim()} 不一致，无法 gather")
    dim_n = dim if dim >= 0 else x_c.dim() + dim
    if not (0 <= dim_n < x_c.dim()):
        return fail("structure", f"index_gather: dim={dim} 越界（输入 {x_c.dim()} 维）")

    if idx_c.shape != val_c.shape:
        return fail("structure", f"index_gather: 索引输出形状 {tuple(idx_c.shape)} 与值输出形状 {tuple(val_c.shape)} 不一致")
    if cumulative and idx_c.shape != x_c.shape:
        return fail("structure", "index_gather: Cummin 索引形状必须与输入一致")

    # gather 要求除 dim 外各维 idx.size(d) <= x.size(d)
    for d in range(x_c.dim()):
        if d != dim_n and idx_c.size(d) > x_c.size(d):
            return fail("structure", f"index_gather: 第 {d} 维索引尺寸 {idx_c.size(d)} 超过输入 {x_c.size(d)}")
    checks["structure"] = {"status": "passed"}

    dim_size = x_c.size(dim_n)
    if idx_c.numel() > 0:
        lo = int(idx_c.min().item())
        hi = int(idx_c.max().item())
        if lo < 0 or hi >= dim_size:
            return fail("range", f"index_gather: 索引越界 [{lo},{hi}]，合法区间 [0,{dim_size})")
    checks["range"] = {"status": "passed"}

    if cumulative:
        shape = [1] * idx_c.dim()
        shape[dim_n] = dim_size
        positions = torch.arange(dim_size).reshape(shape)
        if bool((idx_c > positions).any()):
            return fail("prefix", "index_gather: Cummin 索引引用未来位置，必须满足 indices[..., i, ...] <= i")
        checks["prefix"] = {"status": "passed"}

    gathered = torch.gather(x_c, dim_n, idx_c).to(val_c.dtype)
    # 索引取出的元素必须与候选自报的值严格一致（topk 值即被选元素本身，应逐位相等）。
    # 注意：不能直接用 torch.equal —— 它对 NaN 返回不等（NaN != NaN），会把
    # value_range=[nan,nan] 这类全 NaN 用例（如 top_k case 15）即便索引完全正确也误判失败。
    # 故按位置做 NaN-aware 比较：两侧同为 NaN 视为相等，否则按值相等。
    if gathered.is_floating_point() or val_c.is_floating_point():
        both_nan = torch.isnan(gathered) & torch.isnan(val_c)
        eq = (gathered == val_c) | both_nan
    else:
        eq = gathered == val_c
    if not bool(eq.all()):
        mism = int((~eq).sum().item())
        return fail("gather", f"index_gather: 候选索引指向的元素与其值输出不一致（{mism} 处不符），索引无效")
    checks["gather"] = {"status": "passed"}
    if cumulative:
        if not isinstance(reference_indices, torch.Tensor):
            return fail("reference", "index_gather: Cummin 缺少 golden indices，不能验证 tie/NaN 索引语义")
        reference = reference_indices.detach().cpu()
        if reference.shape != idx_c.shape or reference.dtype != torch.int64:
            return fail("reference", "index_gather: Cummin golden indices 的形状或 dtype 不合法")
        mismatches = int((idx_c != reference).sum().item())
        checks["reference"] = {"status": "failed" if mismatches else "passed",
                               "mismatch_count": mismatches, "total_count": idx_c.numel()}
        if mismatches:
            return False, f"index_gather: Cummin 索引与 golden 不一致（{mismatches} 处），包括 tie/NaN 语义", checks
    return True, "", checks


def validate_index_gather_outputs(
    op_info: Any,
    params: Dict[str, Any],
    case_attrs: Dict[str, Any],
    ai_outputs: Any,
    golden_outputs: Any = None,
) -> Tuple[bool, str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """对算子中所有声明了 index_gather 的输出执行索引校验。

    无此类声明时立即返回 (True, "")，对其他算子零影响。

    Args:
        op_info: 算子规格（需有 .outputs，每个 output 可能带 index_gather）
        params:  按参数名索引的调用入参（含输入张量，如 params['x']）
        case_attrs: 本用例的 attrs（含 dim 等）
        ai_outputs: 候选算子的输出（单个或列表）
    """
    outputs = getattr(op_info, "outputs", None) or []
    name_to_idx = {o.name: i for i, o in enumerate(outputs)}
    ai = _as_list(ai_outputs)
    golden = _as_list(golden_outputs) if golden_outputs is not None else []
    cumulative = str(getattr(op_info, "name", "")).lower() == "cummin"
    recorded = {}

    def fail(index, message):
        recorded[index] = {"structure": {"status": "failed", "diagnostic": message},
                           "range": {"status": "not-run"}, "gather": {"status": "not-run"}}
        if cumulative:
            recorded[index].update(prefix={"status": "not-run"}, reference={"status": "not-run"})
        return False, message, recorded

    for i, out in enumerate(outputs):
        spec: Optional[Dict[str, Any]] = getattr(out, "index_gather", None)
        if not spec:
            continue

        src_name = spec.get("input")
        dim_attr = spec.get("dim_attr", "dim")
        val_name = spec.get("value_output")

        x = params.get(src_name) if isinstance(params, dict) else None
        if not isinstance(x, torch.Tensor):
            return fail(i, f"index_gather[{out.name}]: 找不到源输入张量 {src_name!r}")

        if i >= len(ai) or not isinstance(ai[i], torch.Tensor):
            return fail(i, f"index_gather[{out.name}]: 缺少索引输出 #{i}")
        idx = ai[i]

        if val_name is None or val_name not in name_to_idx:
            return fail(i, f"index_gather[{out.name}]: value_output {val_name!r} 未在输出中声明")
        vi = name_to_idx[val_name]
        if vi >= len(ai) or not isinstance(ai[vi], torch.Tensor):
            return fail(i, f"index_gather[{out.name}]: 缺少配对值输出 {val_name!r} (#{vi})")
        vals = ai[vi]

        # dim 是索引校验的必要信息：缺失不静默兜底为 -1（会在错误维度上 gather、
        # 给出错误的通过/失败），直接判失败暴露配置问题。
        if not isinstance(case_attrs, dict) or dim_attr not in case_attrs:
            return fail(i, f"index_gather[{out.name}]: 缺少维度属性 {dim_attr!r}（index_gather 声明的 dim_attr 未在 case attrs 中找到）")
        dim = case_attrs[dim_attr]
        ok, msg, checks = validate_index_output(
            x, int(dim), idx, vals, cumulative=cumulative,
            reference_indices=golden[i] if i < len(golden) else None,
        )
        recorded[i] = checks
        if not ok:
            return False, msg, recorded

    return True, "", recorded


def apply_index_checks(accuracy_result, op_info, params, case_attrs, ai_outputs, golden_outputs):
    """Attach executed checks to the report and apply their verdict to the case."""
    cumulative = str(getattr(op_info, "name", "")).lower() == "cummin"
    for index, spec in enumerate(getattr(op_info, "outputs", None) or []):
        if not getattr(spec, "index_gather", None) or index >= len(accuracy_result.output_results):
            continue
        output = accuracy_result.output_results[index]
        checks = output.metadata.setdefault("checks", {})
        checks.setdefault("elementwise", {"status": "skipped" if output.passed is None else
                                          "passed" if output.passed else "failed"})
        for name in ("structure", "range", "gather") + (("prefix", "reference") if cumulative else ()):
            checks[name] = {"status": "not-run"}
    if not accuracy_result.passed:
        return accuracy_result
    ok, message, recorded = validate_index_gather_outputs(
        op_info, params, case_attrs, ai_outputs, golden_outputs,
    )
    for index, checks in recorded.items():
        if index >= len(accuracy_result.output_results):
            raise RuntimeError("Index validation requires the checker's output result")
        output = accuracy_result.output_results[index]
        evidence = output.metadata.setdefault("checks", {})
        evidence.setdefault("elementwise", {"status": "skipped" if output.passed is None else
                                            "passed" if output.passed else "failed"})
        evidence.update(checks)
        output.passed = all(c["status"] == "passed" for c in checks.values())
        output.error_msg = "" if output.passed else message
    if not ok:
        accuracy_result.passed = False
        accuracy_result.error_msg = message
    return accuracy_result
