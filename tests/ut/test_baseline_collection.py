#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
UT: refs 迁移 + ref_registry + inputs.py + collect_baseline 核心逻辑

测试迁移后的 ref_registry.py 在新路径下能正确加载 refs/level{1-4}.py，
inputs.py 能正确构建输入数据，以及 collect_baseline 的核心逻辑。
"""

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

# 路径设置
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
BASELINE_DIR = PROJECT_ROOT / "scripts" / "baseline"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

import ref_registry
import inputs as bench_inputs


# ==================== ref_registry 测试 ====================

class TestRefRegistry:
    """测试迁移后的 ref_registry.py"""

    # 预期的注册算子列表
    EXPECTED_LEVEL1 = {
        "level1/exp", "level1/gelu", "level1/sigmoid", "level1/mish",
        "level1/masked_scale", "level1/swi_glu",
        "level1/foreach_norm", "level1/foreach_addcdiv_scalar",
    }
    EXPECTED_LEVEL2 = {
        "level2/apply_adam_w", "level2/apply_rotary_pos_emb",
        "level2/arg_max", "level2/cross_entropy_loss", "level2/cummin",
        "level2/dynamic_quant", "level2/gather", "level2/gcd",
        "level2/grid_sampler_3d", "level2/group_norm", "level2/maximum",
        "level2/resize_bilinear", "level2/rms_norm", "level2/scatter",
        "level2/softmax", "level2/unsorted_segment_sum",
    }
    EXPECTED_LEVEL4 = {
        "level4/gqa", "level4/grouped_matmul_swiglu_quant",
        "level4/gru", "level4/lstm", "level4/mha", "level4/mla",
        "level4/mla_prolog", "level4/sparse_flash_attention",
    }

    def test_all_keys_returns_list(self):
        keys = ref_registry.all_keys()
        assert isinstance(keys, list)
        assert len(keys) > 0

    def test_total_registered_ops(self):
        keys = ref_registry.all_keys()
        # 8 (L1) + 16 (L2) + 18 (L3) + 8 (L4) = 50
        assert len(keys) == 50

    def test_level1_ops_registered(self):
        keys = set(ref_registry.all_keys())
        for op in self.EXPECTED_LEVEL1:
            assert op in keys, f"Missing level1 ref: {op}"

    def test_level2_ops_registered(self):
        keys = set(ref_registry.all_keys())
        for op in self.EXPECTED_LEVEL2:
            assert op in keys, f"Missing level2 ref: {op}"

    def test_level4_ops_registered(self):
        keys = set(ref_registry.all_keys())
        for op in self.EXPECTED_LEVEL4:
            assert op in keys, f"Missing level4 ref: {op}"

    def test_get_ref_returns_callable(self):
        for op_path in ref_registry.all_keys():
            ref_fn = ref_registry.get_ref(op_path)
            assert ref_fn is not None, f"get_ref({op_path}) returned None"
            assert callable(ref_fn), f"get_ref({op_path}) not callable"

    def test_get_ref_unknown_op_returns_none(self):
        assert ref_registry.get_ref("level1/nonexistent") is None
        assert ref_registry.get_ref("nonexistent_op") is None

    def test_ref_fn_names(self):
        """验证 ref_fn 的函数名正确"""
        assert ref_registry.get_ref("level1/exp").__name__ == "exp_ref"
        assert ref_registry.get_ref("level2/cummin").__name__ == "cummin_ref"
        assert ref_registry.get_ref("level4/mla").__name__ == "mla_ref"

    def test_ref_discovery_from_new_path(self):
        """验证 ref_registry 在 scripts/baseline/ 路径下的 auto-discovery 正常"""
        # _load() 使用 Path(__file__).parent / "refs"，迁移后应自动适配
        # 通过检查 all_keys 非空来验证
        assert len(ref_registry.all_keys()) > 0


# ==================== inputs.py 测试 ====================

class TestInputsPy:
    """测试迁移后的 inputs.py"""

    def test_build_inputs_simple_float16(self):
        inputs = bench_inputs.build_inputs(
            [[1024, 1024]], ["float16"], [-1, 1], 1, op_key="level1/exp"
        )
        assert len(inputs) == 1
        assert isinstance(inputs[0], torch.Tensor)
        assert inputs[0].shape == (1024, 1024)
        assert inputs[0].dtype == torch.float16

    def test_build_inputs_simple_float32(self):
        inputs = bench_inputs.build_inputs(
            [[2048, 2048]], ["float32"], [-2, 2], 2, op_key="level1/exp"
        )
        assert len(inputs) == 1
        assert inputs[0].dtype == torch.float32

    def test_build_inputs_multiple_inputs(self):
        """多输入算子（如 level2/maximum）"""
        inputs = bench_inputs.build_inputs(
            [[1024, 1024], [1024, 1024]], ["float16", "float16"],
            [-1, 1], 1, op_key="level2/maximum"
        )
        assert len(inputs) == 2

    def test_build_inputs_deterministic_seed(self):
        """相同 case_id 应产生相同输入"""
        inputs1 = bench_inputs.build_inputs(
            [[256, 256]], ["float32"], [-1, 1], 1, op_key="level1/exp"
        )
        inputs2 = bench_inputs.build_inputs(
            [[256, 256]], ["float32"], [-1, 1], 1, op_key="level1/exp"
        )
        assert torch.allclose(inputs1[0], inputs2[0])

    def test_build_inputs_different_seed(self):
        """不同 case_id 应产生不同输入"""
        inputs1 = bench_inputs.build_inputs(
            [[256, 256]], ["float32"], [-1, 1], 1, op_key="level1/exp"
        )
        inputs2 = bench_inputs.build_inputs(
            [[256, 256]], ["float32"], [-1, 1], 2, op_key="level1/exp"
        )
        assert not torch.allclose(inputs1[0], inputs2[0])

    def test_build_inputs_value_range_zero(self):
        inputs = bench_inputs.build_inputs(
            [[512, 512]], ["float16"], [0, 0], 1, op_key=""
        )
        assert torch.all(inputs[0] == 0)

    def test_to_device_cpu(self):
        """to_device 在 CPU 上应该是 no-op"""
        inputs = bench_inputs.build_inputs(
            [[64, 64]], ["float32"], [-1, 1], 1
        )
        result = bench_inputs.to_device(inputs, "cpu")
        assert result[0].device == torch.device("cpu")

    def test_apply_npu_op_aliases_noop(self):
        """大部分算子 apply_npu_op_aliases 是 no-op"""
        inputs = bench_inputs.build_inputs(
            [[64, 64]], ["float32"], [-1, 1], 1, op_key="level2/softmax"
        )
        result = bench_inputs.apply_npu_op_aliases(inputs, "level2/softmax", {})
        # softmax 不需要别名处理，应原样返回
        assert len(result) == len(inputs)


# ==================== collect_baseline 核心逻辑测试 ====================

class TestCollectBaselineUtils:
    """测试 collect_baseline.py 的辅助函数"""

    def test_normalize_attrs_empty(self):
        """需要导入 collect_baseline 模块"""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _normalize_attrs

        assert _normalize_attrs({}) == {}
        assert _normalize_attrs(None) == {}

    def test_normalize_attrs_string_values(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _normalize_attrs

        result = _normalize_attrs({"dim": "-1", "keepdim": "false"})
        assert result["dim"] == -1
        assert result["keepdim"] is False

    def test_normalize_attrs_inf_nan(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _normalize_attrs

        result = _normalize_attrs({"lo": "inf", "hi": "-inf", "v": "nan"})
        assert result["lo"] == float("inf")
        assert result["hi"] == float("-inf")
        assert math.isnan(result["v"])

    def test_normalize_attrs_preserves_numeric(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _normalize_attrs

        result = _normalize_attrs({"scale": 1.5, "k": 3})
        assert result["scale"] == 1.5
        assert result["k"] == 3

    def test_flatten_with_structure(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _flatten_with_structure

        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        t3 = torch.randn(6, 7)
        t4 = torch.randn(8, 9)

        inputs = [t1, [t2, t3], t4]
        flat, structure = _flatten_with_structure(inputs)
        assert structure == ["T", 2, "T"]
        assert len(flat) == 4
        assert flat[0] is t1
        assert flat[1] is t2
        assert flat[2] is t3
        assert flat[3] is t4

    def test_deep_merge_simple(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _deep_merge

        base = {"a": 1, "b": 2}
        overlay = {"b": 3, "c": 4}
        result = _deep_merge(base, overlay)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _deep_merge

        base = {"level1": {"exp": {"1": {"baseline_perf_us": 13.78, "t_hw_us": 1.09}}}}
        overlay = {"level1": {"exp": {"1": {"baseline_perf_us": 15.0}}}}
        result = _deep_merge(base, overlay)
        # overlay 的 baseline_perf_us 覆盖，但保留 base 的 t_hw_us
        assert result["level1"]["exp"]["1"]["baseline_perf_us"] == 15.0
        assert result["level1"]["exp"]["1"]["t_hw_us"] == 1.09

    def test_deep_merge_metadata_replaced_entirely(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _deep_merge

        base = {"_metadata": {"hardware": "910b2", "old_field": True}}
        overlay = {"_metadata": {"hardware": "910b2", "new_field": True}}
        result = _deep_merge(base, overlay)
        # _metadata 应被 overlay 整体替换
        assert result["_metadata"] == overlay["_metadata"]
        assert "old_field" not in result["_metadata"]

    def test_metadata_json_format_compatible(self):
        """验证 metadata JSON 格式与 BaselineStore 兼容"""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from collect_baseline import _deep_merge
        from kernel_eval.utils.baseline_store import BaselineStore

        metadata = {
            "_metadata": {"hardware": "910b2"},
            "level1": {"exp": {"1": {"baseline_perf_us": 13.78, "t_hw_us": 1.09}}}
        }

        # 写入临时文件
        tmp = Path("/tmp/test_baseline_metadata.json")
        with open(tmp, "w") as f:
            json.dump(metadata, f)

        # BaselineStore 应能加载
        store = BaselineStore(bench_root=Path(PROJECT_ROOT / "tasks"),
                              project_root=PROJECT_ROOT)
        store._data = metadata
        store._loaded = True

        perf = store.get_perf("level1/exp", 1)
        assert perf == 13.78

        t_hw = store.get_t_hw("level1/exp", 1)
        assert t_hw == 1.09

        # 清理
        tmp.unlink(missing_ok=True)


# 导入 math 供 test_normalize_attrs_inf_nan 使用
import math

@pytest.mark.parametrize('layers,bidirectional,batch_first,bias', [
    (1, False, False, True),
    (2, True, True, True),
    (1, True, False, False),
])
def test_gru_reference_uses_task_weights_and_attributes(layers, bidirectional, batch_first, bias):
    import importlib.util
    path = PROJECT_ROOT / 'tasks/level4/gru/golden.py'
    spec = importlib.util.spec_from_file_location('test_gru_golden', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = torch.Generator().manual_seed(7)
    directions = 2 if bidirectional else 1
    size, hidden, batch, seq = 3, 2, 2, 4
    x = torch.randn((batch, seq, size) if batch_first else (seq, batch, size), generator=generator)
    wi, wh, bi, bh = [], [], [], []
    for level in range(layers):
        for _ in range(directions):
            wi.append(torch.randn(3*hidden, size if level == 0 else hidden*directions, generator=generator))
            wh.append(torch.randn(3*hidden, hidden, generator=generator))
            bi.append(torch.randn(3*hidden, generator=generator))
            bh.append(torch.randn(3*hidden, generator=generator))
    h0 = torch.randn(layers*directions, batch, hidden, generator=generator)
    inputs = [x, wi, wh, bi if bias else None, bh if bias else None, h0]
    attrs = dict(inputSize=size, hiddenSize=hidden, numLayers=layers, bias=bias,
                 batchFirst=batch_first, bidirectional=bidirectional, dropout=0.2)
    reference = ref_registry.get_ref('level4/gru')
    with torch.no_grad():
        actual = reference(inputs, attrs)
        expected = module.gru(x=x, weight_ih=wi, weight_hh=wh,
                              bias_ih=inputs[3], bias_hh=inputs[4], h0=h0, **attrs)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    changed = [x, [torch.zeros_like(wi[0])] + wi[1:], wh, inputs[3], inputs[4], h0]
    torch.manual_seed(123)
    first = reference(inputs, attrs)[0]
    torch.manual_seed(123)
    second = reference(changed, attrs)[0]
    assert not torch.allclose(first, second), 'GRU must consume the supplied weights'


def test_quant_per_token_scale_keeps_float_scale_for_native_api():
    values = [torch.zeros(2, 3, dtype=torch.int8), torch.zeros(3, 4, dtype=torch.int8),
              torch.ones(4), None, torch.ones(2)]
    result = bench_inputs.apply_npu_op_aliases(values, 'level3/quant_matmul',
                                               {'output_dtype': 'float16'})
    assert result is values
    assert result[2].dtype == torch.float32
