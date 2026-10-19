#!/usr/bin/python3
# coding=utf-8

from kernel_eval.base.result import AccuracyResult, PerfResult
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult
from kernel_eval.staged_eval import _case_num_from_value, _merge_results


def _case(case_num, *, success, failure_type=None, accuracy_result=None, perf_result=None):
    return EvalCaseResult(
        case_id=f"level2/dynamic_quant_{case_num}",
        rel_path="level2/dynamic_quant",
        operator="DynamicQuant",
        case_num=case_num,
        success=success,
        accuracy_result=accuracy_result,
        perf_result=perf_result,
        baseline_perf_us=100.0,
        t_hw_us=10.0,
        failure_type=failure_type,
    )


def _op(cases):
    passed = sum(1 for case in cases if case.success)
    return EvalOperatorResult(
        rel_path="level2/dynamic_quant",
        operator="DynamicQuant",
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(cases) - passed,
        skipped_cases=0,
        results=cases,
        pass_rate=passed / len(cases) if cases else 0.0,
        avg_speedup=0.0,
    )


def test_case_num_parses_full_case_id_suffix():
    assert _case_num_from_value("level2/dynamic_quant_17") == 17
    assert _case_num_from_value(18) == 18
    assert _case_num_from_value(None) == 0


def test_merge_results_matches_string_case_id_and_recounts_failures():
    correctness_ops = [
        _op([
            _case("level2/dynamic_quant_17", success=True),
            _case(
                "level2/dynamic_quant_9",
                success=False,
                failure_type="precision_mismatch",
                accuracy_result=AccuracyResult(passed=False),
            ),
            _case(
                "level2/dynamic_quant_6",
                success=False,
                failure_type="compile_runtime_error",
            ),
        ])
    ]
    # Reproduce the old aggregate behavior where runtime failures with no
    # accuracy_result were not counted in failed_cases.
    correctness_ops[0].failed_cases = 1

    performance_ops = [
        _op([
            _case(17, success=True, perf_result=PerfResult(elapsed_us=20.0)),
        ])
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]

    assert merged.passed_cases == 1
    assert merged.failed_cases == 2
    assert merged.skipped_cases == 0
    assert merged.pass_rate == 1 / 3
    assert merged.results[0].perf_result is not None
    assert merged.results[0].perf_result.elapsed_us == 20.0


def test_merge_flags_perf_stage_precision_flip_as_precision_failure():
    """correctness 过、performance 精度复检翻车 → 视为该 case 精度错误 + 打标签。

    输入与 golden 两阶段一致，唯一变量是 NPU kernel 输出，故这类翻车等价于
    非确定性算子，应扣精度分（precision_mismatch）、无性能分，并在 results.json
    的 perf_recheck 字段中留痕。
    """
    correctness_ops = [
        _op([
            _case(1, success=True, accuracy_result=AccuracyResult(passed=True)),
            _case(2, success=True, accuracy_result=AccuracyResult(passed=True)),
        ])
    ]
    performance_ops = [
        _op([
            _case(1, success=True, perf_result=PerfResult(elapsed_us=20.0),
                  accuracy_result=AccuracyResult(passed=True)),
            # 同一 case 在性能阶段精度复检失败
            _case(2, success=False, failure_type="precision_mismatch",
                  accuracy_result=AccuracyResult(passed=False, threshold=1e-3),
                  perf_result=None),
        ])
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]
    by = {case.case_num: case for case in merged.results}

    # case 1: 两阶段都过 → 保留时延
    assert by[1].success is True
    assert by[1].perf_result is not None and by[1].perf_result.elapsed_us == 20.0
    assert by[1].perf_recheck is None

    # case 2: 精度翻车 → 判失败、扣精度分、无性能分、accuracy 反映失败、打标签
    assert by[2].success is False
    assert by[2].failure_type == "precision_mismatch"
    assert by[2].perf_result is None
    assert by[2].accuracy_result is not None and by[2].accuracy_result.passed is False
    assert by[2].perf_recheck is not None
    assert by[2].perf_recheck["status"] == "precision_unstable"
    assert by[2].perf_recheck["correctness_passed"] is True

    # 计数：1 通过 / 1 失败
    assert merged.passed_cases == 1
    assert merged.failed_cases == 1
    assert merged.pass_rate == 1 / 2

    # perf_recheck 必须能 to_dict → from_dict 往返（下游读 results.json 才看得到）
    roundtrip = EvalCaseResult.from_dict(by[2].to_dict())
    assert roundtrip.perf_recheck == by[2].perf_recheck
    assert roundtrip.perf_recheck["status"] == "precision_unstable"


def test_merge_tags_perf_unmeasured_without_failing_case():
    """性能阶段非精度失败（timeout/runtime）→ 沿用 correctness 通过判定，仅标注缺失。"""
    correctness_ops = [
        _op([_case(1, success=True, accuracy_result=AccuracyResult(passed=True))])
    ]
    performance_ops = [
        _op([_case(1, success=False, failure_type="timeout", perf_result=None)])
    ]

    merged = _merge_results(correctness_ops, performance_ops)[0]
    case = merged.results[0]

    assert case.success is True          # 非精度问题，不改判定
    assert case.perf_result is None      # 但没有有效性能数据
    assert case.perf_recheck is not None
    assert case.perf_recheck["status"] == "perf_unmeasured"
    assert case.perf_recheck["perf_failure_type"] == "timeout"
    assert merged.passed_cases == 1
    assert merged.failed_cases == 0


def test_compile_failure_report_needs_no_device(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from kernel_eval.config import get_project_root
    from kernel_eval.staged_eval import _write_compile_failure_report, create_parser
    from kernel_eval.utils.device_manager import DeviceManager

    def forbidden_device_init(*args, **kwargs):
        raise AssertionError("compile failure reporting must not initialize NPU")

    monkeypatch.setattr(DeviceManager, "__init__", forbidden_device_init)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    args = create_parser().parse_args([
        "--operator", "Exp", "--device-id", "1", "--reports-dir", str(tmp_path),
    ])
    _write_compile_failure_report(
        args, str(get_project_root() / "tasks"),
        SimpleNamespace(compile_errors={"exp": "device compiler failed"}),
    )
    reports = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    report = next(value for value in reports if value.get("result_stage") == "compile_failed")
    assert report["contains_performance"] is False
    assert report["operators"][0]["compilation_error"] == "device compiler failed"
    assert report["operators"][0]["passed_cases"] == 0
