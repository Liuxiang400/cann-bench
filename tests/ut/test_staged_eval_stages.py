#!/usr/bin/python3
# coding=utf-8

"""Unit tests for the cann-bench split staged entry.

Covers the launcher contract added for the evaluation-timeout repair:
default ``--stage all`` behavior, compile/correctness/performance/merge
split with wheel reuse and ``--handoff-dir`` artifacts, per-stage timeout
routing, stage-progress trust, bounded ``--case-ids`` recovery markers and
``scripts/run_evaluation.sh`` flag forwarding.

The NPU/process-pool layer is faked here: these are unit tests for the staged
entry contract, not device validation.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_eval.staged_eval as staged
from kernel_eval.base.result import AccuracyResult, PerfResult
from kernel_eval.data.package_manager import PackageInfo, PackageManager
from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult


REPO_ROOT = Path(__file__).resolve().parents[2]


FAKE_CASES = [
    SimpleNamespace(
        rel_path="level1/fake_op",
        operator="FakeOp",
        case_num=case_num,
        case_id=f"level1/fake_op_{case_num}",
    )
    for case_num in (1, 2, 3)
]
FAILED_CASE_NUM = 3


class _FakeCaseLoader:
    def __init__(self, cases):
        self._cases = list(cases)

    def scan_all(self):
        return list(self._cases)


def _args(tmp_path, *extra, source_dir=None, reports_dir=None, eval_code=None):
    argv = ["--bench-name", "cann", "--device", "npu", "--device-id", "0"]
    if source_dir is not None:
        argv += ["--source-dir", str(source_dir)]
    argv += ["--reports-dir", str(reports_dir if reports_dir is not None else tmp_path / "reports")]
    if eval_code:
        argv += ["--eval-code", eval_code]
    argv += list(extra)
    return staged.create_parser().parse_args(argv)


def _comparable(payload):
    """Report payload without per-invocation generation timestamps.

    Reports stamp ``timestamp`` on the report and on every case row at
    generation time, so two invocations can never share those values; scoring,
    counts and all evidence fields must still match.
    """

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key != "timestamp"}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    payload = strip(json.loads(json.dumps(payload)))
    setup_metadata = ((payload.get("setup_info") or {}).get("metadata") or {})
    setup_metadata.pop("date", None)
    return payload


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Fake build/install/evaluate layer around the real staged orchestration."""
    state = {
        "build_calls": 0,
        "build_allowed": True,
        "installs": [],
        "evaluate_calls": [],
    }
    source_dir = tmp_path / "submission"
    wheel_name = "cann_bench-1.0.0-py3-none-any.whl"
    (source_dir / "dist").mkdir(parents=True)
    source_wheel = source_dir / "dist" / wheel_name
    source_wheel.write_bytes(b"fake-cann-bench-wheel")

    def fake_build_packages(self, source, iterative=True):
        assert state["build_allowed"], "compile must not run again after --wheel handoff"
        state["build_calls"] += 1
        target = Path(source) / "dist" / wheel_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_wheel.read_bytes())
        return PackageInfo(source_dir=str(source), whl_path=str(target))

    monkeypatch.setattr(PackageManager, "build_packages", fake_build_packages)

    def fake_install_or_scan(args, cfg, *, wheel=None):
        state["installs"].append(str(wheel) if wheel else None)
        return ["FakeOp"]

    monkeypatch.setattr(staged, "_install_or_scan", fake_install_or_scan)

    def fake_operator_rel_paths(matched, bench_root, selected=None):
        return ["level1/fake_op"]

    monkeypatch.setattr(staged, "_operator_rel_paths", fake_operator_rel_paths)

    import kernel_eval.registry.loader_registry as loader_registry

    monkeypatch.setattr(
        loader_registry, "get_case_loader", lambda *args, **kwargs: _FakeCaseLoader(FAKE_CASES),
    )

    def fake_evaluate_cases(args, cfg, cases, *, enable_profiler, timeout_per_operator=None):
        state["evaluate_calls"].append(
            (bool(enable_profiler), timeout_per_operator, cfg.timeout_per_operator),
        )
        results = []
        for spec in cases:
            if enable_profiler:
                elapsed_us = 10.0 * spec.case_num
                results.append(EvalCaseResult(
                    case_id=spec.case_id,
                    rel_path=spec.rel_path,
                    operator=spec.operator,
                    case_num=spec.case_num,
                    success=True,
                    accuracy_result=AccuracyResult(
                        passed=True, threshold=1e-3, metadata={"trial": 1},
                    ),
                    perf_result=PerfResult(
                        elapsed_us=elapsed_us,
                        op_times={"kernel": {"duration_us": elapsed_us}},
                        metadata={
                            "case_id": spec.case_id,
                            "_repeat": 5,
                            "warmup_used": False,
                            "freq_boost": False,
                        },
                    ),
                    baseline_perf_us=100.0,
                    t_hw_us=5.0,
                ))
            else:
                passed = spec.case_num != FAILED_CASE_NUM
                results.append(EvalCaseResult(
                    case_id=spec.case_id,
                    rel_path=spec.rel_path,
                    operator=spec.operator,
                    case_num=spec.case_num,
                    success=passed,
                    accuracy_result=AccuracyResult(
                        passed=passed,
                        threshold=1e-3,
                        error_msg=None if passed else "precision mismatch",
                        metadata={"trial": 1},
                    ),
                    perf_result=None,
                    baseline_perf_us=100.0,
                    t_hw_us=5.0,
                    failure_type=None if passed else "precision_mismatch",
                    error_msg=None if passed else "precision mismatch",
                ))
        by_operator = {}
        for case in results:
            by_operator.setdefault(case.operator, []).append(case)
        operator_results = []
        for operator, op_cases in by_operator.items():
            passed = sum(1 for case in op_cases if case.success)
            operator_results.append(EvalOperatorResult(
                rel_path=op_cases[0].rel_path,
                operator=operator,
                total_cases=len(op_cases),
                passed_cases=passed,
                failed_cases=len(op_cases) - passed,
                skipped_cases=0,
                results=op_cases,
                pass_rate=passed / len(op_cases),
                avg_speedup=0.0,
            ))
        return operator_results

    monkeypatch.setattr(staged, "_evaluate_cases", fake_evaluate_cases)

    return SimpleNamespace(
        state=state,
        source_dir=source_dir,
        wheel_name=wheel_name,
        wheel=source_wheel,
    )


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_default_stage_all_keeps_legacy_final_report_and_returncode(fake_pipeline, tmp_path):
    reports = tmp_path / "reports_all"
    args = _args(tmp_path, source_dir=fake_pipeline.source_dir, reports_dir=reports, eval_code="legacy")
    assert args.stage == "all"

    returncode = staged.run(args)

    # Legacy contract: exit code == failed cases in the merged final report.
    assert returncode == 1
    assert fake_pipeline.state["build_calls"] == 1
    # correctness + performance invocations, each with the legacy default timeout.
    assert fake_pipeline.state["evaluate_calls"] == [
        (False, 300, 300),
        (True, 300, 300),
    ]

    final_payload = _read_json(reports / "legacy_final.json")
    assert final_payload["result_stage"] == "final"
    assert final_payload["contains_performance"] is True
    assert final_payload["overall_score"] is not None
    assert final_payload["total_cases"] == 3
    assert final_payload["passed_cases"] == 2
    assert final_payload["failed_cases"] == 1
    # Correctness and performance reports are still written by the legacy path.
    assert (reports / "legacy_correctness.json").is_file()
    assert (reports / "legacy_performance.json").is_file()


def test_compile_handoff_persists_wheel_and_correctness_does_not_rebuild(fake_pipeline, tmp_path):
    reports = tmp_path / "reports"
    handoff = tmp_path / "attempt"
    progress = tmp_path / "progress.json"

    compile_args = _args(
        tmp_path,
        "--stage", "compile",
        "--handoff-dir", str(handoff),
        "--stage-progress", str(progress),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports,
    )
    assert staged.run(compile_args) == 0
    assert fake_pipeline.state["build_calls"] == 1

    compile_handoff = _read_json(handoff / "compile.json")
    assert compile_handoff["stage"] == "compile"
    assert compile_handoff["returncode"] == 0
    assert compile_handoff["compile_ok"] is True
    assert compile_handoff["wheel_path"] == str((handoff / fake_pipeline.wheel_name).resolve())
    assert Path(compile_handoff["wheel_path"]).parent == handoff.resolve()
    expected_sha = hashlib.sha256(fake_pipeline.wheel.read_bytes()).hexdigest()
    assert compile_handoff["wheel_sha256"] == expected_sha
    assert compile_handoff["source_dir"] == str(fake_pipeline.source_dir.resolve())
    assert compile_handoff["task_dir"].endswith("/tasks")
    assert compile_handoff["device_id"] == 0
    assert compile_handoff["profiler_level"] == "Level1"
    assert compile_handoff["warmup"] == 3
    assert compile_handoff["repeat"] == 5
    assert compile_handoff["build_log_path"] is None

    progress_payload = _read_json(progress)
    assert progress_payload["stage"] == "compile"
    assert progress_payload["entered"] is True
    assert progress_payload["returncode"] == 0
    assert progress_payload["finished_at"] is not None
    assert progress_payload["stage_index"] == 1

    # The correctness invocation installs the handed-off wheel; a rebuild would
    # trip the fixture's build assertion.
    fake_pipeline.state["build_allowed"] = False
    correctness_args = _args(
        tmp_path,
        "--stage", "correctness",
        "--wheel", compile_handoff["wheel_path"],
        "--handoff-dir", str(handoff),
        "--stage-progress", str(progress),
        "--correctness-timeout-per-case", "111",
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports,
    )
    assert staged.run(correctness_args) == 0
    assert fake_pipeline.state["build_calls"] == 1
    assert fake_pipeline.state["installs"][-1] == compile_handoff["wheel_path"]

    correctness_handoff = _read_json(handoff / "correctness.json")
    assert correctness_handoff["stage"] == "correctness"
    assert correctness_handoff["returncode"] == 0
    assert correctness_handoff["case_count"] == 3
    assert correctness_handoff["total_cases"] == 3
    assert correctness_handoff["passed_cases"] == 2
    assert correctness_handoff["failed_cases"] == 1
    assert correctness_handoff["wheel_sha256"] == expected_sha
    assert correctness_handoff["wheel_path"] == compile_handoff["wheel_path"]
    assert correctness_handoff["inner_timeout_per_case"] == 111
    assert Path(correctness_handoff["correctness_report_path"]).is_file()
    assert len(correctness_handoff["operators"]) == 1
    assert len(correctness_handoff["operators"][0]["results"]) == 3

    progress_payload = _read_json(progress)
    assert progress_payload["stage"] == "correctness"
    assert progress_payload["returncode"] == 0


def test_split_stages_match_single_invocation_final_report(fake_pipeline, tmp_path):
    reports_all = tmp_path / "reports_all"
    reports_split = tmp_path / "reports_split"
    handoff = tmp_path / "attempt"

    all_args = _args(
        tmp_path, source_dir=fake_pipeline.source_dir, reports_dir=reports_all, eval_code="campaign",
    )
    assert staged.run(all_args) == 1
    all_final = _comparable(_read_json(reports_all / "campaign_final.json"))
    # Split runs must not rebuild: the compile stage is the only build.
    assert staged.run(_args(
        tmp_path,
        "--stage", "compile",
        "--handoff-dir", str(handoff),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports_split,
        eval_code="campaign",
    )) == 0
    fake_pipeline.state["build_allowed"] = False

    correctness_handoff_path = handoff / "correctness.json"
    assert staged.run(_args(
        tmp_path,
        "--stage", "correctness",
        "--wheel", str(handoff / fake_pipeline.wheel_name),
        "--handoff-dir", str(handoff),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports_split,
        eval_code="campaign",
    )) == 0
    assert correctness_handoff_path.is_file()
    assert fake_pipeline.state["build_calls"] == 2

    final_path = reports_split / "campaign_final.json"
    performance_args = _args(
        tmp_path,
        "--stage", "performance",
        "--wheel", str(handoff / fake_pipeline.wheel_name),
        "--handoff-dir", str(handoff),
        "--no-merge",
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports_split,
        eval_code="campaign",
    )
    assert staged.run(performance_args) == 0
    assert fake_pipeline.state["build_calls"] == 2
    assert not final_path.exists()
    performance_handoff = _read_json(handoff / "performance.json")
    assert performance_handoff["merged"] is False
    assert performance_handoff["final_report_path"] is None
    assert performance_handoff["merge_skipped_reason"] == "no_merge_flag"
    assert Path(performance_handoff["performance_report_path"]).is_file()
    assert len(performance_handoff["case_timings"]) == 2
    assert {row["case_num"] for row in performance_handoff["case_timings"]} == {1, 2}
    assert all(row["elapsed_us"] > 0 for row in performance_handoff["case_timings"])

    # Explicit merge stage reproduces the single-invocation final report.
    merge_args = _args(
        tmp_path,
        "--stage", "merge",
        "--handoff-dir", str(handoff),
        reports_dir=reports_split,
        eval_code="campaign",
    )
    assert staged.run(merge_args) == 0
    split_final = _comparable(_read_json(final_path))
    assert split_final == all_final
    assert split_final["result_stage"] == "final"
    assert split_final["contains_performance"] is True
    merged_handoff = _read_json(handoff / "performance.json")
    assert merged_handoff["merged"] is True
    assert merged_handoff["final_report_path"] == str(final_path)

    # Implicit merge in the performance stage produces the same report.
    final_path.unlink()
    implicit_args = _args(
        tmp_path,
        "--stage", "performance",
        "--wheel", str(handoff / fake_pipeline.wheel_name),
        "--handoff-dir", str(handoff),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports_split,
        eval_code="campaign",
    )
    assert staged.run(implicit_args) == 0
    assert _comparable(_read_json(final_path)) == all_final

    assert fake_pipeline.state["build_calls"] == 2


def test_per_stage_timeouts_only_reach_their_stage(fake_pipeline, tmp_path):
    staged.run(_args(
        tmp_path,
        "--correctness-timeout-per-case", "111",
        "--performance-timeout-per-case", "222",
        source_dir=fake_pipeline.source_dir,
        reports_dir=tmp_path / "reports_override",
    ))
    assert fake_pipeline.state["evaluate_calls"] == [
        (False, 111, 111),
        (True, 222, 222),
    ]

    fake_pipeline.state["evaluate_calls"].clear()
    staged.run(_args(
        tmp_path,
        source_dir=fake_pipeline.source_dir,
        reports_dir=tmp_path / "reports_default",
    ))
    assert fake_pipeline.state["evaluate_calls"] == [
        (False, 300, 300),
        (True, 300, 300),
    ]


def test_evaluate_cases_wires_resolved_timeout_into_process_config(monkeypatch):
    import kernel_eval.eval.process_pool as process_pool

    captured = {}

    class _FakeCoordinator:
        def __init__(self, base_config, process_config, device_id):
            captured["timeout"] = process_config.timeout_per_operator
            self.card_count = 1

        def evaluate_task_units(self, task_units):
            return []

        def shutdown(self):
            pass

    monkeypatch.setattr(process_pool, "ProcessPoolCoordinator", _FakeCoordinator)
    monkeypatch.setattr(process_pool, "build_task_units", lambda cases_by_operator, card_count: [])
    monkeypatch.setattr(process_pool, "aggregate_by_operator", lambda results: [])

    args = staged.create_parser().parse_args([])
    cfg = staged._make_config(
        args, str(REPO_ROOT / "tasks"), enable_profiler=True, timeout_per_operator=222,
    )
    results = staged._evaluate_cases(
        args, cfg, [SimpleNamespace(operator="FakeOp")],
        enable_profiler=True, timeout_per_operator=222,
    )
    assert results == []
    assert cfg.timeout_per_operator == 222
    assert captured["timeout"] == 222


def test_stage_progress_records_failure(fake_pipeline, tmp_path, monkeypatch):
    progress = tmp_path / "progress.json"
    reports = tmp_path / "reports"

    def boom(*args, **kwargs):
        raise RuntimeError("device unavailable")

    monkeypatch.setattr(staged, "_evaluate_cases", boom)
    args = _args(
        tmp_path,
        "--stage", "correctness",
        "--stage-progress", str(progress),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports,
    )
    with pytest.raises(RuntimeError):
        staged.run(args)

    payload = _read_json(progress)
    assert payload["stage"] == "correctness"
    assert payload["entered"] is True
    assert payload["returncode"] == 1
    assert payload["finished_at"] is not None


def test_compile_failure_handoff_has_no_wheel(fake_pipeline, tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    handoff = tmp_path / "attempt"

    def failing_build(self, source, iterative=True):
        (Path(source) / "_compile.log").write_text("compile error\n", encoding="utf-8")
        return PackageInfo(
            source_dir=str(source),
            whl_path=None,
            build_failed=True,
            compile_errors={"fake_op": "compile error"},
        )

    monkeypatch.setattr(PackageManager, "build_packages", failing_build)
    monkeypatch.setattr(staged, "_write_compile_failure_report", lambda *args, **kwargs: None)

    args = _args(
        tmp_path,
        "--stage", "compile",
        "--handoff-dir", str(handoff),
        source_dir=fake_pipeline.source_dir,
        reports_dir=reports,
    )
    assert staged.run(args) == 1

    payload = _read_json(handoff / "compile.json")
    assert payload["returncode"] == 1
    assert payload["compile_ok"] is False
    assert payload["wheel_path"] is None
    assert payload["wheel_sha256"] is None
    assert payload["build_log_path"] == str(
        (fake_pipeline.source_dir / "_compile.log").resolve()
    )


def test_case_ids_marks_bounded_performance_evidence(fake_pipeline, tmp_path):
    reports = tmp_path / "reports"
    handoff = tmp_path / "attempt"
    wheel = str(fake_pipeline.wheel)

    args = _args(
        tmp_path,
        "--stage", "performance",
        "--wheel", wheel,
        "--handoff-dir", str(handoff),
        "--case-ids", "1", "2",
        "--no-merge",
        reports_dir=reports,
    )
    assert staged.run(args) == 0

    performance_handoff = _read_json(handoff / "performance.json")
    assert performance_handoff["merged"] is False
    assert performance_handoff["final_report_path"] is None
    assert performance_handoff["case_filter"] == [1, 2]
    # No correctness handoff was supplied to this invocation.
    assert performance_handoff["merge_skipped_reason"] == "no_correctness_handoff"
    assert {row["case_num"] for row in performance_handoff["case_timings"]} == {1, 2}
    report_payload = _read_json(performance_handoff["performance_report_path"])
    assert report_payload["case_filter"] == [1, 2]
    assert report_payload["cross_invocation_evidence"] is True
    assert report_payload["result_stage"] == "performance"

    # A bounded recovery invocation can never be merged into a complete report.
    (handoff / "correctness.json").write_text(
        json.dumps({"stage": "correctness", "operators": []}), encoding="utf-8",
    )
    merge_args = _args(
        tmp_path,
        "--stage", "merge",
        "--handoff-dir", str(handoff),
        reports_dir=reports,
    )
    with pytest.raises(RuntimeError):
        staged.run(merge_args)


def test_legacy_case_id_marks_gate_evidence(fake_pipeline, tmp_path):
    reports = tmp_path / "reports"
    handoff = tmp_path / "attempt"
    args = _args(
        tmp_path,
        "--stage", "performance",
        "--wheel", str(fake_pipeline.wheel),
        "--case-id", "2",
        "--handoff-dir", str(handoff),
        "--no-merge",
        reports_dir=reports,
    )
    assert staged.run(args) == 0

    performance_handoff = _read_json(handoff / "performance.json")
    assert performance_handoff["case_filter"] == [2]
    assert [row["case_num"] for row in performance_handoff["case_timings"]] == [2]
    assert performance_handoff["case_timings"][0]["elapsed_us"] == 20.0


def _stub_python(stub_dir, record_path):
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python"
    stub.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$@\" > \"{record_path}\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)


def test_run_evaluation_forwards_staged_flags(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("bash is required for the run_evaluation.sh forwarding test")
    script = REPO_ROOT / "scripts" / "run_evaluation.sh"
    if not script.is_file():
        pytest.skip("run_evaluation.sh is not available")

    stub_dir = tmp_path / "bin"
    record = tmp_path / "argv.txt"
    _stub_python(stub_dir, record)
    reports = tmp_path / "reports"
    handoff = tmp_path / "handoff"
    progress = tmp_path / "progress.json"
    wheel = tmp_path / "prebuilt.whl"
    wheel.write_bytes(b"wheel")

    environment = dict(os.environ)
    environment["PATH"] = f"{stub_dir}:{environment.get('PATH', '')}"

    cmd = [
        "bash", str(script), "--staged",
        "--wheel", str(wheel),
        "--stage", "performance",
        "--handoff-dir", str(handoff),
        "--correctness-timeout-per-case", "111",
        "--performance-timeout-per-case", "222",
        "--case-ids", "5", "18",
        "--stage-progress", str(progress),
        "--no-merge",
        "--operator", "FakeOp",
        "--reports-dir", str(reports),
        "--device-id", "0",
        "--timeout-per-process", "77",
    ]
    result = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=environment, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)

    argv = record.read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-m", "kernel_eval.staged_eval"]

    def value_of(flag):
        index = argv.index(flag)
        return argv[index + 1]

    assert value_of("--stage") == "performance"
    assert value_of("--wheel") == str(wheel)
    assert value_of("--handoff-dir") == str(handoff)
    assert value_of("--correctness-timeout-per-case") == "111"
    assert value_of("--performance-timeout-per-case") == "222"
    assert value_of("--stage-progress") == str(progress)
    assert "--no-merge" in argv
    # Legacy --timeout-per-process is still forwarded as --timeout-per-operator.
    assert value_of("--timeout-per-operator") == "77"
    case_ids_index = argv.index("--case-ids")
    assert argv[case_ids_index + 1:case_ids_index + 3] == ["5", "18"]

    # --single-pass keeps the legacy CLI and does not forward staged-only flags.
    single_pass_cmd = [
        "bash", str(script), "--single-pass",
        "--operator", "FakeOp",
        "--reports-dir", str(tmp_path / "reports_single"),
        "--device-id", "0",
        "--timeout-per-process", "88",
    ]
    result = subprocess.run(
        single_pass_cmd, cwd=str(REPO_ROOT), env=environment,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    argv = record.read_text(encoding="utf-8").splitlines()
    assert argv[:2] == ["-m", "kernel_eval.cli"]
    assert value_of("--timeout-per-operator") == "88"
    assert "--stage" not in argv
