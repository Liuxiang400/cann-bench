#!/usr/bin/python3
# coding=utf-8

"""Default staged evaluator for cann-bench local runs.

Stages:
1. compile: build the submitted cann_bench package.
2. correctness: run precision-only evaluation and keep pass/fail as authority.
3. performance: profile only correctness-passed cases, then merge timings back.

Launcher invocations can also run a single stage (``--stage
compile|correctness|performance|merge``) and hand the cross-stage artifacts to
the next invocation through ``--handoff-dir``:

- ``compile.json``: compile result + wheel identity (the wheel is copied into
  the handoff dir), source/task identity and compilation method.
- ``correctness.json``: serialized correctness operator results, report path,
  case counts and the installed wheel/source/method identity.
- ``performance.json``: serialized performance operator results, per-case
  timings, report paths and the installed wheel/source/method identity. The
  merge stage rewrites it with the final report path.

The performance stage merges the correctness handoff implicitly (unless
``--no-merge`` is passed; a ``--case-ids`` recovery invocation never merges);
the explicit ``merge`` stage restores both handoffs through
``EvalOperatorResult.from_dict`` and calls the same ``_merge_results`` /
``_save_report`` path as the single-invocation ``all`` mode, so the final
report/score is identical.

``--stage-progress PATH`` writes an atomically replaced, trusted stage progress
JSON at stage start and end (also when the stage fails); the adapter uses it to
attribute an outer deadline to the harness stage that was entered.

Return codes: compile returns 0/1 (1 = the build failed); split correctness /
performance / merge stages return 0 whenever their report file was written
(candidate failures are evidence, not stage failures) and non-zero only when
the stage could not produce its report. ``--stage all`` keeps the legacy
return code (number of failed cases in the merged final report).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .config import Config, get_project_root, set_config


CaseKey = Tuple[str, int]

COMPILE_HANDOFF = "compile.json"
CORRECTNESS_HANDOFF = "correctness.json"
PERFORMANCE_HANDOFF = "performance.json"

VALID_STAGES = ("all", "compile", "correctness", "performance", "merge")


# ---------------------------------------------------------------------------
# Small shared helpers: timestamps, hashes, atomic JSON writes, wheel copies.
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """UTC RFC3339 timestamp with a ``Z`` suffix (millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_file(path: Any) -> Optional[str]:
    """sha256 of an existing file; ``None`` for a missing/empty path."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Any, payload: Dict[str, Any]) -> None:
    """Write *payload* as JSON via a same-directory temp file + ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as src:
            shutil.copyfileobj(src, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class _StageProgress:
    """Trusted ``--stage-progress`` writer for one launcher stage."""

    def __init__(
        self,
        path: Optional[str],
        stage: str,
        *,
        stage_index: int = 1,
        stage_count: int = 1,
    ):
        self.path = Path(path) if path else None
        self.stage = stage
        self.stage_index = int(stage_index)
        self.stage_count = int(stage_count)
        self.started_at = _utc_now_iso()

    def _payload(self, *, finished_at: Optional[str], returncode: Optional[int]) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_index": self.stage_index,
            "stage_count": self.stage_count,
            "entered": True,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "returncode": returncode,
        }

    def start(self) -> None:
        if self.path is None:
            return
        _write_json_atomic(self.path, self._payload(finished_at=None, returncode=None))

    def finish(self, returncode: int) -> None:
        if self.path is None:
            return
        _write_json_atomic(self.path, self._payload(
            finished_at=_utc_now_iso(), returncode=int(returncode),
        ))


def _run_stage(
    args: argparse.Namespace,
    stage: str,
    body,
    *,
    stage_index: int = 1,
    stage_count: int = 1,
) -> int:
    """Run one launcher stage and keep ``--stage-progress`` trustworthy.

    ``body(started_at, started_monotonic)`` performs the stage and returns its
    return code. On an unhandled exception the progress file is finished with
    ``returncode=1`` before the exception is re-raised, so the adapter never
    observes a stage that was entered but neither finished nor failed.
    """
    progress = _StageProgress(
        getattr(args, "stage_progress", None),
        stage,
        stage_index=stage_index,
        stage_count=stage_count,
    )
    progress.start()
    started_monotonic = time.monotonic()
    returncode = 1
    try:
        returncode = int(body(progress.started_at, started_monotonic))
        return returncode
    finally:
        progress.finish(returncode)


def _stage_wheel_source(args: argparse.Namespace) -> Optional[str]:
    """The wheel this invocation installs: ``--wheel`` or the compile handoff."""
    wheel = getattr(args, "wheel", None)
    if wheel:
        return str(wheel)
    compile_handoff = _read_handoff_json(args, COMPILE_HANDOFF)
    if compile_handoff:
        wheel_path = compile_handoff.get("wheel_path")
        if wheel_path:
            return str(wheel_path)
    return None


def _persist_wheel(args: argparse.Namespace, wheel: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Copy the artifact wheel into ``--handoff-dir`` and return (path, sha256).

    A ``--handoff-dir`` is the durable identity store for the attempt, so the
    recorded wheel path always points inside it even when the adapter passes a
    wheel from a temporary build location. Without a handoff dir the source path
    is recorded as-is.
    """
    if not wheel:
        return None, None
    source = Path(wheel)
    if not source.is_file():
        return None, None
    handoff_dir = getattr(args, "handoff_dir", None)
    if not handoff_dir:
        resolved = source.resolve()
        return str(resolved), _sha256_file(resolved)
    target = Path(handoff_dir) / source.name
    if source.resolve() != target.resolve():
        source_sha = _sha256_file(source)
        if not target.is_file() or _sha256_file(target) != source_sha:
            _copy_file_atomic(source, target)
    resolved_target = target.resolve()
    return str(resolved_target), _sha256_file(resolved_target)


def _handoff_identity(
    args: argparse.Namespace,
    bench_root: Optional[str],
    wheel: Optional[str],
    *,
    inner_timeout_per_case: Optional[int],
    case_filter: Optional[List[int]],
) -> Dict[str, Any]:
    """Persistent source/wheel/task/method identity fields for every handoff."""
    source_dir = getattr(args, "source_dir", None)
    wheel_path, wheel_sha256 = _persist_wheel(args, wheel)
    wheel_source = getattr(args, "wheel", None)
    return {
        "source_dir": str(Path(source_dir).resolve()) if source_dir else None,
        "task_dir": str(Path(bench_root).resolve()) if bench_root else None,
        "task_dir_argument": getattr(args, "task_dir", None),
        "device_id": int(getattr(args, "device_id", 0) or 0),
        "profiler_level": getattr(args, "profiler_level", None),
        "warmup": int(getattr(args, "warmup", 0) or 0),
        "repeat": int(getattr(args, "repeat", 0) or 0),
        "wheel_path": wheel_path,
        "wheel_sha256": wheel_sha256,
        "wheel_source_path": str(Path(wheel_source).resolve()) if wheel_source else None,
        "inner_timeout_per_case": (
            int(inner_timeout_per_case) if inner_timeout_per_case is not None else None
        ),
        "case_filter": list(case_filter) if case_filter else None,
    }


# ---------------------------------------------------------------------------
# Case helpers (unchanged behavior).
# ---------------------------------------------------------------------------


def _case_num_from_value(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:^|_)(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def _case_num(case) -> int:
    case_num = _case_num_from_value(getattr(case, "case_num", None))
    if case_num:
        return case_num
    return _case_num_from_value(getattr(case, "case_id", None))


def _case_result_key(case: EvalCaseResult) -> CaseKey:
    return (case.rel_path, _case_num(case))


def _case_spec_key(case) -> CaseKey:
    return (str(getattr(case, "rel_path", "") or ""), _case_num(case))


# ---------------------------------------------------------------------------
# Configuration / stage timeouts.
# ---------------------------------------------------------------------------


def _make_config(
    args: argparse.Namespace,
    bench_root: str,
    *,
    enable_profiler: bool,
    timeout_per_operator: Optional[int] = None,
) -> Config:
    cfg = Config()
    cfg.tasks_root = bench_root
    cfg.bench_name = args.bench_name
    cfg.device_type = args.device
    cfg.device_id = int(args.device_id or 0)
    cfg.warmup = args.warmup
    cfg.repeat = args.repeat
    cfg.enable_profiler = enable_profiler
    cfg.profiler_level = args.profiler_level
    cfg.timeout_per_operator = (
        args.timeout_per_operator if timeout_per_operator is None else int(timeout_per_operator)
    )
    cfg.reports_dir = args.reports_dir
    cfg.processes_per_card = args.processes_per_card
    cfg.eval_seed = None if args.eval_seed == -1 else args.eval_seed
    if args.source_dir:
        cfg.source_dir = args.source_dir
    cfg.agent_skill = getattr(args, 'agent_skill', '') or ''
    cfg.base_model = getattr(args, 'base_model', '') or ''
    cfg.harness = getattr(args, 'harness', '') or ''
    if args.torch_op_guard_mode:
        cfg.torch_op_guard_mode = args.torch_op_guard_mode
    if args.perf_metric_strategy:
        cfg.perf_metric_strategy_override = args.perf_metric_strategy
    set_config(cfg)
    return cfg


def _stage_timeout(args: argparse.Namespace, stage: str) -> int:
    """Per-stage inner per-case budget.

    ``--correctness-timeout-per-case`` / ``--performance-timeout-per-case`` only
    affect their own stage; both fall back to the legacy ``--timeout-per-operator``.
    """
    if stage == "correctness":
        specific = getattr(args, "correctness_timeout_per_case", None)
    else:
        specific = getattr(args, "performance_timeout_per_case", None)
    if specific is not None:
        return int(specific)
    return int(args.timeout_per_operator)


def _operator_rel_paths(
    matched_operators: Iterable[str],
    bench_root: str,
    selected: Optional[Iterable[str]] = None,
) -> List[str]:
    from .benches import cann as _cann_bench  # noqa: F401
    from .registry.loader_registry import get_task_loader

    matched = {str(op).lower() for op in matched_operators}
    selected_set = {str(op).lower() for op in selected} if selected else None
    loader = get_task_loader("cann", tasks_root=bench_root)
    rel_paths = []
    for spec in loader.list_tasks():
        names = {
            str(getattr(spec, "name", "")).lower(),
            str(spec.get_function_name()).lower(),
        }
        if not (names & matched):
            continue
        if selected_set is not None:
            all_names = names | {
                str(spec.rel_path).lower(),
                Path(spec.rel_path).name.lower(),
            }
            if not (all_names & selected_set):
                continue
        rel_paths.append(spec.rel_path)
    return sorted(set(rel_paths))


# ---------------------------------------------------------------------------
# Install / case loading / evaluation.
# ---------------------------------------------------------------------------


def _install_or_scan(args: argparse.Namespace, cfg: Config, *, wheel: Optional[str] = None) -> List[str]:
    """Install exactly one artifact and scan the installed interfaces.

    ``wheel`` (from ``--wheel``) installs a prebuilt submission wheel instead of
    compiling; without it the legacy behavior is kept (install the wheel the
    compile stage left in ``<source-dir>/dist``, or scan what is installed).
    """
    from .benches import cann as _cann_bench  # noqa: F401
    from .data.package_manager import PackageManager, PackageInfo

    pm = PackageManager(config=cfg)
    if wheel:
        whl_path = Path(wheel)
        if not whl_path.is_file():
            raise FileNotFoundError(f"--wheel 指定的 wheel 不存在: {wheel}")
        from .security.api_guard import APIGuard

        package_info = PackageInfo(
            source_dir=str(args.source_dir or whl_path.parent),
            whl_path=str(whl_path),
        )
        guard = APIGuard()
        guard.snapshot()
        if not pm.install_packages(package_info):
            raise RuntimeError("package install failed")
        matched = pm.prepare_skip_build()
        guard.verify()
        return matched

    if not args.source_dir:
        return pm.prepare_skip_build()

    from .security.api_guard import APIGuard

    package_info = pm.scan_source_dir(args.source_dir)
    if not package_info.whl_path:
        raise RuntimeError(
            "compile stage did not produce a cann_bench wheel; "
            "run --stage compile first or pass --wheel"
        )

    guard = APIGuard()
    guard.snapshot()
    if not pm.install_packages(package_info):
        raise RuntimeError("package install failed")
    matched = pm.prepare_skip_build()
    guard.verify()
    return matched


def _load_cases(
    args: argparse.Namespace,
    bench_root: str,
    rel_paths: List[str],
    *,
    filter_prefix: str,
    allowlist: Optional[Set[CaseKey]] = None,
    case_ids: Optional[Set[int]] = None,
) -> List:
    from .benches import cann as _cann_bench  # noqa: F401
    from .registry.loader_registry import get_case_loader

    loader = get_case_loader(args.bench_name, tasks_root=bench_root)
    all_cases = []
    rel_path_set = set(rel_paths)
    for case in loader.scan_all():
        if rel_path_set and case.rel_path not in rel_path_set:
            continue
        if filter_prefix and not (case.rel_path == filter_prefix or case.rel_path.startswith(filter_prefix + "/")):
            continue
        if args.operator and str(case.operator).lower() != args.operator.lower():
            continue
        if args.case_id is not None and _case_num(case) != int(args.case_id):
            continue
        if case_ids is not None and _case_num(case) not in case_ids:
            continue
        if allowlist is not None and _case_spec_key(case) not in allowlist:
            continue
        all_cases.append(case)
    return all_cases


def _evaluate_cases(
    args: argparse.Namespace,
    cfg: Config,
    cases: List,
    *,
    enable_profiler: bool,
    timeout_per_operator: Optional[int] = None,
) -> List[EvalOperatorResult]:
    from .eval.process_pool import (
        ProcessConfig,
        ProcessPoolCoordinator,
        aggregate_by_operator,
        build_task_units,
    )

    if not cases:
        return []

    effective_timeout = (
        args.timeout_per_operator if timeout_per_operator is None else int(timeout_per_operator)
    )
    cases_by_operator: Dict[str, List] = defaultdict(list)
    for case in cases:
        cases_by_operator[str(case.operator)].append(case)

    coordinator = ProcessPoolCoordinator(
        base_config=cfg,
        process_config=ProcessConfig(
            processes_per_card=args.processes_per_card,
            timeout_per_operator=effective_timeout,
            enable_profiler=enable_profiler,
        ),
        device_id=args.device_id,
    )
    try:
        task_units = build_task_units(cases_by_operator, coordinator.card_count)
        return aggregate_by_operator(coordinator.evaluate_task_units(task_units))
    finally:
        coordinator.shutdown()


def _save_report(
    args: argparse.Namespace,
    cfg: Config,
    operator_results: List[EvalOperatorResult],
    *,
    stage: str,
    contains_performance: bool,
) -> Tuple[dict, object, Dict[str, Path]]:
    from .report.report_generator import ReportGenerator

    generator = ReportGenerator(
        output_dir=args.reports_dir,
        eval_code=f"{args.eval_code}_{stage}" if args.eval_code else None,
        semantic_prefix=f"{args.bench_name}_{stage}",
        config=cfg,
    )
    for op_result in operator_results:
        generator.add_operator_result(op_result)
    report = generator.generate()
    payload = report.to_dict()
    payload["result_stage"] = stage
    payload["contains_performance"] = contains_performance
    if not contains_performance:
        payload["overall_score"] = None
        payload.setdefault("summary", {})["overall_score"] = None
        payload["summary"]["score_unavailable_reason"] = "performance stage has not completed"
    paths = generator.save_all(report)
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, report, paths


def _annotate_report_json(payload: Dict[str, Any], report_paths: Optional[Dict[str, Path]],
                          extra: Dict[str, Any]) -> None:
    """Add adapter-facing markers to an already saved report JSON payload."""
    payload.update(extra)
    if report_paths and report_paths.get("json"):
        report_paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _passed_case_keys(operator_results: List[EvalOperatorResult]) -> Set[CaseKey]:
    return {
        _case_result_key(case)
        for op_result in operator_results
        for case in op_result.results
        if case.success
    }


def _case_timings(operator_results: List[EvalOperatorResult]) -> List[Dict[str, Any]]:
    """Compact per-case timing evidence for the performance handoff.

    Every evaluated case keeps a row; ``elapsed_us`` is explicitly ``null`` when
    the performance stage produced no valid timing, so the adapter can tell a
    missing measurement from an omitted row.
    """
    rows: List[Dict[str, Any]] = []
    for op_result in operator_results:
        for case in op_result.results:
            rows.append({
                "operator": case.operator,
                "rel_path": case.rel_path,
                "case_id": case.case_id,
                "case_num": _case_num(case),
                "success": bool(case.success),
                "failure_type": case.failure_type,
                "elapsed_us": case.perf_result.elapsed_us if case.perf_result else None,
                "baseline_perf_us": case.baseline_perf_us,
                "speedup": case.get_speedup(),
            })
    return rows


def _merge_results(
    correctness_ops: List[EvalOperatorResult],
    performance_ops: List[EvalOperatorResult],
) -> List[EvalOperatorResult]:
    from .eval.results import EvalOperatorResult
    from .base.result import (
        FAILURE_TYPE_PRECISION_MISMATCH,
        is_precision_failure_type,
    )

    # 全量索引性能阶段所有 case（含失败），用途有二：
    #  1) 成功 → 回填时延；
    #  2) 精度失败 → 识别"correctness 过、performance 精度翻车"的 case。
    #     两次跑输入与 golden 完全一致（已定种子），唯一变量是 NPU kernel 自身输出，
    #     故这类翻车基本等价于"算子非确定"。按策略视为该 case 的精度错误。
    perf_all = {
        _case_result_key(case): case
        for op_result in performance_ops
        for case in op_result.results
    }

    merged: List[EvalOperatorResult] = []
    for op_result in correctness_ops:
        cases = []
        for case in op_result.results:
            if not case.success:
                # correctness 阶段本就失败：保持原判定，无性能分。
                case.perf_result = None
                cases.append(case)
                continue

            perf_case = perf_all.get(_case_result_key(case))
            if perf_case is not None and perf_case.success:
                # 性能阶段精度复检同样通过 → 回填时延。
                case.perf_result = perf_case.perf_result
                case.ai_run_result = perf_case.ai_run_result
            elif perf_case is not None and is_precision_failure_type(perf_case.failure_type):
                # correctness 过、performance 精度翻车 → 视为该 case 精度错误：
                # success=False + precision_mismatch，按原公式扣精度分（不扣编译分），
                # 且无对应性能分；同时打标签便于在 results.json 中定位疑似非确定算子。
                case.success = False
                case.failure_type = FAILURE_TYPE_PRECISION_MISMATCH
                if perf_case.accuracy_result is not None:
                    case.accuracy_result = perf_case.accuracy_result
                case.error_msg = (
                    "性能阶段精度复检失败（correctness 阶段已通过）——疑似 NPU 非确定性算子: "
                    + (perf_case.error_msg or "")
                )
                case.perf_result = None
                case.perf_recheck = {
                    "status": "precision_unstable",
                    "correctness_passed": True,
                    "perf_failure_type": perf_case.failure_type,
                    "note": (
                        "passed precision in the correctness stage but failed the "
                        "precision re-check in the performance stage; likely a "
                        "non-deterministic NPU kernel."
                    ),
                }
            else:
                # 性能阶段无法测量（timeout / runtime / 未重跑）：非精度问题，
                # 沿用 correctness 通过判定，仅标注缺失原因，无性能分。
                case.perf_result = None
                if perf_case is not None and not perf_case.success:
                    case.perf_recheck = {
                        "status": "perf_unmeasured",
                        "correctness_passed": True,
                        "perf_failure_type": perf_case.failure_type,
                        "note": (
                            "passed correctness but the performance stage could not "
                            "produce a valid timing (e.g. timeout / runtime error)."
                        ),
                    }
            cases.append(case)

        speedups = [case.get_speedup() for case in cases if case.success and case.get_speedup() > 0]
        total_cases = max(op_result.total_cases, len(cases))
        passed_cases = sum(1 for case in cases if case.success)
        skipped_cases = sum(
            1 for case in cases
            if not case.success and case.failure_type in ("cascade_device", "skipped")
        )
        failed_cases = sum(
            1 for case in cases
            if not case.success and case.failure_type not in ("cascade_device", "skipped")
        )
        merged.append(EvalOperatorResult(
            rel_path=op_result.rel_path,
            operator=op_result.operator,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=max(op_result.skipped_cases, skipped_cases),
            results=cases,
            pass_rate=passed_cases / total_cases if total_cases else 0.0,
            avg_speedup=sum(speedups) / len(speedups) if speedups else 0.0,
            compilation_error=op_result.compilation_error,
            subprocess_failure_reason=op_result.subprocess_failure_reason,
        ))
    return merged


# ---------------------------------------------------------------------------
# Cross-stage handoff artifacts (atomic JSON files under --handoff-dir).
# ---------------------------------------------------------------------------


def _handoff_path(args: argparse.Namespace, name: str) -> Optional[Path]:
    handoff_dir = getattr(args, "handoff_dir", None)
    if not handoff_dir:
        return None
    return Path(handoff_dir) / name


def _read_handoff_json(args: argparse.Namespace, name: str, *, required: bool = False) -> Optional[Dict[str, Any]]:
    path = _handoff_path(args, name)
    if path is None or not path.is_file():
        if required:
            raise FileNotFoundError(f"--handoff-dir 缺少 {name}: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 handoff 文件 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"handoff 文件不是 JSON object: {path}")
    return payload


def _write_compile_handoff(
    args: argparse.Namespace,
    bench_root: Optional[str],
    wheel: Optional[str],
    *,
    build_log_path: Optional[str],
    returncode: int,
    compile_ok: bool,
    started_at: str,
    started_monotonic: float,
) -> Optional[Dict[str, Any]]:
    path = _handoff_path(args, COMPILE_HANDOFF)
    if path is None:
        return None
    payload = {
        "stage": "compile",
        "returncode": int(returncode),
        "compile_ok": bool(compile_ok),
        "build_log_path": str(Path(build_log_path).resolve()) if build_log_path else None,
        **_handoff_identity(
            args, bench_root, wheel,
            inner_timeout_per_case=None, case_filter=None,
        ),
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    }
    if not compile_ok:
        # A failed build has no reusable wheel; never point at a partial artifact.
        payload["wheel_path"] = None
        payload["wheel_sha256"] = None
    _write_json_atomic(path, payload)
    return payload


def _write_correctness_handoff(
    args: argparse.Namespace,
    bench_root: Optional[str],
    operator_results: List[EvalOperatorResult],
    report_payload: Optional[Dict[str, Any]],
    report_paths: Optional[Dict[str, Path]],
    *,
    returncode: int,
    wheel: Optional[str],
    inner_timeout_per_case: int,
    started_at: str,
    started_monotonic: float,
) -> Optional[Dict[str, Any]]:
    path = _handoff_path(args, CORRECTNESS_HANDOFF)
    if path is None:
        return None
    report_payload = report_payload or {}
    payload = {
        "stage": "correctness",
        "returncode": int(returncode),
        "correctness_report_path": str(report_paths["json"]) if report_paths else None,
        "case_count": sum(len(op_result.results) for op_result in operator_results),
        "total_cases": int(report_payload.get("total_cases") or 0),
        "passed_cases": int(report_payload.get("passed_cases") or 0),
        "failed_cases": int(report_payload.get("failed_cases") or 0),
        "operators": [op_result.to_dict() for op_result in operator_results],
        **_handoff_identity(
            args, bench_root, wheel,
            inner_timeout_per_case=inner_timeout_per_case, case_filter=None,
        ),
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    }
    _write_json_atomic(path, payload)
    return payload


def _write_performance_handoff(
    args: argparse.Namespace,
    bench_root: Optional[str],
    operator_results: List[EvalOperatorResult],
    report_paths: Optional[Dict[str, Path]],
    *,
    final_report_paths: Optional[Dict[str, Path]],
    merged: bool,
    merge_skipped_reason: Optional[str],
    returncode: int,
    wheel: Optional[str],
    case_filter: Optional[List[int]],
    inner_timeout_per_case: int,
    correctness_handoff_used: bool,
    started_at: str,
    started_monotonic: float,
) -> Optional[Dict[str, Any]]:
    path = _handoff_path(args, PERFORMANCE_HANDOFF)
    if path is None:
        return None
    payload = {
        "stage": "performance",
        "returncode": int(returncode),
        "merged": bool(merged),
        "merge_skipped_reason": merge_skipped_reason,
        "performance_report_path": str(report_paths["json"]) if report_paths else None,
        "final_report_path": str(final_report_paths["json"]) if final_report_paths else None,
        "case_count": sum(len(op_result.results) for op_result in operator_results),
        "case_timings": _case_timings(operator_results),
        "operators": [op_result.to_dict() for op_result in operator_results],
        "correctness_handoff": bool(correctness_handoff_used),
        **_handoff_identity(
            args, bench_root, wheel,
            inner_timeout_per_case=inner_timeout_per_case, case_filter=case_filter,
        ),
        "started_at": started_at,
        "finished_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
    }
    _write_json_atomic(path, payload)
    return payload


def _handoff_operators(handoff: Dict[str, Any], name: str) -> List[Dict[str, Any]]:
    operators = handoff.get("operators")
    if not isinstance(operators, list):
        raise RuntimeError(f"{name} 缺少 operators 列表，无法恢复评测结果")
    return operators


def _restore_operator_results(raw_operators: List[Dict[str, Any]]) -> List[EvalOperatorResult]:
    """Restore ``EvalOperatorResult`` objects serialized with ``to_dict``.

    ``EvalOperatorResult.from_dict`` / ``EvalCaseResult.from_dict`` are the only
    public deserializers used here. ``EvalCaseResult.from_dict`` fills default
    ``perf.metadata`` keys (case_id/baseline_us/t_hw_us) that the in-memory
    single-invocation path never added; restore the exact serialized metadata so
    the merged report is identical to the ``--stage all`` report.
    """
    from .eval.results import EvalOperatorResult

    restored: List[EvalOperatorResult] = []
    for raw_op in raw_operators:
        if not isinstance(raw_op, dict):
            raise RuntimeError("handoff operators 条目必须是 JSON object")
        op_result = EvalOperatorResult.from_dict(raw_op)
        raw_cases = raw_op.get("results") or []
        for case, raw_case in zip(op_result.results, raw_cases):
            if not isinstance(raw_case, dict):
                continue
            raw_perf = raw_case.get("perf")
            if case.perf_result is not None and isinstance(raw_perf, dict):
                case.perf_result.metadata = dict(raw_perf.get("metadata") or {})
        restored.append(op_result)
    return restored


def _write_compile_failure_report(
    args: argparse.Namespace,
    bench_root: str,
    package_info,
) -> None:
    cfg = _make_config(args, bench_root, enable_profiler=False)
    from . import benches as _benches  # noqa: F401 -- register metadata loaders
    from .eval.failure_synthesizer import FailureSynthesizer
    from .registry.loader_registry import get_case_loader, get_task_loader
    from .registry.matcher_registry import get_operator_matcher

    # Reporting a failed build needs task metadata, not a device. In a masked
    # process the external device ID may not even be a valid logical NPU ID.
    operator_loader = get_task_loader(args.bench_name, tasks_root=cfg.tasks_root)
    operator_matcher = get_operator_matcher(args.bench_name, operator_loader=operator_loader)
    synthesizer = FailureSynthesizer(get_case_loader(args.bench_name, tasks_root=cfg.tasks_root))
    op_filter = [args.operator] if args.operator else None
    failures = synthesizer.synthesize_all_compile_failures(
        operator_matcher,
        package_info,
        operator_filter=op_filter,
    )
    _save_report(args, cfg, failures, stage="compile_failed", contains_performance=False)


def _compile(args: argparse.Namespace, bench_root: str) -> Tuple[int, object]:
    from .data.package_manager import PackageManager

    if not args.source_dir:
        print("[staged_eval] compile: no --source-dir, using installed cann_bench")
        return 0, None

    cfg = _make_config(args, bench_root, enable_profiler=False)
    pm = PackageManager(config=cfg)
    package_info = pm.build_packages(args.source_dir, iterative=not args.no_iterative_compile)
    if getattr(package_info, "build_failed", False) or not package_info.whl_path:
        _write_compile_failure_report(args, bench_root, package_info)
        return 1, package_info
    print(f"[staged_eval] compile: built {Path(package_info.whl_path).name}")
    return 0, package_info


# ---------------------------------------------------------------------------
# Stage implementations.
# ---------------------------------------------------------------------------


def _compile_stage(
    args: argparse.Namespace,
    bench_root: str,
    *,
    stage_index: int = 1,
    stage_count: int = 1,
) -> int:
    def body(started_at: str, started_monotonic: float) -> int:
        wheel_source: Optional[str] = None
        build_log_path: Optional[str] = None
        if getattr(args, "wheel", None):
            wheel_path = Path(args.wheel)
            if not wheel_path.is_file():
                raise FileNotFoundError(f"--wheel 指定的 wheel 不存在: {args.wheel}")
            print(f"[staged_eval] compile: using prebuilt wheel {wheel_path.name}, skip rebuild")
            wheel_source = str(wheel_path)
            returncode = 0
            compile_ok = True
        else:
            returncode, package_info = _compile(args, bench_root)
            built = getattr(package_info, "whl_path", None) if package_info is not None else None
            wheel_source = str(built) if built else None
            # No --source-dir means "use the installed cann_bench": nothing to build.
            compile_ok = returncode == 0 and (
                built is not None or not getattr(args, "source_dir", None)
            )
        source_dir = getattr(args, "source_dir", None)
        if source_dir:
            candidate_log = Path(source_dir) / "_compile.log"
            if candidate_log.is_file():
                build_log_path = str(candidate_log)
        _write_compile_handoff(
            args,
            bench_root,
            wheel_source,
            build_log_path=build_log_path,
            returncode=returncode,
            compile_ok=compile_ok,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        return returncode

    return _run_stage(args, "compile", body, stage_index=stage_index, stage_count=stage_count)


def _correctness_stage(
    args: argparse.Namespace,
    bench_root: str,
    filter_prefix: Optional[str],
    *,
    stage_index: int = 1,
    stage_count: int = 1,
) -> Tuple[int, List[EvalOperatorResult], Optional[dict], Optional[Dict[str, Path]]]:
    """Run the correctness stage.

    Returns ``(returncode, operator_results, report_payload, report_paths)``.
    The return code is 0 whenever the correctness report was written; failed
    candidate cases are evidence, not a stage failure (the adapter classifies
    from the report).
    """
    state: Dict[str, Any] = {
        "results": None,
        "payload": None,
        "report_paths": None,
    }

    def body(started_at: str, started_monotonic: float) -> int:
        timeout = _stage_timeout(args, "correctness")
        cfg = _make_config(args, bench_root, enable_profiler=False, timeout_per_operator=timeout)
        matched = _install_or_scan(args, cfg, wheel=getattr(args, "wheel", None))
        rel_paths = _operator_rel_paths(matched, bench_root, selected=args.selected_operators)
        cases = _load_cases(args, bench_root, rel_paths, filter_prefix=filter_prefix)
        operator_results = _evaluate_cases(
            args, cfg, cases, enable_profiler=False, timeout_per_operator=timeout,
        )
        payload, _, report_paths = _save_report(
            args, cfg, operator_results, stage="correctness", contains_performance=False,
        )
        _write_correctness_handoff(
            args,
            bench_root,
            operator_results,
            payload,
            report_paths,
            returncode=0,
            wheel=_stage_wheel_source(args),
            inner_timeout_per_case=timeout,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        state.update(results=operator_results, payload=payload, report_paths=report_paths)
        return 0

    returncode = _run_stage(
        args, "correctness", body, stage_index=stage_index, stage_count=stage_count,
    )
    return returncode, state["results"] or [], state["payload"], state["report_paths"]


def _performance_stage(
    args: argparse.Namespace,
    bench_root: str,
    filter_prefix: Optional[str],
    correctness_ops: Optional[List[EvalOperatorResult]],
    *,
    correctness_handoff_used: bool,
    respect_no_merge: bool,
    stage_index: int = 1,
    stage_count: int = 1,
) -> Tuple[int, Optional[dict]]:
    """Run the performance stage.

    Returns ``(returncode, final_report_payload)``; the return code is 0 whenever
    the performance report was written (see the module docstring). The final
    report payload is ``None`` when no merge happened.
    """
    state: Dict[str, Any] = {"final_payload": None}

    def body(started_at: str, started_monotonic: float) -> int:
        timeout = _stage_timeout(args, "performance")
        cfg = _make_config(args, bench_root, enable_profiler=True, timeout_per_operator=timeout)
        matched = _install_or_scan(args, cfg, wheel=getattr(args, "wheel", None))
        allowlist = _passed_case_keys(correctness_ops) if correctness_ops is not None else None
        if allowlist is not None:
            print(f"[staged_eval] performance: {len(allowlist)} correctness-passed cases")
        selection = [int(value) for value in getattr(args, "case_ids", None) or []]
        if not selection and getattr(args, "case_id", None) is not None:
            selection = [int(args.case_id)]
        case_ids = set(selection) if selection else None
        case_filter = sorted(selection) if selection else None
        rel_paths = _operator_rel_paths(matched, bench_root, selected=args.selected_operators)
        cases = _load_cases(
            args,
            bench_root,
            rel_paths,
            filter_prefix=filter_prefix,
            allowlist=allowlist,
            case_ids=case_ids,
        )
        performance_ops = _evaluate_cases(
            args, cfg, cases, enable_profiler=True, timeout_per_operator=timeout,
        )
        performance_payload, _, performance_report_paths = _save_report(
            args, cfg, performance_ops, stage="performance", contains_performance=True,
        )

        # A filtered performance invocation is bounded recovery/gate evidence, not
        # a full-task performance report. Mark the report and never merge it into a
        # complete final report.
        if respect_no_merge and case_filter is not None:
            _annotate_report_json(
                performance_payload,
                performance_report_paths,
                {"case_filter": case_filter, "cross_invocation_evidence": True},
            )

        no_merge = bool(getattr(args, "no_merge", False)) if respect_no_merge else False
        skip_merge_for_filter = bool(case_filter) and respect_no_merge
        merged = correctness_ops is not None and not no_merge and not skip_merge_for_filter
        merge_skipped_reason: Optional[str] = None
        if not merged:
            if correctness_ops is None:
                merge_skipped_reason = "no_correctness_handoff"
            elif no_merge:
                merge_skipped_reason = "no_merge_flag"
            elif skip_merge_for_filter:
                merge_skipped_reason = "case_filter"
        final_report_paths: Optional[Dict[str, Path]] = None
        if merged:
            merged_ops = _merge_results(correctness_ops, performance_ops)
            final_payload, _, final_report_paths = _save_report(
                args, cfg, merged_ops, stage="final", contains_performance=True,
            )
            state["final_payload"] = final_payload

        _write_performance_handoff(
            args,
            bench_root,
            performance_ops,
            performance_report_paths,
            final_report_paths=final_report_paths,
            merged=merged,
            merge_skipped_reason=merge_skipped_reason,
            returncode=0,
            wheel=_stage_wheel_source(args),
            case_filter=case_filter,
            inner_timeout_per_case=timeout,
            correctness_handoff_used=correctness_handoff_used,
            started_at=started_at,
            started_monotonic=started_monotonic,
        )
        return 0

    returncode = _run_stage(
        args, "performance", body, stage_index=stage_index, stage_count=stage_count,
    )
    return returncode, state["final_payload"]


def _merge_stage(
    args: argparse.Namespace,
    bench_root: str,
    *,
    stage_index: int = 1,
    stage_count: int = 1,
) -> int:
    def body(started_at: str, started_monotonic: float) -> int:
        if not getattr(args, "handoff_dir", None):
            raise SystemExit("--stage merge requires --handoff-dir")
        correctness_handoff = _read_handoff_json(args, CORRECTNESS_HANDOFF, required=True)
        performance_handoff = _read_handoff_json(args, PERFORMANCE_HANDOFF, required=True)
        if performance_handoff.get("case_filter"):
            raise RuntimeError(
                "performance handoff 带 case_filter（有界补测/gate），不能合并为完整报告"
            )
        correctness_ops = _restore_operator_results(
            _handoff_operators(correctness_handoff, CORRECTNESS_HANDOFF)
        )
        performance_ops = _restore_operator_results(
            _handoff_operators(performance_handoff, PERFORMANCE_HANDOFF)
        )
        timeout = _stage_timeout(args, "performance")
        cfg = _make_config(args, bench_root, enable_profiler=True, timeout_per_operator=timeout)
        merged_ops = _merge_results(correctness_ops, performance_ops)
        _, _, final_report_paths = _save_report(
            args, cfg, merged_ops, stage="final", contains_performance=True,
        )
        _update_performance_handoff_after_merge(
            args, final_report_paths["json"],
            started_at=started_at, started_monotonic=started_monotonic,
        )
        return 0

    return _run_stage(args, "merge", body, stage_index=stage_index, stage_count=stage_count)


def _update_performance_handoff_after_merge(
    args: argparse.Namespace,
    final_report_path: Path,
    *,
    started_at: str,
    started_monotonic: float,
) -> None:
    """Record the merged final report in the existing performance handoff."""
    path = _handoff_path(args, PERFORMANCE_HANDOFF)
    if path is None or not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["merged"] = True
    payload["merge_skipped_reason"] = None
    payload["final_report_path"] = str(final_report_path)
    payload["merged_at"] = _utc_now_iso()
    payload["merge_elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
    payload.setdefault("merge_started_at", started_at)
    _write_json_atomic(path, payload)


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------


def _run_all(args: argparse.Namespace, bench_root: str, filter_prefix: Optional[str]) -> int:
    print("[staged_eval] stage 1/3: compile")
    compile_rc = _compile_stage(args, bench_root, stage_index=1, stage_count=3)
    if compile_rc != 0:
        return compile_rc

    print("[staged_eval] stage 2/3: correctness")
    _, correctness_ops, correctness_payload, _ = _correctness_stage(
        args, bench_root, filter_prefix, stage_index=2, stage_count=3,
    )

    if args.no_perf:
        failed = int((correctness_payload or {}).get("failed_cases") or 0)
        return min(failed, 255)

    print("[staged_eval] stage 3/3: performance")
    _, final_payload = _performance_stage(
        args,
        bench_root,
        filter_prefix,
        correctness_ops,
        correctness_handoff_used=False,
        # Legacy single-invocation behavior always merges; --no-merge is a
        # performance-stage-only flag and does not change --stage all.
        respect_no_merge=False,
        stage_index=3,
        stage_count=3,
    )
    failed = int((final_payload or {}).get("failed_cases") or 0)
    return min(failed, 255)


def _performance_only(
    args: argparse.Namespace,
    bench_root: str,
    filter_prefix: Optional[str],
) -> int:
    correctness_handoff = _read_handoff_json(args, CORRECTNESS_HANDOFF)
    correctness_ops: Optional[List[EvalOperatorResult]] = None
    if correctness_handoff is not None:
        correctness_ops = _restore_operator_results(
            _handoff_operators(correctness_handoff, CORRECTNESS_HANDOFF)
        )
    returncode, _ = _performance_stage(
        args,
        bench_root,
        filter_prefix,
        correctness_ops,
        correctness_handoff_used=correctness_handoff is not None,
        respect_no_merge=True,
    )
    return returncode


def run(args: argparse.Namespace) -> int:
    from .utils.path_resolver import resolve_task_dir

    if args.bench_name != "cann":
        raise SystemExit("staged_eval currently supports --bench-name cann only")
    if args.device != "npu":
        raise SystemExit("staged_eval currently supports NPU evaluation only")

    stage = getattr(args, "stage", "all") or "all"
    project_root = get_project_root()
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, project_root)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    if stage == "all":
        return _run_all(args, bench_root, filter_prefix)
    if stage == "compile":
        print("[staged_eval] stage: compile")
        return _compile_stage(args, bench_root)
    if stage == "correctness":
        print("[staged_eval] stage: correctness")
        return _correctness_stage(args, bench_root, filter_prefix)[0]
    if stage == "performance":
        print("[staged_eval] stage: performance")
        return _performance_only(args, bench_root, filter_prefix)
    if stage == "merge":
        print("[staged_eval] stage: merge")
        return _merge_stage(args, bench_root)
    raise SystemExit(f"unknown --stage: {stage}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cann-bench in compile/correctness/performance stages")
    parser.add_argument("--bench-name", default="cann")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--task-dir", default="tasks")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--case-id", type=int, default=None)
    parser.add_argument("--selected-operators", nargs="*", default=None,
                        help="仅评测指定算子（匹配 name / function_name / rel_path / 目录名，大小写不敏感）")
    parser.add_argument("--device", choices=["npu"], default="npu")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--processes-per-card", type=int, default=2)
    parser.add_argument("--timeout-per-operator", type=int, default=300)
    parser.add_argument("--correctness-timeout-per-case", type=int, default=None,
                        help="correctness 阶段单 case 内层超时（秒）；缺省回退 --timeout-per-operator")
    parser.add_argument("--performance-timeout-per-case", type=int, default=None,
                        help="performance 阶段单 case 内层超时（秒）；缺省回退 --timeout-per-operator")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--eval-code", default=None)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--profiler-level", choices=["Level1", "Level2"], default="Level1")
    parser.add_argument("--perf-metric-strategy", default=None)
    parser.add_argument("--torch-op-guard-mode", choices=["off", "warn", "block"], default=None)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--no-iterative-compile", action="store_true")
    parser.add_argument("--agent-skill", default="", help="评测对象 Agent/Skill 标签，写入报告元信息")
    parser.add_argument("--base-model", default="", help="评测对象 BaseModel 标签，写入报告元信息")
    parser.add_argument("--harness", default="", help="评测对象 Harness 标签，写入报告元信息")

    # Staged-entry contract (defaults keep the legacy single-invocation behavior).
    parser.add_argument("--stage", choices=list(VALID_STAGES), default="all",
                        help="只运行指定阶段；默认 all = 编译/精度/性能三阶段一次跑完")
    parser.add_argument("--wheel", default=None,
                        help="安装预编译的 cann_bench wheel，跳过编译")
    parser.add_argument("--handoff-dir", default=None,
                        help="跨阶段产物目录（compile.json/correctness.json/performance.json）")
    parser.add_argument("--case-ids", type=int, nargs="+", default=None,
                        help="performance 阶段仅评测这些 case 编号（用于缺失 case 的有界补测）")
    parser.add_argument("--stage-progress", default=None,
                        help="可信阶段进度 JSON 路径（阶段开始/结束时原子替换写入）")
    parser.add_argument("--no-merge", action="store_true",
                        help="performance 阶段不合并 correctness 结果（用于有界性能 gate）")
    return parser


def main() -> int:
    return run(create_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
