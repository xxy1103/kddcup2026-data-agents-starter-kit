from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    gold_csv_path: Path
    prediction_csv_path: Path | None
    score: int
    gold_column_count: int
    prediction_column_count: int
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "gold_csv_path": str(self.gold_csv_path),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "score": self.score,
            "gold_column_count": self.gold_column_count,
            "prediction_column_count": self.prediction_column_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RunScoreSummary:
    run_id: str
    run_output_dir: Path
    score_path: Path
    task_count: int
    total_score: int
    accuracy: float
    tasks: list[TaskScore]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_output_dir": str(self.run_output_dir),
            "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_count": self.task_count,
            "total_score": self.total_score,
            "accuracy": self.accuracy,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _load_csv_columns(path: Path) -> list[list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return []

    expected_width = len(rows[0])
    columns = [[] for _ in range(expected_width)]
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != expected_width:
            raise ValueError(
                f"CSV row width mismatch in {path} at line {row_index}: "
                f"expected {expected_width}, got {len(row)}."
            )
        for column_index, value in enumerate(row):
            columns[column_index].append(value)
    return columns


def _column_signature(values: list[str]) -> tuple[str, ...]:
    # 官方规则按“列的无序值向量”匹配，因此忽略列名与行顺序，但保留重复值数量。
    return tuple(sorted(values))


def _score_task(task_id: str, gold_csv_path: Path, prediction_csv_path: Path | None) -> TaskScore:
    if prediction_csv_path is None or not prediction_csv_path.exists():
        gold_columns = _load_csv_columns(gold_csv_path)
        return TaskScore(
            task_id=task_id,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            score=0,
            gold_column_count=len(gold_columns),
            prediction_column_count=0,
            reason="prediction.csv is missing.",
        )

    try:
        gold_columns = _load_csv_columns(gold_csv_path)
        prediction_columns = _load_csv_columns(prediction_csv_path)
    except ValueError as exc:
        return TaskScore(
            task_id=task_id,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            score=0,
            gold_column_count=0,
            prediction_column_count=0,
            reason=str(exc),
        )

    gold_counter = Counter(_column_signature(column) for column in gold_columns)
    prediction_counter = Counter(_column_signature(column) for column in prediction_columns)
    missing_gold_columns = sum(
        max(required_count - prediction_counter.get(signature, 0), 0)
        for signature, required_count in gold_counter.items()
    )

    if missing_gold_columns > 0:
        return TaskScore(
            task_id=task_id,
            gold_csv_path=gold_csv_path,
            prediction_csv_path=prediction_csv_path,
            score=0,
            gold_column_count=len(gold_columns),
            prediction_column_count=len(prediction_columns),
            reason=f"{missing_gold_columns} gold column(s) are missing or mismatched.",
        )

    return TaskScore(
        task_id=task_id,
        gold_csv_path=gold_csv_path,
        prediction_csv_path=prediction_csv_path,
        score=1,
        gold_column_count=len(gold_columns),
        prediction_column_count=len(prediction_columns),
        reason=None,
    )


def resolve_score_run_dir(runs_root: Path, run_id: str | None = None) -> tuple[str, Path]:
    if run_id is not None:
        normalized = run_id.strip()
        if not normalized:
            raise ValueError("run_id must not be empty.")
        if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("run_id must be a single directory name, not a path.")
        run_output_dir = runs_root / normalized
        if not run_output_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_output_dir}")
        return normalized, run_output_dir

    candidate_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No run directories found under {runs_root}")

    # 默认 run_id 使用 UTC 时间戳，按目录名排序能稳定代表“最新一次运行”，
    # 也避免在写入 score.json 后把旧 run 的目录 mtime 刷新成最新。
    latest_run_dir = max(candidate_dirs, key=lambda path: path.name)
    return latest_run_dir.name, latest_run_dir


def score_run_outputs(*, run_output_dir: Path, gold_root: Path) -> RunScoreSummary:
    if not gold_root.is_dir():
        raise FileNotFoundError(f"Gold output directory not found: {gold_root}")

    tasks: list[TaskScore] = []
    for gold_task_dir in sorted(path for path in gold_root.iterdir() if path.is_dir()):
        task_id = gold_task_dir.name
        gold_csv_path = gold_task_dir / "gold.csv"
        if not gold_csv_path.exists():
            continue

        prediction_csv_path = run_output_dir / task_id / "prediction.csv"
        if not prediction_csv_path.exists():
            prediction_csv_path = None
        tasks.append(_score_task(task_id, gold_csv_path, prediction_csv_path))

    total_score = sum(task.score for task in tasks)
    task_count = len(tasks)
    accuracy = (total_score / task_count) if task_count > 0 else 0.0
    score_path = run_output_dir / "score.json"
    summary = RunScoreSummary(
        run_id=run_output_dir.name,
        run_output_dir=run_output_dir,
        score_path=score_path,
        task_count=task_count,
        total_score=total_score,
        accuracy=accuracy,
        tasks=tasks,
    )
    score_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
