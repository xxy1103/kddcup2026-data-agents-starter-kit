from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run import runner as runner_module
from data_agent_baseline.run.runner import TaskRunArtifacts, run_benchmark


def _create_task(input_root: Path, task_id: str, difficulty: str = "easy") -> None:
    task_dir = input_root / task_id
    (task_dir / "context").mkdir(parents=True, exist_ok=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": f"Question for {task_id}",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_benchmark_summary_includes_runtime_and_agent_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "data" / "public" / "input"
    output_root = tmp_path / "artifacts" / "runs"
    _create_task(dataset_root, "task_1")
    _create_task(dataset_root, "task_2", difficulty="hard")

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(max_steps=48, temperature=0.3),
        run=RunConfig(
            output_dir=output_root,
            run_id="summary-test-run",
            max_workers=7,
            task_timeout_seconds=321,
        ),
    )

    def fake_run_single_task(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        model=None,
        tools=None,
    ) -> TaskRunArtifacts:
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}", encoding="utf-8")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=(task_id == "task_1"),
            failure_reason=None if task_id == "task_1" else "failed",
        )

    monkeypatch.setattr(runner_module, "run_single_task", fake_run_single_task)

    run_output_dir, artifacts = run_benchmark(config=config, model=object())

    assert run_output_dir.name == "summary-test-run"
    assert len(artifacts) == 2

    summary_payload = json.loads((run_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["max_workers"] == 1
    assert summary_payload["task_timeout_seconds"] == 321
    assert summary_payload["max_steps"] == 48
    assert summary_payload["temperature"] == 0.3
    assert summary_payload["succeeded_task_count"] == 1
