from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


# trace.json 中记录的单步执行信息。
@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    # 转为普通字典，便于 JSON 序列化写出。
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Agent 在一次任务执行过程中的可变状态。
@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


# 单个任务运行结束后的最终结果。
@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    # 只有拿到答案且没有失败原因时，才视为任务成功。
    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    # 转为 runner 可直接落盘的字典结构。
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }
