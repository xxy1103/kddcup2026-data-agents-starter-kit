from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


# ReAct agent 的运行参数，目前只控制单个任务允许的最大推理步数。
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


# 从模型响应中剥离 ```json ... ``` 或普通代码块外壳，便于后续做 JSON 解析。
def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


# 解析单个 JSON 对象，并拒绝对象后面仍然带有额外文本的响应。
def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        # 有些模型会在 JSON 后附带转义换行，这里做一次宽松清理；
        # 若清理后仍有内容，则说明输出不符合“只返回一个 JSON 对象”的协议。
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


# 把模型原始文本解析成统一的结构化步骤，供后续工具调用和 trace 记录使用。
def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    # 这里显式校验字段类型，避免后面的工具调度阶段收到畸形输入。
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    # 初始化 agent 时注入模型、工具集合以及可覆盖的 system prompt。
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        # 未显式指定时，使用 prompt.py 中定义的默认 ReAct system prompt。
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    # 把 system prompt、任务问题和历史 observation 组装成当前轮次的消息列表。
    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            # assistant 侧保留模型上一步的原始 JSON 输出，完整还原对话历史。
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            # user 侧把工具执行结果作为 observation 喂回模型，驱动下一步 ReAct 推理。
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    # ReAct 主循环：模型决定动作，工具执行动作，再把 observation 反馈回模型。
    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            # 每一步都基于当前完整上下文重新请求模型输出下一条 JSON action。
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                # observation 是下一轮推理的唯一反馈面：是否成功、调用了哪个工具、拿到了什么内容。
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                # 将 thought / action / observation 全量记录下来，便于 trace 复盘。
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    # 一旦工具返回终止信号（通常是 answer 工具），保存答案并结束循环。
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                # 无论是模型输出格式不合法，还是工具执行报错，都作为失败 observation 回灌到历史中；
                # 这样 trace.json 可以完整保留失败现场，模型在后续步数里也有机会自我修正。
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            # 超过最大步数仍未调用 answer，则视为本次任务失败。
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # 输出最终结构化结果，供 runner 写入 trace.json 和 prediction.csv。
        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
