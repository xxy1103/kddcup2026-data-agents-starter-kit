from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI


# 聊天模型所需的最小消息结构。
@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


# 模型单步输出解析后的结构化结果。
@dataclass(frozen=True, slots=True)
class ModelStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str


# 模型适配器协议，便于替换不同后端或测试桩实现。
class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError


# 基于 OpenAI 兼容接口的模型适配器。
class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    # 发送完整消息列表给模型，并返回第一条候选文本。
    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content


# 用于测试的脚本化模型适配器，按预设顺序逐条返回响应。
class ScriptedModelAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def complete(self, messages: list[ModelMessage]) -> str:
        # 这里不依赖真实上下文，只消费预先写好的响应脚本。
        del messages
        if not self._responses:
            raise RuntimeError("No scripted model responses remaining.")
        return self._responses.pop(0)
