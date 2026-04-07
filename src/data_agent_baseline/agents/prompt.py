from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


# ReAct agent 的基础 system prompt，定义角色、工具使用边界和输出格式约束。
REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a task from a public dataset. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call the `answer` tool.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.

Keep reasoning concise and grounded in the observed data.
""".strip()

# 给模型看的标准输出示例，用来强化 JSON action 协议和结束动作格式。
RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table.","action":"answer","action_input":{"columns":["average_long_shots"],"rows":[["63.5"]]}}
```
""".strip()


# 组合 system prompt、工具说明和输出示例，形成每轮请求的完整系统提示。
def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    # 允许调用方覆盖默认系统提示词，未提供时回退到内置的 ReAct prompt。
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


# 为当前任务构造用户提示，包含题目本身和最终回答方式的提醒。
def build_task_prompt(task: PublicTask) -> str:
    return (
        f"Question: {task.question}\n"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


# 把工具返回的 observation 序列化为文本，作为下一轮推理的输入上下文。
def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
