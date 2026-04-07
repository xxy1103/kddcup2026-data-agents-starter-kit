from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 项目根目录用于解析默认数据路径和相对配置路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 默认公开数据集目录。
def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


# 默认运行产物目录。
def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


# 数据集相关配置。
@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


# Agent 相关配置，包括模型信息和 ReAct 步数。
@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    api_key_env: str | None = None
    max_steps: int = 16
    temperature: float = 0.0
    enable_thinking: bool = False


def _optional_string_value(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    return normalized or None


def _bool_value(raw_value: object, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got: {raw_value!r}")


def _dotenv_value(dotenv_path: Path, env_var_name: str) -> str | None:
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        if name.strip() != env_var_name:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.strip()

    return None


def _resolve_api_key(raw_api_key: object, raw_api_key_env: object, default_api_key: str) -> tuple[str, str | None]:
    api_key = str(raw_api_key if raw_api_key is not None else default_api_key).strip()
    api_key_env = _optional_string_value(raw_api_key_env)
    if api_key or api_key_env is None:
        return api_key, api_key_env

    dotenv_api_key = _dotenv_value(PROJECT_ROOT / ".env", api_key_env)
    return (dotenv_api_key or "").strip(), api_key_env


# 运行时配置，包括输出目录、并发度和任务超时。
@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


# 顶层应用配置，把 dataset / agent / run 三组配置聚合在一起。
@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


# 把 YAML 中的路径字段解析成 Path；相对路径默认相对于项目根目录。
def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


# 从 YAML 配置文件加载应用配置，并对缺省值和相对路径做统一处理。
def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    api_key, api_key_env = _resolve_api_key(
        agent_payload.get("api_key", agent_defaults.api_key),
        agent_payload.get("api_key_env"),
        agent_defaults.api_key,
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=api_key,
        api_key_env=api_key_env,
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        enable_thinking=_bool_value(agent_payload.get("enable_thinking"), agent_defaults.enable_thinking),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        # 空字符串视为未指定 run_id，交由 runner 自动生成。
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
