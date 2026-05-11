from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class ExecutionMode(str, Enum):
    WORKSPACE = "workspace"
    INSTALLED = "installed"


ENV_EXECUTION_MODE = "CONTEXTUAL_DRAG_EXECUTION_MODE"
ENV_REPO_ROOT = "CONTEXTUAL_DRAG_REPO_ROOT"
ENV_DATA_ROOT = "CONTEXTUAL_DRAG_DATA_ROOT"
ENV_OUTPUT_ROOT = "CONTEXTUAL_DRAG_OUTPUT_ROOT"
ENV_PROMPT_TEMPLATE_ROOT = "CONTEXTUAL_DRAG_PROMPT_TEMPLATE_ROOT"
ENV_SPLIT_ROOT = "CONTEXTUAL_DRAG_SPLIT_ROOT"
ENV_TIKTOKEN_ENCODINGS_BASE = "TIKTOKEN_ENCODINGS_BASE"


def _is_repo_root(path: Path) -> bool:
    markers = [
        path / "README.md",
        path / "data_generation",
        path / "utils" / "general_inference",
        path / "prompt_templates",
    ]
    return all(marker.exists() for marker in markers)


def find_repo_root(start: Path | None = None) -> Path | None:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if _is_repo_root(candidate):
            return candidate
    return None


def resolve_execution_mode(explicit: str | None = None, start: Path | None = None) -> ExecutionMode:
    if explicit:
        return ExecutionMode(explicit)
    env_value = os.environ.get(ENV_EXECUTION_MODE)
    if env_value:
        return ExecutionMode(env_value)
    return ExecutionMode.WORKSPACE if find_repo_root(start=start) else ExecutionMode.INSTALLED


def get_repo_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    env_root = os.environ.get(ENV_REPO_ROOT)
    if env_root:
        return Path(env_root).expanduser().resolve()
    repo_root = find_repo_root(start=start)
    mode = resolve_execution_mode(execution_mode, start=start)
    if repo_root is None and required and mode == ExecutionMode.WORKSPACE:
        raise RuntimeError("Workspace mode requires a checked-out repository root, but none was detected.")
    return repo_root


def _resolve_workspace_or_env(
    env_name: str,
    repo_relative: str,
    execution_mode: str | None = None,
    start: Path | None = None,
    required: bool = False,
    description: str = "path",
) -> Path | None:
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value).expanduser().resolve()

    mode = resolve_execution_mode(execution_mode, start=start)
    repo_root = get_repo_root(execution_mode=execution_mode, start=start)
    if mode == ExecutionMode.WORKSPACE and repo_root is not None:
        return (repo_root / repo_relative).resolve()

    if required:
        raise RuntimeError(
            f"{description} is not available in installed mode unless you pass it explicitly "
            f"or set {env_name}."
        )
    return None


def get_data_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        ENV_DATA_ROOT, "data", execution_mode=execution_mode, start=start, required=required, description="data root"
    )


def get_output_root(execution_mode: str | None = None, start: Path | None = None, required: bool = False) -> Path | None:
    return _resolve_workspace_or_env(
        ENV_OUTPUT_ROOT,
        "outputs",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="output root",
    )


def get_prompt_template_root(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    return _resolve_workspace_or_env(
        ENV_PROMPT_TEMPLATE_ROOT,
        "prompt_templates",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="prompt template root",
    )


def get_default_split_root(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    return _resolve_workspace_or_env(
        ENV_SPLIT_ROOT,
        "data/big_math_rl_verified/train_split/detailed_splits",
        execution_mode=execution_mode,
        start=start,
        required=required,
        description="split root",
    )


def resolve_split_file(
    split: str,
    split_root: str | Path | None = None,
    execution_mode: str | None = None,
    start: Path | None = None,
) -> list[Path]:
    split_names = split.split("+")
    for item in split_names:
        if item not in {"sft", "rl", "val"}:
            raise ValueError("Data split must be one of sft, rl, val or a '+' combination of them.")

    if split_root is not None:
        root = Path(split_root).expanduser().resolve()
    else:
        root = get_default_split_root(execution_mode=execution_mode, start=start, required=True)
        assert root is not None

    return [root / f"{item}_ids.json" for item in split_names]


def get_tiktoken_encodings_base(
    execution_mode: str | None = None, start: Path | None = None, required: bool = False
) -> Path | None:
    env_value = os.environ.get(ENV_TIKTOKEN_ENCODINGS_BASE)
    if env_value:
        return Path(env_value).expanduser().resolve()

    mode = resolve_execution_mode(execution_mode, start=start)
    repo_root = get_repo_root(execution_mode=execution_mode, start=start)
    if mode == ExecutionMode.WORKSPACE and repo_root is not None:
        return repo_root / "misc" / "gpt-oss-harmony" / "encodings"
    if required:
        raise RuntimeError(
            "TIKTOKEN encodings path is not available in installed mode unless you set "
            f"{ENV_TIKTOKEN_ENCODINGS_BASE}."
        )
    return None
