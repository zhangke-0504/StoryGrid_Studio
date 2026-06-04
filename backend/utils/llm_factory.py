"""Centralized factory for LangChain ChatOpenAI clients.

Reads `config/openai/config.yaml` and constructs a ChatOpenAI. The HTTP layer
is left to the openai SDK defaults — system network stack (global VPN) handles
proxying. Retries for transient errors are handled at the tool layer via
``utils.retry.async_retry``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError


RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
	APIConnectionError,
	APITimeoutError,
	RateLimitError,
	InternalServerError,
	httpx.ConnectError,
	httpx.ReadError,
	httpx.WriteError,
	httpx.RemoteProtocolError,
	httpx.PoolTimeout,
	httpx.ConnectTimeout,
	httpx.ReadTimeout,
)


def with_llm_retry(runnable: Runnable, max_attempts: int = 3) -> Runnable:
	"""Wrap any Runnable (chat model, structured-output chain, ...) with a
	LangChain-level retry that survives transient connection / timeout errors
	from the openai stack."""
	return runnable.with_retry(
		retry_if_exception_type=RETRY_EXCEPTIONS,
		stop_after_attempt=max_attempts,
		wait_exponential_jitter=True,
	)


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    markers = {"pyproject.toml", ".git"}
    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            return Path.cwd()
        current = current.parent


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_config_path(config_path: str) -> Path:
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_openai_config(config_path: str = "config/openai/config.yaml") -> Dict[str, Any]:
    resolved = _resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not config:
        raise ValueError(f"OpenAI 配置为空: {resolved}")
    return config


def build_chat_openai(
    config_path: str = "config/openai/config.yaml",
    *,
    temperature: Optional[float] = 0,
    model: Optional[str] = None,
    **overrides: Any,
) -> ChatOpenAI:
    """Build a ChatOpenAI configured by the project's openai yaml.

    `overrides` are forwarded directly to `ChatOpenAI(...)` and take
    precedence over the config values.
    """
    config = load_openai_config(config_path)

    kwargs: Dict[str, Any] = {
        "model": model or config.get("model", "gpt-4.1-mini"),
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if config.get("timeout") is not None:
        kwargs["timeout"] = float(config["timeout"])
    if config.get("max_retries") is not None:
        kwargs["max_retries"] = int(config["max_retries"])
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)
