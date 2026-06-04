import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Type

import yaml
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field


class GPTClient:
    """OpenAI / DeepSeek API 客户端工具类。"""

    def __init__(self, config_path: Optional[str] = None):
        if not config_path:
            config_path = "config/openai/config.yaml"

        if not os.path.isabs(config_path):
            if getattr(sys, "frozen", False):
                project_root = os.path.dirname(os.path.dirname(sys.executable))
                config_path = os.path.join(project_root, config_path)
            else:
                project_root = self._find_project_root()
                config_path = os.path.join(project_root, config_path)

        self.config_path = config_path

        parent = os.path.dirname(config_path)
        os.makedirs(parent, exist_ok=True)
        if not os.path.exists(config_path):
            placeholder = {
                "api_key": "xxxxxx",
                "model": "gpt-4.1-mini",
            }
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(placeholder, file, allow_unicode=True, sort_keys=False)

        try:
            cfg = self._load_config(config_path)
        except Exception as exc:
            print(f"OpenAI: failed to load config from {config_path}: {exc}")
            raise
        self.config_info = cfg
        self.api_key = self.config_info.get("api_key")
        self.model = self.config_info.get("model", "gpt-4.1-mini")
        self.base_url = self.config_info.get("base_url", None)

        if self.api_key:
            os.environ["OPENAI_API_KEY"] = str(self.api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _find_project_root(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        markers = ("pyproject.toml", ".git")

        while True:
            if any(os.path.exists(os.path.join(current_dir, marker)) for marker in markers):
                return current_dir

            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir

        return os.getcwd()

    def _mask_secret(self, secret: Optional[str]) -> str:
        if not secret:
            return "<empty>"
        if len(secret) <= 8:
            return "*" * len(secret)
        return f"{secret[:4]}...{secret[-4:]}"

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                if not config:
                    raise ValueError("配置为空")
                return config
        except Exception as exc:
            raise RuntimeError(f"加载配置失败: {exc}") from exc

    async def async_non_stream_response(
        self,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        text_format: Optional[Type[BaseModel]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        if messages is None:
            messages_payload = [
                {"role": "system", "content": instructions or ""},
                {"role": "user", "content": prompt or ""},
            ]
        else:
            messages_payload = messages

        extra_args = {}
        if text_format:
            extra_args["response_format"] = text_format

        response = await self.async_client.chat.completions.parse(
            model=self.model,
            messages=messages_payload,
            **extra_args,
        )

        content = response.choices[0].message.content

        if text_format:
            try:
                json_dict = json.loads(content)
            except Exception as exc:
                raise RuntimeError(f"解析 JSON 失败: {content}") from exc
            return text_format.model_validate(json_dict)

        return content

    async def async_stream_response(
        self,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        text_format: Optional[Type[BaseModel]] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncIterable[Any]:
        if messages is None:
            messages_payload = [
                {"role": "system", "content": instructions or ""},
                {"role": "user", "content": prompt or ""},
            ]
        else:
            messages_payload = messages

        kwargs = {}
        if text_format:
            kwargs["response_format"] = text_format
        async with self.async_client.chat.completions.stream(
            model=self.model,
            messages=messages_payload,
            **kwargs,
        ) as stream:
            async for event in stream:
                yield event


if __name__ == "__main__":

    class TotalShotsData(BaseModel):
        total_shots: int = Field(..., description="分镜总数")
        duration_list: List[float] = Field(..., description="每个镜头时长")

    class TotalShots(BaseModel):
        total_shots: int
        duration_list: List[float]

    async def test_async_non_stream_response():
        client = GPTClient()

        instructions = """
        你是一名专业的动漫分镜时长设计师。
        任务：根据总时长60秒生成分镜方案，并输出 JSON。
        严格按照 JSON 输出如下格式（只输出 JSON，不要包含额外解释）：
        {
            \"total_shots\": 整数,
            \"duration_list\": [浮点数列表]
        }
        """

        result = await client.async_non_stream_response(
            prompt="一个中国的道士下山捉妖",
            instructions=instructions,
            text_format=TotalShots,
        )
        print("非流式结构化结果:", result)
        result_dict = result.model_dump()
        print("result_dict:", result_dict)

    async def test_multi_turn():
        client = GPTClient()
        messages = [{"role": "user", "content": "世界上最高的山是什么？"}]
        resp_text = await client.async_non_stream_response(messages=messages)
        print("Round1 assistant:", resp_text)
        messages.append({"role": "assistant", "content": resp_text})
        messages.append({"role": "user", "content": "第二高的呢？"})
        resp2 = await client.async_non_stream_response(messages=messages)
        print("Round2 assistant:", resp2)

    async def test_stream_multi_turn():
        client = GPTClient()
        messages = [{"role": "user", "content": "请输出一个短篇故事的开头。"}]
        print("流式 Round1:")
        buffer = []
        async for event in client.async_stream_response(messages=messages):
            if hasattr(event, "delta") and event.delta:
                piece = event.delta if isinstance(event.delta, str) else str(event.delta)
                buffer.append(piece)
                print(piece, end="", flush=True)
        full = "".join(buffer).strip()
        print("\n-- full:", full)

    async def main():
        await test_async_non_stream_response()

    asyncio.run(main())
