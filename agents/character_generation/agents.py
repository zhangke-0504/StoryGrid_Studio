import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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

from config.prompt.create_character.prompt import (  # noqa: E402
    CHARACTER_SYSTEM_PROMPT,
    DEFAULT_STYLE_NAME,
    build_character_user_prompt,
)
from tools.prompt_generation.character_prompt_generation import (  # noqa: E402
    generate_subject_prompts_tool,
)
from tools.image_generation.doubao.image_generation import (  # noqa: E402
    generate_doubao_images_tool,
)
from tools.image_generation.doubao.image_reconstruction import (  # noqa: E402
    audit_and_reconstruct_images_tool,
)


CHARACTER_REACT_SYSTEM_PROMPT = """
你是一个主体设定 ReAct 智能体，负责基于已有主体清单补全设定图提示词、生成图片，并在需要时做图片审核重构。

执行规则：
1. 严格遵循 ReAct，每一轮最多调用 1 个工具。
2. 不允许改写或丢失已有主体的 name、setting、appearance、type、voice_id。
3. 当主体缺少 image_prompt 时，优先调用 generate_subject_prompts_tool。
4. 当 request.generate_images=true 且主体已有 image_prompt 但缺少 image_url 时，调用 generate_doubao_images_tool。
5. 当 request.audit_images=true 且主体已有 image_url 时，调用 audit_and_reconstruct_images_tool。
6. 不要伪造图片链接、审核结果或重构结果，必须来自工具返回。
7. 全部必要步骤完成后，只输出最终 JSON，格式必须满足 CharacterGenerationResult schema，禁止输出额外解释。
""".strip()


def _resolve_path(path: str) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved)


def _load_config(config_path: str = "config/openai/config.yaml") -> Dict[str, Any]:
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    with resolved_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(f"OpenAI 配置为空: {resolved_path}")
    return config


def _build_llm(config_path: str = "config/openai/config.yaml") -> ChatOpenAI:
    config = _load_config(config_path)
    return ChatOpenAI(
        model=config.get("model", "gpt-4.1-mini"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        temperature=0,
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("agent 未返回可解析的 JSON")

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _parse_tool_message_content(content: Any) -> Any:
    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return stripped
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return content


def _format_generate_subject_prompts_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    subjects = payload.get("subjects", [])
    if not subjects:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    lines: list[str] = []
    for subject in subjects:
        name = subject.get("name", "<unknown>")
        image_prompt = subject.get("image_prompt", "")
        lines.append(f"- {name}:\n{image_prompt}")
    return "\n\n".join(lines)


def _format_generate_images_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    subjects = payload.get("subjects", [])
    if not subjects:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    lines: list[str] = []
    for subject in subjects:
        name = subject.get("name", "<unknown>")
        image_url = subject.get("image_url") or "<empty>"
        image_error = subject.get("image_error")
        if image_error:
            lines.append(f"- {name}: 生成失败 -> {image_error}")
        else:
            lines.append(f"- {name}: {image_url}")
    return "\n".join(lines)


def _format_audit_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    subjects = payload.get("subjects", [])
    if not subjects:
        return json.dumps(payload, ensure_ascii=False, indent=2)

    blocks: list[str] = []
    for subject in subjects:
        name = subject.get("name", "<unknown>")
        audit_points = subject.get("audit_points") or []
        understanding_result = subject.get("understanding_result") or ""
        reconstruction_prompt = subject.get("reconstruction_prompt") or ""
        reconstructed_image_url = subject.get("reconstructed_image_url") or ""
        final_image_url = subject.get("final_image_url") or ""
        error = subject.get("error")

        parts = [f"- 主体: {name}"]
        if error:
            parts.append(f"  错误: {error}")
        if audit_points:
            parts.append("  审核分析:")
            parts.extend(f"    - {point}" for point in audit_points)
        if understanding_result:
            parts.append("  图片理解+审核:")
            parts.append(f"    {understanding_result}")
        if reconstruction_prompt:
            parts.append("  提示词重构:")
            parts.append(f"    {reconstruction_prompt}")
        if reconstructed_image_url:
            parts.append(f"  重构图片URL: {reconstructed_image_url}")
        if final_image_url and final_image_url != reconstructed_image_url:
            parts.append(f"  最终图片URL: {final_image_url}")
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


def _format_tool_result(tool_name: str, content: Any) -> str:
    payload = _parse_tool_message_content(content)
    if tool_name == "generate_subject_prompts_tool":
        return _format_generate_subject_prompts_result(payload)
    if tool_name == "generate_doubao_images_tool":
        return _format_generate_images_result(payload)
    if tool_name == "audit_and_reconstruct_images_tool":
        return _format_audit_result(payload)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)


class CharacterItem(BaseModel):
    name: str = Field(..., description="主体名字")
    setting: str = Field(..., description="主体设定")
    appearance: str = Field(..., description="主体外观描述")
    type: int = Field(..., description="主体类型：0人物，1物品，2场景")
    voice_id: str = Field(..., description="人物音色 ID，非人物为空字符串")
    image_prompt: Optional[str] = Field(default=None, description="主体设定图提示词，未生成时为空")
    image_url: Optional[str] = Field(default=None, description="主体设定图 URL，未生成时为空")
    image_error: Optional[str] = Field(default=None, description="主体设定图生成错误，成功时为空")
    audit_points: list[str] = Field(default_factory=list, description="图片审核重点")
    understanding_result: Optional[str] = Field(default=None, description="图片理解审核结果")
    should_reconstruct: bool = Field(default=False, description="是否需要图生图重构")
    reconstruction_prompt: Optional[str] = Field(default=None, description="图生图重构提示词")
    repair_focus: list[str] = Field(default_factory=list, description="图片重构重点")
    reconstructed_image_url: Optional[str] = Field(default=None, description="图生图重构后的图片 URL")
    final_image_url: Optional[str] = Field(default=None, description="最终可用的图片 URL")


class CharacterGenerationResult(BaseModel):
    characters: list[CharacterItem]


class CharacterGenerationInput(BaseModel):
    theme: str = Field(..., description="故事大纲")
    language: str = Field(default="zh-CN", description="输出语言")
    style_name: str = Field(default=DEFAULT_STYLE_NAME, description="风格名")
    custom_characters: Optional[Dict[Any, Any]] = Field(default=None, description="自定义主体数量约束")
    generate_images: bool = Field(default=True, description="是否在生成提示词后继续生成主体设定图")
    audit_images: bool = Field(default=True, description="是否对已生成图片执行审核与图生图重构")
    image_model: Optional[str] = Field(default=None, description="豆包生图模型名，默认取配置文件中的 image_model")
    image_size: str = Field(default="2K", description="主体设定图尺寸")
    image_watermark: bool = Field(default=False, description="主体设定图是否带水印")


class CharacterGenerationAgent:
    def __init__(
        self,
        config_path: str = "config/openai/config.yaml",
        doubao_config_path: str = "config/doubao/config.yaml",
    ):
        self.config_path = _resolve_path(config_path)
        self.doubao_config_path = _resolve_path(doubao_config_path)
        self.llm = _build_llm(self.config_path)
        self.lang2voice_id = {
            "zh-CN": {
                "精英青年音色": "d474d67b-08a9-4b5d-a0f9-e318433ae634",
                "少女音色": "6d62022f-c1bf-4981-8080-432b7eec5383",
            },
            "en-US": {
                "Gentle-voiced man": "7c52e822-db6a-4690-818f-34350b218bfe",
                "Sweet Girl": "6c65ca82-6792-4b70-be17-47cb42bd2397",
            },
            "ja-JP": {
                "Kind Lady": "60109a1a-7b88-4603-9797-acdeac9f91d4",
                "Innocent Boy": "8a5ff5af-9ac1-4241-8ddb-8e3dc8961e49",
            },
        }
        self.agent = create_agent(
            model=self.llm,
            tools=[
                generate_subject_prompts_tool,
                generate_doubao_images_tool,
                audit_and_reconstruct_images_tool,
            ],
            system_prompt=CHARACTER_REACT_SYSTEM_PROMPT,
        )

    async def _generate_base_characters(self, request: CharacterGenerationInput) -> CharacterGenerationResult:
        voice_id_dict = self.lang2voice_id.get(request.language, self.lang2voice_id["zh-CN"])
        structured_llm = self.llm.with_structured_output(CharacterGenerationResult)
        return await structured_llm.ainvoke(
            [
                {"role": "system", "content": CHARACTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_character_user_prompt(
                        theme=request.theme,
                        language=request.language,
                        voice_id_dict=voice_id_dict,
                        custom_characters=request.custom_characters,
                        style_name=request.style_name,
                    ),
                },
            ]
        )

    def _build_react_user_prompt(
        self,
        request: CharacterGenerationInput,
        base_result: CharacterGenerationResult,
    ) -> str:
        return """请基于下面的 request 和 current_characters 继续完成主体设定生成。

硬性要求：
1. 保留 current_characters 中所有现有字段，不允许删除主体或篡改主体基础信息。
2. 如果有主体缺少 image_prompt，必须先调用 generate_subject_prompts_tool，一次性补齐所有缺失主体的提示词。
3. 如果 request.generate_images=true，并且主体已有 image_prompt 但没有 image_url，调用 generate_doubao_images_tool。
4. 如果 request.audit_images=true，并且主体已有 image_url，调用 audit_and_reconstruct_images_tool。
5. 每轮最多调用一个工具；工具调用完成后再决定下一步。
6. 完成后只输出最终 JSON，不要输出解释。

request:
{request_json}

current_characters:
{characters_json}
""".strip().format(
            request_json=request.model_dump_json(indent=2, ensure_ascii=False),
            characters_json=base_result.model_dump_json(indent=2, ensure_ascii=False),
        )

    async def _normalize_agent_output(
        self,
        request: CharacterGenerationInput,
        base_result: CharacterGenerationResult,
        agent_text: str,
    ) -> CharacterGenerationResult:
        structured_llm = self.llm.with_structured_output(CharacterGenerationResult)
        return await structured_llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": "你负责将 ReAct agent 的最终结果整理成 CharacterGenerationResult schema，只输出符合 schema 的 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        "请将下面内容整理为 CharacterGenerationResult。\n\n"
                        f"request:\n{request.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        f"base_result:\n{base_result.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        f"react_agent_final_text:\n{agent_text}"
                    ),
                },
            ]
        )

    async def _run_react_enrichment(
        self,
        request: CharacterGenerationInput,
        base_result: CharacterGenerationResult,
        verbose: bool = False,
    ) -> CharacterGenerationResult:
        user_prompt = self._build_react_user_prompt(request, base_result)
        final_text = ""
        seen_tool_messages: set[str] = set()

        async for chunk in self.agent.astream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            stream_mode="values",
        ):
            latest_message = chunk["messages"][-1]
            tool_calls = getattr(latest_message, "tool_calls", None) or []
            if verbose and tool_calls:
                for tool_call in tool_calls:
                    print(f"tool_call: {tool_call['name']}")

            if verbose and latest_message.__class__.__name__ == "ToolMessage":
                tool_message_id = getattr(latest_message, "id", None) or getattr(latest_message, "tool_call_id", None)
                if tool_message_id not in seen_tool_messages:
                    seen_tool_messages.add(tool_message_id)
                    tool_name = getattr(latest_message, "name", "unknown_tool")
                    formatted_result = _format_tool_result(tool_name, getattr(latest_message, "content", ""))
                    print(f"tool_result: {tool_name}\n{formatted_result}")

            message_text = _message_text(latest_message)
            if message_text:
                final_text = message_text

        if not final_text:
            return base_result

        try:
            return CharacterGenerationResult.model_validate(_extract_json_object(final_text))
        except Exception:
            return await self._normalize_agent_output(request, base_result, final_text)

    async def ainvoke(
        self,
        payload: Dict[str, Any] | CharacterGenerationInput,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        request = payload if isinstance(payload, CharacterGenerationInput) else CharacterGenerationInput.model_validate(payload)
        base_result = await self._generate_base_characters(request)
        result = await self._run_react_enrichment(request, base_result, verbose=verbose)
        return result.model_dump(mode="json")

    def invoke(
        self,
        payload: Dict[str, Any] | CharacterGenerationInput,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(payload, verbose=verbose))
        raise RuntimeError("当前已存在事件循环，请改用 ainvoke() 调用 CharacterGenerationAgent")


def main():
    agent = CharacterGenerationAgent()
    sample_request = {
        "theme": "在一个被永夜笼罩的海港小镇，少女灯塔守望者意外发现一封来自未来的求救信，于是决定踏上寻找失踪父亲的旅程。",
        "language": "zh-CN",
        "style_name": "奇幻动画电影",
        "generate_images": True,
        "audit_images": True,
        "image_size": "2K",
        "image_watermark": False,
    }
    result = agent.invoke(sample_request, verbose=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()