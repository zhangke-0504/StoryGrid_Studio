import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator


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

from tools.prompt_generation.shot_prompt_generation import generate_shot_prompts_tool  # noqa: E402
from tools.video_generation.vidu.video_generation import generate_vidu_videos_tool  # noqa: E402
from agents.shared.event_emitter import emit_event  # noqa: E402


SHOT_REACT_SYSTEM_PROMPT = """
你是一个分镜与视频生成 ReAct 智能体，负责先生成多分镜脚本，再根据主体图片为每个分镜生成视频。

执行规则：
1. 严格遵循 ReAct，每一轮最多调用 1 个工具。
2. 如果当前没有分镜脚本，优先调用 generate_shot_prompts_tool。
3. 如果 request.generate_videos=true，且已经有分镜脚本，则调用 generate_vidu_videos_tool。
4. 不允许伪造分镜内容、任务 ID、视频 URL 或失败状态，必须来自工具返回。
5. 如果主体没有可用图片 URL，不要伪造图片，允许视频生成结果返回失败原因。
6. 全部必要步骤完成后，只输出最终 JSON，格式必须满足 ShotAgentFinalResult schema，禁止输出额外解释。
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
    from utils.llm_factory import build_chat_openai

    return build_chat_openai(config_path=config_path)


def _structured(llm: ChatOpenAI, schema):  # noqa: D401
    from utils.llm_factory import with_llm_retry

    return with_llm_retry(llm.with_structured_output(schema, method="function_calling"))


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


def _format_shot_prompt_tool_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    plan = payload.get("plan", {})
    shots = payload.get("shots", [])
    lines = [f"分镜规划: total_shots={plan.get('total_shots')} duration_list={plan.get('duration_list')}"]
    for shot in shots:
        lines.append(
            "\n".join(
                [
                    f"- 镜头 {shot.get('sort')}",
                    f"  时长: {shot.get('duration')}s",
                    f"  主旨: {shot.get('theme')}",
                    f"  描述: {shot.get('description')}",
                ]
            )
        )
    return "\n\n".join(lines)


def _format_vidu_tool_result(payload: Any) -> str:
    if not isinstance(payload, dict):
        return str(payload)
    shots = payload.get("shots", [])
    lines: list[str] = []
    for shot in shots:
        lines.append(
            "\n".join(
                [
                    f"- 镜头 {shot.get('sort')} {shot.get('theme')}",
                    f"  状态: {shot.get('status')}",
                    f"  task_id: {shot.get('task_id')}",
                    f"  video_url: {shot.get('video_url')}",
                    f"  error: {shot.get('error')}",
                ]
            )
        )
    return "\n\n".join(lines)


def _format_tool_result(tool_name: str, content: Any) -> str:
    payload = _parse_tool_message_content(content)
    if tool_name == "generate_shot_prompts_tool":
        return _format_shot_prompt_tool_result(payload)
    if tool_name == "generate_vidu_videos_tool":
        return _format_vidu_tool_result(payload)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return str(payload)


class ShotCharacter(BaseModel):
    uid: str = Field(..., description="主体 UID")
    name: str = Field(..., description="主体名")
    setting: str = Field(default="", description="主体设定")
    appearance: str = Field(..., description="主体外观")
    type: int = Field(..., description="主体类型：0人物，1物品，2场景")
    voice_id: str = Field(default="", description="音色 ID")
    image_url: Optional[str] = Field(default=None, description="主体图片 URL")
    reconstructed_image_url: Optional[str] = Field(default=None, description="重构图片 URL")
    final_image_url: Optional[str] = Field(default=None, description="最终图片 URL")


class ShotDurationPlan(BaseModel):
    total_shots: int = Field(..., description="分镜总数")
    duration_list: List[int] = Field(..., description="每个分镜的时长列表")


class ShotItem(BaseModel):
    duration: int = Field(..., description="分镜时长，单位秒")
    sort: int = Field(..., description="分镜顺序，从 0 开始")
    theme: str = Field(..., description="分镜主旨")
    description: str = Field(..., description="分镜详细描述")


class ShotVideoItem(BaseModel):
    sort: int = Field(..., description="分镜序号")
    theme: str = Field(..., description="分镜主题")
    prompt: str = Field(..., description="提交给 Vidu 的提示词")
    subjects: List[Dict[str, Any]] = Field(default_factory=list, description="提交给 Vidu 的主体列表")
    task_id: Optional[str] = Field(default=None, description="Vidu 任务 ID")
    status: str = Field(..., description="submitted、success、failed")
    video_url: Optional[str] = Field(default=None, description="生成后视频 URL")
    error: Optional[str] = Field(default=None, description="失败原因")


class ShotAgentFinalResult(BaseModel):
    plan: ShotDurationPlan
    shots: List[ShotItem]
    videos: List[ShotVideoItem] = Field(default_factory=list)


class ShotGenerationInput(BaseModel):
    theme: str = Field(..., description="故事大纲")
    characters: List[ShotCharacter] = Field(..., min_length=1, description="角色与主体列表")
    duration: int = Field(..., ge=4, description="故事总时长")
    language: str = Field(default="zh-CN", description="输出语言")
    ratio: str = Field(default="16:9", description="视频画幅比例")
    generate_videos: bool = Field(default=True, description="是否继续为分镜生成视频")
    video_resolution: str = Field(default="540p", description="Vidu 视频分辨率")
    video_model: str = Field(default="viduq3", description="Vidu 模型名")
    video_audio: bool = Field(default=True, description="是否生成音频")
    wait_for_completion: bool = Field(default=True, description="是否等待 Vidu 任务完成")
    poll_interval: int = Field(default=10, description="视频生成轮询间隔")

    @model_validator(mode="after")
    def validate_scene_exists(self):
        if not any(character.type == 2 for character in self.characters):
            raise ValueError("characters 中至少需要包含 1 个场景主体(type=2)")
        return self


class ShotGenerationAgent:
    def __init__(self, config_path: str = "config/openai/config.yaml"):
        self.config_path = _resolve_path(config_path)
        self.llm = _build_llm(self.config_path)
        self.agent = create_agent(
            model=self.llm,
            tools=[generate_shot_prompts_tool, generate_vidu_videos_tool],
            system_prompt=SHOT_REACT_SYSTEM_PROMPT,
        )

    def _build_react_user_prompt(self, request: ShotGenerationInput) -> str:
        return """请基于下面的 request 完成多分镜脚本和视频生成。

硬性要求：
1. 先调用 generate_shot_prompts_tool 生成分镜规划和分镜脚本。
2. 如果 request.generate_videos=true，再调用 generate_vidu_videos_tool 为每个分镜生成视频。
3. 视频生成必须直接使用 request.characters 中已有的图片字段，不能伪造图片链接。
4. 最终仅输出 ShotAgentFinalResult 对应的 JSON。

request:
{request_json}
""".strip().format(request_json=request.model_dump_json(indent=2, ensure_ascii=False))

    async def _normalize_agent_output(self, request: ShotGenerationInput, agent_text: str) -> ShotAgentFinalResult:
        structured_llm = _structured(self.llm, ShotAgentFinalResult)
        return await structured_llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": "你负责将 ReAct agent 的最终结果整理成 ShotAgentFinalResult schema，只输出符合 schema 的 JSON。",
                },
                {
                    "role": "user",
                    "content": (
                        "请将下面内容整理为 ShotAgentFinalResult。\n\n"
                        f"request:\n{request.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
                        f"react_agent_final_text:\n{agent_text}"
                    ),
                },
            ]
        )

    async def _run_react(
        self,
        request: ShotGenerationInput,
        verbose: bool = False,
        log_prefix: str = "",
    ) -> ShotAgentFinalResult:
        final_text = ""
        seen_tool_messages: set[str] = set()
        user_prompt = self._build_react_user_prompt(request)
        prefix = f"[{log_prefix}] " if log_prefix else ""
        seen_tool_call_ids: set[str] = set()
        vidu_videos: List[Dict[str, Any]] = []
        shot_prompt_payload: Optional[Dict[str, Any]] = None

        async for chunk in self.agent.astream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            stream_mode="values",
        ):
            latest_message = chunk["messages"][-1]
            tool_calls = getattr(latest_message, "tool_calls", None) or []
            if tool_calls:
                for tool_call in tool_calls:
                    tc_id = tool_call.get("id") or f"{tool_call.get('name')}-{len(seen_tool_call_ids)}"
                    if tc_id in seen_tool_call_ids:
                        continue
                    seen_tool_call_ids.add(tc_id)
                    if verbose:
                        print(f"{prefix}tool_call: {tool_call['name']}")
                    await emit_event(
                        {
                            "event": "subagent_tool_call",
                            "agent": log_prefix or "shot_subagent",
                            "name": tool_call.get("name", "unknown_tool"),
                            "payload": {
                                "args": tool_call.get("args", {}),
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type"),
                            },
                        }
                    )

            if latest_message.__class__.__name__ == "ToolMessage":
                tool_message_id = getattr(latest_message, "id", None) or getattr(latest_message, "tool_call_id", None)
                if tool_message_id not in seen_tool_messages:
                    seen_tool_messages.add(tool_message_id)
                    tool_name = getattr(latest_message, "name", "unknown_tool")
                    parsed_payload = _parse_tool_message_content(getattr(latest_message, "content", ""))
                    if tool_name == "generate_vidu_videos_tool" and isinstance(parsed_payload, dict):
                        vidu_videos = parsed_payload.get("shots", []) or []
                    elif tool_name == "generate_shot_prompts_tool" and isinstance(parsed_payload, dict):
                        shot_prompt_payload = parsed_payload
                    if verbose:
                        print(f"{prefix}tool_result: {tool_name}\n{_format_tool_result(tool_name, getattr(latest_message, 'content', ''))}")
                    await emit_event(
                        {
                            "event": "subagent_tool_result",
                            "agent": log_prefix or "shot_subagent",
                            "name": tool_name,
                            "payload": parsed_payload,
                        }
                    )

            message_text = _message_text(latest_message)
            if message_text:
                final_text = message_text

        if not final_text:
            raise ValueError("shot agent 未返回最终结果")

        try:
            final_result = ShotAgentFinalResult.model_validate(_extract_json_object(final_text))
        except Exception:
            final_result = await self._normalize_agent_output(request, final_text)

        if vidu_videos:
            try:
                final_result.videos = [ShotVideoItem.model_validate(item) for item in vidu_videos]
            except Exception:
                pass

        if shot_prompt_payload:
            shots_from_tool = shot_prompt_payload.get("shots") or []
            if shots_from_tool and not final_result.shots:
                try:
                    final_result.shots = [ShotItem.model_validate(item) for item in shots_from_tool]
                except Exception:
                    pass
            plan_from_tool = shot_prompt_payload.get("plan")
            if plan_from_tool and not getattr(final_result, "plan", None):
                try:
                    final_result.plan = ShotDurationPlan.model_validate(plan_from_tool)
                except Exception:
                    pass

        return final_result

    async def ainvoke(
        self,
        payload: Dict[str, Any] | ShotGenerationInput,
        verbose: bool = False,
        log_prefix: str = "",
    ) -> Dict[str, Any]:
        request = payload if isinstance(payload, ShotGenerationInput) else ShotGenerationInput.model_validate(payload)
        result = await self._run_react(request, verbose=verbose, log_prefix=log_prefix)
        return result.model_dump(mode="json")

    def invoke(
        self,
        payload: Dict[str, Any] | ShotGenerationInput,
        verbose: bool = False,
        log_prefix: str = "",
    ) -> Dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(payload, verbose=verbose, log_prefix=log_prefix))
        raise RuntimeError("当前已存在事件循环，请改用 ainvoke() 调用 ShotGenerationAgent")


def main():
    agent = ShotGenerationAgent()
    sample_request = {
        "theme": "在一个被永夜笼罩的海港小镇，少女灯塔守望者意外发现一封来自未来的求救信，于是决定踏上寻找失踪父亲的旅程。",
        "duration": 24,
        "language": "zh-CN",
        "ratio": "16:9",
        "generate_videos": True,
        "characters": [
            {
                "uid": "char-luna",
                "name": "露娜",
                "setting": "年轻的灯塔守望者，警觉、勇敢，对父亲失踪一事始终无法释怀。",
                "appearance": "黑色短发，穿旧海军外套和防风长靴，随身携带黄铜怀表。",
                "type": 0,
                "voice_id": "d474d67b-08a9-4b5d-a0f9-e318433ae634",
                "final_image_url": "https://example.com/assets/characters/luna-final.jpeg"
            },
            {
                "uid": "prop-letter",
                "name": "未来来信",
                "setting": "",
                "appearance": "边缘烧蚀的信纸，纸面闪烁微弱蓝光。",
                "type": 1,
                "voice_id": "",
                "final_image_url": "https://example.com/assets/props/future-letter.jpeg"
            },
            {
                "uid": "scene-harbor",
                "name": "雾港灯塔",
                "setting": "",
                "appearance": "终年被浓雾笼罩的海港灯塔，海风猛烈，远处隐约可见停泊的旧船。",
                "type": 2,
                "voice_id": "",
                "final_image_url": "https://example.com/assets/scenes/foggy-harbor-lighthouse.jpeg"
            }
        ]
    }
    result = agent.invoke(sample_request, verbose=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()