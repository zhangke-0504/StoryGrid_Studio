import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from langchain.agents import create_agent
from langchain_core.tools import tool
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

from agents.character_generation.agents import (  # noqa: E402
	CharacterGenerationAgent,
	CharacterGenerationResult,
	DEFAULT_STYLE_NAME,
)
from agents.shot_generation.agents import (  # noqa: E402
	ShotAgentFinalResult,
	ShotCharacter,
	ShotGenerationAgent,
	ShotItem,
	ShotDurationPlan,
	ShotVideoItem,
)
from agents.shared.event_emitter import emit_event  # noqa: E402


SUPERVISOR_SYSTEM_PROMPT = """
你是故事视频生产总控 Supervisor Agent。

你的职责是把 theme 变成最终的多分镜视频结果，且必须通过两个子智能体协作完成：
1. 先调用 generate_story_characters_subagent，为故事生成主体并补齐可用图片。
2. 再调用 generate_story_videos_subagent，基于上一步返回的 characters 生成分镜脚本并生成视频。
3. 每一轮最多调用 1 个工具，禁止跳步，禁止伪造 characters、shots、task_id、video_url。
4. 当两个步骤都完成后，只输出符合 StoryVideoGenerationResult schema 的最终 JSON。
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


def _format_tool_result(content: Any) -> str:
	payload = _parse_tool_message_content(content)
	if isinstance(payload, (dict, list)):
		return json.dumps(payload, ensure_ascii=False, indent=2)
	return str(payload)


def _build_character_uid(character_type: int, index: int) -> str:
	prefix = {0: "char", 1: "prop", 2: "scene"}.get(character_type, "subject")
	return f"{prefix}-{index}"


class SupervisorCharacterBatch(BaseModel):
	characters: List[ShotCharacter]


class StoryVideoGenerationInput(BaseModel):
	theme: str = Field(..., description="故事大纲")
	duration: int = Field(default=24, ge=4, description="故事总时长")
	language: str = Field(default="zh-CN", description="输出语言")
	ratio: str = Field(default="16:9", description="视频画幅比例")
	style_name: str = Field(default=DEFAULT_STYLE_NAME, description="主体设计风格")
	custom_characters: Optional[Dict[str, Any]] = Field(default=None, description="自定义主体数量约束")
	generate_character_images: bool = Field(default=True, description="是否生成主体图片")
	audit_character_images: bool = Field(default=True, description="是否审核并重构主体图片")
	character_image_model: Optional[str] = Field(default=None, description="主体生图模型")
	character_image_size: str = Field(default="2K", description="主体图片尺寸")
	character_image_watermark: bool = Field(default=False, description="主体图片是否带水印")
	generate_videos: bool = Field(default=True, description="是否继续生成分镜视频")
	video_resolution: str = Field(default="540p", description="视频分辨率")
	video_model: str = Field(default="viduq3", description="视频模型名")
	video_audio: bool = Field(default=True, description="是否生成音频")
	wait_for_completion: bool = Field(default=True, description="是否等待视频任务完成")
	poll_interval: int = Field(default=10, description="视频轮询间隔秒数")


class StoryVideoGenerationResult(BaseModel):
	characters: List[ShotCharacter]
	plan: ShotDurationPlan
	shots: List[ShotItem]
	videos: List[ShotVideoItem] = Field(default_factory=list)


def _adapt_character_generation_result(payload: Dict[str, Any]) -> SupervisorCharacterBatch:
	result = CharacterGenerationResult.model_validate(payload)
	type_counters = {0: 0, 1: 0, 2: 0}
	characters: list[ShotCharacter] = []

	for item in result.characters:
		type_counters[item.type] = type_counters.get(item.type, 0) + 1
		characters.append(
			ShotCharacter(
				uid=_build_character_uid(item.type, type_counters[item.type]),
				name=item.name,
				setting=item.setting,
				appearance=item.appearance,
				type=item.type,
				voice_id=item.voice_id,
				image_url=item.image_url,
				reconstructed_image_url=item.reconstructed_image_url,
				final_image_url=item.final_image_url,
			)
		)

	return SupervisorCharacterBatch(characters=characters)


class StoryVideoSupervisorAgent:
	def __init__(
		self,
		config_path: str = "config/openai/config.yaml",
		doubao_config_path: str = "config/doubao/config.yaml",
		vidu_config_path: str = "config/vidu/config.yaml",
	):
		self.config_path = _resolve_path(config_path)
		self.doubao_config_path = _resolve_path(doubao_config_path)
		self.vidu_config_path = _resolve_path(vidu_config_path)
		self.llm = _build_llm(self.config_path)
		self.character_agent = CharacterGenerationAgent(
			config_path=self.config_path,
			doubao_config_path=self.doubao_config_path,
		)
		self.shot_agent = ShotGenerationAgent(config_path=self.config_path)
		self.agent = create_agent(
			model=self.llm,
			tools=self._build_tools(),
			system_prompt=SUPERVISOR_SYSTEM_PROMPT,
		)
		self._subagent_verbose = False

	def _build_tools(self) -> list[Any]:
		@tool(parse_docstring=True)
		async def generate_story_characters_subagent(
			theme: str,
			language: str = "zh-CN",
			style_name: str = DEFAULT_STYLE_NAME,
			custom_characters: Optional[Dict[str, Any]] = None,
			generate_images: bool = True,
			audit_images: bool = True,
			image_model: Optional[str] = None,
			image_size: str = "2K",
			image_watermark: bool = False,
		) -> str:
			"""调用 character_generation 子智能体，返回带 UID 的主体列表 JSON。

			Args:
				theme: 故事大纲。
				language: 输出语言。
				style_name: 主体设计风格。
				custom_characters: 自定义主体数量约束。
				generate_images: 是否生成主体图片。
				audit_images: 是否审核并重构主体图片。
				image_model: 主体生图模型。
				image_size: 主体图片尺寸。
				image_watermark: 主体图片是否带水印。
			"""
			await emit_event({"event": "subagent_start", "agent": "character_subagent", "name": "generate_story_characters_subagent"})
			result = await self.character_agent.ainvoke(
				{
					"theme": theme,
					"language": language,
					"style_name": style_name,
					"custom_characters": custom_characters,
					"generate_images": generate_images,
					"audit_images": audit_images,
					"image_model": image_model,
					"image_size": image_size,
					"image_watermark": image_watermark,
				},
				verbose=self._subagent_verbose,
				log_prefix="character_subagent",
			)
			adapted = _adapt_character_generation_result(result)
			await emit_event({"event": "subagent_end", "agent": "character_subagent", "name": "generate_story_characters_subagent"})
			return adapted.model_dump_json(ensure_ascii=False)

		@tool(parse_docstring=True)
		async def generate_story_videos_subagent(
			theme: str,
			characters_payload_json: str,
			duration: int,
			language: str = "zh-CN",
			ratio: str = "16:9",
			generate_videos: bool = True,
			video_resolution: str = "540p",
			video_model: str = "viduq3",
			video_audio: bool = True,
			wait_for_completion: bool = True,
			poll_interval: int = 10,
		) -> str:
			"""调用 shot_generation 子智能体，返回分镜与视频结果 JSON。

			Args:
				theme: 故事大纲。
				characters_payload_json: 上一个子智能体返回的 characters JSON 字符串。
				duration: 故事总时长。
				language: 输出语言。
				ratio: 视频画幅比例。
				generate_videos: 是否生成视频。
				video_resolution: 视频分辨率。
				video_model: 视频模型名。
				video_audio: 是否生成音频。
				wait_for_completion: 是否等待视频任务完成。
				poll_interval: 视频轮询间隔秒数。
			"""
			await emit_event({"event": "subagent_start", "agent": "shot_subagent", "name": "generate_story_videos_subagent"})
			characters_payload = SupervisorCharacterBatch.model_validate(json.loads(characters_payload_json))
			result = await self.shot_agent.ainvoke(
				{
					"theme": theme,
					"characters": [character.model_dump(mode="json") for character in characters_payload.characters],
					"duration": duration,
					"language": language,
					"ratio": ratio,
					"generate_videos": generate_videos,
					"video_resolution": video_resolution,
					"video_model": video_model,
					"video_audio": video_audio,
					"wait_for_completion": wait_for_completion,
					"poll_interval": poll_interval,
				},
				verbose=self._subagent_verbose,
				log_prefix="shot_subagent",
			)
			await emit_event({"event": "subagent_end", "agent": "shot_subagent", "name": "generate_story_videos_subagent"})
			return json.dumps(result, ensure_ascii=False)

		return [generate_story_characters_subagent, generate_story_videos_subagent]

	def _build_react_user_prompt(self, request: StoryVideoGenerationInput) -> str:
		return """请基于下面的 request 生成故事主体、分镜脚本与分镜视频。

硬性要求：
1. 必须先调用 generate_story_characters_subagent。
2. 再把上一步返回的完整 JSON 字符串原样传给 generate_story_videos_subagent 的 characters_payload_json。
3. 不允许跳过任何子智能体，不允许伪造主体图片、分镜、任务 ID 或视频 URL。
4. 完成后只输出 StoryVideoGenerationResult 对应的 JSON。

request:
{request_json}
""".strip().format(request_json=request.model_dump_json(indent=2, ensure_ascii=False))

	async def _normalize_agent_output(
		self,
		request: StoryVideoGenerationInput,
		agent_text: str,
	) -> StoryVideoGenerationResult:
		structured_llm = self.llm.with_structured_output(StoryVideoGenerationResult)
		return await structured_llm.ainvoke(
			[
				{
					"role": "system",
					"content": "你负责将 Supervisor Agent 的结果整理成 StoryVideoGenerationResult schema，只输出符合 schema 的 JSON。",
				},
				{
					"role": "user",
					"content": (
						"请将下面内容整理为 StoryVideoGenerationResult。\n\n"
						f"request:\n{request.model_dump_json(indent=2, ensure_ascii=False)}\n\n"
						f"supervisor_final_text:\n{agent_text}"
					),
				},
			]
		)

	async def _run_react(self, request: StoryVideoGenerationInput, verbose: bool = False) -> StoryVideoGenerationResult:
		final_text = ""
		seen_tool_messages: set[str] = set()
		character_result: Optional[SupervisorCharacterBatch] = None
		shot_result: Optional[ShotAgentFinalResult] = None
		user_prompt = self._build_react_user_prompt(request)
		self._subagent_verbose = verbose

		try:
			async for chunk in self.agent.astream(
				{"messages": [{"role": "user", "content": user_prompt}]},
				stream_mode="values",
			):
				latest_message = chunk["messages"][-1]
				tool_calls = getattr(latest_message, "tool_calls", None) or []
				if verbose and tool_calls:
					for tool_call in tool_calls:
						print(f"tool_call: {tool_call['name']}")

				if latest_message.__class__.__name__ == "ToolMessage":
					tool_message_id = getattr(latest_message, "id", None) or getattr(latest_message, "tool_call_id", None)
					if tool_message_id not in seen_tool_messages:
						seen_tool_messages.add(tool_message_id)
						tool_name = getattr(latest_message, "name", "unknown_tool")
						parsed_content = _parse_tool_message_content(getattr(latest_message, "content", ""))
						if tool_name == "generate_story_characters_subagent":
							character_result = SupervisorCharacterBatch.model_validate(parsed_content)
						elif tool_name == "generate_story_videos_subagent":
							shot_result = ShotAgentFinalResult.model_validate(parsed_content)

						if verbose:
							print(f"tool_result: {tool_name}\n{_format_tool_result(parsed_content)}")

				message_text = _message_text(latest_message)
				if message_text:
					final_text = message_text

			if final_text:
				try:
					return StoryVideoGenerationResult.model_validate(_extract_json_object(final_text))
				except Exception:
					pass

			if character_result and shot_result:
				return StoryVideoGenerationResult(
					characters=character_result.characters,
					plan=shot_result.plan,
					shots=shot_result.shots,
					videos=shot_result.videos,
				)

			if not final_text:
				raise ValueError("supervisor agent 未返回最终结果")

			return await self._normalize_agent_output(request, final_text)
		finally:
			self._subagent_verbose = False

	async def ainvoke(
		self,
		payload: Dict[str, Any] | StoryVideoGenerationInput,
		verbose: bool = False,
	) -> Dict[str, Any]:
		request = payload if isinstance(payload, StoryVideoGenerationInput) else StoryVideoGenerationInput.model_validate(payload)
		result = await self._run_react(request, verbose=verbose)
		return result.model_dump(mode="json")

	def invoke(
		self,
		payload: Dict[str, Any] | StoryVideoGenerationInput,
		verbose: bool = False,
	) -> Dict[str, Any]:
		try:
			asyncio.get_running_loop()
		except RuntimeError:
			return asyncio.run(self.ainvoke(payload, verbose=verbose))
		raise RuntimeError("当前已存在事件循环，请改用 ainvoke() 调用 StoryVideoSupervisorAgent")


def main():
	agent = StoryVideoSupervisorAgent()
	sample_request = {
		"theme": "在一个被永夜笼罩的海港小镇，少女灯塔守望者意外发现一封来自未来的求救信，于是决定踏上寻找失踪父亲的旅程。",
		"duration": 24,
		"language": "zh-CN",
		"ratio": "16:9",
		"style_name": "奇幻动画电影",
		"generate_character_images": True,
		"audit_character_images": True,
		"generate_videos": True,
		"video_resolution": "540p",
		"video_model": "viduq3",
		"video_audio": True,
		"wait_for_completion": True,
		"poll_interval": 10,
	}
	result = agent.invoke(sample_request, verbose=True)
	print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
