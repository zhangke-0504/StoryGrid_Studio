import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
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

from config.prompt.create_shot.prompt import (  # noqa: E402
	SHOT_DURATION_PLANNER_SYSTEM_PROMPT,
	SHOT_GENERATION_SYSTEM_PROMPT,
	build_shot_duration_planner_user_prompt,
	build_shot_generation_user_prompt,
)
from utils.create_model_response.openai.create_model_response import GPTClient  # noqa: E402


def _resolve_path(path: str) -> str:
	resolved = Path(path)
	if not resolved.is_absolute():
		resolved = PROJECT_ROOT / resolved
	return str(resolved)


class ShotToolCharacter(BaseModel):
	uid: str = Field(..., description="主体 UID")
	name: str = Field(..., description="主体名")
	setting: str = Field(default="", description="主体设定")
	appearance: str = Field(..., description="主体外观")
	type: int = Field(..., description="主体类型：0人物，1物品，2场景")
	voice_id: str = Field(default="", description="音色 ID")
	image_url: str | None = Field(default=None, description="主体图片 URL")
	reconstructed_image_url: str | None = Field(default=None, description="重构后的图片 URL")
	final_image_url: str | None = Field(default=None, description="最终可用图片 URL")


class ShotDurationPlan(BaseModel):
	total_shots: int = Field(..., description="分镜总数")
	duration_list: List[int] = Field(..., description="每个分镜的时长列表")


class ShotToolItem(BaseModel):
	duration: int = Field(..., description="分镜时长，单位秒")
	sort: int = Field(..., description="分镜顺序，从 0 开始")
	theme: str = Field(..., description="分镜主旨")
	description: str = Field(..., description="分镜详细描述")


class ShotGenerationResult(BaseModel):
	shots: List[ShotToolItem]


class ShotPromptToolInput(BaseModel):
	theme: str = Field(..., description="故事大纲")
	characters: List[ShotToolCharacter] = Field(..., min_length=1, description="角色与主体列表")
	duration: int = Field(..., ge=4, description="故事总时长")
	language: str = Field(default="zh-CN", description="输出语言")
	ratio: str = Field(default="16:9", description="视频画幅比例")

	@model_validator(mode="after")
	def validate_scene_exists(self):
		if not any(character.type == 2 for character in self.characters):
			raise ValueError("characters 中至少需要包含 1 个场景主体(type=2)")
		return self


class ShotPromptToolResult(BaseModel):
	plan: ShotDurationPlan
	shots: List[ShotToolItem]


def _normalize_durations(total_duration: int, preferred_count: int) -> List[int]:
	min_shots = math.ceil(total_duration / 10)
	max_shots = total_duration // 4
	if max_shots < 1:
		raise ValueError("duration 过短，无法按 4-10 秒分镜规则生成脚本")

	shot_count = preferred_count if min_shots <= preferred_count <= max_shots else min_shots
	durations = [4] * shot_count
	remaining = total_duration - sum(durations)
	index = 0
	while remaining > 0:
		add_value = min(10 - durations[index], remaining)
		durations[index] += add_value
		remaining -= add_value
		index = (index + 1) % shot_count
	return durations


def _repair_duration_plan(request: ShotPromptToolInput, plan: ShotDurationPlan) -> ShotDurationPlan:
	raw_durations = [int(item) for item in plan.duration_list if int(item) > 0]
	if (
		plan.total_shots == len(raw_durations)
		and raw_durations
		and all(4 <= item <= 10 for item in raw_durations)
		and sum(raw_durations) == request.duration
	):
		return ShotDurationPlan(total_shots=plan.total_shots, duration_list=raw_durations)

	normalized = _normalize_durations(request.duration, plan.total_shots)
	return ShotDurationPlan(total_shots=len(normalized), duration_list=normalized)


@tool(parse_docstring=True)
async def generate_shot_prompts_tool(
	theme: str,
	characters: List[Dict[str, Any]],
	duration: int,
	language: str = "zh-CN",
	ratio: str = "16:9",
	config_path: str = "config/openai/config.yaml",
) -> str:
	"""生成多分镜脚本与分镜提示词。

	Args:
		theme: 故事大纲。
		characters: 主体列表，必须至少包含 1 个场景主体。
		duration: 视频总时长，单位秒。
		language: 输出语言。
		ratio: 视频画幅比例。
		config_path: OpenAI 配置路径。
	"""
	request = ShotPromptToolInput.model_validate(
		{
			"theme": theme,
			"characters": characters,
			"duration": duration,
			"language": language,
			"ratio": ratio,
		}
	)
	client = GPTClient(config_path=_resolve_path(config_path))

	raw_plan = await client.async_non_stream_response(
		prompt=build_shot_duration_planner_user_prompt(
			theme=request.theme,
			duration=request.duration,
			language=request.language,
			ratio=request.ratio,
		),
		instructions=SHOT_DURATION_PLANNER_SYSTEM_PROMPT,
		text_format=ShotDurationPlan,
	)
	plan = _repair_duration_plan(request, raw_plan)

	result = await client.async_non_stream_response(
		prompt=build_shot_generation_user_prompt(
			theme=request.theme,
			duration=request.duration,
			language=request.language,
			ratio=request.ratio,
			total_shots=plan.total_shots,
			duration_list=plan.duration_list,
			characters=[character.model_dump(mode="json") for character in request.characters],
		),
		instructions=SHOT_GENERATION_SYSTEM_PROMPT,
		text_format=ShotGenerationResult,
	)

	repaired_shots: List[ShotToolItem] = []
	for index, shot in enumerate(result.shots[: plan.total_shots]):
		repaired_shots.append(
			ShotToolItem(
				duration=plan.duration_list[index],
				sort=index,
				theme=shot.theme,
				description=shot.description,
			)
		)

	return ShotPromptToolResult(plan=plan, shots=repaired_shots).model_dump_json(ensure_ascii=False)
