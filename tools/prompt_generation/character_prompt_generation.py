import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import tool
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

from utils.create_model_response.openai.create_model_response import GPTClient  # noqa: E402


def _resolve_path(path: str) -> str:
	resolved = Path(path)
	if not resolved.is_absolute():
		resolved = PROJECT_ROOT / resolved
	return str(resolved)


class SubjectPromptToolItem(BaseModel):
	name: str = Field(..., description="主体名字")
	setting: str = Field(default="", description="主体设定")
	appearance: str = Field(..., description="主体外观描述")
	type: int = Field(..., description="主体类型：0人物，1物品，2场景")


class SubjectPromptToolInput(BaseModel):
	subjects: List[SubjectPromptToolItem] = Field(..., min_length=1, description="需要生成设定图提示词的主体列表")
	style_name: str = Field(..., description="图片风格名")
	language: str = Field(default="zh-CN", description="输出语言")


class SubjectPromptToolResultItem(SubjectPromptToolItem):
	image_prompt: str = Field(..., description="主体设定图提示词")


class SubjectPromptToolResult(BaseModel):
	subjects: List[SubjectPromptToolResultItem]


SUBJECT_PROMPT_TOOL_INSTRUCTIONS = """
你是一名影视概念设计提示词工程师。你的任务是为主体生成可直接用于图像模型的设定图提示词。

输出要求：
1. 必须只返回符合 schema 的 JSON。
2. 每个主体都必须产出 `image_prompt`，并保留原有 `name`、`setting`、`appearance`、`type` 字段。
3. type=0 时，提示词必须强调角色三视图、角色设定稿、正侧背一致性、随身物件细节、纯净背景。
4. type=1 时，提示词必须强调单一道具设定图、结构细节、材质细节、局部放大或多角度小视图、纯净背景。
5. type=2 时，提示词必须强调单一场景设定图、空间布局、关键地标、材质氛围、纯净背景。
6. 提示词中必须明确加入图片风格 `style_name`。
7. 所有文字使用指定 language，但保留必要的专业视觉描述，不要输出多余解释。
""".strip()


@tool(parse_docstring=True)
async def generate_subject_prompts_tool(
	subjects: List[Dict[str, Any]],
	style_name: str,
	language: str = "zh-CN",
	config_path: str = "config/openai/config.yaml",
) -> str:
	"""批量生成人物、物品、场景的设定图提示词。

	Args:
		subjects: 主体列表，每个主体需要包含 name、setting、appearance、type。
		style_name: 图片风格名，会强制写入每条设定图提示词。
		language: 输出语言。
		config_path: OpenAI 配置路径。
	"""
	payload = SubjectPromptToolInput.model_validate(
		{
			"subjects": subjects,
			"style_name": style_name,
			"language": language,
		}
	)
	client = GPTClient(config_path=_resolve_path(config_path))
	prompt = (
		"请根据以下主体列表生成设定图提示词，输出 JSON。\n\n"
		f"style_name: {payload.style_name}\n"
		f"language: {payload.language}\n"
		f"subjects: {json.dumps([subject.model_dump(mode='json') for subject in payload.subjects], ensure_ascii=False, indent=2)}"
	)
	result = await client.async_non_stream_response(
		prompt=prompt,
		instructions=SUBJECT_PROMPT_TOOL_INSTRUCTIONS,
		text_format=SubjectPromptToolResult,
	)
	return result.model_dump_json(ensure_ascii=False)
