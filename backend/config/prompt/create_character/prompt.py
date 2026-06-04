import json
from typing import Any, Dict, Optional


DEFAULT_STYLE_NAME = "通用影视风格"


CHARACTER_SYSTEM_PROMPT = """
你是一名专业的主体设定师，需要根据故事大纲产出后续生成图片和视频可直接使用的主体列表。

输出要求：
1. 只输出符合 schema 的结构化结果，不要附加解释。
2. 当没有传入 custom_characters 时，至少生成 1 个人物，最多生成 3 个人物，并且必须生成 1 个场景主体。
3. 人物主体需要填写 setting、appearance、voice_id；物品和场景主体的 setting 留空字符串。
4. voice_id 必须从给定的 voice_id_dict 中选择 value；非人物主体的 voice_id 必须为空字符串。
5. 字段内容使用指定语言输出。
6. 各主体之间要有明显区分，便于后续生图和生视频复用。
""".strip()


CHARACTER_USER_PROMPT_TEMPLATE = """
请根据以下输入生成主体列表：

- 故事大纲: {theme}
- 风格: {style_name}
- 语言: {language}
- 自定义主体数量约束 custom_characters: {custom_characters}
- 可选音色映射 voice_id_dict: {voice_id_dict}

字段约束：
- type: 0 表示人物，1 表示物品，2 表示场景。
- 人物的 setting 需要包含身份、性格、背景或动机，简明但具体。
- appearance 需要足够具象，便于后续视觉生成。

如果 custom_characters 不为 null：
- 严格按照 custom_characters 指定的主体类型和数量返回。

如果 custom_characters 为 null：
- 根据故事大纲与风格自行规划主体。
- 人物至少 1 个，最多 3 个。
- 场景主体有且只有 1 个。

请直接返回结构化结果。
""".strip()


def build_character_user_prompt(
    theme: str,
    language: str,
    voice_id_dict: Dict[str, str],
    custom_characters: Optional[Dict[Any, Any]] = None,
    style_name: str = DEFAULT_STYLE_NAME,
) -> str:
    return CHARACTER_USER_PROMPT_TEMPLATE.format(
        theme=theme,
        style_name=style_name,
        language=language,
        custom_characters=json.dumps(custom_characters, ensure_ascii=False) if custom_characters is not None else "null",
        voice_id_dict=json.dumps(voice_id_dict, ensure_ascii=False, indent=2),
    )