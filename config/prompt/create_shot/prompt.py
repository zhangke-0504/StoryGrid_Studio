import json
from typing import Any, Dict, List


SHOT_DURATION_PLANNER_SYSTEM_PROMPT = """
你是一名专业的动漫分镜时长设计师。

你的任务是根据视频总时长规划分镜数量与每个分镜的时长。

要求：
1. 所有分镜时长之和必须等于用户给定的视频总时长。
2. 每个分镜时长应尽量控制在 4 到 10 秒之间。
3. 只输出符合 schema 的结构化结果，不要附加解释。
""".strip()


SHOT_DURATION_PLANNER_USER_PROMPT_TEMPLATE = """
请为以下视频规划分镜时长：

- 故事大纲: {theme}
- 视频总时长: {duration} 秒
- 语言: {language}
- 画幅比例: {ratio}

请返回分镜总数 total_shots 与对应时长列表 duration_list。
""".strip()


SHOT_GENERATION_SYSTEM_PROMPT = """
你是一名专业的动漫分镜设计师，需要根据故事大纲和给定主体列表生成可直接用于视频生成的分镜脚本。

输出要求：
1. 只输出符合 schema 的结构化结果，不要附加解释。
2. 只能使用给定主体列表中的主体，不得发明新主体。
3. description 中提到的主体必须使用 @^uid^ 的形式引用，并保留空格便于后处理。
4. 每个分镜必须包含至少一个场景主体。
5. 分镜之间要形成连续叙事，且每个镜头都紧扣故事大纲。
6. 每个分镜的台词控制在 2 句以内。
7. 字段内容使用指定语言输出。
""".strip()


SHOT_GENERATION_USER_PROMPT_TEMPLATE = """
请根据以下输入生成完整分镜脚本：

- 故事大纲: {theme}
- 视频总时长: {duration} 秒
- 语言: {language}
- 画幅比例: {ratio}
- 分镜总数: {total_shots}
- 每个分镜时长列表: {duration_list}
- 可用主体列表: {characters_json}

主体字段说明：
- uid: 主体唯一标识，description 中必须使用 @^uid^ 的格式引用。
- name: 主体名。
- setting: 设定描述，可为空。
- appearance: 外观描述。
- type: 0 表示人物，1 表示物品，2 表示场景。

分镜字段要求：
- duration: 必须与时长列表中的对应元素一致。
- sort: 从 0 开始递增。
- theme: 当前分镜的核心剧情内容，直接写自然语言，不使用 uid 占位。
- description: 详细描述镜头内容，引用主体时必须使用 @^uid^ 形式；且至少包含一个场景主体。

请直接返回结构化结果。
""".strip()


def build_shot_duration_planner_user_prompt(
    theme: str,
    duration: int,
    language: str,
    ratio: str,
) -> str:
    return SHOT_DURATION_PLANNER_USER_PROMPT_TEMPLATE.format(
        theme=theme,
        duration=duration,
        language=language,
        ratio=ratio,
    )


def build_shot_generation_user_prompt(
    theme: str,
    duration: int,
    language: str,
    ratio: str,
    total_shots: int,
    duration_list: List[int],
    characters: List[Dict[str, Any]],
) -> str:
    return SHOT_GENERATION_USER_PROMPT_TEMPLATE.format(
        theme=theme,
        duration=duration,
        language=language,
        ratio=ratio,
        total_shots=total_shots,
        duration_list=json.dumps(duration_list, ensure_ascii=False),
        characters_json=json.dumps(characters, ensure_ascii=False, indent=2),
    )