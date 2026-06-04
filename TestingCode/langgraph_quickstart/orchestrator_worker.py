"""
在协调器-工作器配置中，协调器：
将任务分解成子任务
将子任务委派给工人
将工人的产出综合成最终结果
编排器-工作流模式提供了更大的灵活性，通常用于无法像并行化那样预先定义子任务的情况。
这在需要编写代码或更新多个文件内容的流程中很常见。
例如，需要更新数量未知的文档中多个 Python 库的安装说明的流程就可能使用这种模式。
"""
import os
import getpass
from typing import Any, Dict, TypedDict, Annotated, List
from typing_extensions import Literal
import yaml
from langchain_openai import ChatOpenAI
# 用于结构化输出的 Schema
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain.messages import HumanMessage, SystemMessage
import operator


def _load_config(config_path: str = "config/openai/config.yaml") -> Dict[str, Any]:
        """从 YAML 配置文件加载 API Key 等配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if not config:
                    raise ValueError("配置为空")
                return config
        except Exception as e:
            raise RuntimeError(f"加载配置失败: {e}")

llm = ChatOpenAI(
    model="gpt-4.1",  # 或 gpt-4o、gpt-4o-mini 等
    temperature=0,
    api_key=_load_config()["api_key"]
)


# 用于规划阶段的结构化输出 Schema
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


# 为 LLM 增加结构化输出能力
planner = llm.with_structured_output(Sections)