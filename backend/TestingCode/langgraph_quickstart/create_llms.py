import os
import getpass
from typing import Any, Dict
import yaml
from langchain_openai import ChatOpenAI
# 用于结构化输出的 Schema
from pydantic import BaseModel, Field


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


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# 为 LLM 增加结构化输出能力
structured_llm = llm.with_structured_output(SearchQuery)

# 调用增强后的 LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")

# 定义一个工具
def multiply(a: int, b: int) -> int:
    return a * b

# 为 LLM 绑定工具
llm_with_tools = llm.bind_tools([multiply])

# 使用会触发工具调用的输入来调用 LLM
msg = llm_with_tools.invoke("What is 2 times 3?")

# 获取工具调用结果
print(msg)