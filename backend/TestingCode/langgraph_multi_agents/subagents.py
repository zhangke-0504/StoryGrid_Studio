"""
在子代理架构中，中央主代理（通常称为监督器）通过调用子代理作为工具来协调它们。
主代理决定调用哪个子代理、提供什么输入以及如何组合结果。
子代理是无状态的——它们不记忆过去的交互，所有对话记忆都由主代理维护。
这实现了上下文隔离：每次子代理调用都在一个干净的上下文窗口中运行，从而防止主对话中出现上下文膨胀。
"""

from langchain.tools import tool
from langchain.agents import create_agent
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

# Create a subagent
subagent = create_agent(model=llm, tools=[...])

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(model=llm, tools=[call_research_agent])