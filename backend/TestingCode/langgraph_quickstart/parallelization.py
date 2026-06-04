"""
通过并行化，LLM（逻辑逻辑管理器）可以同时处理同一任务。这可以通过同时运行多个独立的子任务，或者多次运行同一任务以检查不同的输出来实现。并行化通常用于：
将子任务拆分并并行运行，这样可以提高速度。
多次运行任务以检查不同的输出，这可以提高置信度。
例如：
运行一个子任务来处理文档中的关键词，以及第二个子任务来检查格式错误。
多次运行一项任务，该任务根据不同的标准（例如引用次数、使用的来源数量和来源质量）对文档的准确性进行评分。
"""
import os
import getpass
from typing import Any, Dict, TypedDict
import yaml
from langchain_openai import ChatOpenAI
# 用于结构化输出的 Schema
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


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


# 图状态
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


# 节点定义
def call_llm_1(state: State):
    """第一次调用 LLM，生成笑话"""

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: State):
    """第二次调用 LLM，生成故事"""

    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: State):
    """第三次调用 LLM，生成诗歌"""

    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State):
    """将笑话、故事和诗歌组合成一个输出"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# 构建工作流
parallel_builder = StateGraph(State)

# 添加节点
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# 添加用于连接节点的边
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# 展示工作流
display(Image(parallel_workflow.get_graph().draw_mermaid_png()))

# 调用
state = parallel_workflow.invoke({"topic": "cats"})
print(state["combined_output"])