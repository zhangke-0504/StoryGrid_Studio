"""
提示链式调用是指每个 LLM 调用都处理前一个调用的输出。它通常用于执行定义明确、可以分解为更小、可验证步骤的任务。例如：
将文件翻译成不同语言
验证生成内容的一致性
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
    improved_joke: str
    final_joke: str


# 节点定义
def generate_joke(state: State):
    """第一次调用 LLM，生成初始笑话"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_punchline(state: State):
    """用于检查笑话是否有包袱的门控函数"""

    # 简单检查：笑话里是否包含 "?" 或 "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"


def improve_joke(state: State):
    """第二次调用 LLM，对笑话进行改进"""

    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """第三次调用 LLM，做最终润色"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


# 构建工作流
workflow = StateGraph(State)

# 添加节点
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

# 添加用于连接节点的边
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# 编译
chain = workflow.compile()

# 展示工作流
display(Image(chain.get_graph().draw_mermaid_png()))

# 调用
state = chain.invoke({"topic": "cats"})
print("Initial joke:")
print(state["joke"])
print("\n--- --- ---\n")
if "improved_joke" in state:
    print("Improved joke:")
    print(state["improved_joke"])
    print("\n--- --- ---\n")

    print("Final joke:")
    print(state["final_joke"])
else:
    print("Final joke:")
    print(state["joke"])