"""
在评估器-优化器工作流程中，一个LLM调用生成响应，另一个调用评估该响应。
如果评估器或人工参与判断认为响应需要改进，则会提供反馈并重新生成响应。
此循环持续进行，直至生成可接受的响应。
当任务有特定的成功标准，但需要迭代才能满足这些标准时，通常会使用评估器-优化器工作流程。
例如，在两种语言之间翻译文本时，并非总能找到完美匹配的译文。可能需要多次迭代才能生成在两种语言中含义相同的译文。
"""
import os
import getpass
from typing import Any, Dict, TypedDict
from typing_extensions import Literal
import yaml
from langchain_openai import ChatOpenAI
# 用于结构化输出的 Schema
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain.messages import HumanMessage, SystemMessage


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
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


# Schema for structured output to use in evaluation
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not.",
    )
    feedback: str = Field(
        description="If the joke is not funny, provide feedback on how to improve it.",
    )


# 为 LLM 增加结构化输出能力
evaluator = llm.with_structured_output(Feedback)


# 节点定义
def llm_call_generator(state: State):
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""

    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


# 条件边函数：根据评估器反馈决定回到笑话生成节点还是结束
def route_joke(state: State):
    """Route back to joke generator or end based upon feedback from the evaluator"""

    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


# 构建工作流
optimizer_builder = StateGraph(State)

# 添加节点
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

# 添加用于连接节点的边
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {  # Name returned by route_joke : Name of next node to visit
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# 编译工作流
optimizer_workflow = optimizer_builder.compile()

# 展示工作流
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# 调用
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])