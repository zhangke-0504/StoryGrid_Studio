"""
智能体通常以逻辑逻辑模型（LLM）的形式实现，使用工具执行操作。
它们在持续的反馈循环中运行，适用于问题和解决方案不可预测的情况。
智能体比工作流拥有更大的自主性，可以自行决定使用哪些工具以及如何解决问题。
您仍然可以定义可用的工具集以及智能体的行为准则。
"""
import os
import getpass
from typing import Any, Dict, TypedDict
from typing_extensions import Literal
import yaml
from langchain_openai import ChatOpenAI
# 用于结构化输出的 Schema
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessagesState
from IPython.display import Image, display
from langchain.messages import HumanMessage, SystemMessage, ToolMessage


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


from langchain.tools import tool


# 定义工具
@tool
def multiply(a: int, b: int) -> int:
    """将 `a` 和 `b` 相乘。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """将 `a` 和 `b` 相加。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """将 `a` 除以 `b`。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a / b


# 为 LLM 绑定工具
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


# 节点定义
def llm_call(state: MessagesState):
    """由 LLM 决定是否调用工具"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState):
    """执行工具调用"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# 条件边函数：根据 LLM 是否发起工具调用，决定路由到工具节点还是结束
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """根据 LLM 是否发起工具调用，决定继续循环还是停止"""

    messages = state["messages"]
    last_message = messages[-1]

    # 如果 LLM 发起工具调用，就执行相应操作
    if last_message.tool_calls:
        return "tool_node"

    # 否则停止并回复用户
    return END


# 构建工作流
agent_builder = StateGraph(MessagesState)

# 添加节点
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 添加用于连接节点的边
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# 编译智能体
agent = agent_builder.compile()

# 展示智能体
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# 调用
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()