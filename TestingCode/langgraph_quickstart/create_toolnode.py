"""
ToolNode是一个预构建节点，用于在 LangGraph 工作流中执行工具。它能够自动处理并行工具执行、错误处理和状态注入。
ToolNode当您需要对图的执行工具方式进行细粒度控制时，请使用此功能。它是许多 LangGraph 代理模式中工具执行的基础构建模块。
"""
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode([search, calculator]))
# ... add other nodes and edges
graph = builder.compile()