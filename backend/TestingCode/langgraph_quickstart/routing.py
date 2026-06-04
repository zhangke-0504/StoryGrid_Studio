"""
路由工作流会处理输入，然后将其定向到特定上下文的任务。
这允许您为复杂任务定义专门的流程。
例如，一个用于回答产品相关问题的工作流可能会先处理问题类型，然后将请求路由到定价、退款、退货等特定流程。
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


# 用于作为路由逻辑的结构化输出 Schema
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None, description="The next step in the routing process"
    )


# 为 LLM 增加结构化输出能力
router = llm.with_structured_output(Route)


# 状态
class State(TypedDict):
    input: str
    decision: str
    output: str


# 节点定义
def llm_call_1(state: State):
    """生成一个故事"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State):
    """生成一个笑话"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State):
    """生成一首诗"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State):
    """将输入路由到合适的节点"""

    # 运行带结构化输出能力的 LLM，将其作为路由逻辑
    decision = router.invoke(
        [
            SystemMessage(
                content="Route the input to story, joke, or poem based on the user's request."
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}


# 用于路由到合适节点的条件边函数
def route_decision(state: State):
    # 返回下一步要访问的节点名称
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


# 构建工作流
router_builder = StateGraph(State)

# 添加节点
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# 添加用于连接节点的边
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # route_decision 返回的名称 : 下一步要访问的节点名称
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# 编译工作流
router_workflow = router_builder.compile()

# 展示工作流
display(Image(router_workflow.get_graph().draw_mermaid_png()))

# 调用
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])