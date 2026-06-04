"""
编排器-工作节点工作流很常见，LangGraph 内置了对它们的支持。
APISend允许您动态创建工作节点并向其发送特定的输入。
每个工作节点都有自己的状态，所有工作节点的输出都会写入一个共享的状态键，编排器图可以访问该状态键。
这使得编排器能够访问所有工作节点的输出，并将它们合成为最终输出。
下面的示例遍历一个节列表，并使用SendAPI 将节发送给每个工作节点。
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
from langgraph.types import Send


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


# 图状态
class State(TypedDict):
    topic: str  # 报告主题
    sections: list[Section]  # 报告章节列表
    completed_sections: Annotated[
        list, operator.add
    ]  # 所有工作节点并行写入这个键
    final_report: str  # 最终报告


# 工作节点状态
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


# 节点定义
def orchestrator(state: State):
    """生成报告计划的编排器"""

    # 生成章节规划
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState):
    """工作节点负责撰写报告的一个章节"""

    # 生成章节内容
    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting."
            ),
            HumanMessage(
                content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"
            ),
        ]
    )

    # 将生成后的章节写入已完成章节列表
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """将各章节合成为完整报告"""

    # 已完成章节列表
    completed_sections = state["completed_sections"]

    # 将已完成章节格式化为字符串，用于最终报告
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# 条件边函数：创建 llm_call 工作节点，每个节点负责撰写一个章节
def assign_workers(state: State):
    """为计划中的每个章节分配一个工作节点"""

    # 通过 Send() API 并行启动章节写作任务
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


# 构建工作流
orchestrator_worker_builder = StateGraph(State)

# 添加节点
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# 添加用于连接节点的边
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# 编译工作流
orchestrator_worker = orchestrator_worker_builder.compile()

# 展示工作流
# display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

# 调用
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
print(state["final_report"])
# from IPython.display import Markdown
# Markdown(state["final_report"])