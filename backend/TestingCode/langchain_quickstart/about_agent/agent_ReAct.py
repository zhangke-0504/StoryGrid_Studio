from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
import yaml

# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@tool(description="获取体重，返回值是整数，单位千克")
def get_weight(name: str) -> str:
    """查询体重"""
    return f"{name}的体重是70千克"


@tool(description="获取身高，返回值是整数，单位厘米")
def get_height(name: str) -> str:
    """查询身高"""
    return f"{name}的身高是170厘米" 

agent = create_agent(
    model=ChatTongyi(model=config["chat_model"], api_key=config["DASHSCOPE_API_KEY"]),
    tools=[get_weight, get_height],
    system_prompt="""
你是严格遵循ReAct框架的智能体，必须按[思考->行动->观察->再思考]的流程解决问题，
且**每轮仅能思考并调用1个工具**， 禁止单次调用多个工具。
并告知我你的思考过程，工具的调用原因，按思考，行动，观察三个结构告知我
"""
)

for chunk in agent.stream(
    {"messages": [
        {"role": "user", "content": "计算我的BMI"},
    ]},
    stream_mode="values"
):
    latest_message = chunk['messages'][-1]

    if latest_message.content:
        print(type(latest_message).__name__, latest_message.content)

    try:
        if latest_message.tool_calls:
            print(f"工具调用: { [tc['name'] for tc in latest_message.tool_calls] }")
    except AttributeError as e:
        pass