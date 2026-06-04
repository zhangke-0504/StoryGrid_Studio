from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
import yaml
# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@tool(description="获取骨架，传入股票名称，返回字符串信息")
def get_price(name: str) -> str:
    """查询股票价格"""
    return f"股票{name}的价格是20元"


@tool(description="获取股票信息，传入股票名称，返回字符串信息")
def get_info(name: str) -> str:
    """查询股票信息"""
    return f"股票{name}的基本信息是：这是一个好股票"

agent = create_agent(
    model=ChatTongyi(model=config["chat_model"], api_key=config["DASHSCOPE_API_KEY"]),
    tools=[get_price, get_info],
    system_prompt="你是一个股票查询助手，用户会问你股票相关的问题，你需要调用工具来获取股票信息。"
)

for chunk in agent.stream(
    {"messages": [
        {"role": "user", "content": "A股的股价是多少，介绍一下"},
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