from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
import yaml

# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@tool(description="查询天气")
def get_weather():
    return "晴天"

agent = create_agent(
    model=ChatTongyi(model=config["chat_model"], api_key=config["DASHSCOPE_API_KEY"]),
    tools=[get_weather],
    system_prompt="你是一个天气查询助手，用户会问你天气相关的问题，你需要调用工具来获取天气信息。"
)

res = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "明天深圳天气如何呀？"}
        ]
    }
)

parser = StrOutputParser()

for msg in res["messages"]:
    print(f"{type(msg).__name__}: {parser.invoke(msg)}")