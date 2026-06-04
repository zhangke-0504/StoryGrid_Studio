from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from streamlit.runtime import Runtime
import yaml
from langchain.agents.middleware import AgentState, before_agent, after_agent, before_model, after_model, wrap_model_call, wrap_tool_call

# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

@tool(description="查询天气, 查询城市名称字符串，返回天气信息字符串")
def get_weather(city: str) -> str:
    return f"{city}的天气是晴天"

"""
1. agent执行前
2. agent执行后
3. model执行前
4. model执行后
5. 工具执行中
6. 模型执行中
"""

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[before_agent]agent启动, 并附带{len(state['messages'])}消息")


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_agent]agent结束, 并附带{len(state['messages'])}消息")

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[before_model]模型调用, 并附带{len(state['messages'])}消息")

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_model]模型调用结束, 并附带{len(state['messages'])}消息")

@wrap_model_call
def model_call_hook(request, handler):
    print("模型调用啦")
    return handler(request)

@wrap_tool_call
def monitor_tool(request, handler):
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数:{request.tool_call['args']}")
    return handler(request)


agent = create_agent(
    model=ChatTongyi(model=config["chat_model"], api_key=config["DASHSCOPE_API_KEY"]),
    tools=[get_weather],
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool]
)

res = agent.invoke(
    {"messages": [
        {"role": "user", "content": "深圳今天天气如何呀？"}
    ]}
)

print("*********\n", res)
