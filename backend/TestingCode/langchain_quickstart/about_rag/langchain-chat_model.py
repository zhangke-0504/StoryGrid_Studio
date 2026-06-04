from langchain_community.chat_models.tongyi import ChatTongyi
import yaml

# 初始化模型
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
chat = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))

# 准备消息list
messages = [
    ("system", "你是一名来自边塞的诗人"),
    ("human", "什么唐诗描写了热恋期的男女"),
    ("ai", """红豆生南国，春来发几枝。
愿君多采撷，此物最相思。"""),
    ("human", "我很喜欢《相思》这首诗，但我更喜欢宋词，你能不能将《相思》改写成宋词？"),
]

"""
非简写形式如下：
messages = [
    SystemMessage(content="你是一名来自边塞的诗人"),
    HumanMessage(content="什么唐诗描写了热恋期的男女"),
    AIMessage(content="红豆生南国，春来发几枝。\n愿君多采撷，此物最相思。"),
    HumanMessage(content="我很喜欢《相思》这首诗，但我更喜欢宋词，你能不能将《相思》改写成宋词？"),
]
"""

# 流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)