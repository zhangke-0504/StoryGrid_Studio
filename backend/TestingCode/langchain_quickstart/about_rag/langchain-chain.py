from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
import yaml
from langchain_core.runnables.base import RunnableSerializable

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是我的人工智能助手"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗，再来一个"),
    ("ai", "白日依山尽，黄河入海流，欲穷千里目，更上一层楼"),
]
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))

chain: RunnableSerializable = chat_prompt_template | model
print(type(chain))

print("Runnable接口,invoke方法：")
res = chain.invoke(input={
    "history": history_data
})
print(res.content)

print("\n流式接口,stream方法：")
for chunk in chain.stream(input={
    "history": history_data
}):
    print(chunk.content, end="", flush=True)