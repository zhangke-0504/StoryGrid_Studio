from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

prompt_text = chat_prompt_template.invoke({
    "history": history_data
}).to_string()

print(prompt_text)

# 读取配置
import yaml
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
from langchain_community.chat_models.tongyi import ChatTongyi
model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))
res = model.invoke(input=prompt_text)
print(res.content)