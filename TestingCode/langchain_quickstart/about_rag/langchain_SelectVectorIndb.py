from langchain_community.chat_models import ChatTongyi
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 读取配置
import yaml
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简介且专业的回答用户问题，参考资料：{context}"),
        ("user", "用户提问：{input}"),
    ]
)

vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(dashscope_api_key=config.get("DASHSCOPE_API_KEY"))
)

# 准备一下资料
vector_store.add_texts(["减肥就是要少吃多练","不在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来","跑步是很好的运动哦"])

input_text = "怎么减肥？"

# 检索向良库
result = vector_store.similarity_search(input_text, k=2)
reference_text = "["

for doc in result:
    reference_text += doc.page_content + "\n"
reference_text += "]"

print("参考资料：", reference_text)

def print_prompt(prompt):
    print("="*20, prompt.to_string(), "="*20)
    return prompt

chain = prompt | print_prompt | model | StrOutputParser()

res = chain.invoke({
    "context": reference_text,
    "input": input_text
})

print("模型回答：", res)