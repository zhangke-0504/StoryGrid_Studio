from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import yaml
# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

str_parser = StrOutputParser()
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})

model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}， 刚生了{gender}, 请起名，仅告诉我名字，不要额外信息"
)

second_prompt = PromptTemplate.from_template(
    "姓名是：{name}，请告诉我这个名字的含义是什么？"
)

chain = first_prompt | model | my_func | second_prompt | model | str_parser
res: str = chain.invoke({
    "lastname": "张",
    "gender": "女儿"
})
print(res)

# 流式输出
for chunk in chain.stream({
    "lastname": "张",
    "gender": "女儿"
}):
    print(chunk, end="", flush=True)
