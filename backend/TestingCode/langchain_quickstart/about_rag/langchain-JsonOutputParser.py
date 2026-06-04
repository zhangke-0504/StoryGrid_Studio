from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import yaml

# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

str_parser = StrOutputParser()
json_parser = JsonOutputParser()

model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}， 刚生了{gender}, 请起名，并封装到json格式返回给我，"
    "要求key是name， value是起的名字，请严格遵守格式要求"
)

second_prompt = PromptTemplate.from_template(
    "姓名是：{name}，请告诉我这个名字的含义是什么？"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

res: str = chain.invoke({
    "lastname": "张",
    "gender": "女儿"
})
print(res)
print(type(res))

# 流式输出
for chunk in chain.stream({
    "lastname": "张",
    "gender": "女儿"
}):
    print(chunk, end="", flush=True)