from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
import yaml

parser = StrOutputParser()
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}， 刚生了{gender}, 请起名，仅告知我名字无需其他内容"
)
chain = prompt | model | parser | model
res = chain.invoke({
    "lastname": "张",
    "gender": "女儿"
})
print(res.content)