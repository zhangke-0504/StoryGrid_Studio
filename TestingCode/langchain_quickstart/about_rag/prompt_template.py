from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import yaml

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}，你帮我取个名字"
)

# prompt_text = prompt_template.format(lastname="张", gender="女儿")
# print(prompt_text)
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
model = Tongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))
# res = model.invoke(input=prompt_text)
# print(res)
# 链方式
chain = prompt_template | model

res = chain.invoke(input={
    "lastname": "张",
    "gender": "女儿"
})
print(res)

