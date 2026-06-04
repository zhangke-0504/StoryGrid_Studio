from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.llms.tongyi import Tongyi
import yaml

example_prompt = PromptTemplate.from_template(
    "单词：{word}, 反义词：{antonym}"
)

expample_data = [
    {"word": "高兴", "antonym": "难过"},
    {"word": "大", "antonym": "小"},
]

few_shot_prompt = FewShotPromptTemplate(
    examples=expample_data,
    example_prompt=example_prompt,
    prefix="给出给定次的反义词，有如下示例：",
    suffix="基于示例告诉我： {input_word}的反义词是什么？",
    input_variables=["input_word"]
)

prompt_text = few_shot_prompt.invoke(input={"input_word": "快乐"}).to_string()
print(prompt_text)

# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
# 实例化模型
model = Tongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))
res = model.invoke(input=prompt_text)
print(res)
    