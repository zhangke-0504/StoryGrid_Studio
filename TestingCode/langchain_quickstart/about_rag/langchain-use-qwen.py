# langchain_community
from langchain_community.llms.tongyi import Tongyi
import yaml
"""
从项目根目录的config/aliyun/config.yaml文件中读取DASHSCOPE_API_KEY配置项的值，并将其赋值给变量api_key。
"""
with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)
aliyun_api_key = config.get("DASHSCOPE_API_KEY")
# print("aliyun_api_key:", aliyun_api_key)
model = Tongyi(model="qwen-max", api_key=aliyun_api_key)
res = model.invoke(input="请介绍一下你自己")
print(res)