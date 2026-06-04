# from langchain_community.embeddings import DashScopeEmbeddings
# import yaml

# with open("config/aliyun/config.yaml", "r") as f:
#     config = yaml.safe_load(f)
# embed = DashScopeEmbeddings(dashscope_api_key=config.get("DASHSCOPE_API_KEY"))

# print(embed.embed_query("我喜欢你"))  # 单次转换
# print(embed.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))  # 批量转换


# ============ 用ollama本地文本嵌入模型 =================
from langchain_ollama import OllamaEmbeddings
embed = OllamaEmbeddings(model="qwen3-embedding:4b")
print(embed.embed_query("我喜欢你"))  # 单次转换
print(embed.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))  # 批量转换