from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader
# from langchain_chroma import Chroma
import yaml
from pathlib import Path


with open("config/aliyun/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# In-memory vector store (kept for reference)
# -----------------------------
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(dashscope_api_key=config.get("DASHSCOPE_API_KEY"))
)

loader = CSVLoader(file_path="data/sample_data.csv", encoding="utf-8")
documents = loader.load()

vector_store.add_documents(
    documents=documents,
    ids=["id"+str(i) for i in range(1, len(documents)+1)]
)

# 删除
vector_store.delete(ids=["id1"])

# 检索
results = vector_store.similarity_search(
    query="这是一个测试的行",
    k=2  # 检索的结果数量
)

print(results)

# loader = CSVLoader(file_path="data/sample_data.csv", encoding="utf-8")
# documents = loader.load()

# embeddings = DashScopeEmbeddings(dashscope_api_key=config.get("DASHSCOPE_API_KEY"))


# # Use Chroma persistent store
# persist_dir = Path("data/chroma_db")
# persist_dir.mkdir(parents=True, exist_ok=True)

# chroma_store = Chroma(
#     collection_name="sample_collection",
#     embedding_function=embeddings,
#     persist_directory=str(persist_dir),
# )

# ids = ["id" + str(i) for i in range(1, len(documents) + 1)]
# chroma_store.add_documents(documents=documents, ids=ids)

# if hasattr(chroma_store, "_client") and hasattr(chroma_store._client, "persist"):
#     try:
#         chroma_store._client.persist()
#     except Exception:
#         pass

# results = chroma_store.similarity_search(query="这是一个测试的行", k=2)
# print("Chroma results:", results)