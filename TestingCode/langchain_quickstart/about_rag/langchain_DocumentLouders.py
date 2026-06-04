print("#############CSVLoader#############")
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data/sample_data.csv", 
    encoding="utf-8"
)
documents = loader.load()
print(documents)

documents = loader.load()
print(documents)

print("=============CSV 批量加载==============")
for document in documents:
    print(type(document), document)

print("=============CSV 懒加载==============")
for document in loader.lazy_load():
    print(type(document), document)

print("#############JSONLoader#############")
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path="data/sample_data.json", 
    jq_schema=".content",
    text_content=True,  # 抽取的是否是字符串，默认为True
    json_lines=True,  # 是否是JsonLines文件（每一行都是json文件）
)

documents = loader.load()
print(documents)

print("=============JSON 批量加载==============")
for document in documents:
    print(type(document), document)

print("=============JSON 懒加载==============")
for document in loader.lazy_load():
    print(type(document), document)


print("#############TextTLoader#############")
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    file_path="data/sample_data.txt", 
    encoding="utf-8"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,   # 分段的最大长度
    chunk_overlap=0,  # 分段之间的重叠长度
    separators=["\n\n", "\n", " ", ""],  # 分段的分隔符列表，优先级从高到低
    length_function=len,  # 计算文本长度的函数，默认为len
)

split_docs = splitter.split_documents(docs)
for doc in split_docs:
    print(type(doc), doc)

print("############PyqPDFLoader#############")
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="data/sample_data.pdf",
    mode="page", # 加载模式，默认为"page"，表示按页加载；如果设置为"text"，则将整个PDF作为一个文档加载
    # password="password"
)

i = 0
for doc in loader.lazy_load():
    print(type(doc), doc)
    i += 1
    print(doc)
    print("=" * 20, i)

