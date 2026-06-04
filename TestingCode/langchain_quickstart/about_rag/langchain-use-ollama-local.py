from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

res = model.invoke(input="请介绍一下你自己")
print(res)