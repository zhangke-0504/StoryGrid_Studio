
from pathlib import Path
import json
import yaml

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


class FileChatMessageHistory(BaseChatMessageHistory):
	"""File-backed chat history persisting messages per session to JSON files.

	Implements the `BaseChatMessageHistory` interface so it can be used by
	`RunnableWithMessageHistory` which expects a `BaseChatMessageHistory`.
	"""

	def __init__(self, session_id: str, dir_path: str = "data/chat_history"):
		self.session_id = session_id
		self.dir_path = Path(dir_path)
		self.dir_path.mkdir(parents=True, exist_ok=True)
		self.file_path = self.dir_path / f"{session_id}.json"

	@property
	def messages(self):
		try:
			raw = json.loads(self.file_path.read_text(encoding="utf-8"))
			return messages_from_dict(raw)
		except FileNotFoundError:
			return []
		except Exception:
			return []

	def add_messages(self, messages_list) -> None:
		try:
			existing = list(self.messages)
			existing.extend(messages_list)
			serialized = [message_to_dict(m) for m in existing]
			self.file_path.parent.mkdir(parents=True, exist_ok=True)
			self.file_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
		except Exception:
			# best-effort fallback: ignore write errors
			return

	def clear(self) -> None:
		try:
			self.file_path.parent.mkdir(parents=True, exist_ok=True)
			self.file_path.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
		except Exception:
			return


# 读取配置
with open("config/aliyun/config.yaml", "r") as f:
	config = yaml.safe_load(f)


def print_prompt(full_prompt):
	print("=" * 20, full_prompt.to_string(), "=" * 20)
	return full_prompt


model = ChatTongyi(model="qwen-max", api_key=config.get("DASHSCOPE_API_KEY"))

prompt = ChatPromptTemplate.from_messages(
	[
		("system", "你需要根据对话历史回应用户问题"),
		MessagesPlaceholder("chat_history"),
		("human", "用户当前输入：{input}, 请给出回应"),
	]
)

base_chain = prompt | print_prompt | model | StrOutputParser()

# 存放多个会话id对应的历史会话记录（内存缓存，但内容持久化到文件）
chat_history_store = {}


def get_history(session_id: str):
	if session_id not in chat_history_store:
		chat_history_store[session_id] = FileChatMessageHistory(session_id)
	return chat_history_store[session_id]


conversation_chain = RunnableWithMessageHistory(
	base_chain,
	get_history,
	input_messages_key="input",
	history_messages_key="chat_history",
)


if __name__ == "__main__":
	session_config = {"configurable": {"session_id": "user_002"}}

	print(conversation_chain.invoke({"input": "小明有一只猫"}, config=session_config))

	print(conversation_chain.invoke({"input": "小刚有两只猫"}, config=session_config))

	print(conversation_chain.invoke({"input": "小明和小刚一共有几只猫？"}, config=session_config))
