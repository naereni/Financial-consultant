import logging
import time
from typing import Dict, Any, List

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler

import src.settings as settings
from src.rag import RAG

logging.basicConfig()
if settings.VERBOSE:
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    logging.getLogger("langchain_community.chat_models.gigachat").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)

# logging.getLogger("langchain.chains.llm").setLevel(logging.ERROR)
_logger = logging.getLogger(__name__)


class Credit:
    def __init__(self, llm, emdeddings, memory=None):
        self.prompt_handler = CustomHandler()
        self.memory = memory or ConversationBufferMemory(
            memory_key="history", input_key="question", output_key="text"
        )
        self.rag: RetrievalQA = RAG(llm, emdeddings, self.memory).chain

    def get_answer(self, question):
        chain_invoke = self.rag.invoke(
            {"question": question, "history": self.memory.buffer_as_str},
            config={"callbacks": [self.prompt_handler]},
        )
        return chain_invoke.get("text")

    async def aget_answer(self, question):
        chain_invoke = await self.rag.ainvoke(
            {"question": question, "history": self.memory.buffer_as_str},
            config={"callbacks": [self.prompt_handler]},
        )
        return chain_invoke.get("text")

    def get_last_prompt(self):
        return self.prompt_handler.last_prompt

    def get_last_route(self):
        return self.prompt_handler.last_route


class CustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.last_route = None
        self.last_prompt = None

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        formatted_prompts = "\n".join(prompts)
        self.last_prompt = formatted_prompts

    def on_text(self, text: str, **kwargs: Any) -> Any:
        if any(text.startswith(s) for s in ["common", "capacity"]):
            self.last_route = text.split(": ")[0]


if __name__ == "__main__":
    settings = settings.GigaSettings(stand="ext")
    giga = DepositHelper(settings.chat_model, settings.embeddings)
    while True:
        user_input = input("Вопрос: ")
        start = time.time()
        answer = giga.get_answer(user_input)
        print(f"Время ответа: {time.time() - start} сек.")
        print(f"Ответ: {answer}")
        # print("Начало исходного документа: ", res.get("source_documents"))
        # print(qa_chain.combine_documents_chain.memory)
        print()
