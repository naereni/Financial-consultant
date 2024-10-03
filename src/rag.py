from typing import Optional

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.vectorstores import VectorStoreRetriever

import src.settings as settings
from src.common_prompt import FINAL_PROMPT_START
from src.documents_db import get_vector_db


class RAG:
    _retriever: Optional[VectorStoreRetriever] = None

    def __init__(self, llm, emdeddings, memory=None):
        self.memory = memory or ConversationBufferMemory(
            memory_key="history", input_key="question", output_key="text"
        )
        self.chain: RetrievalQA = RetrievalQA.from_chain_type(
            llm,
            retriever=RAG.get_retriever(emdeddings),
            chain_type="stuff",
            # Should be one of "stuff", "map_reduce", # "map_rerank", and "refine". ,
            return_source_documents=True,
            verbose=settings.VERBOSE,
            input_key="question",
            output_key="text",
            chain_type_kwargs={
                "verbose": settings.VERBOSE,
                "prompt": RAG.get_prompt(),
                "memory": self.memory,
                "output_key": "text",
            },
        )

    @classmethod
    def get_retriever(cls, emdeddings):
        if not cls._retriever:
            cls._retriever = get_vector_db(emdeddings).as_retriever(
                search_type="mmr",
                # Can be "similarity" (default), "mmr", or "similarity_score_threshold"
                search_kwargs={"k": 3, "score_threshold": 0.9, "lambda_mult": 1},
            )
        return cls._retriever

    @staticmethod
    def get_prompt():
        system_prompt_template = f"""{FINAL_PROMPT_START}
----------------
При подготовке ответа клиенту используй следующую информацию, как основополагающую.
{{context}}
----------------
Текущий разговор:
{{history}}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt
