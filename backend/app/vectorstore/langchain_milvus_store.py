from __future__ import annotations

from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings


class LangChainMilvusFactory:
    """Optional LangChain adapter for teams that want retriever wrappers on top of pymilvus storage."""

    def __init__(self, uri: str, api_key: str, base_url: str, embedding_model: str) -> None:
        self.uri = uri
        self.embeddings = OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=embedding_model)

    def build_vectorstore(self, collection_name: str) -> Milvus:
        return Milvus(
            embedding_function=self.embeddings,
            connection_args={'uri': self.uri},
            collection_name=collection_name,
            vector_field='embedding',
            text_field='text',
            auto_id=True,
        )
