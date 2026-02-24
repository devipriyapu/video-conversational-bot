from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService


@dataclass
class RagResult:
    answer: str
    sources: list[dict]
    tokens_used: int


class RagService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        milvus_service: MilvusService,
        openai_client: OpenAI,
        chat_model: str,
        chunk_size: int,
        chunk_overlap: int,
        max_context_chunks: int,
    ) -> None:
        self.embedding_service = embedding_service
        self.milvus_service = milvus_service
        self.openai_client = openai_client
        self.chat_model = chat_model
        self.max_context_chunks = max_context_chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', ''],
        )

    def chunk_text(self, text: str) -> list[str]:
        return self.splitter.split_text(text)

    def build_prompt(self, question: str, context_chunks: list[str]) -> str:
        context = '\n\n'.join(context_chunks)
        return (
            'Answer ONLY using the provided context. '
            "If the answer is not in the context, say: 'I don't know based on the provided context.'\n\n"
            f'Context:\n{context}\n\nQuestion: {question}'
        )

    def answer_question(
        self,
        question: str,
        collection_name: str,
        top_k: int | None = None,
    ) -> RagResult:
        query_embedding = self.embedding_service.embed_text(question)
        hits = self.milvus_service.search(collection_name=collection_name, query_vector=query_embedding, top_k=top_k)

        context_hits = hits[: self.max_context_chunks]
        context_chunks = [item['text'] for item in context_hits if item.get('text')]

        if not context_chunks:
            return RagResult(
                answer="I don't know based on the provided context.",
                sources=[],
                tokens_used=0,
            )

        prompt = self.build_prompt(question, context_chunks)
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {'role': 'system', 'content': 'You are a strict RAG assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0,
        )

        answer = response.choices[0].message.content or "I don't know based on the provided context."
        tokens_used = int(response.usage.total_tokens) if response.usage else 0

        return RagResult(answer=answer, sources=context_hits, tokens_used=tokens_used)

    def stream_answer(self, question: str, collection_name: str, top_k: int | None = None):
        query_embedding = self.embedding_service.embed_text(question)
        hits = self.milvus_service.search(collection_name=collection_name, query_vector=query_embedding, top_k=top_k)

        context_hits = hits[: self.max_context_chunks]
        context_chunks = [item['text'] for item in context_hits if item.get('text')]

        if not context_chunks:
            def _empty_stream():
                if False:
                    yield None

            return _empty_stream(), []

        prompt = self.build_prompt(question, context_chunks)

        stream = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {'role': 'system', 'content': 'You are a strict RAG assistant.'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0,
            stream=True,
        )
        return stream, context_hits
