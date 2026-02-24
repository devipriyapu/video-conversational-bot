from __future__ import annotations

import re
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
        context_hits = self._retrieve_context_hits(question, collection_name, top_k)
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
        context_hits = self._retrieve_context_hits(question, collection_name, top_k)
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

    def _retrieve_context_hits(self, question: str, collection_name: str, top_k: int | None) -> list[dict]:
        list_intent = self._is_list_or_type_question(question)
        base_top_k = top_k or self.milvus_service.top_k
        candidate_top_k = max(base_top_k, 10 if list_intent else 8)

        variants = self._query_variants(question)
        merged_hits: list[dict] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for variant in variants:
            query_embedding = self.embedding_service.embed_text(variant)
            hits = self.milvus_service.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=candidate_top_k,
            )
            for hit in hits:
                metadata = hit.get('metadata') or {}
                key = (
                    str(metadata.get('video_id', '')),
                    str(metadata.get('chunk_index', '')),
                    str(hit.get('text', '')),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_hits.append(hit)

        ranked = sorted(merged_hits, key=lambda item: self._rank_score(question, item), reverse=True)

        if list_intent:
            ranked = self._boost_type_definition_chunks(question, ranked)

        return ranked[: self.max_context_chunks]

    def _query_variants(self, question: str) -> list[str]:
        normalized = question.strip()
        if not normalized:
            return ['']

        simplified = re.sub(r'\s+', ' ', normalized.lower())
        variants = [normalized, simplified]

        digit_expanded = simplified
        for digit, word in {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five'}.items():
            digit_expanded = re.sub(rf'\b{digit}\b', word, digit_expanded)
        variants.append(digit_expanded)

        if self._is_list_or_type_question(simplified):
            variants.extend(
                [
                    f'{digit_expanded} ani agi asi',
                    'three types of artificial intelligence',
                    'artificial narrow intelligence artificial general intelligence artificial superintelligence',
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            key = variant.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _rank_score(self, question: str, hit: dict) -> float:
        base = float(hit.get('score') or 0.0)
        text = str(hit.get('text') or '').lower()
        metadata = hit.get('metadata') or {}

        q_tokens = [token for token in re.findall(r'[a-z0-9]+', question.lower()) if len(token) > 2]
        if not q_tokens:
            return base

        overlap = sum(1 for token in q_tokens if token in text)
        lexical_bonus = overlap / max(len(q_tokens), 1)

        intent_bonus = 0.0
        if self._is_list_or_type_question(question):
            if all(keyword in text for keyword in ['narrow', 'general', 'super']):
                intent_bonus += 0.25
            if any(keyword in text for keyword in ['ani', 'agi', 'asi']):
                intent_bonus += 0.15
            chunk_index = str(metadata.get('chunk_index', ''))
            if chunk_index == '0':
                intent_bonus += 0.03

        return base + (0.2 * lexical_bonus) + intent_bonus

    def _boost_type_definition_chunks(self, question: str, ranked: list[dict]) -> list[dict]:
        def _priority(item: dict) -> int:
            text = str(item.get('text') or '').lower()
            if all(keyword in text for keyword in ['narrow', 'general', 'super']):
                return 0
            if any(keyword in text for keyword in ['ani', 'agi', 'asi']):
                return 1
            return 2

        return sorted(ranked, key=lambda item: (_priority(item), -self._rank_score(question, item)))

    @staticmethod
    def _is_list_or_type_question(question: str) -> bool:
        lowered = question.lower()
        return any(token in lowered for token in ['type', 'types', 'kind', 'kinds', 'list']) and (
            ' ai' in lowered
            or 'artificial intelligence' in lowered
            or 'ani' in lowered
            or 'agi' in lowered
            or 'asi' in lowered
        )
