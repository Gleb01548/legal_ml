from functools import partial
from typing import List

from loguru import logger
from qdrant_client import QdrantClient, models
from FlagEmbedding import FlagLLMReranker

from src.utils.embeddnig_api import EmbeddingAPI
from src.features.retrievers import sparse_queary, dense_query, dense_sparse_query


search_dict = {
    "sparse_queary": sparse_queary,
    "dense_query": dense_query,
    "dense_sparse_query": partial(dense_sparse_query, limit_dense=100),
}
# string_context = """
# "Системой провиден поиск и найдены похожие вопросы от ДРУГИХ пользователей "
# "и ответы на них других юристов. "
# "Изучи их, чтобы лучше понять как отвечать на вопрос пользователя:\n"
# """


class Rag:
    def __init__(
        self,
        collection_name: str,
        qdrant_url: str,
        embedding_model: str,
        retriver_types: str = dense_query,
        reranker: FlagLLMReranker = None,
        limit: int = 5,
    ):
        self.collection_name = collection_name
        self.reranker = reranker
        self.qdrant_client = QdrantClient(qdrant_url)
        self.retriver = search_dict[retriver_types]
        self.embeding_model = EmbeddingAPI(embedding_model)
        self.limit = limit

    def query_records(self, vector: dict) -> list:

        return self.qdrant_client.query_points(
            collection_name=self.collection_name,
            prefetch=self.retriver(vector),
            limit=100,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            timeout=10_000,
        ).points

    def reranker_func(self, question, points):
        texts = [i.payload["question"] for i in points]
        try:
            score = self.reranker.compute_score([[question, i] for i in texts])
        except Exception as e:
            logger.info(f"Ошибка: {e}. {texts}")
            logger.info(f"Ошибка: {[[question, i] for i in texts]}")

            raise

        index_score = sorted(
            [(index, score) for index, score in enumerate(score)],
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_points = [points[i[0]] for i in index_score]

        return reranked_points

    def find_record(self, user_message: str) -> List[dict]:
        vector = self.embeding_model.get_embeddings(user_message)
        vector = {
            "dense": vector["dense_vecs"],
            "text-sparse": models.SparseVector(
                indices=list(vector["lexical_weights"].keys()),
                values=list(vector["lexical_weights"].values()),
            ),
        }

        points = self.query_records(vector=vector)

        print(points)

        if self.reranker:
            points = self.reranker_func(question=user_message, points=points)

        return [i.payload for i in points[: self.limit]]

    def create_context(self, user_message) -> str:
        points = self.find_record(user_message=user_message)

        context = ""

        for i in points:
            context += "Вопрос пользователя:\n"
            context += i["question"] + "\n\n"
            context += "Ответ юриста:\n"
            context += i["answer"] + "\n\n"

        return context
