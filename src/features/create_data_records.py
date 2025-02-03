from typing import List

from loguru import logger
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from FlagEmbedding import FlagLLMReranker

from src.features.retrievers import sparse_queary, dense_query, dense_sparse_query


search_dict = {
    "sparse_queary": sparse_queary,
    "dense_query": dense_query,
    "dense_sparse_query": dense_sparse_query,
}


class CreateDataRecords:
    def __init__(self, reranker: FlagLLMReranker, qdrant_url: str):
        self.reranker = reranker
        self.qdrant_client = QdrantClient(qdrant_url)

    def query_records(
        self, collection_name: str, retriver_type: str, vector: dict, limit: int
    ) -> list:

        try:
            search_type = search_dict[retriver_type]
        except Exception as e:
            logger.error(
                f"Нет такого типа ретривера {retriver_type}. Доступны {list(search_dict.keys())}."
                f"Ошибка {e}"
            )
            raise
        return self.qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=search_type(vector),
            limit=limit + 5,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            timeout=10_000,
        ).points[1:]

    def reranker_func(self, question, points):
        texts = [i.payload["question"] for i in points]
        score = self.reranker.compute_score([[question, i] for i in texts])
        index_score = sorted(
            [(index, score) for index, score in enumerate(score)],
            key=lambda x: x[1],
            reverse=True,
        )
        reranked_points = [points[i[0]] for i in index_score]

        return reranked_points

    def create_record(
        self,
        collection_name: str,
        id_record: str,
        retriver_type: str,
        use_reranker: bool = False,
        limit: int = 3,
    ) -> List[dict]:
        record = self.qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[id_record],
            with_vectors=True,
            with_payload=True,
        )[0]

        vector = record.vector
        payload = record.payload

        points = self.query_records(
            collection_name=collection_name,
            retriver_type=retriver_type,
            vector=vector,
            limit=limit,
        )

        points = [i for i in points if i.payload["question"] != payload["question"]]

        if use_reranker:
            points = self.reranker_func(payload["question"], points)

        record_result = {"question_answer": payload, "context": [i.payload for i in points]}

        return record_result

    def create_data_records(
        self,
        collection_name: str,
        ids: List[int],
        retriver_type: str,
        use_reranker: bool,
        limit: int,
    ):
        records = []
        for id in tqdm(ids):
            records.append(
                self.create_record(collection_name, id, retriver_type, use_reranker, limit)
            )
        return records
