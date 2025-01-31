from typing import List

from loguru import Logger
from qdrant_client import QdrantClient, models
from FlagEmbedding import FlagLLMReranker

from src.features.retrievers import sparse_queary, dense_query, dense_sparse_query


search_dict = {
    "sparse_queary": sparse_queary,
    "dense_query": dense_query,
    "dense_sparse_query": dense_sparse_query,
}


class CreateDataset:
    def __init__(self, reranker: FlagLLMReranker, collection_name: str, qdrant_url: str):
        self.reranker = reranker
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(qdrant_url)

    def query_records(self, collection_name: str, retriver_type: str, vector: dict, limit) -> list:

        try:
            search_type = search_dict[retriver_type]
        except Exception as e:
            Logger.error(
                f"Нет такого типа ретривера {retriver_type}. Доступны {list(search_dict.keys())}."
                f"Ошибка {e}"
            )
            raise
        return self.qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=search_type(vector),
            limit=limit,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            timeout=10_000,
        ).points

    def create_dataset(id_record: str, retriver_types: List[str], use_reranker: bool) -> List[dict]:
        

        pass
