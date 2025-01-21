from typing import List, Optional

from FlagEmbedding import BGEM3FlagModel


class EmbeddingAPI:
    def __init__(self, model: Optional[str] = None):
        self.model = BGEM3FlagModel(model)

    def get_embeddings(self, texts: List[str]) -> List[List[str]]:
        return self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
