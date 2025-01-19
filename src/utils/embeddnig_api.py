import requests

from typing import List

class EmbeddingAPI:
    def __init__(self, url:str):
        self.url = url

    def get_embeddings(self, texts: List[str]) -> List[List[str]]:
        return requests.post(self.url, json={"inputs": texts}, timeout=10).json()

