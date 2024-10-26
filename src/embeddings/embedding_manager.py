from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def get_embeddings(self, texts: List[str]):
        return self.model.encode(texts)