import chromadb
from chromadb.config import Settings
from typing import List, Dict
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = r"C:\Users\danie\OneDrive - SCP Ozona SL\Escritorio\RAG\db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("knowledge_base")
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        if metadata is None:
            metadata = [{}] * len(documents)
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    def query(self, query_embedding: List[float], n_results: int = 5):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results