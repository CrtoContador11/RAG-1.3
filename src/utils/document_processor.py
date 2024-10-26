import os
from typing import List, Dict
from pathlib import Path

class DocumentProcessor:
    def __init__(self, base_path: str = r"C:\Users\danie\OneDrive - SCP Ozona SL\Escritorio\RAG\knowledge"):
        self.base_path = Path(base_path)
    
    def scan_documents(self) -> List[Dict]:
        documents = []
        for file_path in self.base_path.rglob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {
                        "path": str(file_path),
                        "category": file_path.parent.name
                    }
                })
        return documents