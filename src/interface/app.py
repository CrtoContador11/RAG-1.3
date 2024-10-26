import gradio as gr
from src.embeddings.embedding_manager import EmbeddingManager
from src.db.vector_store import VectorStore
from src.llm.model_manager import LLMManager
from src.utils.document_processor import DocumentProcessor

class AIAssistant:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.llm_manager = LLMManager()
        self.doc_processor = DocumentProcessor()
        
    def process_query(self, query: str) -> str:
        # Get query embedding
        query_embedding = self.embedding_manager.get_embeddings([query])[0]
        
        # Retrieve relevant documents
        results = self.vector_store.query(query_embedding)
        
        # Construct prompt with context
        context = "\n".join(results["documents"][0])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate response
        response = self.llm_manager.generate_response(prompt)
        return response

def create_interface():
    assistant = AIAssistant()
    
    interface = gr.Interface(
        fn=assistant.process_query,
        inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
        outputs=gr.Textbox(lines=8),
        title="AI Assistant",
        description="Ask questions about your knowledge base",
        theme="default",
        examples=[
            ["What are the best practices for software development?"],
            ["Explain the concept of machine learning."]
        ]
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)