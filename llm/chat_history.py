import chromadb
from sentence_transformers import SentenceTransformer

class ChatHistoryManager:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Tiny 80MB model
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("banking_chats")
    
    def add_message(self, role: str, content: str, metadata: dict = None):
        embedding = self.embedder.encode(content).tolist()
        self.collection.add(
            documents=content,
            embeddings=[embedding],
            metadatas=metadata or {},
            ids=str(len(self.collection.get()["ids"]) + 1)
        )
    
    def get_similar_messages(self, query: str, k=2):
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(query).tolist()],
            n_results=k
        )
        return [{"content": doc, "metadata": meta} 
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])]