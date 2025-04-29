
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pymilvus import connections, utility
from langchain_milvus import Milvus
import os
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import torch
import numpy as np
# class BGEEmbeddingWrapper:
   # def __init__(self, device="cuda"):
       # self.model = BGEM3EmbeddingFunction(device=device)
    #def embed_query(self, text: str):
        # For basic RAG (standard vector search), return only dense vector
        #result = self.model([text])
        #return result["dense"][0]
   # def embed_documents(self, texts: list):
        #result = self.model(texts)
        #return result["dense"]
# === USE THIS FOR BASIC RAG ===
#def setup_embedding():
    #return BGEEmbeddingWrapper(device="cuda")
class BGEEmbeddingWrapper:
    def __init__(self, model_name, device="cuda"):
        self.model = BGEM3EmbeddingFunction(
            model_name=model_name,
            use_fp16=True,
            device=device if torch.cuda.is_available() else "cpu"
        )
    def embed_query(self, text: str):
        result = self.model([text])
        dense_vec = result["dense"][0]
        return np.array(dense_vec, dtype=np.float32)

    def embed_documents(self, texts: list):
        result = self.model(texts)
        dense_vecs = result["dense"]
        return np.array(dense_vecs, dtype=np.float32)
    
# Update your setup function to use your fine-tuned model
def setup_embedding():
    finetuned_model_path = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge/bge-m3-med-ft"
    return BGEEmbeddingWrapper(model_name=finetuned_model_path, device="cuda")
#def setup_embedding():
#    """Sets up NVIDIA embeddings for document processing."""
#    embedding_model = NVIDIAEmbeddings(
#        model_name="nvidia/nv-embedqa-e5-v5",
#        base_url="http://localhost:8089" 
#    )
#    return embedding_model
#    return BGEM3EmbeddingFunction(device="cuda")
def setup_chat_model():
    """Sets up NVIDIA chat completion model."""
    chat_model = ChatNVIDIA(
        model_name="meta/llama-3.1-nemotron-70b-instruct:latest",
        base_url="http://localhost:8088",
        max_tokens = 32768,
        temperature=0.1,  # Lower temperature for more focused responses
        model_kwargs={
            "do_sample": True,
            "top_p": 0.9
        }
    )
    return chat_model

def check_milvus_content():
    """Simple function to check what's in Milvus"""
    try:
        # Connect to Milvus
        connections.connect(alias="default", host="localhost", port="19530")
        # List all collections
        print("\nCollections in Milvus:")
        collections = utility.list_collections()
        print(collections)
        # Use same embedding model as setup
        embedding_model = setup_embedding()
        # Create Milvus instance with proper embedding
        vectorstore = Milvus(
            embedding_function=embedding_model,
            connection_args={"host": "localhost", "port": "19530"}
        )
        # Now we can do similarity search
        docs = vectorstore.similarity_search(
            query="test",  # simple query
            k=3
        )
        print("\nSample documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connections.disconnect("default")
# Run it
if __name__ == "__main__":
    check_milvus_content()
