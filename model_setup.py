from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pymilvus import connections
from langchain_milvus import Milvus

def setup_milvus(embedding_model, docs, batch_size=10):
    """Sets up the Milvus vector store using the NVIDIA embedding model."""
    
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")
    
    # Create a Milvus vector store
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embedding_model,  
        connection_args={
            "host": "localhost",
            "port": "19530",
        },
        drop_old=True  # Drop the old Milvus collection if it exists to avoid duplication
    )
    
    return vectorstore


def setup_embedding():
    """Sets up NVIDIA embeddings for document processing."""
    embedding_model = NVIDIAEmbeddings(
        model_name="nvidia/nv-embedqa-e5-v5",
        base_url="http://localhost:8001" 
    )
    return embedding_model


def setup_chat_model():
    """Sets up NVIDIA chat completion model."""
    chat_model = ChatNVIDIA(
        model_name="meta/llama-3.1-8b-instruct",
        base_url="http://localhost:8000"  
    )
    return chat_model
