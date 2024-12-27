from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pymilvus import connections, utility
from langchain_milvus import Milvus
import os

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
        model_name="meta/llama-3.1-nemotron-70b-instruct:latest",
        base_url="http://localhost:8000",
        max_tokens = 32768,
        temperature=0.1,  # Lower temperature for more focused responses
        model_kwargs={
            "do_sample": True,
            "top_p": 0.9
        }  
    )
    return chat_model

def get_pdf_paths(folder_path):
    """
    Collects all PDF file paths within a specified folder, including subfolders.
    """
    pdf_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

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