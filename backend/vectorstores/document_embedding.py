# vectorestores/document_embedding.py 
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility, connections
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from pathlib import Path

# Add project root to path 
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from model_setup import setup_embedding, setup_chat_model

from backend.processors.document_processor import process_note_sections, debug_print
from backend.retrievers.hybrid_search import *

def create_collection(collection_name: str, dim: int, with_metadata: bool = True) -> Collection:
    """Create Milvus collection with specified schema and optimized for hybrid search."""
    connections.connect(alias="default", host="localhost", port="19530")
    
    if utility.has_collection(collection_name):
        return Collection(collection_name)
        
    # Base fields
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    ]
    
    # Metadata fields
    if with_metadata:
        metadata_fields = [
            FieldSchema(name="note_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hadm_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="charttime", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="storetime", dtype=DataType.VARCHAR, max_length=100),
            # Add structured keyword fields for better search
            FieldSchema(name="keywords_array", dtype=DataType.ARRAY, max_capacity=50, dtype_of_sub=DataType.VARCHAR),
            FieldSchema(name="keyword_weights", dtype=DataType.ARRAY, max_capacity=50, dtype_of_sub=DataType.FLOAT)
        ]
        fields.extend(metadata_fields)
    
    schema = CollectionSchema(fields=fields)
    collection = Collection(collection_name, schema)
    
    # Create indexes
    # Vector index
    vector_index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", vector_index_params)
    
    # Create indexes on metadata fields for efficient filtering
    if with_metadata:
        collection.create_index("note_id", {"index_type": "INVERTED"})
        collection.create_index("hadm_id", {"index_type": "INVERTED"})
        collection.create_index("subject_id", {"index_type": "INVERTED"})
        collection.create_index("section", {"index_type": "INVERTED"})
        collection.create_index("keywords_array", {"index_type": "INVERTED"})
    
    return collection

def load_json_note(file_path: str) -> dict:
    """Load and validate a single JSON note."""
    try:
        with open(file_path, 'r') as f:
            note = json.load(f)
        return note
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def chunks_to_documents(chunks: List[Dict], include_metadata: bool = True) -> List[Document]:
    """Convert chunks to Langchain Document objects."""
    documents = []
    for chunk in chunks:
        metadata = {}
        if include_metadata:
            metadata = {
                "note_id": str(chunk['metadata']['note_id']),
                "hadm_id": str(chunk['metadata']['hadm_id']),
                "subject_id": str(chunk['metadata']['subject_id']),
                "section": chunk['section'],
                "charttime": str(chunk['metadata']['charttime']),
                "storetime": str(chunk['metadata']['storetime'])
            }
            if 'keywords' in chunk:
                metadata['keywords'] = chunk['keywords']
        
        doc = Document(
            page_content=chunk['content'],
            metadata=metadata
        )
        documents.append(doc)
    return documents

def save_basic_chunks(chunks: List[Dict], embedding_model, collection_name: str = "medical_notes_basic"):
    """Save chunks with just content and embeddings."""
    print(f"\nDEBUG: Starting save_basic_chunks with {len(chunks)} chunks")
    
    documents = chunks_to_documents(chunks, include_metadata=True)
    print(f"\nDEBUG: Converted to {len(documents)} documents")
    
    try:
        print("\nDEBUG: Creating vectorstore...")
        vectorstore = setup_milvus(
            embedding_model=embedding_model,
            docs=documents
        )
        print("DEBUG: Vectorstore created successfully")
        return vectorstore, None
        
    except Exception as e:
        print(f"DEBUG: Error creating vectorstore: {str(e)}")
        raise


def save_detailed_chunks(
    chunks: List[Dict], 
    embedding_model, 
    collection_name: str,
    enable_hybrid: bool = False
) -> Tuple[Any, Optional[Exception]]:
    try:
        documents = chunks_to_documents(chunks, include_metadata=True)
        print(f"\nDEBUG: Processing {len(documents)} documents")
        
        if enable_hybrid:
            print("\nDEBUG: Setting up hybrid search...")
            search_config = SearchConfig(
                collection_name=collection_name,
                dense_weight=0.4,
                sparse_weight=0.3,
                rerank_weight=0.3
            )
            
            # Initialize hybrid search
            hybrid_search = EnhancedHybridSearch(search_config)
            
            # Insert documents into collection
            print("\nDEBUG: Inserting documents into hybrid search...")
            data = []
            for doc in documents:
                # Get BGE embeddings for the document
                embeddings = hybrid_search.ef([doc.page_content])
                dense_vector = embeddings['dense'][0].tolist()  # Convert to list
                sparse_vector = embeddings['sparse'][0].tolist()  # Convert to list
                
                # Verify dimensions
                assert len(dense_vector) == hybrid_search.ef.dim["dense"], \
                    f"Dense vector dimension mismatch: expected {hybrid_search.ef.dim['dense']}, got {len(dense_vector)}"
                assert len(sparse_vector) == hybrid_search.ef.dim["sparse"], \
                    f"Sparse vector dimension mismatch: expected {hybrid_search.ef.dim['sparse']}, got {len(sparse_vector)}"
    
                data.append({
                    'content': doc.page_content,
                    'dense_vector': dense_vector,
                    'sparse_vector': sparse_vector,
                    'note_id': doc.metadata.get('note_id'),
                    'hadm_id': doc.metadata.get('hadm_id'),
                    'subject_id': doc.metadata.get('subject_id'),
                    'section': doc.metadata.get('section')
                })
            
            # Insert data into collection
            hybrid_search.collection.insert(data)
            hybrid_search.collection.flush()
            print(f"\nDEBUG: Inserted {len(data)} documents into hybrid search")
            
            return hybrid_search, None
            
        else:
            print("\nDEBUG: Setting up standard vectorstore...")
            vectorstore = setup_milvus(
                embedding_model=embedding_model,
                docs=documents
            )
            return vectorstore, None
            
    except Exception as e:
        print(f"\nDEBUG: Error in save_detailed_chunks: {e}")
        raise

    
def display_chunks(chunks, max_display=10):
    """Display chunks with detailed information."""
    print("\n" + "="*80)
    print(f"Displaying up to {max_display} chunks")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:max_display]):
        print(f"\nChunk {i+1}:")
        print("-"*40)
        print(f"Section: {chunk['section']}")
        print(f"Metadata: note_id={chunk['metadata']['note_id']}, " 
              f"hadm_id={chunk['metadata']['hadm_id']}")
        print("Content:")
        print("-"*20)
        print(chunk['content'])
        if 'keywords' in chunk:
            print("Keywords:")
            print(chunk['keywords'])
        print("-"*40)
    
    if len(chunks) > max_display:
        print(f"\n... and {len(chunks) - max_display} more chunks")

