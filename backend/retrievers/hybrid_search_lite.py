import faiss
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from langchain.schema import Document
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import logging

logger = logging.getLogger(__name__)

class RerankedResult:
    """Structure for storing search results with multiple scoring components"""
    def __init__(self, document, dense_score, sparse_score, rerank_score, final_score):
        self.document = document
        self.dense_score = dense_score
        self.sparse_score = sparse_score
        self.rerank_score = rerank_score
        self.final_score = final_score

@dataclass
class LiteSearchConfig:
    """Configuration for Milvus Lite hybrid search"""
    collection_name: str
    milvus_lite_db: str = "./milvus_medical.db"
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    rerank_weight: float = 0.3

class HybridSearchLite:
    """Hybrid search implementation using Milvus Lite for dense and FAISS for sparse vectors"""
    
    def __init__(self, config: LiteSearchConfig):
        self.config = config
        
        # Initialize embedding and reranking functions
        self.ef = BGEM3EmbeddingFunction(
            use_fp16=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reranker = BGERerankFunction(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Milvus Lite client
        self.client = MilvusClient(self.config.milvus_lite_db)
        print(f"Using Milvus Lite with DB: {self.config.milvus_lite_db}")
        
        # Initialize FAISS index for sparse vectors
        self.faiss_index = None
        self.faiss_mapping = {}  # Map FAISS indices to document IDs
        self.setup_collection()
        self.setup_faiss()
    
    def setup_collection(self):
        """Set up Milvus Lite collection for dense vectors"""
        # Check if collection exists
        collections = self.client.list_collections()
        if self.config.collection_name not in collections:
            # Create collection with schema for hybrid search
            print(f"Creating collection {self.config.collection_name} for hybrid search")
            self.client.create_collection(
                collection_name=self.config.collection_name,
                dimension=1024  # For dense vectors
            )
        
        # Load collection
        self.client.load_collection(self.config.collection_name)
    
    def setup_faiss(self):
        """Initialize FAISS index for sparse vectors"""
        print("Initializing FAISS index for sparse vectors...")
        self.faiss_index = faiss.IndexFlatIP(250002)  # BGE-M3 sparse dimension
        print("FAISS index initialized")
    
    def insert_documents(self, documents: List[Dict]) -> bool:
        """Insert documents into both Milvus Lite and FAISS"""
        try:
            print(f"Processing batch of {len(documents)} documents for hybrid search")
            
            # Prepare data for Milvus Lite (dense vectors)
            milvus_data = []
            sparse_vectors = []
            
            # Process each document
            for doc in documents:
                # Get embeddings
                content = doc['content']
                embeddings = self.ef([content])
                
                # Extract dense vector
                dense_vector = embeddings['dense'][0].tolist()
                
                # Extract sparse vector
                sparse_vector = embeddings['sparse'][0]
                if hasattr(sparse_vector, 'toarray'):
                    sparse_vector = sparse_vector.toarray()[0]
                sparse_vector = sparse_vector.astype('float32')
                sparse_vectors.append(sparse_vector)
                
                # Prepare data for Milvus Lite
                milvus_data.append({
                    'content': content,
                    'vector': dense_vector,
                    'note_id': doc['metadata'].get('note_id', ''),
                    'hadm_id': doc['metadata'].get('hadm_id', ''),
                    'subject_id': doc['metadata'].get('subject_id', ''),
                    'section': doc['section']
                })
            
            # Insert into Milvus Lite
            self.client.insert(
                collection_name=self.config.collection_name,
                data=milvus_data
            )
            print(f"Inserted {len(milvus_data)} dense vectors into Milvus Lite")
            
            # Insert into FAISS
            start_idx = self.faiss_index.ntotal
            sparse_vectors = np.array(sparse_vectors).astype('float32')
            self.faiss_index.add(sparse_vectors)
            print(f"Inserted {len(sparse_vectors)} sparse vectors into FAISS")
            
            # Update mapping
            for i, doc in enumerate(documents):
                self.faiss_mapping[start_idx + i] = doc['content']
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 10
    ) -> List[RerankedResult]:
        """Perform hybrid search using both Milvus Lite and FAISS"""
        try:
            # Generate query embeddings
            print(f"Starting hybrid search for query: {query}")
            query_embeddings = self.ef([query])
            
            # Search dense vectors in Milvus Lite
            filter_expr = None
            if filters:
                # Convert filters dict to Milvus filter expression
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} == '{value}'")
                    elif isinstance(value, (list, tuple)):
                        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                        filter_parts.append(f"{key} in [{values_str}]")
                    else:
                        filter_parts.append(f"{key} == {value}")
                filter_expr = " && ".join(filter_parts)
            
            dense_results = self.client.search(
                collection_name=self.config.collection_name,
                data=[query_embeddings['dense'][0].tolist()],
                filter=filter_expr,
                limit=k,
                output_fields=["content", "note_id", "hadm_id", "subject_id", "section"]
            )
            print(f"Found {len(dense_results)} dense vector matches")
            
            # Search sparse vectors in FAISS
            sparse_vector = query_embeddings['sparse'][0]
            if hasattr(sparse_vector, 'toarray'):
                sparse_vector = sparse_vector.toarray()[0]
            sparse_vector = sparse_vector.astype('float32').reshape(1, -1)
            
            sparse_scores, sparse_indices = self.faiss_index.search(sparse_vector, k)
            print(f"Found {len(sparse_indices[0])} sparse vector matches")
            
            # Process and combine results
            results = []
            
            # Process dense results
            for i, dense_hit in enumerate(dense_results[0]):
                dense_score = dense_hit.get('distance', 0.0)
                content = dense_hit.get('content', '')
                
                # Find matching sparse result
                sparse_score = 0.0
                for j, idx in enumerate(sparse_indices[0]):
                    if idx < len(self.faiss_mapping) and self.faiss_mapping[idx] == content:
                        sparse_score = float(sparse_scores[0][j])
                        break
                
                # Calculate combined score
                final_score = (
                    self.config.dense_weight * dense_score +
                    self.config.sparse_weight * sparse_score
                )
                
                # Create document
                document = Document(
                    page_content=content,
                    metadata={
                        'note_id': dense_hit.get('note_id', ''),
                        'hadm_id': dense_hit.get('hadm_id', ''),
                        'subject_id': dense_hit.get('subject_id', ''),
                        'section': dense_hit.get('section', ''),
                    }
                )
                
                results.append(RerankedResult(
                    document=document,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rerank_score=0.0,  # No reranking for now
                    final_score=final_score
                ))
            
            # Sort by final score
            results.sort(key=lambda x: x.final_score, reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise 