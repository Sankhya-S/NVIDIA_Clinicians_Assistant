# backend/retrievers/hybrid_search_faiss.py

import faiss
import numpy as np
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from .hybrid_search import EnhancedHybridSearch, SearchConfig, RerankedResult, logger

class EnhancedHybridSearchFAISS(EnhancedHybridSearch):
    """Enhanced hybrid search using Milvus for dense and FAISS for sparse vectors"""
    
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.faiss_index = None
        self.faiss_mapping = {}  # Map FAISS indices to document IDs
        self.setup_faiss()
        
    def setup_faiss(self):
        """Initialize FAISS index for sparse vectors"""
        try:
            print("DEBUG: Initializing FAISS index for sparse vectors...")
            
            # Example: Dynamic determination of sparse vector dimensionality
            sparse_vector_dim = None
            if self.faiss_mapping:
                first_key = next(iter(self.faiss_mapping))
                sparse_vector_dim = len(self.faiss_mapping[first_key])
            else:
                sparse_vector_dim = 768  # Default value if no data exists
    
            self.faiss_index = faiss.IndexFlatIP(sparse_vector_dim)
            print(f"DEBUG: FAISS index initialized with dimension {sparse_vector_dim}")
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise


    def insert_batch(self, batch_data: List[Dict]) -> bool:
        """Insert a batch of documents into both stores"""
        try:
            print(f"\nDEBUG: Processing batch of {len(batch_data)} documents")
            
            # Prepare data for Milvus (dense vectors)
            milvus_data = [{
                'content': item['content'],
                'dense_vector': item['dense_vector'],
                'note_id': item['note_id'],
                'hadm_id': item['hadm_id'],
                'subject_id': item['subject_id'],
                'section': item['section']
            } for item in batch_data]
            
            # Insert into Milvus
            milvus_result = self.collection.insert(milvus_data)
            print("DEBUG: Inserted dense vectors into Milvus")
            
            # Prepare sparse vectors for FAISS
            start_idx = self.faiss_index.ntotal
            sparse_vectors = np.array([item['sparse_vector'] for item in batch_data]).astype('float32')
            
            # Insert into FAISS
            self.faiss_index.add(sparse_vectors)
            print("DEBUG: Inserted sparse vectors into FAISS")
            
            # Update mapping
            for i, item in enumerate(batch_data):
                self.faiss_mapping[start_idx + i] = item['content']
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting batch: {str(e)}")
            return False

    def hybrid_search(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 10) -> List[RerankedResult]:
        """Perform hybrid search using both Milvus and FAISS"""
        try:
            print("\nDEBUG: Starting hybrid search...")
            print(f"DEBUG: Query: {query}")
    
            # Generate query embeddings
            query_embeddings = self.ef.embed([query])
            print(f"DEBUG: Query embeddings structure: {query_embeddings}")
    
            # Search dense vectors in Milvus
            dense_results = self.collection.search(
                data=[query_embeddings['dense'][0].tolist()],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=k,
                output_fields=["content", "note_id", "hadm_id", "subject_id", "section"]
            )
            print(f"DEBUG: Found {len(dense_results[0])} dense vector matches")
    
            # Process sparse vector for FAISS
            sparse_vector = query_embeddings.get('sparse')
            if sparse_vector is None or not isinstance(sparse_vector, np.ndarray):
                raise ValueError("Sparse embedding is missing or invalid")
            if hasattr(sparse_vector, 'toarray'):
                sparse_vector = sparse_vector.toarray()[0]
            sparse_vector = sparse_vector.astype('float32').reshape(1, -1)
    
            # Search sparse vectors in FAISS
            sparse_scores, sparse_indices = self.faiss_index.search(sparse_vector, k)
            print(f"DEBUG: Found {len(sparse_indices[0])} sparse vector matches")
    
            # Combine results
            results = []
            for dense_hit, (sparse_score, sparse_idx) in zip(dense_results[0], zip(sparse_scores[0], sparse_indices[0])):
                dense_score = dense_hit.score
                final_score = (
                    self.config.dense_weight * dense_score +
                    self.config.sparse_weight * float(sparse_score)
                )
                document = Document(
                    page_content=dense_hit.entity.get('content'),
                    metadata={
                        'note_id': dense_hit.entity.get('note_id'),
                        'hadm_id': dense_hit.entity.get('hadm_id'),
                        'subject_id': dense_hit.entity.get('subject_id'),
                        'section': dense_hit.entity.get('section'),
                    }
                )
                results.append(RerankedResult(
                    document=document,
                    dense_score=dense_score,
                    sparse_score=float(sparse_score),
                    rerank_score=0.2,
                    final_score=final_score
                ))
            print(f"\nDEBUG: Final result count: {len(results)}")
            return self._apply_diversity_ranking(results, k)
    
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
