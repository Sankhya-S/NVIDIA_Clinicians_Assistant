# backend/retrievers/hybrid_search_faiss.py

import faiss
import numpy as np
from typing import List, Dict, Optional, Any
from langchain.schema import Document
from .hybrid_search import EnhancedHybridSearch, SearchConfig, RerankedResult, logger

class EnhancedHybridSearchFAISS(EnhancedHybridSearch):
    def __init__(self, config: SearchConfig):
        super().__init__(config)
        self.faiss_index = None
        self.faiss_mapping = {}
        self.setup_faiss()

    def setup_faiss(self):
        """Initialize the FAISS index."""
        try:
            sparse_vector_dim = 768  # Default dimension
            if self.faiss_mapping:
                first_key = next(iter(self.faiss_mapping))
                sparse_vector_dim = len(self.faiss_mapping[first_key])
            self.faiss_index = faiss.IndexFlatIP(sparse_vector_dim)
            logger.info(f"FAISS index initialized with dimension {sparse_vector_dim}")
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise

    def insert_batch(self, batch_data: List[Dict]) -> bool:
        """Insert a batch of documents into Milvus and FAISS."""
        try:
            logger.debug(f"Processing batch of {len(batch_data)} documents")

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
            self.collection.insert(milvus_data)
            logger.debug("Inserted dense vectors into Milvus")

            # Prepare sparse vectors for FAISS
            start_idx = self.faiss_index.ntotal
            sparse_vectors = np.array([item['sparse_vector'] for item in batch_data]).astype('float32')

            # Insert into FAISS
            self.faiss_index.add(sparse_vectors)
            logger.debug("Inserted sparse vectors into FAISS")

            # Update FAISS mapping
            for i, item in enumerate(batch_data):
                self.faiss_mapping[start_idx + i] = item['content']

            return True
        except Exception as e:
            logger.error(f"Error inserting batch: {str(e)}")
            return False

    def hybrid_search(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 10) -> List[RerankedResult]:
        """Perform hybrid search using both Milvus and FAISS."""
        try:
            logger.debug(f"Starting hybrid search with query: {query}")

            # Generate query embeddings
            query_embeddings = self.ef.embed([query])
            dense_vector = query_embeddings.get("dense")
            sparse_vector = query_embeddings.get("sparse")

            if dense_vector is None or sparse_vector is None:
                raise ValueError("Query embeddings missing 'dense' or 'sparse' components")

            # Search dense vectors in Milvus
            dense_results = self.collection.search(
                data=[dense_vector[0].tolist()],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=k,
                output_fields=["content", "note_id", "hadm_id", "subject_id", "section"]
            )
            logger.debug(f"Found {len(dense_results[0])} dense vector matches")

            # Search sparse vectors in FAISS
            sparse_vector = sparse_vector.astype("float32").reshape(1, -1)
            sparse_scores, sparse_indices = self.faiss_index.search(sparse_vector, k)
            logger.debug(f"Found {len(sparse_indices[0])} sparse vector matches")

            # Combine results
            results = []
            for dense_hit, (sparse_score, sparse_idx) in zip(
                dense_results[0], 
                zip(sparse_scores[0], sparse_indices[0])
            ):
                dense_score = dense_hit.score
                sparse_score = float(sparse_score)

                final_score = (
                    self.config.dense_weight * dense_score +
                    self.config.sparse_weight * sparse_score
                )

                document = Document(
                    page_content=dense_hit.entity.get("content"),
                    metadata={
                        "note_id": dense_hit.entity.get("note_id"),
                        "hadm_id": dense_hit.entity.get("hadm_id"),
                        "subject_id": dense_hit.entity.get("subject_id"),
                        "section": dense_hit.entity.get("section"),
                    }
                )

                results.append(RerankedResult(
                    document=document,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    rerank_score=0.0,  # Update if reranking is added
                    final_score=final_score
                ))

            logger.debug(f"Final results count: {len(results)}")
            return self._apply_diversity_ranking(results, k)
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
