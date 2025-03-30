from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from .base_retriever import BaseMedicalRetriever, RetrieverConfig
from .hybrid_search_lite import HybridSearchLite, LiteSearchConfig

class HybridMedicalRetrieverLite(BaseMedicalRetriever):
    """Medical retriever using hybrid search with Milvus Lite"""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        
        # Create search config
        search_config = LiteSearchConfig(
            collection_name=config.collection_name,
            milvus_lite_db=config.milvus_lite_db,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
            rerank_weight=config.rerank_weight
        )
        
        # Initialize hybrid search
        self.hybrid_search = HybridSearchLite(search_config)
    
    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents for a query"""
        results = self.get_relevant_documents_with_scores(query, filters, **kwargs)
        return [doc for doc, _ in results]
    
    def get_relevant_documents_with_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with scores for a query"""
        k = kwargs.get('k', self.config.k_documents)
        
        # Perform hybrid search
        search_results = self.hybrid_search.hybrid_search(
            query=query,
            filters=filters,
            k=k
        )
        
        # Filter by score threshold
        filtered_results = [
            (result.document, result.final_score)
            for result in search_results
            if result.final_score >= self.config.score_threshold
        ]
        
        return filtered_results 