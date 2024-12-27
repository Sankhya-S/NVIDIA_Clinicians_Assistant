# backend/retrievers/basic_retriever.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from dataclasses import dataclass
import logging
from .hybrid_search import EnhancedHybridSearch, SearchConfig, RerankedResult

logger = logging.getLogger(__name__)

@dataclass
class RetrieverConfig:
    """Configuration settings for document retrieval."""
    k_documents: int = 10
    score_threshold: float = 0.7
    
    # Hybrid search specific settings
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    rerank_weight: float = 0.3
    cache_size: int = 1000

class BaseMedicalRetriever(ABC):
    """Base class defining the interface for medical document retrievers."""
    
    @abstractmethod
    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    def get_relevant_documents_with_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with their relevance scores."""
        pass

class BasicMedicalRetriever(BaseMedicalRetriever):
    """Standard retriever using basic vector similarity search."""
    
    def __init__(self, vectorstore, config: Optional[RetrieverConfig] = None):
        """Initialize the basic medical retriever.
        
        Args:
            vectorstore: The vectorstore containing document embeddings
            config: Configuration for retrieval behavior
        """
        self.vectorstore = vectorstore
        self.config = config or RetrieverConfig()
        self._retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.config.k_documents,
                "score_threshold": self.config.score_threshold,
            }
        )

    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using basic vector similarity."""
        try:
            k = kwargs.get('k', self.config.k_documents)
            search_kwargs = {"k": k}
            if filters:
                search_kwargs["filter"] = filters
            
            documents = self._retriever.get_relevant_documents(query, **search_kwargs)
            logger.info(f"Retrieved {len(documents)} documents using basic search")
            return documents
            
        except Exception as e:
            logger.error(f"Error in basic retrieval: {e}")
            raise

    def get_relevant_documents_with_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        try:
            k = kwargs.get('k', self.config.k_documents)
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filters
            )
            return results
            
        except Exception as e:
            logger.error(f"Error in basic scored retrieval: {e}")
            raise

class HybridMedicalRetriever(BaseMedicalRetriever):
    """Advanced retriever using hybrid search and reranking."""
    
    def __init__(self, vectorstore, config: Optional[RetrieverConfig] = None):
        """Initialize the hybrid medical retriever.
        
        Args:
            vectorstore: The vectorstore containing document embeddings
            config: Configuration for hybrid search behavior
        """
        self.vectorstore = vectorstore
        self.config = config or RetrieverConfig()
        
        # Initialize hybrid search
        search_config = SearchConfig(
            collection_name=self.vectorstore.collection.name,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            rerank_weight=self.config.rerank_weight,
            cache_size=self.config.cache_size
        )
        self.hybrid_search = EnhancedHybridSearch(search_config)
        
        # basic retriever as fallback
        self._basic_retriever = BasicMedicalRetriever(vectorstore, config)

    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""
        try:
            k = kwargs.get('k', self.config.k_documents)
            results = self.hybrid_search.hybrid_search(
                query=query,
                filters=filters,
                k=k
            )
            return [result.document for result in results]
            
        except Exception as e:
            logger.error(f"Error in hybrid search, falling back to basic: {e}")
            return self._basic_retriever.get_relevant_documents(
                query,
                filters=filters,
                k=k
            )

    def get_relevant_documents_with_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve documents with hybrid search scores."""
        try:
            k = kwargs.get('k', self.config.k_documents)
            results = self.hybrid_search.hybrid_search(
                query=query,
                filters=filters,
                k=k
            )
            return [(result.document, result.final_score) for result in results]
            
        except Exception as e:
            logger.error(f"Error in hybrid scored search, falling back to basic: {e}")
            return self._basic_retriever.get_relevant_documents_with_scores(
                query,
                filters=filters,
                k=k
            )

def create_retriever(
    vectorstore,
    use_hybrid: bool = False,
    config: Optional[RetrieverConfig] = None
) -> BaseMedicalRetriever:
    """Factory function to create the appropriate retriever.
    
    Args:
        vectorstore: The vectorstore containing document embeddings
        use_hybrid: Whether to use hybrid search capabilities
        config: Configuration for retrieval behavior
        
    Returns:
        An instance of the appropriate retriever type
    """
    if use_hybrid:
        return HybridMedicalRetriever(vectorstore, config)
    return BasicMedicalRetriever(vectorstore, config)