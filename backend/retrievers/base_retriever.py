# backend/retrievers/basic_retriever.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from dataclasses import dataclass
import logging
from .hybrid_search import EnhancedHybridSearch, SearchConfig, RerankedResult
from pymilvus import MilvusClient
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import EmbeddingsRedundantFilter, LongContextReorder

logger = logging.getLogger(__name__)

@dataclass
class RetrieverConfig:
    """Configuration settings for document retrieval."""
    k_documents: int = 10
    score_threshold: float = 0.7

    milvus_lite_db: str = "./milvus_medical.db"  # Path to the Milvus Lite DB file
    collection_name: str = "medical_notes"       # Collection name to use
    use_milvus_lite: bool = False                # Whether to use Milvus Lite instead of regular Milvus

    
    # Hybrid search specific settings
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    rerank_weight: float = 0.3
    cache_size: int = 1000

    # New enhanced retrieval settings
    use_multi_query: bool = False  
    use_contextual_compression: bool = False  
    compression_similarity_threshold: float = 0.75  
    llm: Optional[Any] = None #for multi-query retriever

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
    
    def __init__(self, vectorstore, config: Optional[RetrieverConfig] = None, embedding_model=None):
        """Initialize the basic medical retriever.
        """
        self.config = config or RetrieverConfig()
        self.embedding_model = embedding_model
    
        if self.config.use_milvus_lite:
            # Use Milvus Lite instead of standard vectorstore
            self.vectorstore = None
            self.milvus_client = MilvusClient(self.config.milvus_lite_db)
            self.collection_name = self.config.collection_name
            logger.info(f"Initialized Milvus Lite retriever with db: {self.config.milvus_lite_db}")
        else:
            # Use standard vectorstore 
            self.vectorstore = vectorstore
            self.milvus_client = None
            
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": self.config.k_documents,
                    "score_threshold": self.config.score_threshold,
                }
            )
            # Store the base retriever for fallback
            self._base_retriever = base_retriever
            self._retriever = base_retriever

            # Apply document transformers and contextual compression if configured
            if getattr(self.config, 'use_contextual_compression', False) and embedding_model is not None:
                # Create document transformer pipeline
                redundant_filter = EmbeddingsRedundantFilter(
                    embeddings=embedding_model,
                    similarity_threshold=self.config.compression_similarity_threshold
                )
                reordering = LongContextReorder()
                
                # Create pipeline of transformers
                compressor_pipeline = DocumentCompressorPipeline(
                    transformers=[redundant_filter, reordering]
                )
                
                # Create compression retriever
                self._retriever = ContextualCompressionRetriever(
                    base_compressor=compressor_pipeline,
                    base_retriever=self._retriever
                )
                logger.info("Initialized retriever with contextual compression")

            # Apply multi-query enhancement if configured
            if getattr(self.config, 'use_multi_query', False):
                try:
                    if self.config.llm is not None:
                        # Create multi-query retriever with the provided LLM
                        self._retriever = MultiQueryRetriever.from_llm(
                            retriever=self._retriever,
                            llm=self.config.llm
                        )
                        logger.info("Initialized retriever with multi-query capability")
                    else:
                        logger.warning("Multi-query retrieval enabled but no LLM provided")
                except Exception as e:
                    logger.error(f"Failed to initialize multi-query retriever: {e}")
                    # Fallback to previous retriever
                    pass
            


    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using basic vector similarity.
        
        Args:
            query: The query text to search for
            filters: Optional metadata filters to apply
            **kwargs: Additional arguments including:
                - k: Number of documents to retrieve (overrides config)
                - query_vector: Pre-computed query vector (required for Milvus Lite)
        
        Returns:
            List of relevant documents
        """
        try:
            k = kwargs.get('k', self.config.k_documents)
            
            if self.config.use_milvus_lite:
                # Use Milvus Lite client for search
                # We need to get the vector embedding for the query
                if 'query_vector' not in kwargs:
                    raise ValueError("When using Milvus Lite, you must provide 'query_vector' in kwargs")
                
                query_vector = kwargs['query_vector']
                filter_expr = None
                
                if filters:
                    # Convert filters dict to Milvus filter expression
                    filter_parts = []
                    for key, value in filters.items():
                        if isinstance(value, str):
                            filter_parts.append(f"{key} == '{value}'")
                        elif isinstance(value, (list, tuple)):
                            # Handle IN operator for lists
                            values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                            filter_parts.append(f"{key} in [{values_str}]")
                        else:
                            filter_parts.append(f"{key} == {value}")
                    filter_expr = " && ".join(filter_parts)
                
                # Search in Milvus Lite
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    filter=filter_expr,
                    limit=k,
                    output_fields=["content", "note_id", "hadm_id", "subject_id", "section", "charttime", "storetime"]
                )
                
                # Convert Milvus Lite results to Documents
                documents = []
                if search_results and len(search_results) > 0:
                    for hit in search_results[0]:
                        metadata = {
                            "note_id": hit.get("note_id", ""),
                            "hadm_id": hit.get("hadm_id", ""),
                            "subject_id": hit.get("subject_id", ""),
                            "section": hit.get("section", ""),
                            "score": hit.get("score", 0.0)
                        }
                        # Add optional fields if present
                        if "charttime" in hit:
                            metadata["charttime"] = hit["charttime"]
                        if "storetime" in hit:
                            metadata["storetime"] = hit["storetime"]
                        
                        doc = Document(
                            page_content=hit.get("content", ""),
                            metadata=metadata
                        )
                        documents.append(doc)
                
                logger.info(f"Retrieved {len(documents)} documents using Milvus Lite")
                return documents
            else:
                # Use standard vectorstore
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
    document_chunks=None,
    embedding_model=None,
    config: Optional[RetrieverConfig] = None
) -> BaseMedicalRetriever:
    """Factory function to create the appropriate retriever.
    
    Args:
        document_chunks: Optional list of document chunks to initialize vectorstore
        embedding_model: Embedding model for creating vectors
        config: Configuration for retrieval behavior
        
    Returns:
        An instance of the appropriate retriever type
    """
    config = config or RetrieverConfig()
    
    if config.use_milvus_lite:
        # Set up Milvus Lite
        client = MilvusClient(config.milvus_lite_db)
        
        # Check if collection exists, create if not
        if config.collection_name not in client.list_collections():
            logger.info(f"Creating new collection '{config.collection_name}' in Milvus Lite")
            
            # Try to get embedding dimension from model
            dimension = 384  # Default dimension
            if embedding_model:
                try:
                    sample_embedding = embedding_model.embed_query("Test")
                    dimension = len(sample_embedding)
                except:
                    pass
            
            # Create collection with auto_id=True to handle ID generation
            client.create_collection(
                collection_name=config.collection_name,
                dimension=dimension,
                auto_id=True
            )
            
            # If we have document chunks, add them to the collection
            if document_chunks:
                logger.info(f"Adding {len(document_chunks)} document chunks to Milvus Lite collection")
                batch_size = 100
                for i in range(0, len(document_chunks), batch_size):
                    batch = document_chunks[i:i+batch_size]
                    data = []
                    
                    for chunk in batch:
                        # Get vector using embedding model
                        content = chunk.get("content", "")
                        embedding = embedding_model.embed_query(content)
                        
                        # Prepare document data
                        doc_data = {
                            "vector": embedding,
                            "content": content,
                            "note_id": str(chunk["metadata"]["note_id"]),
                            "hadm_id": str(chunk["metadata"]["hadm_id"]),
                            "subject_id": str(chunk["metadata"]["subject_id"]),
                            "section": chunk["section"]
                        }
                        # Add optional fields if they exist
                        if "charttime" in chunk["metadata"]:
                            doc_data["charttime"] = str(chunk["metadata"]["charttime"])
                        if "storetime" in chunk["metadata"]:
                            doc_data["storetime"] = str(chunk["metadata"]["storetime"])
                            
                        data.append(doc_data)
                    
                    # Insert batch
                    client.insert(config.collection_name, data)
        
        # Create retriever - fixed the attribute check
        # Only check hasattr first before accessing the attribute
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            return HybridMedicalRetriever(None, config)
        else:
            return BasicMedicalRetriever(None, config, embedding_model)
    else:
        # Use standard Milvus with vectorstore
        from backend.vectorstores.document_embedding import (
            save_basic_chunks,
            save_detailed_chunks
        )
        
        # If we have document chunks, create vectorstore
        vectorstore = None
        if document_chunks and embedding_model:
            # Create vectorstore using existing implementation
            if hasattr(config, 'use_medical_sections') and config.use_medical_sections:
                vectorstore, _ = save_detailed_chunks(
                    document_chunks,
                    embedding_model,
                    config.collection_name,
                    enable_hybrid=hasattr(config, 'use_hybrid') and config.use_hybrid
                )
            else:
                vectorstore, _ = save_basic_chunks(
                    document_chunks,
                    embedding_model,
                    config.collection_name
                )
        
        # Create retriever based on vectorstore - fixed attribute check
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            return HybridMedicalRetriever(vectorstore, config)
        else:
            return BasicMedicalRetriever(vectorstore, config, embedding_model)
