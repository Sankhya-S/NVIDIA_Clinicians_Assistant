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

def insert_documents_into_milvus_lite(
    client, 
    collection_name: str, 
    document_chunks: list, 
    embedding_model
):
    """Directly insert document chunks into Milvus Lite collection."""
    import logging
    logger = logging.getLogger(__name__)
    print(f"Inserting {len(document_chunks)} chunks into Milvus Lite collection {collection_name}")
    
    # First, check if collection exists
    if collection_name not in client.list_collections():
        print(f"Collection {collection_name} doesn't exist, creating it now...")
        
        # Get embedding dimension
        sample_text = "Sample text for dimension detection"
        sample_embedding = embedding_model.embed_query(sample_text)
        dimension = len(sample_embedding)
        print(f"Using embedding dimension: {dimension}")
        
        # Create collection with explicit schema
        collection_schema = {
            "collection_name": collection_name,
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True, "auto_id": True},
                {"name": "vector", "type": "float_vector", "params": {"dim": dimension}},
                {"name": "content", "type": "varchar", "params": {"max_length": 65535}},
                {"name": "note_id", "type": "varchar", "params": {"max_length": 100}},
                {"name": "hadm_id", "type": "varchar", "params": {"max_length": 100}},
                {"name": "subject_id", "type": "varchar", "params": {"max_length": 100}},
                {"name": "section", "type": "varchar", "params": {"max_length": 100}}
            ]
        }
        
        try:
            client.create_collection(collection_schema)
            print(f"Successfully created collection {collection_name}")
        except Exception as e:
            print(f"Error creating collection with schema: {e}")
            # Fallback to simpler creation
            client.create_collection(
                collection_name=collection_name,
                dimension=dimension
            )
        
        # Create an index for vector search
        try:
            client.create_index(
                collection_name=collection_name,
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE"
            )
            print("Created search index")
        except Exception as e:
            print(f"Error creating index: {e}")
    
    # Load the collection
    try:
        client.load_collection(collection_name)
        print(f"Loaded collection {collection_name}")
    except Exception as e:
        print(f"Error loading collection: {e}")
    
    # Insert documents in batches
    batch_size = 50
    total_inserted = 0
    
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(document_chunks)-1)//batch_size + 1}")
        
        # Prepare data for insertion
        data = []
        for chunk in batch:
            try:
                # Get content and create embedding
                content = chunk.get("content", "")
                if not content:
                    print("Empty content in chunk, skipping")
                    continue
                
                # Create embedding
                embedding = embedding_model.embed_query(content)
                
                # Get metadata
                metadata = chunk.get("metadata", {})
                
                # Create entity for insertion
                entity = {
                    "vector": embedding,
                    "content": content,
                    "note_id": str(metadata.get("note_id", "")),
                    "hadm_id": str(metadata.get("hadm_id", "")),
                    "subject_id": str(metadata.get("subject_id", "")),
                    "section": str(chunk.get("section", ""))
                }
                
                data.append(entity)
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        # Insert batch if we have data
        if data:
            try:
                print(f"Inserting batch of {len(data)} documents")
                # Debug data format
                print(f"First entity fields: {list(data[0].keys())}")
                print(f"Vector dimension: {len(data[0]['vector'])}")
                
                # Use insert method
                result = client.insert(collection_name, data)
                print(f"Insertion result: {result}")
                
                if isinstance(result, dict) and "insert_count" in result:
                    total_inserted += result["insert_count"]
                    print(f"Batch inserted: {result['insert_count']} documents")
                else:
                    print(f"Unexpected insert result format: {result}")
            except Exception as e:
                print(f"Error inserting batch: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
    
    # Flush to ensure data is written
    try:
        client.flush()
        print("Flushed collection to ensure data persistence")
    except Exception as e:
        print(f"Error flushing collection: {e}")
    
    # Verify insertion
    try:
        stats = client.get_collection_stats(collection_name)
        row_count = stats.get("row_count", 0)
        print(f"Collection stats after insertion: {stats}")
        print(f"Total documents in collection: {row_count}")
        
        if row_count > 0:
            print("Documents successfully inserted!")
            
            # Sample query to verify searchability
            try:
                print("Testing search functionality:")
                sample_query = "patient"
                sample_vector = embedding_model.embed_query(sample_query)
                
                search_results = client.search(
                    collection_name=collection_name,
                    data=[sample_vector],
                    limit=5,
                    output_fields=["content", "note_id"]
                )
                
                if search_results and len(search_results) > 0 and len(search_results[0]) > 0:
                    print(f"Search successful! Found {len(search_results[0])} results")
                    print(f"First result: {search_results[0][0]}")
                else:
                    print("Search returned no results")
            except Exception as e:
                print(f"Error testing search: {e}")
        else:
            print("WARNING: No documents inserted!")
    except Exception as e:
        print(f"Error verifying insertion: {e}")
    
    return total_inserted

def create_retriever(
    document_chunks=None,
    embedding_model=None,
    config: Optional[RetrieverConfig] = None
) -> BaseMedicalRetriever:
    """Factory function to create the appropriate retriever."""
    config = config or RetrieverConfig()
    
    if config.use_milvus_lite:
        # Set up Milvus Lite client
        client = MilvusClient(config.milvus_lite_db)
        print(f"Creating retriever for Milvus Lite collection: {config.collection_name}")
        
        # Check if collection exists
        collections = client.list_collections()
        print(f"Available collections: {collections}")
        
        if config.collection_name not in collections:
            print(f"Collection {config.collection_name} not found. You need to create and populate it first.")
            # We'll just create an empty collection as a fallback
            try:
                # Get embedding dimension
                dimension = 384  # Default dimension
                if embedding_model:
                    try:
                        sample_embedding = embedding_model.embed_query("Test")
                        dimension = len(sample_embedding)
                        print(f"Using embedding dimension: {dimension}")
                    except Exception as e:
                        print(f"Error determining embedding dimension: {e}")
                
                # Create collection
                print(f"Creating empty collection {config.collection_name} with dimension {dimension}")
                client.create_collection(
                    collection_name=config.collection_name,
                    dimension=dimension
                )
            except Exception as e:
                print(f"Error creating empty collection: {e}")
        
        # Create appropriate retriever based on config
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            print("Creating hybrid medical retriever with Milvus Lite")
            retriever = HybridMedicalRetriever(None, config)
        else:
            print("Creating basic medical retriever with Milvus Lite")
            retriever = BasicMedicalRetriever(None, config, embedding_model)
        
        return retriever
    else:
        # Use standard Milvus with vectorstore
        from backend.vectorstores.document_embedding import (
            save_basic_chunks,
            save_detailed_chunks
        )
        
        # Create vectorstore if we have document chunks
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
        
        # Create retriever based on vectorstore
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            return HybridMedicalRetriever(vectorstore, config)
        else:
            return BasicMedicalRetriever(vectorstore, config, embedding_model)
