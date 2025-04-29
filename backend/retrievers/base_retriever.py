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

    milvus_lite_db: str = "./milvus_medical_M.db"  # Path to the Milvus Lite DB file
    collection_name: str = "medical_notes"       # Collection name to use
    use_milvus_lite: bool = False                # Whether to use Milvus Lite instead of regular Milvus

     # Hybrid search flag
    use_hybrid: bool = False   
    
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
    embedding_model: Optional[Any] = None

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
                
                # Define all output fields explicitly - IMPORTANT for content retrieval
                output_fields = [
                    "content",      # This is critical - make sure content is included
                    "note_id", 
                    "hadm_id", 
                    "subject_id", 
                    "section", 
                    "charttime", 
                    "storetime"
                ]
                
                # Search in Milvus Lite with explicit output fields
                print(f"Searching Milvus Lite collection with k={k}, filter={filter_expr}")
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    filter=filter_expr,
                    limit=k,
                    output_fields=output_fields  # Explicitly defined fields
                )
                
                # Convert Milvus Lite results to Documents
                documents = []
                if search_results and len(search_results) > 0:
                    for hit in search_results[0]:
                        # Print the full hit for debugging
                        # print(f"Debug - Search hit: {hit}")
                        
                        # Extract content first, ensuring it's present
                        content = hit.get("content", "")
                        if not content and "entity" in hit and isinstance(hit["entity"], dict):
                            # Sometimes content is in the entity field
                            content = hit["entity"].get("content", "")
                        
                        metadata = {
                            "note_id": hit.get("note_id", ""),
                            "hadm_id": hit.get("hadm_id", ""),
                            "subject_id": hit.get("subject_id", ""),
                            "section": hit.get("section", ""),
                            "score": hit.get("distance", 0.0)  # Note: May be "score" or "distance"
                        }
                        
                        # Check if fields are in entity structure
                        if "entity" in hit and isinstance(hit["entity"], dict):
                            if not metadata["note_id"]:
                                metadata["note_id"] = hit["entity"].get("note_id", "")
                            if not metadata["hadm_id"]:
                                metadata["hadm_id"] = hit["entity"].get("hadm_id", "")
                            if not metadata["subject_id"]:
                                metadata["subject_id"] = hit["entity"].get("subject_id", "")
                            if not metadata["section"]:
                                metadata["section"] = hit["entity"].get("section", "")
                        
                        # Add optional fields if present
                        if "charttime" in hit:
                            metadata["charttime"] = hit["charttime"]
                        elif "entity" in hit and "charttime" in hit["entity"]:
                            metadata["charttime"] = hit["entity"]["charttime"]
                            
                        if "storetime" in hit:
                            metadata["storetime"] = hit["storetime"]
                        elif "entity" in hit and "storetime" in hit["entity"]:
                            metadata["storetime"] = hit["entity"]["storetime"]
                        
                        # Create the document only if content is not empty
                        if content:
                            doc = Document(
                                page_content=content,
                                metadata=metadata
                            )
                            documents.append(doc)
                        else:
                            print(f"Warning: Empty content in search result, skipping: {hit}")
                
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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

def create_index_properly(client, collection_name, dimension):
    """Create a proper index for Milvus Lite collection."""
    try:
        # Define proper index parameters as a dictionary
        index_params = {
            "metric_type": "IP",  # Use Inner Product as recommended for BGE-M3
            "index_type": "FLAT",  # Use FLAT instead of AUTOINDEX for better compatibility
            "params": {}
        }
        
        # Create the index with proper params
        client.create_index(
            collection_name=collection_name,
            field_name="vector",
            index_params=index_params
        )
        print(f"Successfully created index for collection {collection_name}")
        return True
    except Exception as e:
        print(f"Error creating index: {e}")
        print(f"Index params type: {type(index_params)}")
        # Continue even if index creation fails - Milvus Lite can still work
        print("Continuing without custom index - Milvus Lite will use default settings")
        return True  # Return True to continue processing

def flush_collection_properly(client, collection_name):
    """Properly flush a Milvus Lite collection."""
    try:
        # Try different flush approaches based on client version
        try:
            # First try with collection name as string (newer clients)
            client.flush(collection_name)
            print(f"Successfully flushed collection {collection_name} (string method)")
        except Exception as e1:
            try:
                # Then try with no arguments (some client versions)
                client.flush()
                print(f"Successfully flushed all collections (no args method)")
            except Exception as e2:
                # As a last resort, try other variations
                print(f"Standard flush methods failed, trying alternatives")
                try:
                    # Some versions expect just the string
                    conn = getattr(client, '_connection', None)
                    if conn and hasattr(conn, 'flush'):
                        conn.flush(collection_name)
                        print(f"Successfully flushed via connection object")
                except Exception as e3:
                    print(f"All flush methods failed: {e3}")

        return True
    except Exception as e:
        print(f"Error in flush_collection_properly: {e}")
        return False

def insert_documents_into_milvus_lite(
    client, 
    collection_name: str, 
    document_chunks: list, 
    embedding_model
):
    """Directly insert document chunks into Milvus Lite collection with fixed index and flush."""
    import logging
    logger = logging.getLogger(__name__)
    print(f"Inserting {len(document_chunks)} chunks into Milvus Lite collection {collection_name}")
    
    # Check if collection exists
    collections = client.list_collections()
    
    # Drop existing collection to ensure clean start
    if collection_name in collections:
        print(f"Collection {collection_name} already exists. Dropping it for clean creation.")
        client.drop_collection(collection_name)
    
    # Get embedding dimension
    sample_text = "Sample text for dimension detection"
    sample_embedding = embedding_model.embed_query(sample_text)
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")
    
    # Create collection with SIMPLE approach first (this is more reliable)
    print(f"Creating collection {collection_name} with dimension {dimension}")
    try:
        # Use the simpler creation method which auto-generates the ID field
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            auto_id=True  # This ensures auto-generation of IDs
        )
        print("Successfully created collection")
    except Exception as e:
        print(f"Error creating collection: {e}")
        import traceback
        print(traceback.format_exc())
        return 0
    
    # Create index using the fixed function
    create_index_properly(client, collection_name, dimension)
    
    # Load the collection
    try:
        client.load_collection(collection_name)
        print(f"Loaded collection {collection_name}")
    except Exception as e:
        print(f"Error loading collection: {e}")
    
    # Get collection schema to understand its structure
    try:
        schema = client.describe_collection(collection_name)
        print(f"Collection schema: {schema}")
        
        # Check field names to confirm id field exists
        field_names = [field['name'] for field in schema.get('fields', [])]
        print(f"Field names in schema: {field_names}")
        
        # Check if ID field is primary and auto-generated
        for field in schema.get('fields', []):
            if field['name'] == 'id':
                print(f"ID field properties: {field}")
                
    except Exception as e:
        print(f"Error getting collection schema: {e}")
    
    # Insert documents in batches
    batch_size = 50
    total_inserted = 0
    
    for i in range(0, len(document_chunks), batch_size):
        batch = document_chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(document_chunks)-1)//batch_size + 1}")
        
        # Prepare data for insertion
        data = []
        for chunk_idx, chunk in enumerate(batch):
            try:
                # Get content and metadata based on document type
                if isinstance(chunk, dict):
                    content = chunk.get("content", "")
                    metadata = chunk.get("metadata", {})
                    section = chunk.get("section", "")
                elif hasattr(chunk, 'page_content'):
                    # Handle LangChain Document objects
                    content = chunk.page_content
                    metadata = chunk.metadata
                    section = metadata.get("section", "")
                else:
                    print(f"Skipping chunk {chunk_idx} - unexpected format: {type(chunk)}")
                    continue
                
                # Skip empty or very short texts
                if not content or len(content.strip()) < 10:
                    print(f"Skipping chunk {chunk_idx} - empty or too short content")
                    continue
                
                # Create embedding - with better error handling
                try:
                    embedding = embedding_model.embed_query(content)
                except Exception as e:
                    print(f"Error creating embedding for chunk {chunk_idx}: {e}")
                    # Try with truncated content if original is too long
                    if len(content) > 8000:
                        try:
                            truncated = content[:8000]
                            print(f"Trying with truncated content (8000 chars)")
                            embedding = embedding_model.embed_query(truncated)
                            content = truncated  # Use truncated version
                        except Exception as e2:
                            print(f"Error creating embedding even with truncated content: {e2}")
                            continue
                    else:
                        continue
                
                # Ensure metadata is a dictionary and convert values to strings
                if not isinstance(metadata, dict):
                    metadata = {}
                
                # Create entity for insertion - DO NOT include ID field
                entity = {
                    "vector": embedding,
                    "content": content,
                    "note_id": str(metadata.get("note_id", "")),
                    "hadm_id": str(metadata.get("hadm_id", "")),
                    "subject_id": str(metadata.get("subject_id", "")),
                    "section": str(section),
                    "charttime": str(metadata.get("charttime", "")),  
                    "storetime": str(metadata.get("storetime", ""))  
                }
                
                data.append(entity)
            except Exception as e:
                import traceback
                print(f"Error processing chunk {chunk_idx}: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                if isinstance(chunk, dict):
                    print(f"Chunk keys: {chunk.keys()}")
                continue
        
        # Insert batch if we have data
        if data:
            try:
                print(f"Inserting batch of {len(data)} documents")
                if data:
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
    
    # Flush to ensure data is written - use the fixed flush function
    flush_collection_properly(client, collection_name)
    
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
                    output_fields=["content", "note_id", "hadm_id", "subject_id", "section"]
                )
                
                if search_results and len(search_results) > 0 and len(search_results[0]) > 0:
                    print(f"Search successful! Found {len(search_results[0])} results")
                    print(f"First result structure: {search_results[0][0].keys()}")
                    
                    # Check if content is directly available or nested in entity
                    first_hit = search_results[0][0]
                    if "content" in first_hit:
                        print(f"Content directly available in hit")
                        content_preview = first_hit["content"][:100] + "..." if len(first_hit["content"]) > 100 else first_hit["content"]
                        print(f"Content preview: {content_preview}")
                    elif "entity" in first_hit and "content" in first_hit["entity"]:
                        print(f"Content available in entity property")
                        content_preview = first_hit["entity"]["content"][:100] + "..." if len(first_hit["entity"]["content"]) > 100 else first_hit["entity"]["content"]
                        print(f"Content preview: {content_preview}")
                    else:
                        print(f"Warning: Content not found in search result")
                        print(f"Available fields: {first_hit.keys()}")
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
    document_chunks: Optional[List[Dict]],
    embedding_model,
    config: RetrieverConfig
) -> BaseMedicalRetriever:
    """Create appropriate retriever based on configuration"""
    
    # Check if we're using Milvus Lite
    if config.use_milvus_lite:
        # Import the hybrid retriever for Milvus Lite
        from .hybrid_retriever_lite import HybridMedicalRetrieverLite
        
        print(f"DEBUG: Creating retriever with Milvus Lite DB: {config.milvus_lite_db}")
        print(f"DEBUG: Collection name: {config.collection_name}")
        print(f"DEBUG: use_hybrid flag: {config.use_hybrid}")
        
        # Create appropriate retriever based on config
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            print("DEBUG: Creating hybrid medical retriever with Milvus Lite")
            retriever = HybridMedicalRetrieverLite(config)
            
            # Insert documents if provided
            if document_chunks:
                print(f"DEBUG: Inserting {len(document_chunks)} documents into Milvus Lite")
                insertion_success = retriever.hybrid_search.insert_documents(document_chunks)
                print(f"DEBUG: Document insertion success: {insertion_success}")
            
            return retriever
        else:
            print("DEBUG: Creating basic medical retriever with Milvus Lite")
            print("DEBUG: Using embedding model type:", type(embedding_model))
            retriever = BasicMedicalRetriever(None, config, embedding_model)
            
            # Insert documents if provided and we're using basic retriever
            if document_chunks:
                print(f"DEBUG: Inserting {len(document_chunks)} documents using insert_documents_into_milvus_lite")
                from pymilvus import MilvusClient
                client = MilvusClient(config.milvus_lite_db)
                insert_documents_into_milvus_lite(client, config.collection_name, document_chunks, embedding_model)
            
            return retriever
    
    # Rest of the function for standard Milvus remains unchanged
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
        
        if hasattr(config, 'use_hybrid') and config.use_hybrid:
            return HybridMedicalRetriever(vectorstore, config,embedding_model)
        else:
            return BasicMedicalRetriever(vectorstore, config)

        
def improved_document_processing(chunk, chunk_idx, embedding_model):
    """Improved method to process document chunks with better error handling."""
    try:
        # Get content and create embedding
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            section = chunk.get("section", "")
        elif hasattr(chunk, 'page_content'):
            # Handle LangChain Document objects
            content = chunk.page_content
            metadata = chunk.metadata
            section = metadata.get("section", "")
        else:
            print(f"Skipping chunk {chunk_idx} - unexpected format: {type(chunk)}")
            return None
        
        # Skip empty or very short texts
        if not content or len(content.strip()) < 10:
            print(f"Skipping chunk {chunk_idx} - empty or too short content")
            return None
        
        # Create embedding - with better error handling
        try:
            embedding = embedding_model.embed_query(content)
        except Exception as e:
            print(f"Error creating embedding for chunk {chunk_idx}: {e}")
            # Try with truncated content if original is too long
            if len(content) > 8000:
                try:
                    truncated = content[:8000]
                    print(f"Trying with truncated content (8000 chars)")
                    embedding = embedding_model.embed_query(truncated)
                    content = truncated  # Use truncated version
                except Exception as e2:
                    print(f"Error creating embedding even with truncated content: {e2}")
                    return None
            else:
                return None
        
        # Ensure metadata is a dictionary and convert values to strings
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Create entity for insertion - DO NOT include ID field
        entity = {
            "vector": embedding,
            "content": content,
            "note_id": str(metadata.get("note_id", "")),
            "hadm_id": str(metadata.get("hadm_id", "")),
            "subject_id": str(metadata.get("subject_id", "")),
            "section": str(section),
            "charttime": str(metadata.get("charttime", "")),  
            "storetime": str(metadata.get("storetime", ""))  
        }
        
        return entity
    except Exception as e:
        import traceback
        print(f"Error processing chunk {chunk_idx}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print(f"Chunk type: {type(chunk)}")
        if isinstance(chunk, dict):
            print(f"Chunk keys: {chunk.keys()}")
        return None
