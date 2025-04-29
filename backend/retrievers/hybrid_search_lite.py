# backend/retrievers/hybrid_search_lite.py

import numpy as np
import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from langchain.schema import Document
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
import logging
import scipy.sparse as sp

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
    milvus_lite_db: str = "./milvus_medical_M.db"
    dense_weight: float = 0.6
    sparse_weight: float = 0.2
    rerank_weight: float = 0.2

class HybridSearchLite:
    """Hybrid search implementation using Milvus Lite for both dense and sparse vectors"""
    
    def __init__(self, config: LiteSearchConfig):
        self.config = config
        
        # Initialize embedding and reranking functions
        self.ef = BGEM3EmbeddingFunction(
            model_name="/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge/bge-m3-med-ft",
            use_fp16=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reranker = BGERerankFunction(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Milvus Lite client
        self.client = MilvusClient(self.config.milvus_lite_db)
        print(f"Using Milvus Lite with DB: {self.config.milvus_lite_db}")
        
        # Detect embedding dimensions from test sample
        self._detect_dimensions()
        
        # Setup collections
        self.setup_collections()
    
    def _detect_dimensions(self):
        """Detect the dimensions of BGE-M3 embeddings."""
        try:
            sample_text = "Sample text for dimension detection"
            self.sample_embeddings = self.ef([sample_text])
            print(f"DEBUG: BGE-M3 sample embeddings: {self.sample_embeddings}")
            
            # Get dense dimension
            if 'dense' in self.sample_embeddings and len(self.sample_embeddings['dense']) > 0:
                self.dense_dim = len(self.sample_embeddings['dense'][0])
                print(f"DEBUG: BGE-M3 dense vector dimension: {self.dense_dim}")
            else:
                self.dense_dim = 1024  # Default if detection fails
                print(f"DEBUG: Using default dense dimension: {self.dense_dim}")
            
            # For sparse vectors, we don't need to explicitly detect dimensions
            # as Milvus handles this automatically with SPARSE_FLOAT_VECTOR
            if 'sparse' in self.sample_embeddings:
                print(f"DEBUG: BGE-M3 sparse vector detected")
                # We can log some info about the sparse vector for debugging
                sparse_vector = self.sample_embeddings['sparse'][0]
                if hasattr(sparse_vector, 'nnz'):
                    print(f"DEBUG: Sample sparse vector has {sparse_vector.nnz} non-zero elements")
                elif hasattr(sparse_vector, 'count_nonzero'):
                    print(f"DEBUG: Sample sparse vector has {sparse_vector.count_nonzero()} non-zero elements")
                
        except Exception as e:
            print(f"DEBUG: Error detecting dimensions: {e}")
            import traceback
            print(f"DEBUG: Dimension detection traceback: {traceback.format_exc()}")
            
            # Default values
            self.dense_dim = 1024  # Default for BGE-M3
            print(f"DEBUG: Using default dense dimension: {self.dense_dim}")

    def convert_to_sparse_dict(self, sparse_vector):
        """Convert sparse vector to dictionary format for Milvus SPARSE_FLOAT_VECTOR"""
        try:
            # For COO sparse matrix format
            if hasattr(sparse_vector, 'row') and hasattr(sparse_vector, 'col') and hasattr(sparse_vector, 'data'):
                result = {int(idx): float(val) for idx, val in zip(sparse_vector.col, sparse_vector.data)}
                
            # For CSR sparse matrix format
            elif isinstance(sparse_vector, sp.csr_matrix):
                coo = sparse_vector.tocoo()
                result = {int(idx): float(val) for idx, val in zip(coo.col, coo.data)}
                
            # For any array-convertible format
            elif hasattr(sparse_vector, 'toarray'):
                array = sparse_vector.toarray()
                if array.ndim > 1:
                    array = array[0]
                indices = np.nonzero(array)[0]
                values = array[indices]
                result = {int(idx): float(val) for idx, val in zip(indices, values)}
                
            # For numpy arrays or lists
            elif isinstance(sparse_vector, (list, np.ndarray)):
                array = np.asarray(sparse_vector)
                indices = np.nonzero(array)[0]
                values = array[indices]
                result = {int(idx): float(val) for idx, val in zip(indices, values)}
                
            else:
                print(f"Warning: Unknown sparse vector type: {type(sparse_vector)}")
                # Provide a minimal valid sparse vector instead of empty dict
                result = {0: 0.0001}
            
            # Always provide at least one element in the sparse vector
            if not result:
                result = {0: 0.0001}
                
            print(f"DEBUG: Final sparse_dict type: {type(result)}, with {len(result)} elements")
            if result and len(result) > 0:
                sample = list(result.items())[:3]
                print(f"DEBUG: Final sparse_dict sample items: {sample}")
                
            return result
            
        except Exception as e:
            print(f"Error converting sparse vector to dictionary: {e}")
            print(f"Using default sparse vector instead")
            # Return a default sparse vector with one element
            return {0: 0.0001}
    
    def setup_collections(self):
        """Set up Milvus Lite collection for hybrid search following official documentation"""
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            print(f"DEBUG: Existing collections: {collections}")
            
            # Create or recreate collection
            collection_name = self.config.collection_name
            if collection_name in collections:
                print(f"DEBUG: Collection {collection_name} already exists. Dropping it.")
                self.client.drop_collection(collection_name)
            
            # Create collection with schema to support both dense and sparse vectors
            from pymilvus import FieldSchema, CollectionSchema
            from pymilvus import DataType
            
            # Define fields for hybrid collection
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                # Dense vector field
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dense_dim),
                # Sparse vector field - use proper datatype
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                # Content and metadata fields
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="note_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="hadm_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="charttime", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="storetime", dtype=DataType.VARCHAR, max_length=100)   
            ]
            
            schema = CollectionSchema(fields)
            
            # Prepare index parameters according to official docs
            try:
                # Check if newer Milvus version with prepare_index_params
                if hasattr(self.client, 'prepare_index_params'):
                    index_params = self.client.prepare_index_params()
                    
                    # Add index for dense vectors
                    index_params.add_index(
                        field_name="vector",
                        index_name="vector_index",
                        index_type="IVF_FLAT",  # Use IVF_FLAT as recommended
                        metric_type="IP",       # Inner Product for BGE-M3
                        params={"nlist": 128},
                    )
                    
                    # Add index for sparse vectors
                    index_params.add_index(
                        field_name="sparse_vector",
                        index_name="sparse_index",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"inverted_index_algo": "DAAT_MAXSCORE"},
                    )
                    
                    # Create collection with schema and index params
                    print(f"DEBUG: Creating collection {collection_name} with schema and index params")
                    create_result = self.client.create_collection(
                        collection_name=collection_name,
                        schema=schema,
                        index_params=index_params
                    )
                    
                    # Save the metric type we used
                    self.metric_type = "IP"
                else:
                    # Older Milvus version without prepare_index_params
                    # Create collection first, then create index
                    print(f"DEBUG: Creating collection {collection_name} with schema only")
                    create_result = self.client.create_collection(
                        collection_name=collection_name,
                        schema=schema
                    )
                    
                    # Create dense vector index
                    dense_index_params = {
                        "index_type": "IVF_FLAT",
                        "metric_type": "IP",
                        "params": {"nlist": 128}
                    }
                    
                    print(f"DEBUG: Creating dense index with params: {dense_index_params}")
                    index_result = self.client.create_index(
                        collection_name=collection_name,
                        field_name="vector",
                        index_params=dense_index_params
                    )
                    print(f"DEBUG: Dense index creation result: {index_result}")
                    
                    # Create sparse vector index
                    sparse_index_params = {
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "IP",
                        "params": {"inverted_index_algo": "DAAT_MAXSCORE"}
                    }
                    
                    print(f"DEBUG: Creating sparse index with params: {sparse_index_params}")
                    sparse_index_result = self.client.create_index(
                        collection_name=collection_name,
                        field_name="sparse_vector",
                        index_params=sparse_index_params
                    )
                    print(f"DEBUG: Sparse index creation result: {sparse_index_result}")
                    
                    # Save the metric type we used
                    self.metric_type = "IP"
            except Exception as e:
                print(f"DEBUG: Error creating collection with index: {e}")
                # Fall back to basic collection creation
                print(f"DEBUG: Falling back to basic collection creation")
                create_result = self.client.create_collection(
                    collection_name=collection_name,
                    schema=schema
                )
                
                # Try to create basic indices
                try:
                    dense_index_params = {
                        "index_type": "FLAT",  # Simple FLAT index for compatibility
                        "metric_type": "IP",
                        "params": {}
                    }
                    
                    print(f"DEBUG: Creating basic dense index with params: {dense_index_params}")
                    index_result = self.client.create_index(
                        collection_name=collection_name,
                        field_name="vector",
                        index_params=dense_index_params
                    )
                    print(f"DEBUG: Basic dense index creation result: {index_result}")
                    
                    # Try creating sparse index
                    sparse_index_params = {
                        "index_type": "SPARSE_INVERTED_INDEX",
                        "metric_type": "IP",
                        "params": {}
                    }
                    
                    print(f"DEBUG: Creating basic sparse index with params: {sparse_index_params}")
                    sparse_index_result = self.client.create_index(
                        collection_name=collection_name,
                        field_name="sparse_vector",
                        index_params=sparse_index_params
                    )
                    print(f"DEBUG: Basic sparse index creation result: {sparse_index_result}")
                    
                    # Save the metric type we used
                    self.metric_type = "IP"
                except Exception as e2:
                    print(f"DEBUG: Error creating basic indices: {e2}")
                    self.metric_type = None  # Will try different metrics at search time
            
            # Load collection
            self.client.load_collection(collection_name)
            print(f"DEBUG: Loaded collection {collection_name}")
            
            # Get schema to verify fields
            schema = self.client.describe_collection(collection_name)
            print(f"DEBUG: Collection schema: {schema}")
            
        except Exception as e:
            print(f"DEBUG: Error setting up collection: {e}")
            import traceback
            print(f"DEBUG: Setup traceback: {traceback.format_exc()}")
        
    def insert_documents(self, documents: List[Any]) -> bool:
        """Insert documents into collection with both dense and sparse vectors"""
        try:
            print(f"DEBUG: Processing batch of {len(documents)} documents")
            
            # Prepare data for insertion
            data = []
            
            # Process each document
            for doc_idx, doc in enumerate(documents):
                try:
                    # Get text and metadata based on document type
                    if hasattr(doc, 'page_content'):
                        # It's a langchain Document object
                        text = doc.page_content
                        metadata = doc.metadata
                    elif isinstance(doc, dict):
                        # It's a dictionary
                        if 'page_content' in doc:
                            text = doc['page_content']
                            metadata = doc.get('metadata', {})
                        elif 'content' in doc:
                            text = doc['content']
                            metadata = doc.get('metadata', {})
                        else:
                            # Skip documents without text
                            print(f"DEBUG: Document {doc_idx} has no content, skipping")
                            continue
                    else:
                        # Skip unsupported document types
                        print(f"DEBUG: Document {doc_idx} has unsupported type {type(doc)}, skipping")
                        continue
                    
                    # Get section
                    section = ""
                    if isinstance(doc, dict) and 'section' in doc:
                        section = doc['section']
                    
                    # Log progress
                    if doc_idx % 20 == 0:  # Log less frequently for better performance
                        print(f"DEBUG: Getting embeddings for document {doc_idx}")
                    
                    # Generate embeddings - both dense and sparse vectors
                    embeddings = self.ef([text])
                    dense_vector = embeddings['dense'][0].tolist()
                    
                    # Process sparse vector into dictionary format for SPARSE_FLOAT_VECTOR
                    sparse_vector = embeddings['sparse'][0]
                    sparse_dict = self.convert_to_sparse_dict(sparse_vector)
                    
                    if doc_idx % 20 == 0:
                        print(f"DEBUG: Document {doc_idx} sparse vector has {len(sparse_dict)} non-zero elements")
                        if sparse_dict:
                            sample_items = list(sparse_dict.items())[:5]
                            print(f"DEBUG: Sample sparse entries: {sample_items}")
                    
                    # Create entity with both dense and sparse vectors
                    entity = {
                        "vector": dense_vector,
                        "sparse_vector": sparse_dict,
                        "content": text,
                        "note_id": str(metadata.get("note_id", "")),
                        "hadm_id": str(metadata.get("hadm_id", "")),
                        "subject_id": str(metadata.get("subject_id", "")),
                        "section": section,
                        "charttime": str(metadata.get("charttime", "")),
                        "storetime": str(metadata.get("storetime", ""))
                    }
                    
                    data.append(entity)
                    
                    if doc_idx % 20 == 0:  # Log less frequently
                        print(f"DEBUG: Successfully added document {doc_idx} to batch")
                            
                except Exception as e:
                    print(f"DEBUG: Error processing document {doc_idx}: {e}")
                    import traceback
                    print(f"DEBUG: Document processing traceback: {traceback.format_exc()}")
                    continue
            
            # Insert data if we have any
            if data:
                try:
                    print(f"DEBUG: Inserting batch of {len(data)} documents")
                    if data:
                        # Show first entity's fields and vector dimension
                        print(f"DEBUG: First entity fields: {list(data[0].keys())}")
                        print(f"DEBUG: Dense vector dimension: {len(data[0]['vector'])}")
                        print(f"DEBUG: Sparse vector non-zeros: {len(data[0]['sparse_vector'])}")
                    
                        # Insert batch
                        insert_result = self.client.insert(
                            collection_name=self.config.collection_name,
                            data=data
                        )
                        print(f"DEBUG: Insertion result: {insert_result}")
                    
                        # Get collection stats
                        try:
                            stats = self.client.get_collection_stats(self.config.collection_name)
                            print(f"DEBUG: Collection stats after insertion: {stats}")
                            row_count = stats.get("row_count", 0)
                            print(f"DEBUG: Total documents in collection: {row_count}")
                        except Exception as e:
                            print(f"DEBUG: Error getting collection stats: {e}")
                    
                    return True
                except Exception as e:
                    print(f"DEBUG: Error inserting batch: {e}")
                    import traceback
                    print(f"DEBUG: Insertion traceback: {traceback.format_exc()}")
                    return False
            else:
                print("DEBUG: No valid documents to insert")
                return False
            
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
        """Perform search using both dense and sparse vectors following official Milvus documentation"""
        try:
            # Generate query embeddings
            print(f"DEBUG: Starting hybrid search for query: {query}")
            query_embeddings = self.ef([query])
            
            # Get dense and sparse vectors
            dense_vector = query_embeddings['dense'][0].tolist()
            sparse_vector = query_embeddings['sparse'][0]
            
            # Convert sparse vector to dictionary format
            sparse_dict = self.convert_to_sparse_dict(sparse_vector)
            
            # Official Milvus hybrid search approach
            if hasattr(self.client, 'hybrid_search') and hasattr(self, 'metric_type') and self.metric_type:
                try:
                    # Import necessary classes
                    from pymilvus import AnnSearchRequest, WeightedRanker
                    
                    # Define output fields
                    output_fields = ["content", "note_id", "hadm_id", "subject_id", "section", 
                                    "charttime", "storetime"]
                    
                    # Create filter expression if needed
                    filter_expr = None
                    if filters:
                        filter_parts = []
                        for key, value in filters.items():
                            if isinstance(value, str):
                                filter_parts.append(f"{key} == '{value}'")
                            elif isinstance(value, (list, tuple)):
                                values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                                filter_parts.append(f"{key} in [{values_str}]")
                            else:
                                filter_parts.append(f"{key} == {value}")
                        
                        if filter_parts:
                            filter_expr = " && ".join(filter_parts)
                    
                    # Create BOTH dense and sparse search requests
                    search_requests = []
                    
                    # 1. Dense vector search request
                    dense_request = AnnSearchRequest(
                        data=[dense_vector],
                        anns_field="vector",
                        param={
                            "metric_type": self.metric_type,
                            "params": {"nprobe": 10}
                        },
                        limit=k * 2,  # Get more candidates for filtering
                        expr=filter_expr
                    )
                    search_requests.append(dense_request)
                    
                    # 2. Sparse vector search request
                    if sparse_dict:
                        sparse_request = AnnSearchRequest(
                            data=[sparse_dict],
                            anns_field="sparse_vector",
                            param={
                                "metric_type": "IP",
                                "params": {"drop_ratio_search": 0.2}  # Optional parameter for sparse search
                            },
                            limit=k * 2,
                            expr=filter_expr
                        )
                        search_requests.append(sparse_request)
                    
                    # Create ranker with appropriate weights
                    weights = []
                    if len(search_requests) == 1:
                        # Only dense search
                        weights = [1.0]
                    elif len(search_requests) == 2:
                        # Both dense and sparse
                        weights = [self.config.dense_weight, self.config.sparse_weight]
                    
                    # Create the ranker with individual float parameters
                    ranker = WeightedRanker(self.config.dense_weight, self.config.sparse_weight)
                    
                    # Execute hybrid search
                    print(f"DEBUG: Executing official hybrid search with {len(search_requests)} requests")
                    print(f"DEBUG: - Weights: {weights}")
                    print(f"DEBUG: - collection_name: {self.config.collection_name}")
                    print(f"DEBUG: Sending search request with dense vector of dimension {len(dense_vector)}")
                    print(f"DEBUG: Sending sparse dict with {len(sparse_dict)} elements")

                    hybrid_results = self.client.hybrid_search(
                        collection_name=self.config.collection_name,
                        reqs=search_requests,
                        ranker=ranker,
                        limit=k,
                        output_fields=output_fields
                    )
                    print(f"DEBUG: First result type: {type(hybrid_results[0])}")
                    print(f"DEBUG: Raw hybrid result structure: {str(hybrid_results)[:200]}...")
                    
                    # Process results
                    results = []
                    print(f"DEBUG: Processing {len(hybrid_results)} hybrid search results")
                    
                    for hits_str in hybrid_results:
                        if isinstance(hits_str, str) and hits_str.startswith('[{'):
                            # Parse stringified list of dictionaries
                            parsed_hits = ast.literal_eval(hits_str)

                            for hit in parsed_hits:
                                entity = hit.get('entity', {})
                                content = entity.get('content', '')
                                
                                if not content:
                                    continue
                                
                                metadata = {
                                    "note_id": entity.get("note_id", ""),
                                    "hadm_id": entity.get("hadm_id", ""),
                                    "subject_id": entity.get("subject_id", ""),
                                    "section": entity.get("section", ""),
                                    "charttime": entity.get("charttime", ""),
                                    "storetime": entity.get("storetime", "")
                                }

                                dense_score = float(hit.get('distance', 0.0))

                                doc = Document(page_content=content, metadata=metadata)

                                results.append(RerankedResult(
                                    document=doc,
                                    dense_score=dense_score,
                                    sparse_score=0.0,
                                    rerank_score=0.0,
                                    final_score=dense_score
                                ))

                        elif isinstance(hits_str, list):  # Handle case if it's already a list of dicts
                            for hit in hits_str:
                                entity = hit.get('entity', {})
                                content = entity.get('content', '')
                                
                                if not content:
                                    continue
                                
                                metadata = {
                                    "note_id": entity.get("note_id", ""),
                                    "hadm_id": entity.get("hadm_id", ""),
                                    "subject_id": entity.get("subject_id", ""),
                                    "section": entity.get("section", ""),
                                    "charttime": entity.get("charttime", ""),
                                    "storetime": entity.get("storetime", "")
                                }

                                dense_score = float(hit.get('distance', 0.0))

                                doc = Document(page_content=content, metadata=metadata)

                                results.append(RerankedResult(
                                    document=doc,
                                    dense_score=dense_score,
                                    sparse_score=0.0,
                                    rerank_score=0.0,
                                    final_score=dense_score
                                ))
                                        
                    # Apply reranking if available
                    if results and self.reranker:
                        try:
                            doc_contents = [result.document.page_content for result in results]
                            print(f"DEBUG: Reranking {len(doc_contents)} documents")
                            
                            rerank_results = self.reranker(query, doc_contents)
                            
                            # Update scores with reranking
                            for i, rerank_result in enumerate(rerank_results):
                                if i < len(results):
                                    rerank_score = rerank_result.score
                                    results[i].rerank_score = rerank_score
                                    
                                    # Update final score with reranking component
                                    current_score = results[i].final_score * (1 - self.config.rerank_weight)
                                    rerank_component = self.config.rerank_weight * rerank_score
                                    results[i].final_score = current_score + rerank_component
                            
                            # Sort by final score
                            results.sort(key=lambda x: x.final_score, reverse=True)
                        except Exception as e:
                            print(f"DEBUG: Error during reranking: {e}")
                    
                    # Log the number of results found
                    print(f"DEBUG: Found {len(results)} documents")
                    
                    # Return top k results
                    return results[:k]
                    
                except (ImportError, AttributeError, Exception) as e:
                    print(f"DEBUG: Official hybrid search failed: {e}")
                    print(f"DEBUG: Falling back to manual hybrid search implementation")
            
            # Manual hybrid search implementation (fallback)
            print("DEBUG: Using manual hybrid search implementation")
            
            # Prepare filter expression
            filter_expr = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} == '{value}'")
                    elif isinstance(value, (list, tuple)):
                        values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                        filter_parts.append(f"{key} in [{values_str}]")
                    else:
                        filter_parts.append(f"{key} == {value}")
                
                if filter_parts:
                    filter_expr = " && ".join(filter_parts)
            
            # Define fields to retrieve
            output_fields = ["content", "note_id", "hadm_id", "subject_id", "section", 
                            "charttime", "storetime"]
            
            # Try different metric types for dense search
            metric_types = ["IP", "COSINE", "L2"]
            if hasattr(self, 'metric_type') and self.metric_type:
                metric_types = [self.metric_type] + [mt for mt in metric_types if mt != self.metric_type]
            
            # Perform dense search
            dense_results = []
            for metric_type in metric_types:
                try:
                    search_params = {
                        "metric_type": metric_type,
                        "params": {"nprobe": 10}
                    }
                    
                    print(f"DEBUG: Trying dense search with metric_type: {metric_type}")
                    results = self.client.search(
                        collection_name=self.config.collection_name,
                        data=[dense_vector],
                        anns_field="vector",
                        filter=filter_expr,
                        limit=k * 2,
                        output_fields=output_fields,
                        search_params=search_params
                    )
                    
                    if results and len(results) > 0 and len(results[0]) > 0:
                        print(f"DEBUG: Dense search successful with {len(results[0])} results")
                        dense_results = results
                        self.metric_type = metric_type  # Save for future searches
                        break
                except Exception as e:
                    print(f"DEBUG: Dense search failed with metric_type {metric_type}: {e}")
                    continue
            
            if not dense_results or len(dense_results[0]) == 0:
                print("DEBUG: No dense search results found")
                return []
            
            # Perform sparse search if we have sparse vector
            sparse_results = []
            if sparse_dict:
                try:
                    sparse_search_params = {
                        "metric_type": "IP",
                        "params": {"drop_ratio_search": 0.2}
                    }
                    
                    print(f"DEBUG: Trying sparse search with {len(sparse_dict)} non-zero elements")
                    sparse_results = self.client.search(
                        collection_name=self.config.collection_name,
                        data=[sparse_dict],
                        anns_field="sparse_vector",
                        filter=filter_expr,
                        limit=k * 2,
                        output_fields=output_fields,
                        search_params=sparse_search_params
                    )
                    
                    if sparse_results and len(sparse_results) > 0 and len(sparse_results[0]) > 0:
                        print(f"DEBUG: Sparse search successful with {len(sparse_results[0])} results")
                except Exception as e:
                    print(f"DEBUG: Sparse search failed: {e}")
                    
            # Process each dense search result with sparse scores if available
            candidates = {}
            doc_contents = []
            
            # Check if dense results are in string format
            if isinstance(dense_results[0], str) and dense_results[0].startswith('[{'):
                import json
                import ast
                
                try:
                    # First try ast.literal_eval which is safer for Python literals
                    parsed_hits = ast.literal_eval(dense_results[0])
                    print(f"DEBUG: Successfully parsed dense results with ast.literal_eval")
                except Exception as e1:
                    try:
                        # Then try json.loads with proper escaping
                        json_str = dense_results[0].replace("'", '"').replace('\n', '\\n')
                        parsed_hits = json.loads(json_str)
                        print(f"DEBUG: Successfully parsed dense results with json.loads")
                    except Exception as e2:
                        print(f"DEBUG: Both parsing methods failed for dense results. Errors: {e1}, {e2}")
                        return []
                
                # Process each hit from the parsed string
                for hit in parsed_hits:
                    # Extract entity
                    entity = hit.get("entity", {})
                    
                    # Extract content
                    content = entity.get("content", "")
                    if not content:
                        continue
                    
                    # Extract metadata
                    metadata = {
                        "note_id": entity.get("note_id", ""),
                        "hadm_id": entity.get("hadm_id", ""),
                        "subject_id": entity.get("subject_id", ""),
                        "section": entity.get("section", ""),
                        "charttime": entity.get("charttime", ""),
                        "storetime": entity.get("storetime", "")
                    }
                    
                    # Get dense score
                    dense_score = float(hit.get("distance", 0.0))
                    
                    # Create document
                    doc = Document(page_content=content, metadata=metadata)
                    
                    # Store candidate with initial dense score
                    candidates[content] = RerankedResult(
                        document=doc,
                        dense_score=dense_score,
                        sparse_score=0.0,  # Will update if sparse results available
                        rerank_score=0.0,  # Will set this later
                        final_score=dense_score * self.config.dense_weight  # Initial weighted score
                    )
                    doc_contents.append(content)
            else:
                # Standard object processing for dense results
                for hit in dense_results[0]:
                    # Extract content and metadata
                    entity = hit.get("entity", {}) if "entity" in hit else hit
                    content = entity.get("content", "")
                    
                    # Skip if no content
                    if not content:
                        continue
                    
                    # Extract metadata
                    metadata = {
                        "note_id": entity.get("note_id", ""),
                        "hadm_id": entity.get("hadm_id", ""),
                        "subject_id": entity.get("subject_id", ""),
                        "section": entity.get("section", ""),
                        "charttime": entity.get("charttime", ""),
                        "storetime": entity.get("storetime", "")
                    }
                    
                    # Get dense score
                    dense_score = float(hit.get("distance", 0.0))
                    if "score" in hit and not "distance" in hit:
                        dense_score = float(hit.get("score", 0.0))
                    
                    # Create document
                    doc = Document(page_content=content, metadata=metadata)
                    
                    # Store candidate with initial dense score
                    candidates[content] = RerankedResult(
                        document=doc,
                        dense_score=dense_score,
                        sparse_score=0.0,  # Will update if sparse results available
                        rerank_score=0.0,  # Will set this later
                        final_score=dense_score * self.config.dense_weight  # Initial weighted score
                    )
                    doc_contents.append(content)
            
            # If we have sparse results, update scores
            if sparse_results and len(sparse_results) > 0:
                # Check if sparse results are in string format
                if isinstance(sparse_results[0], str) and sparse_results[0].startswith('[{'):
                    import json
                    import ast
                    
                    try:
                        parsed_sparse_hits = ast.literal_eval(sparse_results[0])
                        print(f"DEBUG: Successfully parsed sparse results with ast.literal_eval")
                    except Exception as e1:
                        try:
                            json_str = sparse_results[0].replace("'", '"').replace('\n', '\\n')
                            parsed_sparse_hits = json.loads(json_str)
                            print(f"DEBUG: Successfully parsed sparse results with json.loads")
                        except Exception as e2:
                            print(f"DEBUG: Both parsing methods failed for sparse results")
                            parsed_sparse_hits = []
                    
                    # Create a dictionary of sparse scores
                    sparse_scores = {}
                    for hit in parsed_sparse_hits:
                        entity = hit.get("entity", {})
                        content = entity.get("content", "")
                        
                        if content:
                            sparse_score = float(hit.get("distance", 0.0))
                            sparse_scores[content] = sparse_score
                    
                    # Update candidates with sparse scores
                    for content, result in candidates.items():
                        if content in sparse_scores:
                            result.sparse_score = sparse_scores[content]
                            # Update final score with sparse component
                            result.final_score = (
                                self.config.dense_weight * result.dense_score + 
                                self.config.sparse_weight * result.sparse_score
                            )
                else:
                    # Standard object processing for sparse results
                    sparse_scores = {}
                    for hit in sparse_results[0]:
                        entity = hit.get("entity", {}) if "entity" in hit else hit
                        content = entity.get("content", "")
                        
                        if content:
                            sparse_score = float(hit.get("distance", 0.0))
                            if "score" in hit and not "distance" in hit:
                                sparse_score = float(hit.get("score", 0.0))
                            
                            sparse_scores[content] = sparse_score
                    
                    # Update candidates with sparse scores
                    for content, result in candidates.items():
                        if content in sparse_scores:
                            result.sparse_score = sparse_scores[content]
                            # Update final score with sparse component
                            result.final_score = (
                                self.config.dense_weight * result.dense_score + 
                                self.config.sparse_weight * result.sparse_score
                            )
            
            # Convert to list and sort by current score
            results_list = list(candidates.values())
            results_list.sort(key=lambda x: x.final_score, reverse=True)
            
            # Keep top k candidates before reranking
            results_list = results_list[:k]
            doc_contents = [result.document.page_content for result in results_list]
            
            # Apply reranking if available
            if results_list and self.reranker:
                try:
                    print(f"DEBUG: Reranking {len(doc_contents)} documents")
                    rerank_results = self.reranker(query, doc_contents)
                    
                    # Update scores with reranking
                    for i, rerank_result in enumerate(rerank_results):
                        if i < len(results_list):
                            rerank_score = rerank_result.score
                            results_list[i].rerank_score = rerank_score
                            
                            # Update final score with reranking component
                            base_score = (
                                self.config.dense_weight * results_list[i].dense_score + 
                                self.config.sparse_weight * results_list[i].sparse_score
                            )
                            rerank_component = self.config.rerank_weight * rerank_score
                            
                            # Normalize weights
                            total_weight = self.config.dense_weight + self.config.sparse_weight
                            if total_weight > 0:
                                base_weight = 1.0 - self.config.rerank_weight
                                results_list[i].final_score = (base_weight * base_score / total_weight) + rerank_component
                            else:
                                results_list[i].final_score = rerank_score
                            
                            # Print top score details
                            if i == 0:
                                print(f"DEBUG: Top result rerank score: {rerank_score:.4f}")
                                print(f"DEBUG: Top result final score: {results_list[i].final_score:.4f}")
                except Exception as e:
                    print(f"DEBUG: Error during reranking: {e}")
            
            # Sort by final score again after reranking
            results_list.sort(key=lambda x: x.final_score, reverse=True)
            
            # Print details about top result
            if results_list:
                top = results_list[0]
                print(f"DEBUG: Top result - Dense: {top.dense_score:.4f}, Sparse: {top.sparse_score:.4f}, "
                    f"Rerank: {top.rerank_score:.4f}, Final: {top.final_score:.4f}")
                print(f"DEBUG: Top result content (first 100 chars): {top.document.page_content[:100]}...")
            
            print(f"DEBUG: Returning {len(results_list)} results")
            return results_list
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
