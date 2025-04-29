from typing import List, Dict, Optional, Any, Tuple
from langchain.schema import Document
from .base_retriever import BaseMedicalRetriever, RetrieverConfig
from .hybrid_search_lite import HybridSearchLite, LiteSearchConfig

class HybridMedicalRetrieverLite(BaseMedicalRetriever):
    """Medical retriever using hybrid search with Milvus Lite"""
    
    def __init__(self, config: RetrieverConfig):
        self.config = config
        
        print("DEBUG: Initializing HybridMedicalRetrieverLite")
        print(f"DEBUG: Config parameters: {config.__dict__}")
        
        # Create search config
        search_config = LiteSearchConfig(
            collection_name=config.collection_name,
            milvus_lite_db=config.milvus_lite_db,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
            rerank_weight=config.rerank_weight
        )
        
        print(f"DEBUG: Created LiteSearchConfig with collection: {search_config.collection_name}")
        
        # Initialize hybrid search
        try:
            self.hybrid_search = HybridSearchLite(search_config)
            print("DEBUG: Successfully initialized HybridSearchLite")
        except Exception as e:
            print(f"DEBUG: Error initializing HybridSearchLite: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            raise
    
    def get_relevant_documents(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents for a query"""
        k = kwargs.get('k', self.config.k_documents)
        
        print(f"DEBUG: Searching for '{query}' with k={k}")
        
        # Get query vector if provided
        query_vector = kwargs.get('query_vector', None)
        
        try:
            # Perform search
            search_results = self.hybrid_search.hybrid_search(
                query=query,
                filters=filters,
                k=k
            )
            
            # Extract documents
            documents = [result.document for result in search_results]
            print(f"DEBUG: Found {len(documents)} documents")
            
            # Print first document if available
            if documents:
                print(f"DEBUG: First document content (first 100 chars): {documents[0].page_content[:100]}")
                print(f"DEBUG: First document metadata: {documents[0].metadata}")
            
            return documents
        except Exception as e:
            print(f"DEBUG: Error in get_relevant_documents: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return []
    
    def get_relevant_documents_with_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with scores for a query"""
        k = kwargs.get('k', self.config.k_documents)
        
        try:
            # Perform search
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
            
            print(f"DEBUG: Found {len(filtered_results)} documents after score filtering")
            return filtered_results
        except Exception as e:
            print(f"DEBUG: Error in get_relevant_documents_with_scores: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return []
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the collection (for debugging)"""
        try:
            # Use hybrid_search instance to access client
            client = self.hybrid_search.client
            collection_name = self.config.collection_name
            
            print(f"DEBUG: Fetching all documents from collection {collection_name}")
            
            # Query all documents with limit high enough to get everything
            results = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["content", "note_id", "hadm_id", "subject_id", "section"],
                limit=10000  # Adjust if you have more documents
            )
            
            documents = []
            if results:
                print(f"DEBUG: Found {len(results)} total documents")
                
                for item in results:
                    # Extract content and metadata
                    content = item.get("content", "")
                    metadata = {
                        "note_id": item.get("note_id", ""),
                        "hadm_id": item.get("hadm_id", ""),
                        "subject_id": item.get("subject_id", ""),
                        "section": item.get("section", "")
                    }
                    
                    if content:
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"DEBUG: Error getting all documents: {e}")
            return []
