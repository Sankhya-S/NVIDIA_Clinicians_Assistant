# RAG_implementation.py

from pathlib import Path
import logging
import pandas as pd
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# At top of RAG_implementation.py
import sys
import os
from pathlib import Path

# Get project root directory (parent of backend)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Import custom modules
from model_setup import setup_embedding, setup_chat_model
from backend.prompt.prompt import MedicalPrompts
from backend.processors.document_processor import (
    process_note_sections,
    debug_print
)
from backend.vectorstores.document_embedding import (
    save_basic_chunks,
    save_detailed_chunks,
    display_chunks,
    load_json_note
)
from backend.processors.notes_processor import extract_text_from_pdf
from backend.retrievers.base_retriever import (
    RetrieverConfig,
    create_retriever,
    insert_documents_into_milvus_lite,
    BaseMedicalRetriever
)
from backend.retrievers.hybrid_search import EnhancedHybridSearch, SearchConfig
from backend.retrievers.hybrid_search_faiss import EnhancedHybridSearchFAISS  

from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import EmbeddingsRedundantFilter, LongContextReorder

logger = logging.getLogger(__name__)

class RAGProcessor:
    def __init__(self):
        """Initialize the RAG processor with necessary models and configurations."""
        self.chat_model = setup_chat_model()
        self.embedding_model = setup_embedding()
        self.medical_prompts = MedicalPrompts()
        self.collection_name = None
        self.vectorstore = None
        self.qa_chain = None
        self.k_documents = 5

        
        self.use_milvus_lite = True
        self.milvus_lite_db = "./milvus_medical_M.db"

        # New enhanced retrieval settings
        self.use_contextual_compression = True
        self.use_multi_query = True
        self.compression_similarity_threshold = 0.75

    
    def process_document_text(self, text: str, metadata: Dict, chunk_type: str = "detailed") -> List[Dict]:
        """Process document text using either basic or detailed chunking."""
        if chunk_type == "basic":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            text_chunks = text_splitter.split_text(text)
            return [
                {
                    "content": chunk,
                    "metadata": metadata.copy(),
                    "section": "text"
                } for chunk in text_chunks
            ]
        else:
            return process_note_sections(text, metadata, self.chat_model)

    def _setup_qa_chain(self):
        """Configure the QA chain with appropriate retrieval method."""
        if not self.vectorstore:
            raise ValueError("Vectorstore must be initialized before setting up QA chain")
        
        try:
            # Define retriever_func at the outer scope so it's available to enhanced_qa
            retriever_func = None
            
            # Determine search type and set up appropriate retriever
            if hasattr(self, 'use_milvus_lite') and self.use_milvus_lite and isinstance(self.vectorstore, BaseMedicalRetriever):
                print("Configuring Milvus Lite retriever...")
                
                # Apply document transformers if enabled
                if self.use_contextual_compression and hasattr(self.vectorstore, '_base_retriever'):
                    print("Adding contextual compression to retriever...")
                    redundant_filter = EmbeddingsRedundantFilter(
                        embeddings=self.embedding_model,
                        similarity_threshold=self.compression_similarity_threshold
                    )
                    reordering = LongContextReorder()
                    
                    compressor_pipeline = DocumentCompressorPipeline(
                        transformers=[redundant_filter, reordering]
                    )
                    
                    enhanced_retriever = ContextualCompressionRetriever(
                        base_compressor=compressor_pipeline,
                        base_retriever=self.vectorstore._base_retriever
                    )
                    
                    def get_relevant_docs(query):
                        query_str = query["query"] if isinstance(query, dict) else query
                        query_vector = query.get("query_vector") if isinstance(query, dict) else None
                        
                        if query_vector is None:
                            query_vector = self.embedding_model.embed_query(query_str)
                        
                        # Use enhanced retriever if no query vector is needed, otherwise use base
                        if hasattr(enhanced_retriever, 'get_relevant_documents'):
                            return enhanced_retriever.get_relevant_documents(query_str)
                        else:
                            return self.vectorstore.get_relevant_documents(
                                query_str,
                                query_vector=query_vector,
                                k=self.k_documents
                            )
                    
                    retriever_func = get_relevant_docs
                    print("Enhanced Milvus Lite retriever configured with compression")
                else:
                    # Original Milvus Lite retriever code
                    def get_relevant_docs(query):
                        query_str = query["query"] if isinstance(query, dict) else query
                        query_vector = query.get("query_vector") if isinstance(query, dict) else None
                        
                        if query_vector is None:
                            query_vector = self.embedding_model.embed_query(query_str)
                        
                        return self.vectorstore.get_relevant_documents(
                            query_str,
                            query_vector=query_vector,
                            k=self.k_documents
                        )
                    
                    retriever_func = get_relevant_docs
                    print("Standard Milvus Lite retriever configured")
                    
                # Apply multi-query enhancement if enabled
                if self.use_multi_query and retriever_func and hasattr(self.chat_model, 'predict'):
                    try:
                        print("Adding multi-query capability to Milvus Lite retriever...")
                        
                        # Store the base retriever function
                        base_retriever_func = retriever_func
                        
                        # Create a wrapper function that implements multi-query
                        def multi_query_retriever_func(query):
                            query_str = query["query"] if isinstance(query, dict) else query
                            query_vector = query.get("query_vector") if isinstance(query, dict) else None
                            
                            # Generate multiple query variations using the LLM
                            prompt = self.medical_prompts.get_multi_query_prompt(query_str)
                            
                            try:
                                variations_response = self.chat_model.predict(prompt)
                                
                                # Parse the response to extract only the actual query variations
                                query_variations = []
                                
                                # Split the response by lines
                                lines = variations_response.strip().split('\n')
                                
                                # Filter out lines that are not actual queries
                                for line in lines:
                                    line = line.strip()
                                    # Skip empty lines or lines that look like instructions/headers
                                    if not line or "Here are" in line or "alternative" in line:
                                        continue
                                        
                                    # If line starts with a number or bullet point, extract the actual query
                                    if line.startswith(('1.', '2.', '3.', '4.', '5.', '•', '-', '*')):
                                        # Extract the query part - look for text in quotes if present
                                        if '"' in line:
                                            # Extract text between quotes
                                            start = line.find('"')
                                            end = line.rfind('"')
                                            if start != -1 and end != -1 and end > start:
                                                extracted_query = line[start+1:end]
                                                query_variations.append(extracted_query)
                                        else:
                                            # If no quotes, take everything after the first colon or after the bullet/number
                                            parts = line.split(':', 1)
                                            if len(parts) > 1:
                                                query_variations.append(parts[1].strip())
                                            else:
                                                # Try to find the first alphabetic character after the bullet/number
                                                for i, char in enumerate(line):
                                                    if char.isalpha():
                                                        query_variations.append(line[i:].strip())
                                                        break
                                    else:
                                        # If it doesn't look like a header or instruction, include it as a query
                                        query_variations.append(line)
                                
                                # Always include the original query
                                if query_str not in query_variations:
                                    query_variations.append(query_str)
                                
                                # Limit to a reasonable number of variations
                                query_variations = query_variations[:5]
                                
                                print(f"Generated {len(query_variations)} query variations")
                                
                                # Process each query variation
                                all_docs = []
                                for q in query_variations:
                                    print(f"Starting hybrid search for query: {q}")
                                    q_with_vector = {"query": q, "query_vector": self.embedding_model.embed_query(q)}
                                    try:
                                        var_docs = base_retriever_func(q_with_vector)
                                        print(f"Retrieved {len(var_docs)} documents for variation: {q}")
                                        # Debug first document for each query variation
                                        if var_docs:
                                            print(f"First doc content preview: {var_docs[0].page_content[:100]}...")
                                            print(f"First doc metadata: {var_docs[0].metadata}")
                                        all_docs.extend(var_docs)
                                    except Exception as e:
                                        print(f"Error in processing query variation '{q}': {str(e)}")
                                
                                # Remove duplicates while preserving order
                                seen = set()
                                unique_docs = []
                                for doc in all_docs:
                                    doc_id = doc.page_content
                                    if doc_id not in seen:
                                        seen.add(doc_id)
                                        unique_docs.append(doc)
                                
                                print(f"Total unique documents after deduplication: {len(unique_docs)}")
                                return unique_docs[:self.k_documents]
                            except Exception as e:
                                print(f"Error in multi-query processing: {str(e)}, falling back to base retriever")
                                return base_retriever_func(query)
                        
                        # Replace the retriever function with multi-query version
                        retriever_func = multi_query_retriever_func
                        print("Multi-query capability added to Milvus Lite retriever")
                        
                    except Exception as e:
                        print(f"Failed to set up multi-query for Milvus Lite: {e}")
                        # Keep the original retriever_func
                
            # Handle FAISS hybrid search case
            elif isinstance(self.vectorstore, EnhancedHybridSearchFAISS):
                print("Configuring hybrid search retriever with FAISS...")
                hybrid_search = self.vectorstore
                
                def get_relevant_docs(query):
                    query_str = query["query"] if isinstance(query, dict) else query
                    results = hybrid_search.hybrid_search(
                        query=query_str,
                        k=self.k_documents
                    )
                    return [result.document for result in results]
                    
                retriever_func = get_relevant_docs
                print("Hybrid search retriever configured successfully")
                
                # Apply document transformers if enabled
                if self.use_contextual_compression:
                    try:
                        print("Adding contextual compression to hybrid search retriever...")
                        redundant_filter = EmbeddingsRedundantFilter(
                            embeddings=self.embedding_model,
                            similarity_threshold=self.compression_similarity_threshold
                        )
                        reordering = LongContextReorder()
                        
                        # Create a document compressor pipeline
                        compressor = DocumentCompressorPipeline(
                            transformers=[redundant_filter, reordering]
                        )
                        
                        # Store the base retriever function
                        base_retriever_func = retriever_func
                        
                        # Create a function that applies compression
                        def compressed_retriever_func(query):
                            docs = base_retriever_func(query)
                            return compressor.compress_documents(docs, query)
                        
                        # Replace the retriever function
                        retriever_func = compressed_retriever_func
                        print("Compression added to hybrid search retriever")
                    except Exception as e:
                        print(f"Error setting up compression for hybrid search: {e}")
                        # Keep the original retriever_func
            
            # Handle standard vectorstore case
            else:
                print("Configuring standard vector search retriever...")
                
                # Create the base retriever
                base_retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": self.k_documents}
                )
                retriever_func = base_retriever.get_relevant_documents
                
                # Apply multi-query if enabled
                if self.use_multi_query and hasattr(self.chat_model, 'predict'):
                    try:
                        print("Adding multi-query capability to standard retriever...")
                        multi_query_retriever = MultiQueryRetriever.from_llm(
                            retriever=base_retriever,
                            llm=self.chat_model
                        )
                        
                        # Update the retriever function
                        def multi_query_func(query):
                            query_str = query["query"] if isinstance(query, dict) else query
                            return multi_query_retriever.get_relevant_documents(query_str)
                        
                        retriever_func = multi_query_func
                        print("Multi-query retriever configured successfully")
                    except Exception as e:
                        print(f"Error setting up multi-query retriever: {e}, using standard retriever")
                        # Keep the original retriever_func
                
                # Apply contextual compression if enabled
                if self.use_contextual_compression:
                    try:
                        print("Adding contextual compression to standard retriever...")
                        redundant_filter = EmbeddingsRedundantFilter(
                            embeddings=self.embedding_model,
                            similarity_threshold=self.compression_similarity_threshold
                        )
                        reordering = LongContextReorder()
                        
                        compressor_pipeline = DocumentCompressorPipeline(
                            transformers=[redundant_filter, reordering]
                        )
                        
                        # Store the current retriever function
                        base_retriever_func = retriever_func
                        
                        # Create a function that applies compression
                        def compressed_retriever_func(query):
                            query_str = query["query"] if isinstance(query, dict) else query
                            docs = base_retriever_func(query)
                            return compressor_pipeline.compress_documents(docs, query_str)
                        
                        # Replace the retriever function
                        retriever_func = compressed_retriever_func
                        print("Compression added to standard retriever")
                    except Exception as e:
                        print(f"Error setting up compression for standard retriever: {e}")
                        # Keep the original retriever_func
            
            # Verify that we have a retriever function
            if retriever_func is None:
                raise ValueError("Failed to configure a retriever function")
            
            # Now define enhanced_qa using the configured retriever_func
            def enhanced_qa(query):
                """Process queries using the configured retrieval method."""
                query_str = query["query"] if isinstance(query, dict) else query
                
                print("\n==== RETRIEVING DOCUMENTS ====")
                # Get relevant documents
                docs = retriever_func(query)
                
                # Debug the retrieved documents
                print(f"\n==== RETRIEVED {len(docs)} DOCUMENTS ====")
                for i, doc in enumerate(docs[:3]):  # Show first 3 docs for brevity
                    print(f"\nDOCUMENT {i+1}:")
                    print(f"Content (first 300 chars): {doc.page_content[:300]}...")
                    print(f"Metadata: {doc.metadata}")
                
                # Create context using medical prompts
                context = self.medical_prompts.combine_docs_with_metadata(docs)
                print(f"\n==== CONTEXT LENGTH: {len(context)} chars ====")
                print(f"Context preview: {context[:300]}...")
                
                # IMPORTANT: Log full context to debug what's being sent to the LLM
                print(f"\n==== FULL CONTEXT ====")
                print(context)
                
                # Format prompt and get response
                qa_prompt = self.medical_prompts.get_qa_prompt()
                formatted_prompt = qa_prompt.format(
                    context=context,
                    question=query_str
                )
                print(f"\n==== PROMPT LENGTH: {len(formatted_prompt)} chars ====")
                print(f"Prompt preview: {formatted_prompt[:300]}...")
                
                llm_response = self.chat_model.predict(formatted_prompt)
                
                # Return structured response
                return {
                    "result": llm_response,
                    "source_documents": docs,
                    "metadata_summary": [{
                        "note_id": doc.metadata.get("note_id", "N/A"),
                        "subject_id": doc.metadata.get("subject_id", "N/A"),
                        "hadm_id": doc.metadata.get("hadm_id", "N/A"),
                        "charttime": doc.metadata.get("charttime", "N/A"),
                        "storetime": doc.metadata.get("storetime", "N/A")
                    } for doc in docs]
                }
            
            self.qa_chain = enhanced_qa
            print("QA chain setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise
            
    def process_pdf_documents(self, pdf_folder: str, chunk_type: str = "detailed", enable_hybrid: bool = False) -> int:
        """Process PDF documents with specified chunking and search strategy."""
        print(f"\nProcessing PDFs from {pdf_folder}")
        print(f"Using {chunk_type} chunking with {'hybrid' if enable_hybrid else 'standard'} search")
        
        debug_print(f"Processing PDFs from {pdf_folder}")
        all_chunks = []
        
        # Process each PDF
        for pdf_path in get_pdf_paths(pdf_folder):
            try:
                document_text = extract_text_from_pdf(pdf_path)
                metadata = {
                    "note_id": Path(pdf_path).stem,
                    "source_type": "pdf",
                    "file_path": pdf_path
                }
                chunks = self.process_document_text(document_text, metadata, chunk_type)
                all_chunks.extend(chunks)
                display_chunks(chunks, max_display=5)
                print(f"Created {len(chunks)} chunks for {pdf_path.name}")
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")

        if not all_chunks:
            raise ValueError("No valid documents processed from PDF folder")

        search_type = "hybrid" if enable_hybrid else "standard"
        self.collection_name = f"pdf_documents_{chunk_type}_{search_type}"
        
        try:
            if enable_hybrid:
                print("Setting up hybrid search for PDFs...")
                retriever_config = RetrieverConfig(
                    use_hybrid=True,
                    use_medical_sections=(chunk_type == "detailed"),
                    collection_name=self.collection_name
                )
                self.vectorstore = create_retriever(
                    all_chunks,
                    self.embedding_model,
                    retriever_config
                )
            else:
                print("Setting up standard vector search for PDFs...")
                if chunk_type == "detailed":
                    self.vectorstore, _ = save_detailed_chunks(
                        all_chunks,
                        self.embedding_model,
                        self.collection_name,
                        enable_hybrid=False
                    )
                else:  # basic
                    self.vectorstore, _ = save_basic_chunks(
                        all_chunks,
                        self.embedding_model,
                        self.collection_name
                    )

            self._setup_qa_chain()
            return len(all_chunks)
            
        except Exception as e:
            print(f"Error setting up vectorstore: {e}")
            raise

    def process_json_documents(self, json_folder: str, chunk_type: str = "detailed", enable_hybrid: bool = False, use_compression: bool = True, use_multi_query: bool = True) -> int:
        """Process JSON documents with specified chunking and search strategy."""
        
        self.use_contextual_compression = use_compression
        self.use_multi_query = use_multi_query
        
        print(f"\nProcessing JSONs from {json_folder}")
        print(f"Using {chunk_type} chunking with {'hybrid' if enable_hybrid else 'standard'} search")
        if hasattr(self, 'use_milvus_lite') and self.use_milvus_lite:
            print(f"Using Milvus Lite with database: {self.milvus_lite_db}")
        
        debug_print(f"Processing medical JSONs from {json_folder}")
        all_chunks = []
        
        # Process each JSON file
        for json_path in Path(json_folder).glob("**/note*.json"):
            try:
                print(f"\nProcessing medical note: {json_path.name}")
                
                note = load_json_note(str(json_path))
                if not note:
                    print(f"Skipping invalid or empty note: {json_path.name}")
                    continue
                
                metadata = {
                    "note_id": note.get("note_id", ""),
                    "subject_id": note.get("subject_id", ""),
                    "hadm_id": note.get("hadm_id", ""),
                    "charttime": note.get("charttime", ""),
                    "storetime": note.get("storetime", ""),
                    "source_type": "json"
                }
                
                if chunk_type == "basic":
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    text_chunks = text_splitter.split_text(note.get("text", ""))
                    chunks = [
                        {
                            "content": chunk,
                            "metadata": metadata.copy(),
                            "section": "text"
                        } for chunk in text_chunks
                    ]
                    all_chunks.extend(chunks)
                    print(f"Created {len(chunks)} basic chunks from note")
                else:
                    chunks = process_note_sections(note.get("text", ""), metadata, self.chat_model)
                    display_chunks(chunks, max_display=5)
                    all_chunks.extend(chunks)
                    print(f"Created {len(chunks)} detailed chunks with section analysis")
                    
            except Exception as e:
                print(f"Error processing medical note {json_path}: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
    
        if not all_chunks:
            raise ValueError("No valid medical documents were processed from the input folder")
    
        search_type = "hybrid" if enable_hybrid else "standard"
        # Add suffix for Milvus Lite
        lite_suffix = "_lite" if hasattr(self, 'use_milvus_lite') and self.use_milvus_lite else ""
        self.collection_name = f"json_documents_{chunk_type}_{search_type}{lite_suffix}"
    
        try:
            # Check if we're using Milvus Lite
            if hasattr(self, 'use_milvus_lite') and self.use_milvus_lite:
                print(f"\nSetting up {'hybrid' if enable_hybrid else 'standard'} search with Milvus Lite...")
                
                # Configure retriever for Milvus Lite
                retriever_config = RetrieverConfig(
                    collection_name=self.collection_name,
                    use_milvus_lite=True,
                    milvus_lite_db=self.milvus_lite_db,
                    # Add the new configuration options
                    use_contextual_compression=self.use_contextual_compression,
                    use_multi_query=self.use_multi_query,
                    compression_similarity_threshold=self.compression_similarity_threshold,
                    llm=self.chat_model,
                    k_documents=self.k_documents,
                    score_threshold=0.6,  # Lower threshold for better retrieval
                    # Add hybrid search config if enabled
                    use_hybrid=enable_hybrid,
                    dense_weight=0.6,
                    sparse_weight=0.2,
                    rerank_weight=0.2,
                    embedding_model=self.embedding_model
                    )
                                    
                # Create retriever with Milvus Lite
                try:
                    self.vectorstore = create_retriever(
                        all_chunks,  # Pass all chunks for insertion
                        self.embedding_model,
                        retriever_config
                    )
                    
                    print(f"Successfully set up {'hybrid' if enable_hybrid else 'standard'} Milvus Lite retriever with collection: {self.collection_name}")
                except Exception as e:
                    print(f"Error creating retriever: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    raise
            
            elif enable_hybrid:
                print("\nDEBUG: Setting up hybrid search for JSONs...")
                
                # Debug: Print collection name
                print(f"\nDEBUG: Collection name: {self.collection_name}")
                
                search_config = SearchConfig(
                    collection_name=self.collection_name,
                    dense_weight=0.4,
                    sparse_weight=0.3,
                    rerank_weight=0.3
                )
                
                try:
                    # Use EnhancedHybridSearchFAISS instead of EnhancedHybridSearch
                    self.vectorstore = EnhancedHybridSearchFAISS(search_config)
                    print("\nDEBUG: Created EnhancedHybridSearchFAISS instance")
                    
                    # Debug: Print embedding function details
                    print(f"DEBUG: Embedding function type: {type(self.vectorstore.ef)}")
                    print(f"DEBUG: Dense dimension: {self.vectorstore.ef.dim['dense']}")
                    print(f"DEBUG: Sparse dimension: {self.vectorstore.ef.dim['sparse']}")
                    print(f"DEBUG: FAISS index initialized: {self.vectorstore.faiss_index is not None}")
                    
                except Exception as e:
                    print(f"\nDEBUG: Error creating EnhancedHybridSearchFAISS: {str(e)}")
                    raise
                
                # Process documents in batches
                print("\nDEBUG: Processing documents for hybrid search...")
                print(f"DEBUG: Total chunks to process: {len(all_chunks)}")
                
                batch_size = 100
                total_processed = 0
                
                # Debug: Print collection schema
                print("\nDEBUG: Collection Schema:")
                print(f"DEBUG: Collection name: {self.collection_name}")
                print(f"DEBUG: Collection fields: {self.vectorstore.collection.schema}")
                print(f"DEBUG: FAISS current size: {self.vectorstore.faiss_index.ntotal}")
                
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    batch_data = []
                    
                    print(f"\nDEBUG: Processing batch {i//batch_size + 1}, size: {len(batch)}")
                    print(f"DEBUG: Batch start index: {i}")
                    print(f"DEBUG: Batch end index: {min(i + batch_size, len(all_chunks))}")
                    
                    for chunk_idx, chunk in enumerate(batch):
                        try:
                            # Debug: Print detailed chunk info
                            print(f"\nDEBUG: Processing chunk {chunk_idx} in batch (total index: {i + chunk_idx})")
                            print(f"DEBUG: Content length: {len(chunk['content'])}")
                            print(f"DEBUG: Content preview: {chunk['content'][:50]}...")
                            print(f"DEBUG: Metadata keys: {chunk['metadata'].keys()}")
                            print(f"DEBUG: Section: {chunk['section']}")
                            
                            # Get embeddings
                            print("\nDEBUG: Getting embeddings...")
                            embeddings = self.vectorstore.ef([chunk['content']])
                            
                            # Debug: Print detailed embedding info
                            print("\nDEBUG: Embedding details:")
                            print(f"DEBUG: Dense type: {type(embeddings['dense'])}")
                            print(f"DEBUG: Sparse type: {type(embeddings['sparse'])}")
                            
                            if hasattr(embeddings['dense'], 'shape'):
                                print(f"DEBUG: Dense shape: {embeddings['dense'].shape}")
                            else:
                                print(f"DEBUG: Dense is not numpy array, converting...")
                                import numpy as np
                                embeddings['dense'] = np.array(embeddings['dense'])
                                print(f"DEBUG: Dense shape after conversion: {embeddings['dense'].shape}")
                                
                            if hasattr(embeddings['sparse'], 'shape'):
                                print(f"DEBUG: Sparse shape: {embeddings['sparse'].shape}")
                            else:
                                print(f"DEBUG: Sparse is not numpy array, converting...")
                                embeddings['sparse'] = np.array(embeddings['sparse'])
                                print(f"DEBUG: Sparse shape after conversion: {embeddings['sparse'].shape}")
                            
                            # Convert vectors
                            dense_vector = embeddings['dense'][0].tolist()
                            if hasattr(embeddings['sparse'], 'toarray'):
                                sparse_vector = embeddings['sparse'].toarray()[0].astype('float32')
                            else:
                                sparse_vector = embeddings['sparse'][0].astype('float32')
                            
                            # Debug: Print vector details
                            print("\nDEBUG: Vector details:")
                            print(f"DEBUG: Dense vector type: {type(dense_vector)}")
                            print(f"DEBUG: Dense vector length: {len(dense_vector)}")
                            print(f"DEBUG: Dense vector preview: {dense_vector[:5]}...")
                            print(f"DEBUG: Sparse vector type: {type(sparse_vector)}")
                            print(f"DEBUG: Sparse vector length: {len(sparse_vector)}")
                            print(f"DEBUG: Sparse vector preview: {sparse_vector[:5]}...")
                            
                            data = {
                                'content': chunk['content'],
                                'dense_vector': dense_vector,
                                'sparse_vector': sparse_vector,
                                'note_id': str(chunk['metadata']['note_id']),
                                'hadm_id': str(chunk['metadata']['hadm_id']),
                                'subject_id': str(chunk['metadata']['subject_id']),
                                'section': chunk['section'],
                                'charttime': str(chunk['metadata']['charttime']),
                                'storetime': str(chunk['metadata']['storetime'])
                            }
                            
                            # Debug: Print data structure
                            print("\nDEBUG: Data entry details:")
                            print(f"DEBUG: Data keys: {data.keys()}")
                            print(f"DEBUG: Data types: {[(k, type(v)) for k, v in data.items()]}")
                            
                            batch_data.append(data)
                            print(f"DEBUG: Successfully added data to batch. Current batch size: {len(batch_data)}")
                            print(f"DEBUG: FAISS current size: {self.vectorstore.faiss_index.ntotal}")
                            
                        except Exception as e:
                            print(f"\nDEBUG: Error processing chunk {chunk_idx}:")
                            print(f"DEBUG: Error type: {type(e)}")
                            print(f"DEBUG: Error message: {str(e)}")
                            import traceback
                            print(f"DEBUG: Stack trace:\n{traceback.format_exc()}")
                            print(f"DEBUG: Chunk content preview: {chunk['content'][:100]}")
                            continue
                    
                    # Insert batch if we have data
                    if batch_data:
                        try:
                            print(f"\nDEBUG: Attempting to insert batch of {len(batch_data)} documents")
                            print(f"DEBUG: First document content length: {len(batch_data[0]['content'])}")
                            print(f"DEBUG: First document vector lengths: dense={len(batch_data[0]['dense_vector'])}, sparse={len(batch_data[0]['sparse_vector'])}")
                            print(f"DEBUG: FAISS index size before insert: {self.vectorstore.faiss_index.ntotal}")
                            
                            self.vectorstore.insert_batch(batch_data)
                            total_processed += len(batch_data)
                            print(f"DEBUG: Successfully inserted batch. Total processed: {total_processed}")
                            print(f"DEBUG: FAISS index size after insert: {self.vectorstore.faiss_index.ntotal}")
                            
                        except Exception as e:
                            print(f"\nDEBUG: Error inserting batch:")
                            print(f"DEBUG: Error type: {type(e)}")
                            print(f"DEBUG: Error message: {str(e)}")
                            print(f"DEBUG: First entry keys: {batch_data[0].keys()}")
                            print(f"DEBUG: First entry types: {[(k, type(v)) for k, v in batch_data[0].items()]}")
                            import traceback
                            print(f"DEBUG: Stack trace:\n{traceback.format_exc()}")
                            raise
    
            else:
                print("Setting up standard vector search for JSONs...")
                if chunk_type == "detailed":
                    self.vectorstore, _ = save_detailed_chunks(
                        all_chunks,
                        self.embedding_model,
                        self.collection_name,
                        enable_hybrid=False
                    )
                else:  # basic
                    self.vectorstore, _ = save_basic_chunks(
                        all_chunks,
                        self.embedding_model,
                        self.collection_name
                    )
            
            self._setup_qa_chain()
            print(f"Successfully processed and stored {len(all_chunks)} medical note chunks")
            return len(all_chunks)
            
        except Exception as e:
            print(f"Error saving processed documents: {e}")
            raise

    def query(self, query_text: str) -> Dict:
        """Process a query using the configured QA chain."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Please process documents first.")
        
        print("\n==== PROCESSING QUERY ====")
        print(f"Query: {query_text}")
        print(f"Using collection: {self.collection_name}")
        print(f"Search type: {'Hybrid' if hasattr(self.vectorstore, 'hybrid_search') else 'Standard Vector'}")
        print(f"Number of documents to retrieve: {self.k_documents}")
        
        # Process the query through the configured QA chain
        # This will use whatever retriever was set up in _setup_qa_chain
        try:
            response = self.qa_chain(query_text)
            
            # Log information about retrieved documents for debugging
            if 'source_documents' in response and response['source_documents']:
                print(f"\n==== RETRIEVED {len(response['source_documents'])} DOCUMENTS ====")
                for i, doc in enumerate(response['source_documents'][:3], 1):  # Show first 3 docs
                    print(f"\nDocument {i}:")
                    print(f"Content preview: {doc.page_content[:100]}...")
                    print(f"Metadata: {doc.metadata}")
            else:
                print("\nNo documents retrieved")
                
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def process_questions(self, questions_csv: str, output_csv: str, chunk_size: int = 10):
        """Process multiple questions from a CSV and save results."""
        if not self.qa_chain:
            raise ValueError("Please process documents first")
            
        try:
            questions_df = pd.read_csv(questions_csv)
            if 'user_input' not in questions_df.columns:
                raise ValueError("CSV must contain a 'user_input' column")
            
            print(f"Processing questions using collection: {self.collection_name}")
            
            questions_df['answer'] = None
            questions_df['source_content'] = None
            questions_df['source_document_info'] = None
            
            for idx, row in questions_df.iterrows():
                try:
                    # Log for each question
                    print(f"\nProcessing question {idx+1}/{len(questions_df)}: {row['user_input']}")
                
                    response = self.query(row['user_input'])
                    
                    # Debug the response
                    print(f"Response structure: {list(response.keys())}")
                    print(f"Result type: {type(response.get('result'))} Length: {len(str(response.get('result', '')))}")
                    print(f"Number of source documents: {len(response.get('source_documents', []))}")
                    
                    
                    questions_df.at[idx, 'answer'] = response.get('result', '')
                    questions_df.at[idx, 'source_content'] = str([
                        doc.page_content[:200] for doc in response.get('source_documents', [])
                    ])
                    questions_df.at[idx, 'source_document_info'] = str([{
                        'note_id': doc.metadata.get('note_id', 'N/A'),
                        'subject_id': doc.metadata.get('subject_id', 'N/A'),
                        'hadm_id': doc.metadata.get('hadm_id', 'N/A'),
                        'charttime': doc.metadata.get('charttime', 'N/A'),
                        'storetime': doc.metadata.get('storetime', 'N/A')
                    } for doc in response.get('source_documents', [])])
                    
                    if (idx + 1) % chunk_size == 0:
                        questions_df.to_csv(output_csv, index=False)
                        print(f"Processed {idx + 1}/{len(questions_df)} questions")
                        
                except Exception as e:
                    print(f"Error processing question {idx}: {e}")
                    questions_df.at[idx, 'answer'] = f"ERROR: {str(e)}"
                    questions_df.at[idx, 'source_content'] = ""
                    questions_df.at[idx, 'source_document_info'] = ""
            
            questions_df.to_csv(output_csv, index=False)
            print(f"Completed processing. Results saved to {output_csv}")
            
        except Exception as e:
            print(f"Error processing CSV: {e}")

from pymilvus import MilvusClient
from langchain.schema import Document

class BaselineRAG:
    def __init__(self, embedding_model, chat_model, milvus_db_path, collection_name, k=5):
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.k = k
        self.collection_name = collection_name
        self.client = MilvusClient(milvus_db_path)

    def query(self, question: str):
        query_vector = self.embedding_model.embed_query(question)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=self.k,
            output_fields=["content", "note_id", "hadm_id", "subject_id", "section", "charttime", "storetime"]
        )
        
        docs = []
        if results and len(results[0]) > 0:
            for hit in results[0]:
                content = hit.get("content") or hit.get("entity", {}).get("content", "")
                metadata = {
                    "note_id": hit.get("note_id", ""),
                    "hadm_id": hit.get("hadm_id", ""),
                    "subject_id": hit.get("subject_id", ""),
                    "section": hit.get("section", ""),
                    "charttime": hit.get("charttime", hit.get("entity", {}).get("charttime", "")),
                    "storetime": hit.get("storetime", hit.get("entity", {}).get("storetime", ""))
                }
                docs.append(Document(page_content=content, metadata=metadata))
        
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""Use the following medical notes to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
        response = self.chat_model.predict(prompt)
        
        return {
            "result": response,
            "source_documents": docs
        }

def main():
    """Run the RAG processor interactively."""
    processor = RAGProcessor()
    
    PDF_FOLDER = "/raid/sivaks1/BasicRAG/NVIDIA_Clinicians_Assistant/MIMIC_notes_PDF copy"
    JSON_FOLDER = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"

    
    while True:
        print("\nMedical Document Processing System")
        print("=================================")
        print("1. Process PDF documents")
        print("2. Process JSON documents")
        print("3. Ask a question")
        print("4. Process questions from CSV")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ")
        
        if choice in ['1', '2']:
            # First choose chunking strategy
            print("\nSelect Chunking Strategy:")
            print("1. Basic (simple text splitting)")
            print("2. Detailed (medical section analysis)")
            chunk_choice = input("Enter chunking choice (1-2): ")
            chunk_type = "basic" if chunk_choice == '1' else "detailed"
            
            # Then choose search strategy
            print("\nSelect Search Strategy:")
            print("1. Standard Vector Search")
            print("2. Hybrid Search")
            print("3. Baseline RAG (no compression / no multi-query / no hybrid)")
            search_choice = input("Enter search choice (1-3): ")
            enable_hybrid = search_choice == '2'
            use_baseline = search_choice == '3'

            # Log the chosen configuration
            print(f"\nConfiguration:")
            print(f"- Chunking: {chunk_type}")
            print(f"- Search: {'Hybrid' if enable_hybrid else 'Standard Vector'}")
            
            # Process documents based on type
            if choice == '1':
                try:
                    print(f"\nProcessing PDF documents from {PDF_FOLDER}")
                    num_chunks = processor.process_pdf_documents(
                        PDF_FOLDER,
                        chunk_type=chunk_type,
                        enable_hybrid=enable_hybrid
                    )
                    print(f"\nSuccessfully processed {num_chunks} chunks from PDF documents")
                except Exception as e:
                    print(f"Error processing PDF documents: {e}")
            else:
                try:
                    print(f"\nProcessing JSON documents from {JSON_FOLDER}")
                    num_chunks = processor.process_json_documents(
                        JSON_FOLDER,
                        chunk_type=chunk_type,
                        enable_hybrid=enable_hybrid,
                        use_compression=not use_baseline,
                        use_multi_query=not use_baseline
                    )
                    print(f"\nSuccessfully processed {num_chunks} chunks from JSON documents")
                except Exception as e:
                    print(f"Error processing JSON documents: {e}")
                
        elif choice == '3':
            if not processor.qa_chain:
                print("\nError: Please process documents first!")
                continue
                
            try:
                question = input("\nEnter your medical question: ")
                print("\nProcessing question...")
                if use_baseline:
                    baseline = BaselineRAG(
                    embedding_model=processor.embedding_model,
                    chat_model=processor.chat_model,
                    milvus_db_path=processor.milvus_lite_db,
                    collection_name=processor.collection_name,
                    k=5
                    )
                    response = baseline.query(question)
                else:
                    response = processor.query(question)
                
                print("\nAnswer:", response['result'])
                # print("\nSource Documents:")
                # for i, doc in enumerate(response['source_documents'], 1):
                #     print(f"\nSource {i}:")
                #     print(f"Content: {doc.page_content[:200]}...")
                #     print(f"Metadata: {doc.metadata}")
            except Exception as e:
                print(f"Error processing question: {e}")
            
        elif choice == '4':
            if not processor.qa_chain:
                print("\nError: Please process documents first!")
                continue
                
            try:
                # Get processing parameters
                csv_path = input("\nEnter path to questions CSV: ")
                output_path = input("Enter path for results CSV: ")
                chunk_size = int(input("Save progress every N questions (default 10): ") or "10")
                
                print("\nProcessing questions...")
                processor.process_questions(csv_path, output_path, chunk_size)
            except Exception as e:
                print(f"Error processing questions from CSV: {e}")

        elif choice == '5':
            print("\nExiting Medical Document Processing System")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()

