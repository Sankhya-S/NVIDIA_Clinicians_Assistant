from pathlib import Path
import json
import warnings
import pandas as pd
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import from our custom modules
from model_setup import setup_embedding, setup_chat_model, setup_milvus, get_pdf_paths
from document_processor import process_note_sections, debug_print
from document_embedding import (
    create_collection,
    load_json_note,
    chunks_to_documents,
    save_basic_chunks,
    save_detailed_chunks,
    display_chunks
)
from notes_processor import extract_text_from_pdf

class RAGProcessor:
    def __init__(self):
        self.chat_model = setup_chat_model()
        self.embedding_model = setup_embedding()
        self.collection_name = None
        self.vectorstore = None
        self.qa_chain = None
        self.k_documents = 10

    def set_retrieval_count(self, k: int):
        """Set the number of documents to retrieve."""
        self.k_documents = k
        if self.vectorstore:
            self._setup_qa_chain()

    def _create_metadata_aware_prompt(self):
        """Create a prompt template that incorporates metadata analysis."""
        return PromptTemplate(
            input_variables=["context", "metadata", "question"],
            template="""You are analyzing medical documents and their metadata to provide accurate answers.

Context from documents:
{context}

Document Metadata:
{metadata}

Based on both the content and metadata (like timestamps, source types, and IDs), please answer the following question:
{question}

Please provide a comprehensive answer that:
1. Addresses the main question
2. Incorporates relevant metadata insights (especially temporal relationships)
3. Notes if the information comes from multiple source types or time periods
4. Highlights any patterns or inconsistencies across documents
"""
        )


    def _process_metadata(self, source_documents: List):
        """Process and analyze metadata from source documents."""
        metadata_analysis = []
        for doc in source_documents:
            metadata = doc.metadata
            analysis = {
                "source_type": metadata.get("source_type", "unknown"),
                "timestamp": metadata.get("charttime") or metadata.get("storetime"),
                "document_id": metadata.get("note_id"),
                "additional_ids": {
                    "subject_id": metadata.get("subject_id"),
                    "hadm_id": metadata.get("hadm_id")
                }
            }
            metadata_analysis.append(analysis)
        return metadata_analysis


    def process_pdf_documents(self, pdf_folder: str, embedding_type: str = "detailed", enable_hybrid: bool = False):
        """Process PDF documents and store in vector database.
        
        Args:
            pdf_folder: Path to PDF documents
            embedding_type: "basic" (simple splitting) or "detailed" (full processing)
            enable_hybrid: Whether to enable hybrid search
        """
        debug_print(f"Processing PDFs from {pdf_folder}")
        all_chunks = []
        
        pdf_paths = get_pdf_paths(pdf_folder)
        for pdf_path in pdf_paths:
            try:
                # Extract text from PDF
                document_text = extract_text_from_pdf(pdf_path)
                
                metadata = {
                    "note_id": Path(pdf_path).stem,
                    "source_type": "pdf",
                    "file_path": pdf_path
                }
                
                if embedding_type == "basic":
                    # For basic embedding, just do simple text splitting
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=300,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    # Split the text into chunks
                    text_chunks = text_splitter.split_text(document_text)
                    
                    # Convert chunks to documents with metadata
                    chunks = [
                        {
                            "content": chunk,
                            "metadata": metadata.copy(),
                            "section": "text"  # Add default section for consistency
                        } for chunk in text_chunks
                    ]
                    
                    all_chunks.extend(chunks)
                    print(f"Created {len(chunks)} basic chunks for {pdf_path.name}")
                    
                else:
                    # For detailed embedding, use the full processing pipeline
                    chunks = process_note_sections(document_text, metadata, self.chat_model)
                    display_chunks(chunks, max_display=5)
                    all_chunks.extend(chunks)
                    
            except Exception as e:
                print(f"Error processing PDF {pdf_path}: {e}")

        # Determine collection name based on processing type and search method
        search_type = "hybrid" if enable_hybrid else "standard"
        process_type = "basic" if embedding_type == "basic" else "detailed"
        self.collection_name = f"pdf_documents_{process_type}_{search_type}"
        
        # Save chunks with appropriate method
        if embedding_type == "basic":
            if enable_hybrid:
                self.vectorstore, _ = save_detailed_chunks(
                    all_chunks, 
                    self.embedding_model, 
                    self.collection_name,
                    enable_hybrid=True
                )
            else:
                self.vectorstore, _ = save_basic_chunks(
                    all_chunks, 
                    self.embedding_model, 
                    self.collection_name
                )
        else:
            self.vectorstore, _ = save_detailed_chunks(
                all_chunks, 
                self.embedding_model, 
                self.collection_name,
                enable_hybrid=enable_hybrid
            )
        
        self._setup_qa_chain()
        return len(all_chunks)
        

    def process_json_documents(self, json_folder: str, embedding_type: str = "detailed", enable_hybrid: bool = False):
        """Process JSON documents and store in vector database.
        
        Args:
            json_folder: Path to JSON documents
            embedding_type: "basic" (simple splitting) or "detailed" (full processing)
            enable_hybrid: Whether to enable hybrid search
        """
        debug_print(f"Processing JSONs from {json_folder}")
        all_chunks = []
        
        for json_path in Path(json_folder).glob("**/*.json"):
            print(f"\nProcessing {json_path.name}")
            
            note = load_json_note(str(json_path))
            if not note:
                continue
                
            metadata = {
                "note_id": note.get("note_id"),
                "subject_id": note.get("subject_id"),
                "hadm_id": note.get("hadm_id"),
                "charttime": note.get("charttime"),
                "storetime": note.get("storetime"),
                "source_type": "json"
            }
            
            if embedding_type == "basic":
                # For basic embedding, just do simple text splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # Split the text into chunks
                text_chunks = text_splitter.split_text(note.get("text", ""))
                
                # Convert chunks to documents with metadata
                chunks = [
                    {
                        "content": chunk,
                        "metadata": metadata.copy(),
                        "section": "text"  # Add default section for consistency
                    } for chunk in text_chunks
                ]
                
                all_chunks.extend(chunks)
                print(f"Created {len(chunks)} basic chunks")
                
            else:
                # For detailed embedding, use the full processing pipeline
                chunks = process_note_sections(note.get("text", ""), metadata, self.chat_model)
                display_chunks(chunks, max_display=5)
                all_chunks.extend(chunks)

        # Determine collection name based on processing type and search method
        search_type = "hybrid" if enable_hybrid else "standard"
        process_type = "basic" if embedding_type == "basic" else "detailed"
        self.collection_name = f"json_documents_{process_type}_{search_type}"
        
        # Save chunks with appropriate method
        if embedding_type == "basic":
            if enable_hybrid:
                self.vectorstore, _ = save_detailed_chunks(
                    all_chunks, 
                    self.embedding_model, 
                    self.collection_name,
                    enable_hybrid=True
                )
            else:
                self.vectorstore, _ = save_basic_chunks(
                    all_chunks, 
                    self.embedding_model, 
                    self.collection_name
                )
        else:
            self.vectorstore, _ = save_detailed_chunks(
                all_chunks, 
                self.embedding_model, 
                self.collection_name,
                enable_hybrid=enable_hybrid
            )
        
        self._setup_qa_chain()
        return len(all_chunks)

    def _setup_qa_chain(self):
        """Set up the QA chain with metadata debugging."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_documents}
        )
        
        # Format the context to explicitly include metadata
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a medical document assistant analyzing these documents. Each document includes content and metadata.

            {context}

            Current Question: {question}

            Instructions for answering:
            1. For questions about IDs or dates, primarily use the METADATA values
            2. Report exact ID values as shown in metadata
            3. If the same ID appears in multiple documents, mention this
            4. Always prioritize metadata over document content for ID values
            """
        )

        def combine_docs_with_metadata(docs):
            """Combine documents while preserving metadata."""
            combined_docs = []
            for i, doc in enumerate(docs, 1):
                doc_text = f"""
                Document {i}:
                -----------------
                METADATA:
                - Patient/Subject ID: {doc.metadata.get('subject_id', 'N/A')}
                - Note ID: {doc.metadata.get('note_id', 'N/A')}
                - Admission ID: {doc.metadata.get('hadm_id', 'N/A')}
                - Chart Time: {doc.metadata.get('charttime', 'N/A')}
                - Store Time: {doc.metadata.get('storetime', 'N/A')}

                CONTENT:
                {doc.page_content}
                -----------------
                """
                combined_docs.append(doc_text)
            return "\n".join(combined_docs)

        def enhanced_qa(query):
            if isinstance(query, str):
                query_dict = {"query": query}
            else:
                query_dict = query
                
            # Get relevant documents
            docs = retriever.get_relevant_documents(query_dict["query"])
            
            # Create formatted context with metadata
            context = combine_docs_with_metadata(docs)
            
            # Debug print the actual context being sent to the LLM
            # print("\nDEBUG: Context being sent to LLM:")
            # print("=" * 50)
            # print(context[:1000])  # Print first 1000 chars to avoid cluttering
            # print("=" * 50)
            
            # Format prompt with context and question
            formatted_prompt = prompt.format(
                context=context,
                question=query_dict["query"]
            )
            
            # Debug print the full prompt
            # print("\nDEBUG: Full prompt:")
            # print("=" * 50)
            # print(formatted_prompt[:1000])  # Print first 1000 chars
            # print("=" * 50)
            
            # Run query
            llm_response = self.chat_model.predict(formatted_prompt)
            
            # Debug print the response
            # print("\nDEBUG: LLM Response:")
            # print("=" * 50)
            # print(llm_response)
            # print("=" * 50)
            
            # Structure the response
            response = {
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
            
            return response

        self.qa_chain = enhanced_qa

    def query(self, question: str):
        """Helper method to query the QA chain with proper output formatting."""
        if not self.qa_chain:
            raise ValueError("Please process documents first")
        
        response = self.qa_chain(question)
        
        # print("\nAnswer:", response['result'])
        # print("\nSource Documents:")
        
        # Format and display each source document with its metadata
        # for idx, (doc, metadata) in enumerate(zip(response['source_documents'], response['metadata_summary']), 1):
        #     print(f"\nSource {idx}:")
        #     print("Content:", doc.page_content[:200], "...")
        #     print("\nMetadata:")
        #     print(f"- Patient/Subject ID: {metadata['subject_id']}")
        #     print(f"- Note ID: {metadata['note_id']}")
        #     print(f"- Admission ID: {metadata['hadm_id']}")
        #     print(f"- Chart Time: {metadata['charttime']}")
        #     print(f"- Store Time: {metadata['storetime']}")
        #     print("-" * 50)
        
        return response

    def _get_temporal_range(self, metadata_analysis):
        """Calculate the temporal range of the documents."""
        timestamps = [m["timestamp"] for m in metadata_analysis if m["timestamp"]]
        if not timestamps:
            return None
        return {
            "earliest": min(timestamps),
            "latest": max(timestamps),
            "span": len(set(timestamps))
        }

    def process_questions(self, questions_csv: str, output_csv: str, chunk_size: int = 10):
        """Process questions from CSV using the current collection."""
        if not self.qa_chain:
            raise ValueError("No document collection has been processed yet.")
            
        try:
            questions_df = pd.read_csv(questions_csv)
            if 'user_input' not in questions_df.columns:
                raise ValueError("CSV must contain a 'user_input' column")
            
            print(f"Processing questions using collection: {self.collection_name}")
            
            # Enhanced columns for results
            questions_df['answer'] = None
            questions_df['source_content'] = None
            questions_df['source_document_info'] = None
            
            for idx, row in questions_df.iterrows():
                try:
                    response = self.qa_chain(row['user_input'])
                    
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
                        print(f"Processed {idx + 1}/{len(questions_df)} questions...")
                        
                except Exception as e:
                    print(f"Error processing question {idx}: {e}")
                    questions_df.at[idx, 'answer'] = f"ERROR: {str(e)}"
                    questions_df.at[idx, 'source_content'] = ""
                    questions_df.at[idx, 'source_document_info'] = ""
            
            questions_df.to_csv(output_csv, index=False)
            print(f"\nCompleted processing. Results saved to {output_csv}")
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
        
def main():
    processor = RAGProcessor()
    
    # Define paths
    PDF_FOLDER = "/raid/sivaks1/BasicRAG/NVIDIA_Clinicians_Assistant/MIMIC_notes_PDF copy"
    JSON_FOLDER = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"
    
    while True:
        print("\nChoose operation:")
        print("1. Process PDF documents")
        print("2. Process JSON documents")
        print("3. Ask a question")
        print("4. Process questions from CSV")
        print("5. Exit")
        
        choice = input("Enter choice (1-5): ")
        
        if choice in ['1', '2']:
            print("\nChoose embedding type:")
            print("1. Basic (content only)")
            print("2. Detailed (with metadata)")
            print("3. Hybrid Search")
            print("4. Detailed with Hybrid Search")
            embed_choice = input("Enter choice (1-4): ")
            
            if embed_choice == '1':
                embedding_type = "basic"
                enable_hybrid = False
            elif embed_choice == '2':
                embedding_type = "detailed"
                enable_hybrid = False
            elif embed_choice == '3':
                embedding_type = "basic"
                enable_hybrid = True
            else:  # embed_choice == '4'
                embedding_type = "detailed"
                enable_hybrid = True
                
            if choice == '1':
                num_chunks = processor.process_pdf_documents(
                    PDF_FOLDER,
                    embedding_type,
                    enable_hybrid=enable_hybrid)
                print(f"\nProcessed {num_chunks} chunks from PDF documents")
            else:
                num_chunks = processor.process_json_documents(
                    JSON_FOLDER,
                    embedding_type,
                    enable_hybrid=enable_hybrid)
                print(f"\nProcessed {num_chunks} chunks from JSON documents")
                
        elif choice == '3':
            if not processor.qa_chain:
                print("Please process documents first!")
                continue
                
            question = input("\nEnter your question: ")
            response = processor.query(question)
            print("\nAnswer:", response['result'])
            
        elif choice == '4':
            if not processor.qa_chain:
                print("Please process documents first!")
                continue
                
            csv_path = input("Enter path to questions CSV: ")
            output_path = input("Enter path for results CSV: ")
            chunk_size = int(input("Save progress every N questions (default 10): ") or "10")
            processor.process_questions(csv_path, output_path, chunk_size)
            
        elif choice == '5':
            break

if __name__ == "__main__":
    main()