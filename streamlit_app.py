import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Optional
import time
import json

# Import your RAG processor
from RAG_implementation import RAGProcessor

class StreamlitRAGInterface:
    def __init__(self):
        if 'processor' not in st.session_state:
            st.session_state.processor = RAGProcessor()
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'processor_ready' not in st.session_state:
            st.session_state.processor_ready = False
        
    def display_chat_messages(self):
        """Display chat messages from history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {idx}**")
                            
                            # Metadata section
                            st.markdown("**Metadata:**")
                            metadata = source.get("metadata", {})
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("Note ID:", metadata.get("note_id", "N/A"))
                                st.write("Subject ID:", metadata.get("subject_id", "N/A"))
                                st.write("Admission ID:", metadata.get("hadm_id", "N/A"))
                            with cols[1]:
                                st.write("Chart Time:", metadata.get("charttime", "N/A"))
                                st.write("Store Time:", metadata.get("storetime", "N/A"))
                                st.write("Source Type:", metadata.get("source_type", "N/A"))
                            
                            # Display keywords if available
                            if "keywords" in source:
                                st.markdown("**Keywords:**")
                                st.write(source["keywords"])
                            
                            st.markdown("**Content:**")
                            st.markdown(source.get("content", "No content available"))
                            st.divider()
                            
    def process_documents(self, folder_path: str, doc_type: str, processing_choice: str) -> None:
        """Process documents and update session state"""
        with st.spinner(f"Processing {doc_type} documents..."):
            try:
                # Reset processor for new processing
                st.session_state.processor = RAGProcessor()
                
                # Determine embedding type and hybrid search based on choice
                if processing_choice == "Basic (content only)":
                    embedding_type = "basic"
                    enable_hybrid = False
                elif processing_choice == "Detailed (with metadata)":
                    embedding_type = "detailed"
                    enable_hybrid = False
                elif processing_choice == "Hybrid Search":
                    embedding_type = "basic"
                    enable_hybrid = True
                else:  # "Detailed with Hybrid Search"
                    embedding_type = "detailed"
                    enable_hybrid = True
                
                if doc_type == "PDF":
                    num_chunks = st.session_state.processor.process_pdf_documents(
                        folder_path, 
                        embedding_type=embedding_type,
                        enable_hybrid=enable_hybrid
                    )
                else:  # JSON
                    num_chunks = st.session_state.processor.process_json_documents(
                        folder_path, 
                        embedding_type=embedding_type,
                        enable_hybrid=enable_hybrid
                    )
                
                st.session_state.processor_ready = True
                st.success(f"Successfully processed {num_chunks} chunks from {doc_type} documents")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.session_state.processor_ready = False

    def render_ui(self):
        """Render the Streamlit UI"""
        st.title("RAG-powered Medical Chatbot")
        
        # Debug information
        if st.sidebar.checkbox("Show Debug Info"):
            st.sidebar.write("Processor Ready:", st.session_state.processor_ready)
            st.sidebar.write("Has Vectorstore:", hasattr(st.session_state.processor, 'vectorstore') and st.session_state.processor.vectorstore is not None)
            st.sidebar.write("Has QA Chain:", hasattr(st.session_state.processor, 'qa_chain') and st.session_state.processor.qa_chain is not None)
            if hasattr(st.session_state.processor, 'collection_name'):
                st.sidebar.write("Collection Name:", st.session_state.processor.collection_name)
        
        # Sidebar for document processing
        with st.sidebar:
            st.header("Document Processing")
            
            doc_type = st.selectbox("Select Document Type", ["PDF", "JSON"])
            processing_choice = st.selectbox(
                "Select Processing Type",
                [
                    "Basic (content only)",
                    "Detailed (with metadata)",
                    "Hybrid Search",
                    "Detailed with Hybrid Search"
                ]
            )
            
            folder_path = st.text_input(
                "Enter folder path",
                value="/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/MIMIC_notes_PDF" if doc_type == "PDF"
                else "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"
            )
            
            if st.button("Process Documents"):
                self.process_documents(folder_path, doc_type, processing_choice)

        # Display chat interface
        self.display_chat_messages()

        # Enhanced chat input with source display
        if prompt := st.chat_input("Ask a question about the medical documents"):
            if not st.session_state.processor_ready:
                st.error("Please process documents first!")
                return

            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.processor.query(prompt)
                        
                        # Format sources with more details
                        sources = []
                        for doc in response['source_documents']:
                            source_data = {
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            }
                            if "keywords" in doc.metadata:
                                source_data["keywords"] = doc.metadata["keywords"]
                            sources.append(source_data)
                        
                        # Add assistant response to chat
                        message = {
                            "role": "assistant",
                            "content": response['result'],
                            "sources": sources
                        }
                        st.session_state.messages.append(message)
                        
                        # Display response
                        st.markdown(response['result'])
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        import traceback
                        st.error(f"Full error: {traceback.format_exc()}")

def main():
    interface = StreamlitRAGInterface()
    interface.render_ui()

if __name__ == "__main__":
    main()