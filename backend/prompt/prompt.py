from langchain.prompts import PromptTemplate
from typing import List
from langchain.schema import Document

class MedicalPrompts:
    """Manages prompt templates and document formatting for medical document analysis."""

    @staticmethod
    def get_metadata_prompt() -> PromptTemplate:
        """Gets the main metadata-aware prompt template for medical document analysis."""
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

    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """Gets the medical QA prompt template focusing on metadata and document analysis."""
        return PromptTemplate(
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

    @staticmethod
    def combine_docs_with_metadata(docs: List[Document]) -> str:
        """Combines multiple documents with their metadata into a single formatted context.

        Args:
            docs: List of Document objects containing content and metadata

        Returns:
            A formatted string containing all documents with their metadata
        """
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