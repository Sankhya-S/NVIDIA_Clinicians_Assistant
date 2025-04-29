from langchain.prompts import PromptTemplate
from typing import List
from langchain.schema import Document

class MedicalPrompts:
    """Manages prompt templates and document formatting for medical document analysis."""
    
    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """Gets the definitive medical RAG prompt for comprehensive clinical document analysis."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a world-class medical document analyst with deep expertise in clinical documentation interpretation.

{context}

Question: {question}

**HOW TO ANSWER**

RESPONSE GUIDELINES:
1. START with one direct, authoritative sentence that precisely answers the question.
2. PRESENT information in cohesive paragraphs without bullet points or unnecessary formatting.
3. ENSURE FAITHFULNESS to source documents - never add information not present in the context.
4. PRIORITIZE RELEVANCE - focus only on information directly related to the question.
5. SYNTHESIZE across documents chronologically, noting clinical progression when relevant.
6. ACKNOWLEDGE information gaps explicitly rather than making assumptions.
7. CITE evidence by referring to specific document segments when supporting key claims.
8. DISTINGUISH between documented facts (highest confidence) and reasonable clinical inferences.
9. CONSIDER RECENCY - newer clinical information typically supersedes older information.
10. USE appropriate medical terminology while maintaining clarity.
11. PRESENT information in cohesive paragraphs without bullet points or unnecessary formatting.


Your goal is to provide clinically sound, accurate, and useful information that directly addresses the question while remaining absolutely faithful to the source documents.
"""
        )
    
    @staticmethod
    def combine_docs_with_metadata(docs: List[Document]) -> str:
        """Combines multiple documents with their metadata into a single formatted context."""
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
    
    @staticmethod
    def get_multi_query_prompt(query: str) -> str:
        """Gets a prompt for generating multiple query variations to improve retrieval."""
        return f"""
        Generate four diverse medical query reformulations of the following question to maximize relevant document retrieval.
        
        For the original query: "{query}"
        
        Create variations that:
        1. Use alternative medical terminology and phrasing
        2. Expand implicit concepts into explicit medical terms
        3. Consider related symptoms, conditions, and clinical relationships
        4. Address temporal aspects or progression where applicable
        5. Preserve any spelling variations or informal language from the original
        
        Return exactly four alternative queries, one per line, numbered 1-4.
        Do not include explanations, just the reformulated queries.
        """
