from pathlib import Path
import json
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility, connections
from sklearn.feature_extraction.text import TfidfVectorizer
from model_setup import setup_embedding, setup_chat_model, setup_milvus
from document_processor import process_note_sections, debug_print

def create_collection(collection_name: str, dim: int, with_metadata: bool = True, enable_hybrid: bool = False) -> Collection:
    """Create Milvus collection with specified schema."""
    connections.connect(alias="default", host="localhost", port="19530")
    
    if utility.has_collection(collection_name):
        return Collection(collection_name)
        
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    ]
    
    if with_metadata:
        fields.extend([
            FieldSchema(name="note_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hadm_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="charttime", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="storetime", dtype=DataType.VARCHAR, max_length=100)
        ])
        
    if enable_hybrid:
        fields.append(FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=65535))
    
    schema = CollectionSchema(fields=fields)
    collection = Collection(collection_name, schema)
    
    # Create index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index("embedding", index_params)
    
    return collection

def load_json_note(file_path: str) -> dict:
    """Load and validate a single JSON note."""
    try:
        with open(file_path, 'r') as f:
            note = json.load(f)
        return note
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def chunks_to_documents(chunks: List[Dict], include_metadata: bool = True) -> List[Document]:
    """Convert chunks to Langchain Document objects."""
    documents = []
    for chunk in chunks:
        metadata = {}
        if include_metadata:
            metadata = {
                "note_id": str(chunk['metadata']['note_id']),
                "hadm_id": str(chunk['metadata']['hadm_id']),
                "subject_id": str(chunk['metadata']['subject_id']),
                "section": chunk['section'],
                "charttime": str(chunk['metadata']['charttime']),
                "storetime": str(chunk['metadata']['storetime'])
            }
            if 'keywords' in chunk:
                metadata['keywords'] = chunk['keywords']
        
        doc = Document(
            page_content=chunk['content'],
            metadata=metadata
        )
        documents.append(doc)
    return documents

def save_basic_chunks(chunks: List[Dict], embedding_model, collection_name: str = "medical_notes_basic"):
    """Save chunks with just content and embeddings."""
    print(f"\nDEBUG: Starting save_basic_chunks with {len(chunks)} chunks")
    
    documents = chunks_to_documents(chunks, include_metadata=False)
    print(f"\nDEBUG: Converted to {len(documents)} documents")
    
    try:
        print("\nDEBUG: Creating vectorstore...")
        vectorstore = setup_milvus(
            embedding_model=embedding_model,
            docs=documents
        )
        print("DEBUG: Vectorstore created successfully")
        return vectorstore, None
        
    except Exception as e:
        print(f"DEBUG: Error creating vectorstore: {str(e)}")
        raise


class HybridSearchWrapper:
    def __init__(self, vectorstore, keyword_weight: float = 0.4):
        self.vectorstore = vectorstore
        self.keyword_weight = keyword_weight
        self.k = 4

    def hybrid_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform hybrid search with improved keyword matching."""
        k = k or self.k
        
        # Get initial vector results
        vector_results = self.vectorstore.similarity_search_with_score(
            query,
            k=k*2  # Get more results for better ranking
        )
        
        # Process results with keyword matching
        results_with_scores = []
        query_terms = set(query.lower().split())
        
        for doc, score in vector_results:
            # Base score from vector similarity
            final_score = 1 - score  # Convert distance to similarity
            
            # Apply keyword matching if available
            if 'keywords' in doc.metadata:
                keywords = doc.metadata['keywords'].lower()
                matches = sum(1 for term in query_terms if term in keywords)
                if matches:
                    keyword_boost = matches / len(query_terms)
                    final_score = (1 - self.keyword_weight) * final_score + self.keyword_weight * keyword_boost
            
            results_with_scores.append((doc, final_score))
        
        # Return top k results
        sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)[:k]
        return sorted_results

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Standard similarity search interface."""
        results = self.hybrid_search(query, k)
        return [doc for doc, _ in results]

    def similarity_search_with_score(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Similarity search with scores interface."""
        return self.hybrid_search(query, k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Required method for retriever interface."""
        return self.similarity_search(query, self.k)

    def as_retriever(self, **kwargs):
        """Retriever interface compatibility."""
        self.k = kwargs.get('search_kwargs', {}).get('k', 4)
        return self
    

def save_detailed_chunks(chunks: List[Dict], embedding_model, collection_name: str = "medical_notes_detailed", enable_hybrid: bool = False):
    """Save chunks with hybrid search capability."""
    if enable_hybrid:
        try:
            # Initialize vectorizer
            vectorizer = TfidfVectorizer(
                max_features=150,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            # Collect all texts for better TF-IDF computation
            all_texts = [chunk['content'] for chunk in chunks]
            vectorizer.fit(all_texts)
            
            # Extract keywords for each chunk
            for i, chunk in enumerate(chunks):
                tfidf_matrix = vectorizer.transform([chunk['content']])
                feature_array = tfidf_matrix.toarray()[0]
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top scoring terms
                term_scores = list(zip(feature_names, feature_array))
                sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)
                keywords = " ".join([term for term, score in sorted_terms[:15] if score > 0])
                chunks[i]['keywords'] = keywords
                
        except Exception as e:
            print(f"Warning: Error in keyword extraction: {e}")
    
    documents = chunks_to_documents(chunks, include_metadata=True)
    
    try:
        # Create vectorstore using setup_milvus
        vectorstore = setup_milvus(
            embedding_model=embedding_model,
            docs=documents
        )
        
        # Wrap with hybrid search if enabled
        if enable_hybrid:
            vectorstore = HybridSearchWrapper(vectorstore)
        
        print(f"Successfully processed {len(chunks)} chunks")
        return vectorstore, None
        
    except Exception as e:
        print(f"Error in save_detailed_chunks: {e}")
        raise

def display_chunks(chunks, max_display=10):
    """Display chunks with detailed information."""
    print("\n" + "="*80)
    print(f"Displaying up to {max_display} chunks")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:max_display]):
        print(f"\nChunk {i+1}:")
        print("-"*40)
        print(f"Section: {chunk['section']}")
        print(f"Metadata: note_id={chunk['metadata']['note_id']}, " 
              f"hadm_id={chunk['metadata']['hadm_id']}")
        print("Content:")
        print("-"*20)
        print(chunk['content'])
        if 'keywords' in chunk:
            print("Keywords:")
            print(chunk['keywords'])
        print("-"*40)
    
    if len(chunks) > max_display:
        print(f"\n... and {len(chunks) - max_display} more chunks")

def main():
    # Initialize models
    debug_print("Initializing models")
    nvidia_chat = setup_chat_model()
    nvidia_chat.max_tokens = 2000
    embedding_model = setup_embedding()
    chat_model = nvidia_chat
    
    # Set up paths
    json_dir = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"
    all_chunks = []
    
    # Determine note type from json_dir
    note_type = "discharge" if "discharge" in json_dir else "radiology"

    # Process each note
    glob_pattern = f"note_{note_type}_*_*.json"
    for json_path in Path(json_dir).glob(glob_pattern):
        print(f"\nProcessing {json_path.name}")
        
        note = load_json_note(str(json_path))
        if not note:
            continue
            
        metadata = {
            "note_id": note.get("note_id"),
            "subject_id": note.get("subject_id"),
            "hadm_id": note.get("hadm_id"),
            "charttime": note.get("charttime"),
            "storetime": note.get("storetime")
        }
        
        chunks = process_note_sections(note.get("text", ""), metadata, chat_model)
        display_chunks(chunks, max_display=10)
        all_chunks.extend(chunks)
        
        sections = set(chunk["section"] for chunk in chunks)
        print(f"\nFound {len(chunks)} chunks across {len(sections)} sections")
        print("\nSections found:")
        for section in sections:
            section_chunks = [c for c in chunks if c["section"] == section]
            print(f"- {section}: {len(section_chunks)} chunks")
    
    print(f"\nTotal processed: {len(all_chunks)} chunks from {len(list(Path(json_dir).glob('note_discharge_*_*.json')))} notes")

    # Save options
    save_option = input("\nHow would you like to save the chunks?\n1. Basic (content only)\n2. Detailed (with metadata)\n3. Detailed with hybrid search\n4. Skip saving\nEnter choice (1-4): ")

    results = {"basic": None, "detailed": None, "hybrid": None}

    if save_option == '1':
        results["basic"] = save_basic_chunks(all_chunks, embedding_model)
    elif save_option == '2':
        results["detailed"] = save_detailed_chunks(all_chunks, embedding_model)
    elif save_option == '3':
        results["hybrid"] = save_detailed_chunks(all_chunks, embedding_model, enable_hybrid=True)
        
    # Optionally save to JSON
    save_json = input("\nWould you like to also save the processed chunks to JSON? (y/n): ")
    if save_json.lower() == 'y':
        output_file = "processed_chunks.json"
        with open(output_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print(f"Saved chunks to {output_file}")

    return results

if __name__ == "__main__":
    main()