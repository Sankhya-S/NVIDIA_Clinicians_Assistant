import os
import json
import traceback
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer  # Correct import
from ragas.testset.synthesizers.base import QueryStyle
from ragas.testset.persona import Persona
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings as LangOpenAIEmbeddings
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Load JSON Notes ===
def load_json_notes(folder_path: str) -> list:
    docs = []
    for file in Path(folder_path).glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
            content = data.get("text", "").strip()
            if not content:
                continue
            metadata = {
                "note_id": data.get("note_id", ""),
                "hadm_id": data.get("hadm_id", ""),
                "subject_id": data.get("subject_id", ""),
                "charttime": data.get("charttime", ""),
                "storetime": data.get("storetime", ""),
                "source": file.name,
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        except Exception as e:
            print(f"Error loading {file.name}: {e}")
    print(f"✅ Loaded {len(docs)} JSON notes from {folder_path}")
    return docs

# === Token Chunking ===
def num_tokens(text: str) -> int:
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def safe_split_text(text: str, max_tokens: int = 450) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=50,
        length_function=num_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

# === Embed & Annotate Docs ===
def process_document(doc, embedding_model):
    try:
        content = doc.page_content
        chunks = safe_split_text(content)
        embeddings = []
        for chunk in chunks:
            try:
                vec = embedding_model.embed_query(chunk)
                embeddings.append(vec)
            except Exception as e:
                print(f"Embedding error: {e}")
        if not embeddings:
            return None
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding_list = avg_embedding.tolist()  # Critical fix
        doc.metadata["embedding"] = avg_embedding_list  # Critical fix
        return doc
    except Exception as e:
        print(f"Doc processing error: {e}")
        return None

# === Create Knowledge Graph ===
def create_knowledge_graph(docs, chat_llm, embed_wrapper, output_dir):
    print("🔗 Creating Knowledge Graph...")
    kg = KnowledgeGraph()

    # Create nodes for documents
    for doc in docs:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
                "type": NodeType.DOCUMENT,
                "embedding": doc.metadata.get("embedding")
            }
        )
        kg.nodes.append(node)

    # Apply default transformations and calculate keyphrase similarities
    transforms = default_transforms(documents=docs, llm=chat_llm, embedding_model=embed_wrapper)
    apply_transforms(kg, transforms)

    # Applying Jaccard Similarity to calculate 'overlapped_items'
    apply_transforms(kg, [
        JaccardSimilarityBuilder(
            property_name="keyphrases",
            new_property_name="keyphrase_similarity",
            threshold=0.1,
        )
    ])

    # Now assign overlapped_items property to the relationships or nodes
    for rel in kg.relationships:
        if rel.type == 'jaccard_similarity':
            rel.type = 'entities_overlap'
            source_kp = set(rel.source.properties.get('keyphrases', []))
            target_kp = set(rel.target.properties.get('keyphrases', []))
            rel.properties['overlapped_items'] = [[item] for item in source_kp & target_kp]

            # Make sure both nodes involved have the 'overlapped_items' property
            if 'overlapped_items' not in rel.source.properties:
                rel.source.properties['overlapped_items'] = rel.properties['overlapped_items']
            if 'overlapped_items' not in rel.target.properties:
                rel.target.properties['overlapped_items'] = rel.properties['overlapped_items']

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "knowledge_graph.json")
    kg.save(path)
    print(f"✅ Knowledge Graph saved to: {path}")
    return KnowledgeGraph.load(path)

# === Multi-hop Query Synthesizer ===
def generate_qa_dataset(kg, chat_llm, embed_wrapper, testset_size=50, output_dir="./output"):
    personas = [
        Persona(name="Clinical Care Coordinator", role_description="Coordinates patient care and treatment plans."),
        Persona(name="Medical Researcher", role_description="Analyzes medical data and records."),
        Persona(name="Healthcare Provider", role_description="Provides direct clinical care."),
    ]

    # Use the MultiHopSpecificQuerySynthesizer, which handles 2-hop queries
    synth = MultiHopSpecificQuerySynthesizer(llm=chat_llm)

    from ragas.testset import TestsetGenerator
    generator = TestsetGenerator(
        llm=chat_llm,
        embedding_model=embed_wrapper,
        knowledge_graph=kg,
        persona_list=personas
    )
    dist = [(synth, 1.0)]
    qa_pairs = generator.generate(
        testset_size=testset_size,
        query_distribution=dist,
        num_personas=len(personas)
    )

    # Save the generated QA pairs
    df = qa_pairs.to_pandas()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_qa_pairs.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ QA Pairs saved to: {out_path}")
    return df

# === MAIN ===
def main():
    input_folder = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/data/processed/note/json/discharge"
    output_dir = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/eval/KGoutput"
    testset_size = 50  # Increase the testset size to generate more QA pairs

    print("🚀 Initializing models...")
    chat_llm = LangchainLLMWrapper(ChatOpenAI())  # LangChain OpenAI Chat model
    embed_wrapper = LangchainEmbeddingsWrapper(LangOpenAIEmbeddings())  # LangChain OpenAI Embedding model

    docs = load_json_notes(input_folder)

    processed = []
    for i, doc in enumerate(docs):
        print(f"🧠 Processing {i+1}/{len(docs)}")
        processed_doc = process_document(doc, embed_wrapper)
        if processed_doc:
            processed.append(processed_doc)

    if not processed:
        print("❌ No documents successfully processed.")
        return

    kg = create_knowledge_graph(processed, chat_llm, embed_wrapper, output_dir)
    generate_qa_dataset(kg, chat_llm, embed_wrapper, testset_size, output_dir)

if __name__ == "__main__":
    main()

