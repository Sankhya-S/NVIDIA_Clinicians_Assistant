import os
import glob
import warnings
import traceback
from langchain_core.callbacks import Callbacks
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.testset.persona import Persona
import logging
logger = logging.getLogger(__name__)
from ragas.llms.base import BaseRagasLLM
from ragas.testset.synthesizers.prompts import ThemesPersonasMatchingPrompt
import tiktoken
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasMatchingPrompt,
    ThemesPersonasInput,
    PersonaThemesMapping
)
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.synthesizers.multi_hop.prompts import QueryAnswerGenerationPrompt

from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.base import MultiHopQuerySynthesizer, MultiHopScenario


from dataclasses import dataclass
import typing as t
import numpy as np
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryAnswerGenerationPrompt,
    QueryConditions,
    GeneratedQueryAnswer
)
# Import your model setup functions
from model_setup import setup_chat_model, setup_embedding



class CustomQueryAnswerGenerationPrompt(QueryAnswerGenerationPrompt):
    instruction: str = (
        "Generate a medical query and answer based on the given context and persona.\n"
        "Your response must be in this exact JSON format:\n"
        '{"query": "your question here", "answer": "your detailed answer here"}\n\n'
        "Instructions:\n"
        "1. Generate a question that reflects the persona's expertise and interest\n"
        "2. Provide an answer using only information from the given context\n"
        "3. Format as a single JSON object with 'query' and 'answer' fields\n\n"
        "Example:\n"
        '{\n'
        '    "query": "What are the prescribed medication dosages?",\n'
        '    "answer": "According to the medical records, albuterol 2 puffs every 4 hours was prescribed."\n'
        '}'
    )

    def format_context(self, nodes: t.List[Node]) -> t.List[str]:
        """Add metadata to each context before generation."""
        formatted_contexts = []
        for node in nodes:
            metadata = node.properties.get("document_metadata", {})
            content = node.properties.get("page_content", "")
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "")
            
            context_with_metadata = f"[Source: {source}, Page: {page}]\n{content}"
            formatted_contexts.append(context_with_metadata)
            
        return formatted_contexts


    async def generate(
        self,
        llm: BaseRagasLLM,
        data: QueryConditions,
        callbacks: Callbacks = None,
    ) -> GeneratedQueryAnswer:
        # Format context with metadata
        data.context = self.format_context(data.context)
        return await super().generate(llm, data, callbacks)

    def parse_output(self, llm_output: t.Any) -> GeneratedQueryAnswer:
        try:
            # If already correct format
            if isinstance(llm_output, GeneratedQueryAnswer):
                return llm_output

            # Convert StringIO to string if needed
            if hasattr(llm_output, 'read'):
                text = llm_output.read()
            else:
                text = str(llm_output)

            # Try to extract and parse JSON
            import json
            import re
            
            # Find JSON pattern
            json_match = re.search(r'\{[^{}]*\}', text.replace('\n', ' '))
            if json_match:
                data = json.loads(json_match.group())
                if 'query' in data and 'answer' in data:
                    return GeneratedQueryAnswer(
                        query=str(data['query']).strip(),
                        answer=str(data['answer']).strip()
                    )

            # If JSON parsing fails, try to extract query/answer directly
            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', text)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
            
            if query_match and answer_match:
                return GeneratedQueryAnswer(
                    query=query_match.group(1).strip(),
                    answer=answer_match.group(1).strip()
                )

            # Fallback to context if available
            context = getattr(self, 'last_input_data', None)
            if context and hasattr(context, 'context') and context.context:
                first_context = context.context[0][:200]
                return GeneratedQueryAnswer(
                    query="What medical information is described in this context?",
                    answer=first_context + "..."
                )

            # Final fallback
            return GeneratedQueryAnswer(
                query="What medical procedures are described?",
                answer="The context describes medical procedures and treatments."
            )

        except Exception as e:
            logger.warning(f"Error parsing output: {str(e)}")
            return GeneratedQueryAnswer(
                query="What medical information is available?",
                answer="The context provides medical information and procedures."
            )

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")
class CustomThemesPersonasMatchingPrompt(ThemesPersonasMatchingPrompt):
    def parse_output(self, llm_output: str) -> PersonaThemesMapping:
        """Parse the LLM output into a PersonaThemesMapping object."""
        try:
            # First try to parse as direct mapping
            if isinstance(llm_output, PersonaThemesMapping):
                return llm_output

            # Handle StringIO
            if hasattr(llm_output, 'read'):
                llm_output = llm_output.read()

            # If we got a string, try to parse it
            if isinstance(llm_output, str):
                # Clean up the JSON string
                import json
                try:
                    # Try to parse as a JSON first
                    data = json.loads(llm_output)
                    
                    # If we got a mapping directly, use it
                    if "mapping" in data:
                        return PersonaThemesMapping(mapping=data["mapping"])
                    
                    # If we got themes and personas, create a mapping
                    if "themes" in data and "personas" in data:
                        mapping = {}
                        for persona in data["personas"]:
                            mapping[persona["name"]] = data["themes"]
                        return PersonaThemesMapping(mapping=mapping)
                except json.JSONDecodeError:
                    pass

            # Get input data for fallback
            input_data = getattr(self, 'last_input_data', None)
            if input_data:
                # Create default mapping where each persona gets all themes
                mapping = {
                    persona.name: input_data.themes
                    for persona in input_data.personas
                }
                return PersonaThemesMapping(mapping=mapping)

            # If all else fails, create a minimal valid mapping
            return PersonaThemesMapping(mapping={
                "default_persona": ["default_theme"]
            })

        except Exception as e:
            logger.warning(f"Error parsing output: {str(e)}, using default mapping")
            try:
                if input_data:
                    return PersonaThemesMapping(mapping={
                        persona.name: input_data.themes
                        for persona in input_data.personas
                    })
            except:
                pass
            
            # Absolute fallback
            return PersonaThemesMapping(mapping={
                "default_persona": ["default_theme"]
            })


@dataclass
class CustomMultiHopQuery(MultiHopQuerySynthesizer):
    name: str = "custom_multi_hop_query_synthesizer"
    theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()
    generate_query_reference_prompt = CustomQueryAnswerGenerationPrompt()

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: t.List[Persona],
        callbacks: Callbacks,
    ) -> t.List[MultiHopScenario]:
        # Find nodes with entity overlap relationships
        triplets = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                True if rel.type == "entities_overlap" else False
            )
        )

        if len(triplets) == 0:
            raise ValueError(
                "No entities_overlap relationships found in the knowledge graph."
            )

        num_sample_per_cluster = max(1, int(np.ceil(n / len(triplets))))
        scenarios = []

        for triplet in triplets:
            if len(scenarios) < n:
                node_a, node_b = triplet[0], triplet[-1]
                overlapped_items = triplet[1].properties.get("overlapped_items", [])
                
                if overlapped_items:
                    # Extract themes from overlapped items
                    themes = [item[0] if isinstance(item, list) else item for item in overlapped_items]
                    
                    # Create simple mapping where each persona gets all themes
                    mapping = {
                        persona.name: themes 
                        for persona in persona_list
                    }

                    # Create scenarios using base class methods
                    base_scenarios = self.prepare_combinations(
                        nodes=[node_a, node_b],
                        combinations=[[theme] for theme in themes],  # Each theme as a single combination
                        personas=persona_list,
                        persona_item_mapping=mapping,
                        property_name="entities"
                    )

                    # Sample diverse combinations
                    sampled_scenarios = self.sample_diverse_combinations(
                        base_scenarios,
                        num_sample_per_cluster
                    )
                    scenarios.extend(sampled_scenarios)

        return scenarios


def num_tokens(text: str) -> int:
    """Count tokens in text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def safe_split_text(text: str, max_tokens: int = 450) -> list:
    """Split text into token-limited chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=50,
        length_function=num_tokens,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

def process_document(doc, embedding_model):
    """Process document."""
    try:
        content = doc.page_content
        if not content.strip():
            print("Empty document content")
            return None

        # Split content into chunks
        chunks = safe_split_text(content)

        # Process each chunk
        all_embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                # Embed chunk
                chunk_embedding = embedding_model.embed_query(chunk)
                all_embeddings.append(chunk_embedding)
            except Exception as e:
                print(f"Error embedding chunk {i+1}: {str(e)}")
                traceback.print_exc()

        if all_embeddings:
            # Average the embeddings
            avg_embedding = [sum(x)/len(x) for x in zip(*all_embeddings)]

            # Update metadata
            doc.metadata.update({
                "embedding": avg_embedding,
            })

            return doc

        return None

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        return None

def load_multiple_pdfs(folder_path):
    """Load all PDFs from a folder."""
    print(f"\nLoading PDFs from {folder_path}")
    all_docs = []
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    for pdf_path in pdf_files:
        try:
            print(f"Processing: {os.path.basename(pdf_path)}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load_and_split()
            all_docs.extend(docs)
            print(f"Added {len(docs)} chunks from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Error loading {pdf_path}: {str(e)}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs

def create_knowledge_graph(docs, chat_wrapper, embedding_wrapper, output_dir):
    """Create knowledge graph with default transforms."""
    try:
        print("\nCreating Knowledge Graph...")
        kg = KnowledgeGraph()

        # Add documents to the graph
        for doc in docs:
            node_properties = {
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
                "type": NodeType.DOCUMENT
            }
            if "embedding" in doc.metadata:
                node_properties["embedding"] = doc.metadata["embedding"]
            
            kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties=node_properties
                )
            )
        print(f"Added {len(kg.nodes)} nodes to knowledge graph")

        print("\nApplying default transforms...")
        transforms = default_transforms(documents=docs, llm=chat_wrapper, embedding_model=embedding_wrapper)
        apply_transforms(kg, transforms)
        kg.nodes = [node for node in kg.nodes if node.properties.get('keyphrases') is not None]

        relationship_builders = [
            JaccardSimilarityBuilder(
                property_name="keyphrases",
                new_property_name="keyphrase_similarity",
                threshold=0.1,
            ),
        ]
        apply_transforms(kg, relationship_builders)

        # Convert relationships to entities_overlap type
        for rel in kg.relationships:
            if rel.type == 'jaccard_similarity':
                rel.type = 'entities_overlap'
                source_keyphrases = set(rel.source.properties.get('keyphrases', []))
                target_keyphrases = set(rel.target.properties.get('keyphrases', []))
                overlapped = source_keyphrases.intersection(target_keyphrases)
                # Store just the strings
                rel.properties['overlapped_items'] = [[item] for item in overlapped]


        # Save graph
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "knowledge_graph.json")
        kg.save(output_path)
        print(f"Knowledge graph saved to {output_path}")

        loaded_kg = KnowledgeGraph.load(output_path)
        print("Knowledge graph loaded successfully")

        return loaded_kg

    except Exception as e:
        print(f"Error creating knowledge graph: {str(e)}")
        traceback.print_exc()
        return None

def generate_qa_dataset(loaded_kg, chat_wrapper, embedding_wrapper, testset_size=10):
    """Generate QA dataset using default settings."""
    try:
        print("\nGenerating test set...")
        personas = [
            Persona(
                name="Clinical Care Coordinator",
                role_description="A healthcare professional coordinating patient care and medical procedures."
            ),
            Persona(
                name="Medical Researcher",
                role_description="A medical researcher analyzing clinical data and medical findings."
            ),
            Persona(
                name="Healthcare Provider",
                role_description="A medical practitioner providing direct patient care."
            )
        ]

        synthesizer = CustomMultiHopQuery(llm=chat_wrapper)

        testset_generator = TestsetGenerator(
            llm=chat_wrapper,
            embedding_model=embedding_wrapper,
            knowledge_graph=loaded_kg,
            persona_list=personas
        )

        query_distribution = [(synthesizer, 1.0)]

        # Generate QA pairs
        qa_pairs = testset_generator.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            num_personas=3
        )

        # Save QA pairs with proper directory handling
        import os
        output_dir = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "generated_qa_pairs.csv")
        
        if qa_pairs is not None:
            try:
                qa_pairs.to_pandas().to_csv(output_path, index=False)
                print(f"Q&A pairs saved to '{output_path}'")
            except PermissionError:
                alt_output_path = os.path.expanduser("~/generated_qa_pairs.csv")
                print(f"Permission denied for {output_path}, trying to save to {alt_output_path}")
                qa_pairs.to_pandas().to_csv(alt_output_path, index=False)
                print(f"Q&A pairs saved to alternative location: {alt_output_path}")

        return qa_pairs

    except Exception as e:
        print(f"Error generating QA dataset: {str(e)}")
        traceback.print_exc()
        return None

def main():
    try:
        print("Initializing models")
        nvidia_chat = setup_chat_model()
        nvidia_chat.max_tokens = 2000  # Adjust as needed
        embedding_model = setup_embedding()
        chat_wrapper = LangchainLLMWrapper(nvidia_chat)
        embedding_wrapper = LangchainEmbeddingsWrapper(embedding_model)

        # Set directories (update these paths to your actual directories)
        pdf_folder = "/raid/sivaks1/nvidia-test/MIMIC notes processing/documents_for_rag/pdf"
        output_dir = "/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/output"

        # Load all PDFs
        docs = load_multiple_pdfs(pdf_folder)
        if not docs:
            print("No documents loaded. Exiting.")
            return

        # Process documents
        processed_docs = []
        for i, doc in enumerate(docs):
            print(f"\nProcessing document {i+1}/{len(docs)}")
            processed_doc = process_document(doc, embedding_model)
            if processed_doc:
                processed_docs.append(processed_doc)
                print(f"Successfully processed document {i+1}")
            else:
                print(f"Failed to process document {i+1}")

        if not processed_docs:
            print("No documents processed. Exiting.")
            return

        # Create knowledge graph
        loaded_kg = create_knowledge_graph(processed_docs, chat_wrapper, embedding_wrapper, output_dir)

        if loaded_kg:
            # Generate QA pairs
            qa_pairs = generate_qa_dataset(loaded_kg, chat_wrapper, embedding_wrapper, testset_size=50)

            if qa_pairs is not None:
                qa_output_path = os.path.join(output_dir, "generated_qa_pairs.csv")
                qa_pairs.to_pandas().to_csv(qa_output_path, index=False)
                print(f"Q&A pairs saved to {qa_output_path}")
            else:
                print("Failed to generate QA pairs")
        else:
            print("Failed to create knowledge graph")

    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
