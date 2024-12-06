import pandas as pd
import ast
import warnings
import asyncio
import os
import glob
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference, ResponseRelevancy, NoiseSensitivity, Faithfulness
from model_setup import setup_embedding, setup_chat_model
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import nltk
from io import StringIO
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_nvidia_ai_endpoints")

# Base paths
BASE_PATH = '/raid/nvidia-project/NVIDIA_Clinicians_Assistant/NVIDIA_Clinicians_Assistant/eval'
RAG_RESULTS_PATH = os.path.join(BASE_PATH, 'RAG_results')
RAGAS_METRICS_PATH = os.path.join(BASE_PATH, 'RAGAS_metrics')

def clean_text(text):
    """Clean and validate text input"""
    if isinstance(text, StringIO):
        text = text.getvalue()
    if not isinstance(text, str):
        return str(text)
    text = str(text).strip()
    sentences = nltk.sent_tokenize(text)
    return ' '.join(sentences)

def process_list_string(x):
    """Safely process string representations of lists"""
    try:
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            return ast.literal_eval(x)
        return [x] if x else []
    except (ValueError, SyntaxError):
        return [x] if x else []

def get_output_filename(input_filename):
    """Convert RAG results filename to corresponding RAGAS metrics filename"""
    base_name = os.path.basename(input_filename)
    name_without_ext = os.path.splitext(base_name)[0]
    return f"{name_without_ext}_RAGAS_metrics.csv"

async def process_single_file(input_file_path, chat_wrapper, embedding_wrapper):
    """Process a single CSV file and generate metrics"""
    print(f"\n=== Processing file: {os.path.basename(input_file_path)} ===")
    
    # Load and process the CSV
    rag_results = pd.read_csv(input_file_path, dtype={
        'user_input': str,
        'reference': str,
        'source_content': str,
        'answer': str
    })

    # Clean text columns
    for column in ['user_input', 'reference', 'source_content', 'answer']:
        rag_results[column] = rag_results[column].apply(clean_text)

    # Process reference contexts
    rag_results['reference_contexts'] = rag_results['reference_contexts'].apply(process_list_string)

    # Process source document info if present
    if 'source_document_info' in rag_results.columns:
        rag_results['source_document_info'] = rag_results['source_document_info'].apply(process_list_string)

    # Convert to dictionary and create dataset
    data_dict = rag_results.to_dict(orient='records')
    
    # Define evaluation metrics
    context_precision_metric = LLMContextPrecisionWithReference(llm=chat_wrapper)
    relevancy_metric = ResponseRelevancy(llm=chat_wrapper, embeddings=embedding_wrapper)
    
    # Evaluate scores
    relevancy_scores, context_precision_scores = await evaluate_scores(data_dict, context_precision_metric, relevancy_metric)
    
    # Update data dictionary with scores
    for i, (rel_score, cp_score) in enumerate(zip(relevancy_scores, context_precision_scores)):
        data_dict[i]['context_precision_score'] = cp_score
        data_dict[i]['relevancy_score'] = rel_score

    # Convert to DataFrame and save
    df = pd.DataFrame(data_dict)
    output_filename = get_output_filename(input_file_path)
    output_path = os.path.join(RAGAS_METRICS_PATH, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to: {output_filename}")
    
    return output_path

async def evaluate_scores(dataset, context_precision_metric, relevancy_metric):
    """Evaluate scores for a dataset"""
    context_precision_scores = []
    relevancy_scores = []
    
    print("\n=== Starting evaluation ===")
    for idx, entry in enumerate(dataset):
        print(f"\nProcessing entry {idx + 1}")
        
        user_input = clean_text(entry['user_input'])
        reference = clean_text(entry['reference'])
        source_content = clean_text(entry['source_content'])
        answer = clean_text(entry['answer'])
        
        sample1 = SingleTurnSample(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=[source_content]
        )
        
        sample2 = SingleTurnSample(
            user_input=user_input,
            response=answer,
            retrieved_contexts=[source_content]
        )
        
        try:
            # Run metrics concurrently
            context_precision_score, relevancy_score = await asyncio.gather(
                context_precision_metric.single_turn_ascore(sample1),
                relevancy_metric.single_turn_ascore(sample2)
            )

            print(f"Context precision score: {context_precision_score}")
            print(f"Relevancy score: {relevancy_score}")
            
        except Exception as e:
            print(f"\nError during evaluation:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise e

        context_precision_scores.append(context_precision_score)
        relevancy_scores.append(relevancy_score)
        print("Successfully processed entry")
    
    return relevancy_scores, context_precision_scores

async def main():
    print("Starting main function...")
    
    # Set up models
    embedding_model = setup_embedding()
    chat_model = setup_chat_model()
    
    # Wrap models
    chat_wrapper = LangchainLLMWrapper(chat_model)
    embedding_wrapper = LangchainEmbeddingsWrapper(embedding_model)
    
    # Get all CSV files in the RAG_results directory
    csv_files = glob.glob(os.path.join(RAG_RESULTS_PATH, '*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    
    try:
        # Process each file
        for csv_file in csv_files:
            print(f"\nProcessing file: {os.path.basename(csv_file)}")
            await process_single_file(csv_file, chat_wrapper, embedding_wrapper)
            
        print('\nAll files processed successfully')
        
    except Exception as e:
        print(f"\nError in main function:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise e

if __name__ == "__main__":
    # Create RAGAS_metrics directory if it doesn't exist
    os.makedirs(RAGAS_METRICS_PATH, exist_ok=True)
    asyncio.run(main())