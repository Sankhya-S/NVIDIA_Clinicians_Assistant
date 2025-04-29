import pandas as pd
import ast
import asyncio
from pathlib import Path
import os
from ragas.run_config import RunConfig

# === Set your OpenAI API key ===
from dotenv import load_dotenv
load_dotenv()

# === RAGAS core and metrics ===
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

# === LangChain-wrapped OpenAI models ===
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings as LangOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# === File path for test ===
FILE_PATH = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/eval/testset_output/detailed_hybrid_finetune.csv"

run_config = RunConfig(timeout=120.0)  # Increase timeout to 120 seconds

# === Helper to safely parse stringified lists ===
def safe_list(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    return [x] if isinstance(x, str) else x

# === Async Main ===
async def main():
    print(f"Reading from: {FILE_PATH}")
    df = pd.read_csv(FILE_PATH)
    df = df.fillna("")
    print(f"Loaded {len(df)} rows")

    # Handle list-like fields
    if "reference_contexts" in df.columns:
        df["reference_contexts"] = df["reference_contexts"].apply(safe_list)
    df["source_content"] = df["source_content"].apply(lambda x: x if isinstance(x, str) else "")

    # Build samples
    samples = []
    for _, row in df.iterrows():
        sample = SingleTurnSample(
            user_input=row["user_input"],
            response=row["answer"],
            retrieved_contexts=[row["source_content"]],
            reference=row.get("reference", None) or None,
            reference_contexts=row.get("reference_contexts", None) or None,
        )
        samples.append(sample)

    # Wrap samples in EvaluationDatase
    dataset = EvaluationDataset(samples)

    # === Use LangChain-wrapped OpenAI models ===
   # chat_model = LangchainLLMWrapper(ChatOpenAI())  # Defaults to gpt-4o-mini
    chat_model = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embedding_model = LangchainEmbeddingsWrapper(LangOpenAIEmbeddings())  # Defaults to ada-002

    print("\nEvaluating...")
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=chat_model,
        embeddings=embedding_model,
    )

    print("\n=== Evaluation Results ===")
    df2 = results.to_pandas()
    print(df2.head())
    output_path = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/eval/testset_output/final_RAGAS/fine-tuned/detailed_hybrid_finetune_RAGAS_4o.csv"
    df2.to_csv(output_path, index=False)
    print(f"Saved results as a csv to {output_path}")

# === Entry point ===
if __name__ == "__main__":
    asyncio.run(main())

