import pandas as pd
import ast
import asyncio
from pathlib import Path
from io import StringIO
import sys

# Get project root directory (parent of backend)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import model setup
from model_setup import setup_chat_model, setup_embedding

# Import RAGAS evaluation tools
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

# === File path for test ===
FILE_PATH = "/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/eval/RAG_results/Basic_RAG_results.csv"

# === Helper ===
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

    # Wrap samples in EvaluationDataset
    dataset = EvaluationDataset(samples)

    # Load models
    chat_model = setup_chat_model()
    embedding_model = setup_embedding()

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
    df2=results.to_pandas()
    print(df2.head())
    df2.to_csv("/data/p_dsi/nvidia-capstone/NVIDIA_Clinicians_Assistant/eval/RAGAS_metrics/basic_RAG_results_RAGAS.csv", index=False)
    print("Saved results as a csv")

# === Entry point ===
if __name__ == "__main__":
    asyncio.run(main())

