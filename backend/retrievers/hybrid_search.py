from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
import numpy as np
import logging
import torch
from torch.utils.checkpoint import checkpoint
from pymilvus import (
    Collection, FieldSchema, CollectionSchema, DataType,
    utility, connections, AnnSearchRequest
)
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
import requests
response = requests.get("https://huggingface.co/BAAI/bge-m3/...", verify=False)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RerankedResult(NamedTuple):
    """Structure for storing search results with multiple scoring components"""
    document: Document
    dense_score: float
    sparse_score: float
    rerank_score: float
    final_score: float

@dataclass
class SearchConfig:
    """Configuration for optimizing hybrid search and reranking behavior"""
    collection_name: str
    dense_weight: float = 0.4
    sparse_weight: float = 0.3
    rerank_weight: float = 0.3
    num_shards: int = 2
    consistency_level: str = "Strong"

class CustomEmbeddingFunction:
    """Custom embedding function using Hugging Face transformers with memory optimization"""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        self.model = (
            AutoModel.from_pretrained(model_path)
            .to(self.device)
            .half()
            .eval()
        )

    def embed(self, texts: List[str], batch_size: int = 2) -> np.ndarray:
        """Generate embeddings for a list of texts in batches with memory optimization"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = checkpoint(
                    lambda **x: self.model(**x), 
                    **inputs
                )
            
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
            
            # Explicit memory cleanup
            torch.cuda.empty_cache()
        
        return np.vstack(embeddings)

class CustomReranker:
    """Custom reranker using Hugging Face transformers with memory optimization"""
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        self.model = (
            AutoModel.from_pretrained(model_path)
            .to(self.device)
            .half()
            .eval()
        )

    def rerank(self, candidates: List[Dict], query: str, batch_size: int = 2) -> List[Dict]:
        """Rerank candidates with reduced batch size and memory optimization"""
        query_embedding = self.embed([query])[0]
        candidate_texts = [doc["content"] for doc in candidates]
        
        candidate_embeddings = self.embed(candidate_texts, batch_size=batch_size)
        
        scores = np.dot(candidate_embeddings, query_embedding)
        for i, doc in enumerate(candidates):
            doc["score"] = scores[i]

        return sorted(candidates, key=lambda x: x["score"], reverse=True)

    def embed(self, texts: List[str], batch_size: int = 2) -> np.ndarray:
        """Embedding method with memory optimization"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = checkpoint(
                    lambda **x: self.model(**x), 
                    **inputs
                )
            
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
            
            torch.cuda.empty_cache()
        
        return np.vstack(embeddings)

class EnhancedHybridSearch:
    """Enhanced hybrid search with memory-aware implementation"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        model_path = "../models/bge-m3-extracted"
        self.ef = CustomEmbeddingFunction(model_path=model_path)
        self.reranker = CustomReranker(model_path=model_path)
        self.collection = None
        self.setup_collection()

    def setup_collection(self):
        """Set up Milvus collection for dense vectors"""
        connections.connect("default", host="localhost", port="19530")
        
        if utility.has_collection(self.config.collection_name):
            utility.drop_collection(self.config.collection_name)
        
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="note_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hadm_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="subject_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100)
        ]
        
        schema = CollectionSchema(fields)
        self.collection = Collection(
            name=self.config.collection_name,
            schema=schema,
            consistency_level=self.config.consistency_level,
            num_shards=self.config.num_shards
        )
        
        # Create index for dense vectors
        self.collection.create_index(
            field_name="dense_vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "IP",
                "params": {"nlist": 1024}
            }
        )
        
        self.collection.load()

    def _apply_diversity_ranking(
        self,
        results: List[RerankedResult],
        k: int
    ) -> List[RerankedResult]:
        """Apply diversity ranking to avoid similar results"""
        if not results:
            return results
        
        final_results = []
        seen_sections = set()
        
        for result in sorted(results, key=lambda x: x.final_score, reverse=True):
            section = result.document.metadata.get("section")
            if section not in seen_sections and len(final_results) < k:
                final_results.append(result)
                seen_sections.add(section)
        
        remaining_slots = k - len(final_results)
        if remaining_slots > 0:
            remaining_results = [
                result for result in results
                if result.document.metadata.get("section") in seen_sections
            ]
            final_results.extend(
                sorted(remaining_results, 
                       key=lambda x: x.final_score, 
                       reverse=True)[:remaining_slots]
            )
        
        return sorted(final_results, key=lambda x: x.final_score, reverse=True)

    def _build_expression(self, filters: Optional[Dict]) -> Optional[str]:
        """Build Milvus search expression from filters"""
        if not filters:
            return None
        
        expressions = []
        for field, value in filters.items():
            if isinstance(value, (int, float)):
                expr = f"{field} == {value}"
            else:
                expr = f"{field} == '{value}'"
            expressions.append(expr)
        
        return " && ".join(expressions)
