# backend/retrievers/hybrid_search.py

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, NamedTuple
import numpy as np
import logging
import torch
from pymilvus import (
    Collection, FieldSchema, CollectionSchema, DataType,
    utility, connections, AnnSearchRequest
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
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

class EnhancedHybridSearch:
    """Base class for hybrid search implementation"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
        model_path="../models/bge-m3-extracted"
        self.ef = BGEM3EmbeddingFunction(
            model_path=model_path,
            use_fp16=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reranker = BGERerankFunction(
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
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
        
        # First pass: get highest scoring result from each section
        for result in sorted(results, key=lambda x: x.final_score, reverse=True):
            section = result.document.metadata.get("section")
            if section not in seen_sections and len(final_results) < k:
                final_results.append(result)
                seen_sections.add(section)
        
        # Second pass: fill remaining slots with highest scoring results
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
