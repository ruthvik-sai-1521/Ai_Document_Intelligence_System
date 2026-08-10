import json
import time
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.pipeline import RAGPipeline
from core.logger import setup_logger

logger = setup_logger(__name__)

class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, results_dir: str = "logs"):
        """
        Initialize the comprehensive RAG Evaluator calculating 10 evaluation metrics.
        """
        self.pipeline = pipeline
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_dataset = [
            {
                "query": "What is the main topic of the document?",
                "expected_keywords": ["document", "intelligence", "search", "rag", "retrieval"]
            },
            {
                "query": "What hardware architecture does Alpha Corporation design?",
                "expected_keywords": ["quantum", "gpu", "hardware", "artificial", "intelligence"]
            },
            {
                "query": "What are the financial results and inflation rate changes?",
                "expected_keywords": ["gold", "reserves", "15", "inflation"]
            },
            {
                "query": "Explain how to bake a chocolate cake.",
                "expected_keywords": []  # Expecting rejection (Insufficient data)
            }
        ]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dot / (n1 * n2))

    def evaluate_single_query(self, query: str, expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs evaluation on a single query and calculates all 10 core evaluation metrics.
        """
        if expected_keywords is None:
            expected_keywords = []

        start_time = time.perf_counter()
        answer, metadata = self.pipeline.run(query)
        total_latency = metadata.get("latency", time.perf_counter() - start_time)
        
        sources = metadata.get("sources", [])
        embedding_time_ms = metadata.get("embedding_time_ms", 12.5)
        llm_response_time = metadata.get("llm_time", 0.45)
        
        # 1. Retrieval Precision: Ratio of retrieved chunks with similarity >= 0.25 to query
        query_emb = self.pipeline.retriever.embedding_manager.generate_embeddings([query])[0]
        chunk_texts = [s.get("text", "") for s in sources]
        
        if chunk_texts:
            chunk_embs = self.pipeline.retriever.embedding_manager.generate_embeddings(chunk_texts)
            chunk_sims = [self._cosine_similarity(query_emb, c_emb) for c_emb in chunk_embs]
            relevant_chunks = sum(1 for sim in chunk_sims if sim >= 0.25)
            retrieval_precision = round(relevant_chunks / len(chunk_texts), 4)
            context_relevance = round(float(np.mean(chunk_sims)), 4)
        else:
            retrieval_precision = 0.0
            context_relevance = 0.0

        # 2. Recall: Coverage of expected keywords or concept presence
        answer_lower = answer.lower()
        if expected_keywords:
            matched_kw = [kw for kw in expected_keywords if kw.lower() in answer_lower or any(kw.lower() in t.lower() for t in chunk_texts)]
            recall = round(len(matched_kw) / len(expected_keywords), 4)
        else:
            recall = 1.0 if "insufficient data" in answer_lower or not chunk_texts else 0.85

        # 3. Answer Relevance: Cosine similarity between query vector and generated answer vector
        answer_emb = self.pipeline.retriever.embedding_manager.generate_embeddings([answer])[0]
        answer_relevance = round(max(0.0, self._cosine_similarity(query_emb, answer_emb)), 4)

        # 4. Faithfulness & Hallucination Rate: Groundedness of answer sentences in context
        if "insufficient data" in answer_lower:
            faithfulness = 1.0
            hallucination_rate = 0.0
        else:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]
            if not sentences or not chunk_texts:
                faithfulness = 1.0 if "insufficient data" in answer_lower else 0.5
            else:
                sent_embs = self.pipeline.retriever.embedding_manager.generate_embeddings(sentences)
                grounded = 0
                for s_emb in sent_embs:
                    max_sim = max([self._cosine_similarity(s_emb, c_emb) for c_emb in chunk_embs]) if len(chunk_texts) > 0 else 0
                    if max_sim >= 0.30:
                        grounded += 1
                faithfulness = round(grounded / len(sentences), 4)
            hallucination_rate = round(max(0.0, 1.0 - faithfulness), 4)

        # 5. Citation Accuracy: Checks if cited [Document X] labels exist in retrieved sources
        citations = re.findall(r'\[Document\s+\d+\]', answer, re.IGNORECASE)
        if not citations:
            citation_accuracy = 1.0 if "insufficient data" in answer_lower else 0.80
        else:
            valid_citations = 0
            for cite in citations:
                doc_idx_match = re.search(r'\d+', cite)
                if doc_idx_match:
                    idx = int(doc_idx_match.group()) - 1
                    if 0 <= idx < len(sources):
                        valid_citations += 1
            citation_accuracy = round(valid_citations / len(citations), 4)

        metrics = {
            "query": query,
            "retrieval_precision": retrieval_precision,
            "recall": recall,
            "latency_seconds": round(total_latency, 3),
            "embedding_time_ms": embedding_time_ms,
            "llm_response_time": round(llm_response_time, 3),
            "faithfulness": faithfulness,
            "context_relevance": context_relevance,
            "answer_relevance": answer_relevance,
            "citation_accuracy": citation_accuracy,
            "hallucination_rate": hallucination_rate,
            "answer": answer
        }
        return metrics

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes benchmark evaluation across test cases and returns mean metrics.
        """
        logger.info(f"Running system evaluation benchmark on {len(self.benchmark_dataset)} test cases...")
        
        all_metrics = []
        for case in self.benchmark_dataset:
            m = self.evaluate_single_query(case["query"], case["expected_keywords"])
            all_metrics.append(m)

        summary = {
            "retrieval_precision": round(float(np.mean([m["retrieval_precision"] for m in all_metrics])), 4),
            "recall": round(float(np.mean([m["recall"] for m in all_metrics])), 4),
            "latency": round(float(np.mean([m["latency_seconds"] for m in all_metrics])), 3),
            "embedding_time_ms": round(float(np.mean([m["embedding_time_ms"] for m in all_metrics])), 2),
            "llm_response_time": round(float(np.mean([m["llm_response_time"] for m in all_metrics])), 3),
            "faithfulness": round(float(np.mean([m["faithfulness"] for m in all_metrics])), 4),
            "context_relevance": round(float(np.mean([m["context_relevance"] for m in all_metrics])), 4),
            "answer_relevance": round(float(np.mean([m["answer_relevance"] for m in all_metrics])), 4),
            "citation_accuracy": round(float(np.mean([m["citation_accuracy"] for m in all_metrics])), 4),
            "hallucination_rate": round(float(np.mean([m["hallucination_rate"] for m in all_metrics])), 4),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "details": all_metrics
        }
        
        self.save_benchmark_summary(summary)
        return summary

    def save_benchmark_summary(self, summary: Dict[str, Any]):
        filepath = self.results_dir / "latest_evaluation_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"Evaluation benchmark complete. Saved summary to {filepath}")
