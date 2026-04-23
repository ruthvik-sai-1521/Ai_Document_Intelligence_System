import json
import time
from pathlib import Path
from typing import List, Dict, Any
from src.core.pipeline import RAGPipeline
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, results_dir: str = "logs"):
        """
        Initialize the RAG Evaluator.
        """
        self.pipeline = pipeline
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardcoded sample test cases for baseline evaluation
        self.test_cases = [
            {
                "query": "What is the main topic of the document?",
                "expected_keywords": ["ai", "document", "intelligence"]
            },
            {
                "query": "What are the key methodologies used?",
                "expected_keywords": ["retrieval", "hybrid", "search"]
            },
            {
                "query": "A completely irrelevant query about cooking pasta.",
                "expected_keywords": [] # Expecting the pipeline to reject it (Insufficient data)
            }
        ]

    def add_test_case(self, query: str, expected_keywords: List[str]):
        """Add a custom test case programmatically."""
        self.test_cases.append({
            "query": query,
            "expected_keywords": [k.lower() for k in expected_keywords]
        })

    def evaluate(self) -> Dict[str, Any]:
        """Run all test cases, compute accuracy, and save results."""
        logger.info(f"Starting evaluation of {len(self.test_cases)} test cases...")
        
        results = []
        total_score = 0.0
        
        for idx, case in enumerate(self.test_cases):
            query = case["query"]
            expected = case["expected_keywords"]
            
            logger.info(f"Evaluating Case {idx+1}/{len(self.test_cases)}: '{query}'")
            
            start_time = time.time()
            answer, metadata = self.pipeline.run(query)
            latency = time.time() - start_time
            
            # Simple Accuracy Metric: Keyword Recall in the LLM answer
            answer_lower = answer.lower()
            
            if not expected:
                # If no keywords expected, we expect the system to reject the query (hallucination prevention)
                if "insufficient data" in answer_lower:
                    accuracy = 1.0
                else:
                    accuracy = 0.0
                matched_keywords = []
            else:
                matched_keywords = [kw for kw in expected if kw in answer_lower]
                accuracy = len(matched_keywords) / len(expected)
                
            total_score += accuracy
            
            result_record = {
                "test_id": idx + 1,
                "query": query,
                "expected_keywords": expected,
                "matched_keywords": matched_keywords,
                "accuracy_score": round(accuracy, 2),
                "latency_seconds": round(latency, 3),
                "confidence_score": round(metadata.get("confidence", 0.0), 4),
                "generated_answer": answer,
                "sources_used": list(set([s.get("metadata", {}).get("source", "Unknown") for s in metadata.get("sources", [])]))
            }
            results.append(result_record)
            
        final_accuracy = total_score / len(self.test_cases) if self.test_cases else 0.0
        
        summary = {
            "total_test_cases": len(self.test_cases),
            "average_accuracy": round(final_accuracy, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "details": results
        }
        
        self._save_results(summary)
        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """Save evaluation results to a JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Evaluation complete. Average Accuracy: {summary['average_accuracy'] * 100:.2f}%")
        logger.info(f"Results saved to {filepath}")

# Standalone script execution
if __name__ == "__main__":
    from src.core.embedding_manager import EmbeddingManager
    from src.retrieval.keyword_search import KeywordSearch
    from src.retrieval.retriever import HybridRetriever
    from src.llm.generator import LLMGenerator
    from src.core.config import FAISS_INDEX_PATH, CHUNKS_PATH, BM25_INDEX_PATH
    
    # Initialize core pipeline components
    logger.info("Initializing components for evaluation...")
    emb_mgr = EmbeddingManager(index_path=FAISS_INDEX_PATH, chunks_path=CHUNKS_PATH)
    kw_search = KeywordSearch(BM25_INDEX_PATH)
    retriever = HybridRetriever(emb_mgr, kw_search)
    llm = LLMGenerator()
    pipeline = RAGPipeline(retriever, llm)
    
    # Run evaluation
    evaluator = RAGEvaluator(pipeline)
    evaluator.evaluate()
