"""
BizIntel RAG Evaluation Runner.

Runs the full RAG pipeline on every query in the eval dataset,
scores each with the LLM-as-Judge evaluator, and saves results
to JSON + CSV.

Usage:
    uv run python -m bizintel.evaluation.run_eval
    uv run python -m bizintel.evaluation.run_eval --limit 5          # quick test
    uv run python -m bizintel.evaluation.run_eval --output eval_results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from bizintel.embeddings.embedder import StartupEmbedder
from bizintel.vectorstore.base import create_vector_store
from bizintel.rag.retriever import StartupRetriever
from bizintel.config.llm_client import get_llm_client
from bizintel.graph.builder import build_graph
from bizintel.config.settings import RERANK_ENABLED, HYBRID_SEARCH_ENABLED, LLM_MODEL, LLM_PROVIDER

from bizintel.evaluation.eval_dataset import EVAL_DATASET
from bizintel.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
)
logger = logging.getLogger("eval")


def run_evaluation(
    output_dir: Path,
    limit: int | None = None,
) -> None:
    """Run the full evaluation pipeline and save results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialise components ────────────────────────────────────────
    logger.info("Loading embedder…")
    embedder = StartupEmbedder()

    logger.info("Loading vector store…")
    store = create_vector_store()

    doc_count = store.count
    if doc_count == 0:
        logger.error(
            "Vector store is empty. Run batch_embed.py first:\n"
            "  uv run python -m bizintel.pipeline.batch_embed --reset"
        )
        return

    logger.info("Vector store has %d documents", doc_count)

    # ── Optional reranker ────────────────────────────────────────────
    reranker = None
    if RERANK_ENABLED:
        from bizintel.rag.reranker import StartupReranker
        logger.info("Loading cross-encoder reranker…")
        reranker = StartupReranker()

    # ── Optional BM25 index for hybrid search ────────────────────────
    bm25_index = None
    if HYBRID_SEARCH_ENABLED:
        from bizintel.search.bm25_search import BM25Index
        logger.info("Building BM25 index…")
        doc_ids, texts, metadatas = store.get_all_documents()
        bm25_index = BM25Index(doc_ids, texts, metadatas)

    retriever = StartupRetriever(
        embedder, store, reranker=reranker, bm25_index=bm25_index,
    )
    llm_client = get_llm_client()
    graph = build_graph(retriever=retriever, llm_client=llm_client)
    evaluator = RAGEvaluator()

    # ── Select queries ───────────────────────────────────────────────
    queries = EVAL_DATASET[:limit] if limit else EVAL_DATASET
    total = len(queries)
    logger.info("Running evaluation on %d queries…", total)

    # ── Run + Score ──────────────────────────────────────────────────
    results: list[dict] = []
    overall_start = time.perf_counter()
    from bizintel.config.settings import EVAL_QUERY_DELAY
    QUERY_DELAY = EVAL_QUERY_DELAY  # 15s for Groq (30 RPM), 0s for OpenAI

    for i, entry in enumerate(queries, 1):
        query_id = entry["id"]
        query = entry["query"]
        analysis_type = entry["analysis_type"]

        # Respect API rate limits between queries
        if i > 1:
            time.sleep(QUERY_DELAY)

        logger.info("[%d/%d] Evaluating: %s — '%s'", i, total, query_id, query[:60])

        # Run the RAG pipeline
        t0 = time.perf_counter()
        try:
            rag_result = graph.invoke(
                {"user_query": query},
            )
        except Exception as e:
            logger.error("Query %s failed: %s", query_id, e)
            results.append({
                "id": query_id,
                "query": query,
                "analysis_type": analysis_type,
                "error": str(e),
            })
            continue

        latency = time.perf_counter() - t0

        answer = rag_result["answer"]
        sources = rag_result["source_docs"]
        retrieved_docs = [s["text"] for s in sources]

        # Count tokens (from the last LLM call — approximate)
        total_tokens = 0
        # We'll estimate from answer length since chain doesn't expose usage directly
        # A more precise version would modify chain.analyze() to return usage
        total_tokens = len(answer.split()) * 2  # rough estimate

        # Score with evaluator
        scores = evaluator.evaluate(
            query=query,
            analysis_type=analysis_type,
            answer=answer,
            retrieved_docs=retrieved_docs,
            expected_domains=entry["expected_domains"],
            expected_sections=entry["expected_sections"],
            bad_results=entry["bad_results"],
            latency=latency,
            total_tokens=total_tokens,
        )

        result = {
            "id": query_id,
            "query": query,
            "analysis_type": analysis_type,
            "model_name": LLM_MODEL,
            "description": entry["description"],
            "answer_preview": answer[:200] + "…" if len(answer) > 200 else answer,
            "num_sources": len(sources),
            **scores,
        }

        results.append(result)

        logger.info(
            "  → ctx=%.2f  gnd=%.2f  ans=%.2f  prec=%.2f  struct=%.2f  bad=%.1f  "
            "latency=%.1fs",
            scores["context_relevancy"],
            scores["groundedness"],
            scores["answer_relevancy"],
            scores["precision_at_k"],
            scores["structure_score"],
            scores["bad_result_check"],
            scores["latency_seconds"],
        )

    overall_elapsed = time.perf_counter() - overall_start

    # ── Compute aggregates ───────────────────────────────────────────
    scored = [r for r in results if "error" not in r]

    if scored:
        avg = lambda key: sum(r[key] for r in scored) / len(scored)

        summary = {
            "run_date": datetime.now().isoformat(),
            "model_name": LLM_MODEL,
            "llm_provider": LLM_PROVIDER,
            "total_queries": total,
            "successful": len(scored),
            "failed": total - len(scored),
            "total_time_seconds": round(overall_elapsed, 1),
            "avg_latency_seconds": round(avg("latency_seconds"), 2),
            "avg_context_relevancy": round(avg("context_relevancy"), 3),
            "avg_groundedness": round(avg("groundedness"), 3),
            "avg_answer_relevancy": round(avg("answer_relevancy"), 3),
            "avg_precision_at_k": round(avg("precision_at_k"), 3),
            "avg_structure_score": round(avg("structure_score"), 3),
            "avg_bad_result_check": round(avg("bad_result_check"), 3),
        }

        # Per analysis-type breakdown
        type_breakdown = {}
        for atype in ("similar", "swot", "competitor", "comparison", "ecosystem", "auto"):
            type_results = [r for r in scored if r["analysis_type"] == atype]
            if type_results:
                type_breakdown[atype] = {
                    "count": len(type_results),
                    "avg_context_relevancy": round(
                        sum(r["context_relevancy"] for r in type_results) / len(type_results), 3
                    ),
                    "avg_groundedness": round(
                        sum(r["groundedness"] for r in type_results) / len(type_results), 3
                    ),
                    "avg_answer_relevancy": round(
                        sum(r["answer_relevancy"] for r in type_results) / len(type_results), 3
                    ),
                    "avg_precision_at_k": round(
                        sum(r["precision_at_k"] for r in type_results) / len(type_results), 3
                    ),
                }

        summary["by_analysis_type"] = type_breakdown
    else:
        summary = {"error": "All queries failed"}

    # ── Save JSON ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"eval_{timestamp}.json"
    output_data = {"summary": summary, "results": results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved: %s", json_path)

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = output_dir / f"eval_{timestamp}.csv"
    if scored:
        fieldnames = list(scored[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in scored:
                writer.writerow(row)
        logger.info("CSV saved: %s", csv_path)

    # ── Print summary to console ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BizIntel RAG Evaluation — Summary")
    print("=" * 70)

    if "error" not in summary:
        print(f"  Model             : {summary['llm_provider']} / {summary['model_name']}")
        print(f"  Queries evaluated : {summary['successful']}/{summary['total_queries']}")
        print(f"  Total time        : {summary['total_time_seconds']:.1f}s")
        print(f"  Avg latency       : {summary['avg_latency_seconds']:.2f}s per query")
        print()
        print("  ┌────────────────────────┬─────────┐")
        print("  │ Metric                 │  Score  │")
        print("  ├────────────────────────┼─────────┤")
        print(f"  │ Context Relevancy      │  {summary['avg_context_relevancy']:.3f}  │")
        print(f"  │ Groundedness           │  {summary['avg_groundedness']:.3f}  │")
        print(f"  │ Answer Relevancy       │  {summary['avg_answer_relevancy']:.3f}  │")
        print(f"  │ Precision@K            │  {summary['avg_precision_at_k']:.3f}  │")
        print(f"  │ Structure Score        │  {summary['avg_structure_score']:.3f}  │")
        print(f"  │ Bad Result Check       │  {summary['avg_bad_result_check']:.3f}  │")
        print("  └────────────────────────┴─────────┘")

        if type_breakdown:
            print()
            print("  Per Analysis Type:")
            print("  ┌──────────────┬───────┬────────┬──────┬────────┬────────┐")
            print("  │ Type         │ Count │ CtxRel │ Grnd │ AnsRel │ Prec@K │")
            print("  ├──────────────┼───────┼────────┼──────┼────────┼────────┤")
            for atype, data in type_breakdown.items():
                print(
                    f"  │ {atype:<12} │  {data['count']:>3}  │ {data['avg_context_relevancy']:.3f}  "
                    f"│{data['avg_groundedness']:.3f} │ {data['avg_answer_relevancy']:.3f}  │ {data['avg_precision_at_k']:.3f}  │"
                )
            print("  └──────────────┴───────┴────────┴──────┴────────┴────────┘")
    else:
        print("  ❌ All queries failed!")

    print()
    print(f"  Results: {json_path}")
    print(f"           {csv_path}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="BizIntel RAG Evaluation")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for JSON/CSV results (default: eval_results)",
    )
    args = parser.parse_args()

    run_evaluation(
        output_dir=Path(args.output),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
