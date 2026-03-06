"""Run the 6 assignment queries with all 3 models and write TSV files."""
import os
import csv

from config import QUERIES_DIR, TOP_K
from retrieval import structured_search, tfidf_search, bm25_search, get_preview


# Assignment queries (exact text from Assignment_1.pdf)
QUERIES = [
    "to be, or not to be",
    "English Grammar",
    "Philip K Dick",
    "Jabberwocky",
    "Gutenberg",
    "Dornröschen",
]

MODELS = [
    ("structured", structured_search),
    ("tfidf", tfidf_search),
    ("bm25", bm25_search),
]


def save_results_tsv(filepath, results, query, include_preview=True):
    """
    Write TSV: rank, book_id, score, preview (optional), line_number (optional).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["rank", "book_id", "score", "preview", "line_number"])
        for rank, (doc_id, score) in enumerate(results[:TOP_K], 1):
            if include_preview:
                preview, line_num = get_preview(doc_id, query)
                writer.writerow([rank, doc_id, f"{score:.6f}", preview, line_num])
            else:
                writer.writerow([rank, doc_id, f"{score:.6f}"])
    print(f"  Wrote {filepath}")


def run_all_queries():
    """Run each of the 6 queries with each model and save 18 TSV files."""
    os.makedirs(QUERIES_DIR, exist_ok=True)
    for q_num, query in enumerate(QUERIES, 1):
        print(f"Query {q_num}: {query!r}")
        for model_name, search_fn in MODELS:
            results = search_fn(query, top_k=TOP_K)
            filename = f"{q_num}_{model_name}.tsv"
            filepath = os.path.join(QUERIES_DIR, filename)
            save_results_tsv(filepath, results, query, include_preview=True)
    print("Done. 18 TSV files in", QUERIES_DIR)


if __name__ == "__main__":
    run_all_queries()
