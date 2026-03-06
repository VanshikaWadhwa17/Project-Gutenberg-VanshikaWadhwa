"""Flask web interface for the search engine."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request

from config import TOP_K
from retrieval import structured_search, tfidf_search, bm25_search

app = Flask(__name__)


def compare_models(structured_ids, tfidf_ids, bm25_ids):
    """Compute comparison metrics: overlap and counts."""
    s_set = set(r[0] for r in structured_ids)
    t_set = set(r[0] for r in tfidf_ids)
    b_set = set(r[0] for r in bm25_ids)
    overlap_st = len(s_set & t_set)
    overlap_sb = len(s_set & b_set)
    overlap_tb = len(t_set & b_set)
    overlap_all = len(s_set & t_set & b_set)
    return {
        "structured_count": len(structured_ids),
        "tfidf_count": len(tfidf_ids),
        "bm25_count": len(bm25_ids),
        "overlap_structured_tfidf": overlap_st,
        "overlap_structured_bm25": overlap_sb,
        "overlap_tfidf_bm25": overlap_tb,
        "overlap_all_three": overlap_all,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = (request.form.get("query") or "").strip()
        if not query:
            return render_template("index.html", error="Please enter a search query.")

        structured_results = structured_search(query, top_k=TOP_K)
        tfidf_results = tfidf_search(query, top_k=TOP_K)
        bm25_results = bm25_search(query, top_k=TOP_K)

        metrics = compare_models(structured_results, tfidf_results, bm25_results)

        # Optional: average scores for TF-IDF and BM25 (structured has no scores)
        tfidf_scores = [r[1] for r in tfidf_results]
        bm25_scores = [r[1] for r in bm25_results]
        metrics["avg_score_tfidf"] = sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0
        metrics["avg_score_bm25"] = sum(bm25_scores) / len(bm25_scores) if bm25_scores else 0

        return render_template(
            "results.html",
            query=query,
            structured=structured_results,
            tfidf=tfidf_results,
            bm25=bm25_results,
            metrics=metrics,
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
