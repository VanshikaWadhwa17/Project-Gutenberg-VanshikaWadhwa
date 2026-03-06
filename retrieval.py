"""Retrieval models: structured (metadata), TF-IDF vector space, BM25."""
import os
import re
import pickle
import csv
from collections import defaultdict

import numpy as np

from config import (
    METADATA_PATH,
    METADATA_PATH_ALT,
    INDEX_DIR,
    INVERTED_INDEX_PATH,
    TFIDF_VECTORS_PATH,
    TFIDF_VOCAB_PATH,
    DOC_LENGTHS_PATH,
    AVG_DOC_LEN_PATH,
    DOC_IDS_PATH,
    IDF_PATH,
    BM25_K1,
    BM25_B,
    TOP_K,
)
from preprocess import preprocess, _normalize
from indexer import load_metadata


def _metadata_path():
    if os.path.isfile(METADATA_PATH):
        return METADATA_PATH
    return METADATA_PATH_ALT


def _load_metadata_rows():
    path = _metadata_path()
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def structured_search(query: str, metadata=None, top_k: int = TOP_K):
    """
    Search metadata fields (title, author, bookshelf). Returns list of (book_id, score).
    Matches when the full query string appears as substring OR when all query terms
    (after tokenizing) appear in the combined metadata (so "Philip K Dick" matches "Dick, Philip K.").
    """
    if metadata is None:
        metadata = _load_metadata_rows()
    q = _normalize(query.lower().strip())
    if not q:
        return []
    # Tokenize like preprocess so "Dornröschen" -> dornroschen and term match works
    q_terms = [t for t in re.sub(r"[^a-z0-9\s]", " ", q).split() if t]
    results = []
    seen = set()
    for row in metadata:
        gid = row.get("gutenberg_id", "").strip()
        if not gid or gid in seen:
            continue
        title = (row.get("title") or "").lower()
        author = (row.get("author") or "").lower()
        bookshelf = (row.get("gutenberg_bookshelf") or "").lower()
        combined = f"{title} {author} {bookshelf}"
        combined_norm = _normalize(combined)
        # Exact phrase match (normalized)
        if q in combined_norm:
            results.append((gid, 1.0))
            seen.add(gid)
            continue
        # All query terms present in metadata (e.g. "Philip K Dick" -> "dick, philip k."; "Dornröschen" -> dornroschen)
        if q_terms and all(term in combined_norm for term in q_terms):
            results.append((gid, 1.0))
            seen.add(gid)
    return results[:top_k]


def _ensure_index_loaded():
    """Load index files once; used by tfidf and bm25."""
    with open(INVERTED_INDEX_PATH, "rb") as f:
        inv = pickle.load(f)
    with open(TFIDF_VECTORS_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    with open(DOC_IDS_PATH, "rb") as f:
        doc_ids = pickle.load(f)
    with open(DOC_LENGTHS_PATH, "rb") as f:
        doc_lengths = pickle.load(f)
    with open(AVG_DOC_LEN_PATH, "rb") as f:
        avg_doc_len = pickle.load(f)
    with open(IDF_PATH, "rb") as f:
        idf = pickle.load(f)
    term_to_idx = {t: i for i, t in enumerate(vocab)}
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    return inv, tfidf_matrix, vocab, term_to_idx, doc_ids, doc_id_to_idx, doc_lengths, avg_doc_len, idf


def tfidf_search(query: str, top_k: int = TOP_K):
    """
    Vector space model with TF-IDF and cosine similarity.
    Returns list of (book_id, score) sorted by score descending.
    """
    try:
        (inv, tfidf_matrix, vocab, term_to_idx, doc_ids, doc_id_to_idx,
         doc_lengths, avg_doc_len, idf) = _ensure_index_loaded()
    except FileNotFoundError:
        return []
    words = preprocess(query)
    if not words:
        return []
    # Build query vector (same vocab, tf-idf weighting)
    N = len(doc_ids)
    q_vec = np.zeros(len(vocab), dtype=np.float64)
    for w in words:
        if w in term_to_idx:
            j = term_to_idx[w]
            df = len(inv.get(w, {}))
            idf_w = np.log((N + 1) / (df + 1)) + 1.0
            q_vec[j] += idf_w
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []
    q_vec = q_vec / q_norm
    scores = np.asarray(tfidf_matrix.dot(q_vec)).flatten()
    order = np.argsort(-scores)
    out = []
    for i in order:
        if scores[i] <= 0:
            break
        out.append((doc_ids[i], float(scores[i])))
        if len(out) >= top_k:
            break
    return out


def bm25_search(query: str, k1: float = BM25_K1, b: float = BM25_B, top_k: int = TOP_K):
    """
    BM25 ranking. Returns list of (book_id, score) sorted by score descending.
    """
    try:
        (inv, _m, vocab, _t2i, doc_ids, doc_id_to_idx,
         doc_lengths, avg_doc_len, idf) = _ensure_index_loaded()
    except FileNotFoundError:
        return []
    words = preprocess(query)
    if not words or avg_doc_len <= 0:
        return []
    scores = defaultdict(float)
    for term in words:
        if term not in inv:
            continue
        idf_t = idf.get(term, 0.0)
        for doc_id, tf in inv[term].items():
            doc_len = doc_lengths.get(doc_id, 1)
            num = tf * (k1 + 1)
            denom = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            scores[doc_id] += idf_t * (num / denom)
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    return [(doc_id, float(score)) for doc_id, score in sorted_docs[:top_k]]


def get_preview(doc_id: str, query: str, max_chars: int = 150):
    """Get a short preview of the book text around a query match (optional)."""
    from indexer import _books_root
    root = _books_root()
    if not root:
        return "", 0
    path = os.path.join(root, f"{doc_id}.txt")
    if not os.path.isfile(path):
        return "", 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return "", 0
    q_lower = query.lower()
    pos = text.lower().find(q_lower)
    if pos == -1:
        # no exact phrase; take first chunk
        preview = text[:max_chars].replace("\n", " ")
        return preview.strip(), 1
    start = max(0, pos - max_chars // 2)
    end = min(len(text), start + max_chars)
    preview = text[start:end].replace("\n", " ")
    line_num = text[:pos].count("\n") + 1
    return preview.strip(), line_num
