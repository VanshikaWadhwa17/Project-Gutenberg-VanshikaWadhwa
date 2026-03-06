"""Build inverted index, TF-IDF vectors, and BM25 structures from the book collection."""
import os
import pickle
import csv
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, diags

from config import (
    BASE_DIR,
    BOOKS_DIR,
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
)
from preprocess import preprocess


def _metadata_path():
    """Return path to metadata CSV (data/ or project root)."""
    if os.path.isfile(METADATA_PATH):
        return METADATA_PATH
    return METADATA_PATH_ALT


def load_metadata():
    """Load metadata CSV; handle quoted multiline fields."""
    path = _metadata_path()
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _books_root():
    """Return the directory that actually contains .txt files: BOOKS_DIR or BOOKS_DIR/books (nested)."""
    if not os.path.isdir(BOOKS_DIR):
        return None
    # Check for .txt directly in data/books/
    if any(f.endswith(".txt") for f in os.listdir(BOOKS_DIR)):
        return BOOKS_DIR
    nested = os.path.join(BOOKS_DIR, "books")
    if os.path.isdir(nested) and any(f.endswith(".txt") for f in os.listdir(nested)):
        return nested
    return BOOKS_DIR  # default


def get_book_ids_with_text():
    """Return set of gutenberg_id that have has_text true and a .txt in books/ (or books/books/)."""
    meta = load_metadata()
    with_text = {int(r["gutenberg_id"]) for r in meta if r.get("has_text", "").lower() == "true"}
    root = _books_root()
    if not root:
        return with_text
    files = set()
    for name in os.listdir(root):
        if name.endswith(".txt"):
            try:
                files.add(int(name[:-4]))
            except ValueError:
                pass
    return with_text & files if files else with_text


def _metadata_by_id():
    """Return dict: gutenberg_id -> (title, author) using first row per id."""
    meta = load_metadata()
    by_id = {}
    for row in meta:
        gid = row.get("gutenberg_id", "").strip()
        if not gid:
            continue
        if gid not in by_id:
            title = (row.get("title") or "").strip().replace("\n", " ")
            author = (row.get("author") or "").strip()
            by_id[gid] = (title, author)
    return by_id


def load_documents(max_docs=None, include_metadata_in_text=True):
    """
    Load (doc_id, text) for each book.
    If include_metadata_in_text, prepend title and author so TF-IDF/BM25 can match author/title queries.
    """
    book_ids = get_book_ids_with_text()
    if max_docs:
        book_ids = set(list(book_ids)[:max_docs])
    meta_by_id = _metadata_by_id() if include_metadata_in_text else {}
    root = _books_root()
    if not root:
        return []
    documents = []
    for gid in book_ids:
        path = os.path.join(root, f"{gid}.txt")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            continue
        gid_str = str(gid)
        if include_metadata_in_text and gid_str in meta_by_id:
            title, author = meta_by_id[gid_str]
            text = f"{title} {author} {text}"
        documents.append((gid_str, text))
    return documents


def build_inverted_index(documents):
    """Build inverted index: term -> {doc_id: frequency}."""
    index = defaultdict(lambda: defaultdict(int))
    for doc_id, text in documents:
        words = preprocess(text)
        for w in words:
            index[w][doc_id] += 1
    return dict(index)


def build_tfidf_and_doc_lengths(documents, inverted_index):
    """
    Build TF-IDF matrix (sparse via term->doc_id->tfidf), vocabulary, doc lengths for BM25,
    and IDF vector. All computed from the inverted index.
    """
    N = len(documents)
    doc_ids = sorted([d[0] for d in documents])
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # Vocabulary: sorted list of terms
    vocab = sorted(inverted_index.keys())
    term_to_idx = {t: i for i, t in enumerate(vocab)}

    # Document lengths (number of terms after preprocessing)
    doc_lengths = {}
    for doc_id, text in documents:
        words = preprocess(text)
        doc_lengths[doc_id] = len(words)
    avg_doc_len = np.mean(list(doc_lengths.values())) if doc_lengths else 0.0

    # IDF: idf[t] = log((N+1)/(df_t+1)) + 1
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = np.log((N + 1) / (df + 1)) + 1.0

    # TF-IDF matrix (sparse, docs x terms) for cosine similarity — avoids 800+ GiB dense array
    n_terms = len(vocab)
    n_docs = len(doc_ids)
    rows, cols, data = [], [], []
    for term, postings in inverted_index.items():
        j = term_to_idx[term]
        idf_t = idf[term]
        for doc_id, tf in postings.items():
            i = doc_id_to_idx.get(doc_id)
            if i is not None:
                rows.append(i)
                cols.append(j)
                data.append(tf * idf_t)
    matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float64)

    # Normalize rows for cosine similarity (unit vectors)
    row_norms = np.array(matrix.multiply(matrix).sum(axis=1)).flatten() ** 0.5
    row_norms[row_norms == 0] = 1.0
    matrix = diags(1.0 / row_norms) @ matrix

    return {
        "tfidf_matrix": matrix,
        "vocab": vocab,
        "term_to_idx": term_to_idx,
        "doc_ids": doc_ids,
        "doc_id_to_idx": doc_id_to_idx,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def save_index(inverted_index, tfidf_data):
    """Save all index structures to INDEX_DIR."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INVERTED_INDEX_PATH, "wb") as f:
        pickle.dump(inverted_index, f)
    with open(TFIDF_VECTORS_PATH, "wb") as f:
        pickle.dump(tfidf_data["tfidf_matrix"], f)
    with open(TFIDF_VOCAB_PATH, "wb") as f:
        pickle.dump(tfidf_data["vocab"], f)
    with open(DOC_IDS_PATH, "wb") as f:
        pickle.dump(tfidf_data["doc_ids"], f)
    with open(DOC_LENGTHS_PATH, "wb") as f:
        pickle.dump(tfidf_data["doc_lengths"], f)
    with open(AVG_DOC_LEN_PATH, "wb") as f:
        pickle.dump(tfidf_data["avg_doc_len"], f)
    with open(IDF_PATH, "wb") as f:
        pickle.dump(tfidf_data["idf"], f)
    print("Index saved to", INDEX_DIR)


def load_index():
    """Load inverted index and TF-IDF structures."""
    with open(INVERTED_INDEX_PATH, "rb") as f:
        inverted_index = pickle.load(f)
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
    return inverted_index, {
        "tfidf_matrix": tfidf_matrix,
        "vocab": vocab,
        "term_to_idx": term_to_idx,
        "doc_ids": doc_ids,
        "doc_id_to_idx": doc_id_to_idx,
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def run_indexing(max_docs=None):
    """Load documents, build index, save. Call from command line or app."""
    print("Loading documents...")
    documents = load_documents(max_docs=max_docs)
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        print("No documents found. Place .txt files in data/books/ and ensure metadata.csv lists them.")
        return
    print("Building inverted index...")
    inverted_index = build_inverted_index(documents)
    print("Building TF-IDF and doc lengths...")
    tfidf_data = build_tfidf_and_doc_lengths(documents, inverted_index)
    save_index(inverted_index, tfidf_data)
    print("Done.")


if __name__ == "__main__":
    run_indexing()
