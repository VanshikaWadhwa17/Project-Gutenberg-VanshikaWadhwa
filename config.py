"""Configuration paths and constants for the search engine."""
import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
BOOKS_DIR = os.path.join(DATA_DIR, "books")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
# Fallback if metadata is in project root (e.g. before moving into data/)
METADATA_PATH_ALT = os.path.join(BASE_DIR, "metadata.csv")

# Index paths
INDEX_DIR = os.path.join(BASE_DIR, "index")
INVERTED_INDEX_PATH = os.path.join(INDEX_DIR, "inverted_index.pkl")
TFIDF_VECTORS_PATH = os.path.join(INDEX_DIR, "tfidf_vectors.pkl")
TFIDF_VOCAB_PATH = os.path.join(INDEX_DIR, "tfidf_vocab.pkl")
DOC_LENGTHS_PATH = os.path.join(INDEX_DIR, "doc_lengths.pkl")
AVG_DOC_LEN_PATH = os.path.join(INDEX_DIR, "avg_doc_len.pkl")
DOC_IDS_PATH = os.path.join(INDEX_DIR, "doc_ids.pkl")
IDF_PATH = os.path.join(INDEX_DIR, "idf.pkl")

# Queries output
QUERIES_DIR = os.path.join(BASE_DIR, "queries")

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# Top-K results to return
TOP_K = 100
