"""Download required NLTK data (run once after pip install)."""
import nltk
nltk.download("stopwords", quiet=True)
print("NLTK stopwords ready.")
