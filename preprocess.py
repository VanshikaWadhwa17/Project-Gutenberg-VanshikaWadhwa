"""Text preprocessing for indexing and querying."""
import re
import unicodedata
from nltk.corpus import stopwords

try:
    _stop_words = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    _stop_words = set(stopwords.words("english"))


def _normalize(text: str) -> str:
    """Normalize unicode (e.g. ö->o) so 'Dornröschen' matches in index and query."""
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def preprocess(text: str, remove_stopwords: bool = False) -> list[str]:
    """
    Preprocess text for indexing and querying.
    Steps: normalize unicode, lowercase, remove punctuation, tokenize.
    Set remove_stopwords=True to drop stopwords (default False so queries like
    "to be or not to be" keep terms and match).
    """
    if not text or not isinstance(text, str):
        return []
    text = _normalize(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [w for w in words if len(w) > 0]
    if remove_stopwords:
        words = [w for w in words if w not in _stop_words]
    return words
