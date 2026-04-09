from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_vectorizer(use_tfidf=True, max_features=5000):
    if use_tfidf:
        return TfidfVectorizer(stop_words="english", max_features=max_features)
    return CountVectorizer(stop_words="english", max_features=max_features)
