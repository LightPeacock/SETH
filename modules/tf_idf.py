# Code for TF-IDF vectorization of the text dataset.
#term frequency-inverse document frequency is a test done to see the importance of the word.

from sklearn.feature_extraction.text import TfidfVectorizer
from train_test import X_train, X_test

vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)