# Code for TF-IDF vectorization of the text dataset.
#term frequency-inverse document frequency is a test done to see the importance of the word.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from csv_import import df
from train_test import X_test,X_train,y_test,y_train

df = df.dropna(subset=['sentence', 'emotion'])# dropping waste sentences to reduce the size of the dataset

tfidf = TfidfVectorizer(
    max_features=10000,           # Limit feature set to reduce noise
    ngram_range=(1, 3),          # Capture bigrams for more context
    stop_words='english',
    min_df = 2,
    max_df = 0.75       
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"X_train_vec shape: {X_train_vec.shape}")
print(f"y_train shape: {y_train.shape}")
