import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# Load dataset
df = pd.read_csv("preprocessed_emotions.csv")

# Remove rare labels (those appearing <2 times)
df = df[df.groupby("emotion")["emotion"].transform("count") > 1]

X = df["sentence"]
y = df["emotion"]

# Dynamically get unique labels from dataset (fix for unseen labels issue)
unique_labels = sorted(df["emotion"].unique())  # Ensures all labels are considered

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)  # Fit with actual dataset labels
y = label_encoder.transform(y)

# Split data (after filtering out rare classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train Decision Tree model
model = DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42)
model.fit(X_train_vec, y_train)

# Evaluate accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Save model & vectorizer
with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Debugging: Print label mappings
label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
reverse_mapping = {idx: label for label, idx in label_mapping.items()}
print("Emotion Mapping:", label_mapping)
print("Reverse Mapping:", reverse_mapping)

# Function to predict emotion
def predict_emotion(text):
    text_vec = vectorizer.transform([text])
    predicted_label = model.predict(text_vec)[0]
    return reverse_mapping[predicted_label]

# Interactive Testing
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exit emotion detection")
        break

    predicted_emotion = predict_emotion(user_input)
    print(f"Mapped Emotion: {predicted_emotion}")
