from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
from csv_import import X, y, label_encoder
from tf_idf import X_train_vec, X_test_vec, vectorizer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
from train_test import X_train, X_test, y_train, y_test


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

save_path = "/Users/clnarayanan/Documents/MyWork/SETH/Model"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(os.path.join(save_path,"emotion_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(save_path,"tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(save_path,"label_encoder.pkl"), "wb") as f:
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
