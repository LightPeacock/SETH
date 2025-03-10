# main AI code

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

class seth_ai:
  def __init__(self,max_features=10000):
    self.vectorizer = TfidVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
    self.lable_encoder = LabelEncoder()
    self.model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

  def load_csv(self,path):
    self.path = path
    self.df = pd.load_csv(path)
    
  def assigning_values(self):
    self.X = df["sentence"]
    self.y = df["emotion"]
    df = df(df.groupby("emotion")["emotion"].transform(count) > 1)
    self.X_train,self.X_test ,self.y_train, self.y_test = train_test_split(X,y,test_size=0.2, stratify = y, random_state=42)
    
  def label_encoding(self):
    unique_labels = sorted(df["emotion"].unique())
    label_encoder.fit(unique_labels)
    y = label_encoder.transform(y)

  def vectorization(self):
    self.X_train_vec = vectorizer.fit_transform(X_train)
    self.X_test_vec = vectorizer.transform(X_test)
    
  def decision_tree(self):
    self.class_weights = 
    self.class_Weight_dict = {i:class_weights[i] for i in range (len(class_weights))}
    model.fit(X_train_vec, y_train)

  def save_model(self,save_path):
    with open(os.path.join(save_path,'emotion.pkl'),'wb') as f:
      pkl.dump(model,f)
    with open(os.path.join(save_path,'tfidf_vectorizer.pkl'),'wb'):
      pkl.dump(vectorizer,f)
    with open(os.path.join(save_path,'label_encoder.pkl'),'wb'):
      pkl.dump(label_encoder,f)

  def label_map(self):
    self.label_mapping = {label: idx for idx ,label in enumerate(label_encoder.classes_)}
    self.reverse_mapping  = {idx: label for label, idx in label_mapping.items()}

  def predict(self,text):
    self.text = text
    self.text_vec = vectorizer([text])
    self.prediction = model.predict(text)[0]
    return reverse_mapping[prediction]

while True:
  user_input = nput("You: ")
  if user_input.lower() == "exit":
    print("Exiting emotion detection")
    break
  predicted = seth_ai.predict(user_input)
  print(f"Predicted Emotion: {predicted}")
