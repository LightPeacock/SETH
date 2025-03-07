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
