# Import the csv file and kept as a df
# Load dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("../datasets/resampled_emotions.csv")

X = df["sentence"]
y = df["emotion"]

# Dynamically get unique labels from dataset (fix for unseen labels issue)
unique_labels = sorted(df["emotion"].unique())  # Ensures all labels are considered

# Label encoding
label_encoder = LabelEncoder()
label_encoder.fit(unique_labels)  # Fit with actual dataset labels
y = label_encoder.transform(y)
