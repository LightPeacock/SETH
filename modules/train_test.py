# Split data

from sklearn.model_selection import train_test_split
from csv_import import X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
