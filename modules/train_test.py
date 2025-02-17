#Code for training and testing the dataset


from sklearn.model_selection import train_test_split
from csv_import import df


X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['emotion_label'], test_size=0.2, random_state=42)

# print(f"Training samples: {X_train.shape[0]}")
# print(f"Testing samples: {X_test.shape[0]}")

# print(df['emotion'].value_counts())  # Check class distribution
# print(df.isnull().sum())              # Check for NaNs again
# print(df.head(10))                    # Check text after preprocessing
