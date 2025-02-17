#Code for performing logistic regression on the dataset

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 
from train_test import X_train, X_test, y_train, y_test
from tf_idf import X_train_vec, X_test_vec,tfidf

model = XGBClassifier(
    max_depth = 6,
    n_estimators=100,
    learning_rate = 0.2,
    subsample = 0.7,
    colsample_bytree = 0.7,
    random_state = 42,
    tree_method='hist',
    n_jobs = -1,
    eval_metric = 'mlogloss'
)

#training the decision tree
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


print(classification_report(y_test, y_pred))

try: 
    joblib.dump(model, '../Model/emotion_training_model.pkl') # saving model for future use
    joblib.dump(tfidf, '../Model/tfidf_vectorizer.pkl') # saving vectorizer for future use
    print("\nVEctoerizer saved as 'tfidf_vectorizer.pkl' in the Model folder") # confirmation message for saving
    print("\nModel Saved as 'emotion_training_model.pkl' in the Model folder") # confirmation message for saving

except Exception as e:
    print("\nModel not saved") # error message for saving
    print("Exception:\n", e) # display error message