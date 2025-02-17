# Import the csv file and kept as a df
import pandas as pd 
import re

df = pd.read_csv('../datasets/preprocessed_emotions.csv')

#preprocess data

def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '',text) # removes non alphabetic characters
    text = text.lower() # convert to lowercase
    text = re.sub(r'\s+','',text).strip() # remove extra spaces
    return text

df['sentence'] = df['sentence'].fillna('')
# Clean dataset
df = df.dropna(subset=['sentence', 'emotion'])
df = df.drop_duplicates()
# print("Unique emotions:", df['emotion'].unique())
# print("Unique emotion labels:", df['emotion_label'].unique())

df['sentence'] = df['sentence'].apply(preprocess_text)

df.to_csv('../datasets/preprocessed_emotions.csv', index=False)

emotion_mapping = {
    emotion: idx for idx, emotion in enumerate(df['emotion'].unique())
}
df['emotion_label'] = df['emotion'].map(emotion_mapping)
# print("Emotion Mapping:", emotion_mapping)