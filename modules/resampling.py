import pandas as pd
import numpy
from sklearn.utils import resample

df = pd.read_csv('../datasets/merged_emotions.csv')
df.dropna(subset=['sentence','emotion'],inplace=True)
df['emotion'] = df['emotion'].replace('guit','guilty')
label_counts = df['emotion'].value_counts()
print("Original label distribution:\n",label_counts)

MAX_SAMPLES = 12000
MIN_SAMPLES = 5000

def get_target_samples(count,min_count,max_count):
    return int(((count-min_count)/(max_count - min_count)) * (MAX_SAMPLES - MIN_SAMPLES) + MIN_SAMPLES )

min_count = label_counts.min()
max_count = label_counts.max()

target_counts = {label:get_target_samples(count,min_count,max_count) for label,count in label_counts.items()}
print("Target label distribution:\n",target_counts)

resamples_df = []
for emotions, target_Size in target_counts.items():
    df_emotion = df[df['emotion'] == emotions]

    if len(df_emotion) > target_Size:
        df_emotion = resample(df_emotion, replace=False, n_samples=target_Size, random_state=42)
    elif len(df_emotion) < target_Size:
        df_emotion = resample(df_emotion, replace=True, n_samples=target_Size, random_state=42)
    
    resamples_df.append(df_emotion)
df_resampled = pd.concat(resamples_df)
print("Resampled label distribution:\n",df_resampled['emotion'].value_counts())

df_resampled.to_csv('../datasets/resampled_emotions.csv',index=False)