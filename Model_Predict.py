print("Predict Start!")

import pandas as pd
import numpy as np
import tqdm
import tensorflow as tf

import librosa
import os

### READ AND CONCAT DATA ###

father_path = os.getcwd()
df = pd.read_csv(father_path + r"\Assets\Res\HistoryDataFromUnity.csv", header=None).dropna().T

print(df.shape)
print(df.info)

### APPLY CLIPPING AND FIRST ORDER DIFFERECE ###

df = df.values
df = np.clip(np.diff(df, axis=1), -15, 15)
print(df.shape)

### APPLY MFCC TRANSFORMATION FOR EACH SIGNAL ###
df_mfcc = []

for i, sample in enumerate(tqdm.tqdm(df)):
    sample_mfcc = librosa.feature.mfcc(np.asfortranarray(sample), sr=40000)
    df_mfcc.append(sample_mfcc)

df_spectre = np.asarray(df_mfcc)
df_spectre = df_spectre.transpose(0, 2, 1)
print(df_spectre.shape)

### Use ACTION ###
model_path = father_path + r"\Assets\Res\dl_model.h5"
model = tf.keras.models.load_model(model_path)
print(model.summary())
pred_test = np.argmax(model.predict(df_spectre), axis=1)
print(pred_test)

result = pd.DataFrame(pred_test)
result.to_csv(father_path + r"\Assets\Res\ProcessedDataFromPython.csv", na_rep='NULL', index=False, header=False)
