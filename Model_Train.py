import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import itertools
import librosa
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.utils import *




def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)


### READ AND CONCAT DATA ###

father_path = os.getcwd()

label = pd.read_csv(father_path + r"\Assets\Res\label_train.csv")
label.replace({'outer': 0, 'roller': 1, 'inner': 2, 'normal': 3}, inplace=True)
# label.columns = ['','','','','']
print("___Label Info___")
print(label.shape)
print(label.info)

df = pd.read_csv(father_path + r"\Assets\Res\data_train.csv", header=None).dropna().T
print(df.shape)
print(df.info)

### PLOT ORIGINAL VIBRATE ACCELERATE ###

df.iloc[2, :].plot(figsize=(8, 5), title='Original Vibrate accelerate')
plt.ylabel('Vibrate accelerate')
plt.xlabel('Time')

### APPLY CLIPPING AND FIRST ORDER DIFFERECE ###

df = df.values
df = np.clip(np.diff(df, axis=1), -15, 15)
print(df.shape)

### PLOT STANDARDIZED DATA ###

plt.figure(figsize=(9, 6))
plt.plot(df[2])
plt.title('Standardized Vibrate accelerate')
plt.ylabel('First Difference');
plt.xlabel('Time')
np.set_printoptions(False)

### LABEL DISTRIBUTION ###

label = label['Type']
print(label.value_counts())  # Series or DataFrame?

### LABEL ENCODING ###

diz_label = {0: 'outer', 1: 'roller', 2: 'inner', 3: 'normal'}
y = to_categorical(label)

### APPLY MFCC TRANSFORMATION FOR EACH SIGNAL ###

df_mfcc = []

for i, sample in enumerate(tqdm.tqdm(df)):
    sample_mfcc = librosa.feature.mfcc(np.asfortranarray(sample), sr=40000)
    df_mfcc.append(sample_mfcc)

df_spectre = np.asarray(df_mfcc)
df_spectre = df_spectre.transpose(0, 2, 1)
print(df_spectre.shape)

### PLOT MFCC FOR A SINGLE SIGNAL ###

plt.figure(figsize=(9, 6))

plt.plot(df_spectre[2])
plt.legend(['mfcc_' + str(i) for i in range(df_spectre.shape[-1])],
           loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('MFCC features')
plt.ylabel('Amplitudes');
plt.xlabel('Time')

np.set_printoptions(False)

### TRAIN TEST SPLIT ###

X_train, X_test, y_train, y_test = train_test_split(df_spectre, y, random_state=42, test_size=0.2)


### DEFINE RESNET CNN ###

def get_model(data):
    def residual_block(init, hidden_dim):
        init = Conv1D(hidden_dim, 3, activation='relu', padding="same")(init)

        x = Conv1D(hidden_dim, 3, activation='relu', padding="same")(init)
        x = Conv1D(hidden_dim, 3, activation='relu', padding="same")(x)
        x = Conv1D(hidden_dim, 3, activation='relu', padding="same")(x)
        skip = Add()([x, init])

        return skip

    inp = Input(shape=(data.shape[1], data.shape[2]))

    x = residual_block(inp, 256)
    x = residual_block(x, 128)
    x = residual_block(x, 32)
    x = GlobalMaxPool1D()(x)

    out = Dense(len(diz_label), activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


tf.random.set_seed(33)
os.environ['PYTHONHASHSEED'] = str(33)
np.random.seed(33)
random.seed(33)

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(),
    config=session_conf
)
tf.compat.v1.keras.backend.set_session(sess)

es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=15)

model = get_model(X_train)
model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=2, validation_split=0.1, callbacks=[es])

### Save ACTION ###
model_path = father_path + r"\Assets\Res\dl_model.h5"
model.save(model_path)

### GET PREDICTED CLASS ON TEST ###

pred_test = np.argmax(model.predict(X_test), axis=1)
print(classification_report([diz_label[np.argmax(label)] for label in y_test],
                            [diz_label[label] for label in pred_test]))
cnf_matrix = confusion_matrix([diz_label[np.argmax(label)] for label in y_test],
                              [diz_label[label] for label in pred_test])

plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=list(diz_label.values()))
plt.show()

### DUMMY CLASSIFIER ACCURACY (always stable) ###

sum([diz_label[np.argmax(label)] == 'stable' for label in y_test]) / len(pred_test)
