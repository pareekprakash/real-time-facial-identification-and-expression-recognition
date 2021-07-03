# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:51:52 2021

@author: prakash pareek
"""

import numpy as np
import cv2
import os 

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as plt_con_mat

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from keras.utils import plot_model

ROW = 32
COL = 32

path = "E:/cnn_age_gender-main/cnn_age_gender-main/UTKFace"
file_names = os.listdir(path)

gender = [i.split('_')[1] for i in file_names]
y_data = np.array([int(i) for i in gender])
y_data = np.expand_dims(y_data, axis = -1)

print(y_data.shape)

X_data = np.array([cv2.resize(cv2.imread(os.path.join(path,file)), (ROW, COL)) for file in file_names])
print(X_data.shape)

y_data = to_categorical(y_data)

X_temp, X_val, y_temp, y_val = train_test_split(X_data, y_data, test_size = 0.1, shuffle = True, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size = 0.1, random_state = 1)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

idxs = [0,1]
x_ticks = ['male', 'female']

plt.bar(idxs, [len([i for i in X_train if np.argmax(i) == 0]), 
                len([i for i in X_train if np.argmax(i) == 1])
               ],
       color = ['b','r'])
plt.xticks(idxs, x_ticks)
plt.title('Training')
plt.show()

plt.bar(idxs, [len([i for i in X_val if np.argmax(i) == 0]), 
                len([i for i in X_val if np.argmax(i) == 1])
               ],
       color = ['b','r'])
plt.xticks(idxs, x_ticks)
plt.title('Validation data')
plt.show()

plt.bar(idxs, [len([i for i in X_val if np.argmax(i) == 0]), 
                len([i for i in X_val if np.argmax(i) == 1])
               ],
       color = ['b','r'])
plt.xticks(idxs, x_ticks)
plt.title('Testing data')
plt.show()

m = 0
f = 0

male_images = []
female_images = []

for idx, label in enumerate(y_train):

    if(m <= 5 and np.argmax(label) == 0):
        male_images.append(idx)
        m += 1
    elif(f <= 5):
        female_images.append(idx)
        f += 1
        
    if(m == 5 and f == 5):
        break
        
fig, ax = plt.subplots(5, 2, figsize = (15, 15))

ax[0, 0].title.set_text("Male")
ax[0, 1].title.set_text("Female")

for i in range(5):
    
    ax[i, 0].imshow(X_train[male_images[i]])
    ax[i, 0].axis('off')
    ax[i, 1].imshow(X_train[female_images[i]])
    ax[i, 1].axis('off')

plt.show()

model = Sequential()

model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = (ROW, COL, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation = 'softmax'))
model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy', 'Precision', 'Recall'])

history = model.fit(X_train, y_train, 
                    validation_data = (X_val, y_val), 
                    epochs = 15,
                    batch_size = 64)

model.save("E:/Face-and-Emotion-Recognition-master/gender_CNN2.h5")

sns.set()
fig = plt.figure(0, (12, 4))

ax = plt.subplot(1, 2, 1)
sns.lineplot(history.epoch, history.history['acc'], label = 'train')
sns.lineplot(history.epoch, history.history['val_acc'], label = 'validation')
plt.title('Accuracy')
plt.tight_layout()

ax = plt.subplot(1, 2, 2)
sns.lineplot(history.epoch, history.history['loss'], label = 'train')
sns.lineplot(history.epoch, history.history['val_loss'], label = 'validation')
plt.title('Loss')
plt.tight_layout()
plt.savefig('epoch_history.png')
plt.show()
'''preds = model.predict_classes(X_test)
y_true = np.argmax(y_test, axis=1)

plt_con_mat(y_true, preds, figsize=(14,14))
plt.show()'''