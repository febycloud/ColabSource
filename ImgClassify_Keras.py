# Setting up Colab
#https://mp.weixin.qq.com/s/e2L9nba3NQiRn8QDTjjnDA

!pip install PyDrive

import os

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials

auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)

# Replace the id and filename in the below codes

download = drive.CreateFile({'id': '1ZCzHDAfwgLdQke_GNnHp_4OheRRtNPs-'})

download.GetContentFile('Train_UQcUa52.zip')

!unzip Train_UQcUa52.zip

# Importing libraries

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm

train = pd.read_csv('train.csv')

# Reading the training images

train_image = []

for i in tqdm(range(train.shape[0])):

    img = image.load_img('Images/train/'+train['filename'][i], target_size=(28,28,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)

# Creating the target variable

y=train['label'].values

y = to_categorical(y)

# Creating validation set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Define the model structure

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

# Compile the model

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Training the model

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

download = drive.CreateFile({'id': '1zHJR6yiI06ao-UAh_LXZQRIOzBO3sNDq'})

download.GetContentFile('Test_fCbTej3.csv')

test_file = pd.read_csv('Test_fCbTej3.csv')

test_image = []

for i in tqdm(range(test_file.shape[0])):

    img = image.load_img('Images/test/'+test_file['filename'][i], target_size=(28,28,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)

test = np.array(test_image)

prediction = model.predict_classes(test)

download = drive.CreateFile({'id': '1nRz5bD7ReGrdinpdFcHVIEyjqtPGPyHx'})

download.GetContentFile('Sample_Submission_lxuyBuB.csv')

sample = pd.read_csv('Sample_Submission_lxuyBuB.csv')

sample['filename'] = test_file['filename']

sample['label'] = prediction

sample.to_csv('sample.csv', header=True, index=False)