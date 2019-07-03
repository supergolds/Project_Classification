# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:10:42 2019

@author: DSP
"""

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

pickle_in3 = open("C:\\Project\\test_x.pickle", "rb")
test_x = pickle.load(pickle_in3)

pickle_in4 = open("C:\\Project\\test_y.pickle", "rb")
test_y = pickle.load(pickle_in4)

pickle_in = open("C:\\Project\\train_x.pickle", "rb")
train_x = pickle.load(pickle_in)
train_x[6] = np.zeros((128, 72))

train_x[92] = np.zeros((128, 72))

train_x[161] = np.zeros((128, 72))

train_x[210] = np.zeros((128, 72))

train_x[709] = np.zeros((128, 72))

train_x[783] = np.zeros((128, 72))

train_x[794] = np.zeros((128, 72))


train_x = np.array(train_x)
test_x = np.array(test_x)

pickle_in2 = open("C:\\Project\\train_y.pickle", "rb")
train_y = pickle.load(pickle_in2)

train_x = train_x.reshape(800, 128, 72, 1)
test_x = test_x.reshape(281, 128, 72, 1)

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = train_x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(564))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer = "adam", metrics=["accuracy"])

model.fit(train_x, train_y, batch_size = 100, epochs = 20, validation_split=0.2)
print(model.summary())

loss_and_metrics = model.evaluate(test_x, test_y, batch_size = 20)
print(loss_and_metrics)