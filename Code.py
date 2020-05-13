# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:32:40 2019

@author: DSP
"""
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Reshape, Lambda
from tensorflow.python.keras.layers import Input, GRU, Bidirectional, BatchNormalization, Conv1D, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from tensorflow.python.keras import backend as K


n_frame = 600
n_mel = 128


data = []
label = []

pic1 = open(".\Data\\other_data(19.7.29).p", "rb")
data = pickle.load(pic1)
pic1.close()

pic2 = open(".\\Data\\other_label(19.7.29).p", "rb")
label = pickle.load(pic2)
pic2.close()

x_train, y_train, x_label, y_label = train_test_split(data, label, test_size = 0.1, shuffle = True)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.transpose(x_train, (0, 2, 1))
y_train = np.transpose(y_train, (0, 2, 1))

x_label = np.array(x_label)
y_label = np.array(y_label)

x_train = x_train.reshape(x_train.shape + (1,))
y_train = y_train.reshape(y_train.shape + (1,))
x_label = x_label.reshape(x_label.shape + (1,))
y_label = y_label.reshape(y_label.shape + (1,))


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



print("?쒖옉")

def CRNN(input_shape):
    Input_Tr = Input(input_shape, dtype = 'float', name = 'Input_Tr')
    
    conv_layer1 = Conv2D(32, kernel_size = 3, strides = 1, padding = 'SAME')(Input_Tr)
    batch_layer1 = BatchNormalization(axis=-1)(conv_layer1)
    conv_layer1_out = Activation('relu')(batch_layer1)
    
    pooling_layer1 = MaxPooling2D((1, 4))(conv_layer1_out)
    dropout_layer1 = Dropout(0.5)(pooling_layer1)
    
    conv_layer2 = Conv2D(64, kernel_size = 3, strides = 1, padding = 'SAME')(dropout_layer1)
    batch_layer2 = BatchNormalization(axis=-1)(conv_layer2)
    conv_layer2_out = Activation('relu')(batch_layer2)
    
    pooling_layer2 = MaxPooling2D((1, 4))(conv_layer2_out)
    dropout_layer2 = Dropout(0.5)(pooling_layer2)
    
    print(dropout_layer2.shape)
    
    reshape_layer3 = Reshape((600, 64*int(round(n_mel/4/4))))(dropout_layer2)
    print(reshape_layer3.shape)
    bidir_layer3 = Bidirectional(GRU(64, return_sequences = True, activation = 'tanh'))(reshape_layer3)
    output = TimeDistributed(Dense(1, activation = 'sigmoid'))(bidir_layer3)
    
    model = Model(inputs = [Input_Tr], outputs = [output])
    return model


model = CRNN(input_shape=(n_frame, n_mel, 1))

adam = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999,decay=0.0)
model.compile(loss="binary_crossentropy", optimizer = adam, metrics=['accuracy', f1_m, precision_m, recall_m])
hist = model.fit(x_train, x_label, epochs = 30, batch_size = 30, validation_split = 0.1)

model.summary()

loss, acc, f1_score, precision, recall = model.evaluate(y_train, y_label, batch_size = 15)
print("evaluation 寃곌낵")
print("loss : {:.3f}\n, acc : {:.3f}\n, f1_score : {:.3f}\n, precision : {:.3f}\n, recall : {:.3f}\n".format(loss, acc, f1_score, precision, recall))



plt.plot(hist.history['acc'], label='train')
plt.plot(hist.history['val_acc'], label='val')
plt.legend()
plt.show()


model.save('.\\Data\\Keras_Model\\(19.08.09).h5')

