# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:45:24 2020

@author: DSP
"""

from tensorflow.keras.layers import Concatenate, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Multiply, BatchNormalization, Activation, Input
from tensorflow.keras.layers import Reshape, Bidirectional, GRU, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

frames = 240
bands = 64

def GCRNN(input_shape):
    Input_First = Input(input_shape, dtype = 'float', name = 'First')
    Input_Second = Input(input_shape, dtype = 'float', name = 'Second')
    
    #Input First
    conv_layerS1 = Conv2D(64, kernel_size = 3, strides = 1, padding = 'SAME')(Input_First)
    batch_layerS1 = BatchNormalization(axis=-1)(conv_layerS1)
    convoutS1 = Activation('linear')(batch_layerS1)
    
    #Input Second
    conv_layerS2 = Conv2D(64, kernel_size = 3, strides = 1, padding = 'SAME')(Input_Second)
    batch_layerS2 = BatchNormalization(axis=-1)(conv_layerS2)
    convoutS2 = Activation('sigmoid')(batch_layerS2)
    
    conv_layer = Multiply([convoutS1, convoutS2])
    
    pooling_layer = MaxPooling2D((1, 4))(conv_layer)
    dropout_layer = Dropout(0.5)(pooling_layer)
    
    
    reshape_layer = Reshape((600, 64*int(round(bands/4/4))))(dropout_layer)
    bidir_layer = Bidirectional(GRU(64, return_sequences = True, activation = 'tanh'))(reshape_layer)
    
    output = TimeDistributed(Dense(1, activation = 'sigmoid'))(bidir_layer)
    
    
    model = Model(inputs = [Input_Tr], outputs = [output])
    
    
    return model


model = GCRNN(input_shape=(frames, bands, 1))

adam = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999,decay=0.0)
model.compile(loss="binary_crossentropy", optimizer = adam, metrics=['accuracy', f1_m, precision_m, recall_m])
hist = model.fit(x_train, x_label, epochs = 30, batch_size = 30, validation_split = 0.1)

model.summary()