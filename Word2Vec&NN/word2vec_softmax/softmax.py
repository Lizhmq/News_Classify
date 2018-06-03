# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:56:03 2018

@author: Robbery
"""

import numpy as np
import random
import keras
from keras.utils import np_utils  
from keras.models import Sequential  
from keras.layers import Dense, Activation  


labels=['auto','health','it','learning','sports','stock','travel','yule']

def BuildData():
    trainX = []
    trainY = []
    testX = []
    testY = []
    for i in range(0,8):
        label = labels[i]
        filename = "vec/vec"+label+".txt"
        # with open(filename, errors='ignore', encoding='utf-8') as f:
        with open(filename) as f:
            vec_list = eval(f.read())
            random.shuffle(vec_list)
            size = len(vec_list)
            trainsize = int(size*0.7)
            testsize = size - trainsize
            trainy = np.zeros((8,trainsize))
            testy = np.zeros((8,testsize))
            for j in range(0,trainsize):
                trainy[i][j] = 1.0
            for j in range(0,testsize):
                testy[i][j] = 1.0
            trainX.append(np.transpose(np.array(vec_list[:trainsize])))
            testX.append(np.transpose(np.array(vec_list[trainsize:])))
            trainY.append(trainy)
            testY.append(testy)
    trainX = np.transpose(np.hstack(tuple(trainX)))
    trainY = np.transpose(np.hstack(tuple(trainY)))
    testX = np.transpose(np.hstack(tuple(testX)))
    testY = np.transpose(np.hstack(tuple(testY)))
    return trainX,trainY,testX,testY

trainX,trainY,testX,testY = BuildData()

# model = Sequential()
# model.add(Dense(8,input_dim = 400,kernel_initializer='he_normal'))
# model.add(Activation('softmax'))
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
# model.fit(trainX , trainY, epochs=10, batch_size=50, verbose=1)
model = keras.models.load_model('model_softmax')
loss, accuracy = model.evaluate(testX, testY)
print('Test loss:', loss)
print('Accuracy:', accuracy)
# model.save('model_softmax')


