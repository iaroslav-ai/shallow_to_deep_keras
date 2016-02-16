'''
Compares shallow nets to deep ones initialized randomly and with supervised init
'''
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

from datasets import abalone_dataset,chess_dataset

#Tr, Val, Ts = abalone_dataset([0.5,0.25,0.25])
Tr, Val, Ts = chess_dataset([0.5,0.25,0.25])


model = Sequential()

# hidden layer

for param in [5,10,20,30,40]:
    
    width = 10;
    depth = param;
    
    for i in range(depth):
        
        if i == 0:
            model.add(Dense(width, input_dim=Tr["InputSize"]));
        else:
            model.add(Dense(width));
            
        model.add(Activation('relu'))
        #model.add(Dropout(0.25))
    
    # output layer
    model.add(Dense(Tr["OutputSize"]));
    
    model.compile(loss='mean_absolute_error',
                  optimizer='adam')
    
    tmp = model.fit(Tr["X"], Tr["Y"], batch_size=512, nb_epoch=1024,
              validation_data=(Val["X"], Val["Y"]))
    loss = tmp.history["val_loss"][-1];
    
    with open("results.txt", "a") as myfile:
        myfile.write("Loss/depth: " + str(loss) + " / " + str(depth) + "\n")