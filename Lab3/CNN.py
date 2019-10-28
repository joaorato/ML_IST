import keras.layers as layer
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from keras import Sequential
from keras import optimizers as optimizer
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(layer.Conv2D(filters = 16, kernel_size = 3, input_shape=(28,28,1), activation='relu'))
model.add(layer.MaxPooling2D(pool_size = (2, 2)))
model.add(layer.Conv2D(filters = 32, kernel_size = 3, activation='relu'))
model.add(layer.MaxPooling2D(pool_size = (2, 2)))
model.add(layer.Flatten())
model.add(layer.Dense(64, activation='relu'))
model.add(layer.Dense(10, activation='softmax'))

print(model.summary())

