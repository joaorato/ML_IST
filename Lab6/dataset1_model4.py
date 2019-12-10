#MLP

import keras.layers as layer
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from keras import Sequential
from keras import optimizers as optimizer
from keras.callbacks import EarlyStopping

#LOADING AND TREATING DATA
x_train = np.load('dataset1_xtrain.npy')
y_train = np.load('dataset1_ytrain.npy')

x_test = np.load('dataset1_xtest.npy')
y_test = np.load('dataset1_ytest.npy')

train_instances = x_train.shape[0]
test_instances = x_test.shape[0]
features = x_train.shape[1]
class_dimensions = y_train.shape[1]

y_train_matrix = keras.utils.to_categorical(y_train, num_classes = 2, dtype = 'int32')
y_test_matrix = keras.utils.to_categorical(y_test, num_classes = 2, dtype = 'int32')

X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(x_train, y_train_matrix, test_size=0.30, random_state=42)

print(X_train.shape)

model = Sequential()
model.add(layer.Dense(64, input_dim = features, activation='relu'))
model.add(layer.Dense(128, activation='relu'))
model.add(layer.Dense(2, activation='softmax'))

print(model.summary())

earlyStopping = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
model.compile(loss='categorical_crossentropy', optimizer=optimizer.Adam(lr = 0.0001, clipnorm = 1))
history = model.fit(X_train, Y_train, batch_size=100, epochs=400, callbacks=earlyStopping, validation_data=(X_val, Y_val))
#history = model.fit(X_train, Y_train, batch_size=300, epochs=400, validation_data=(X_val, Y_val)) #without early stopping

#print(history.history.keys())
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_predicted = model.predict(x_test)
print(y_predicted.shape)
for i in range(test_instances):
    index = y_predicted[i].argmax()
    y_predicted[i] = [0,0]
    y_predicted[i,index] = 1

accuracy = sklearn.metrics.accuracy_score(y_test_matrix, y_predicted)
print("accuracy = ", accuracy)
conf_matrix = sklearn.metrics.confusion_matrix(y_test_matrix.argmax(axis=1), y_predicted.argmax(axis=1))
print("Confusion matrix : \n", conf_matrix)
