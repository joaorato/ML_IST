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
x_train = np.load('mnist_train_data.npy')
y_train = np.load('mnist_train_labels.npy')

x_test = np.load('mnist_test_data.npy')
y_test = np.load('mnist_test_labels.npy')

# plt.imshow(x_train[1,:,:,0])
# plt.show()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#from greyscale (0-255) to (0-1)
x_train = x_train/255
x_test = x_test/255

#print(x_train[1,15,20,:])
#print(y_train)

#from integer classification to one-hot encoding
y_train_matrix = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'int32')
y_test_matrix = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'int32')

#print(y_train[2], y_train_matrix[2])

X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(x_train, y_train_matrix, test_size=0.30, random_state=42)

# plt.imshow(x_train[1,:,:,0])
# plt.show()

print(X_train.shape)

model = Sequential()
model.add(layer.Flatten(input_shape=(28,28,1)))
model.add(layer.Dense(64, activation='relu'))
model.add(layer.Dense(128, activation='relu'))
model.add(layer.Dense(10, activation='softmax'))

print(model.summary())

earlyStopping = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
model.compile(loss='categorical_crossentropy', optimizer=optimizer.Adam(lr = 0.01, clipnorm = 1))
#history = model.fit(X_train, Y_train, batch_size=300, epochs=400, callbacks=earlyStopping, validation_data=(X_val, Y_val))
history = model.fit(X_train, Y_train, batch_size=300, epochs=400, validation_data=(X_val, Y_val)) #without early stopping

#print(history.history.keys())
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_predicted = model.predict(x_test)

for i in range(500):
    index = y_predicted[i].argmax()
    y_predicted[i] = [0,0,0,0,0,0,0,0,0,0]
    y_predicted[i,index] = 1

accuracy = sklearn.metrics.accuracy_score(y_test_matrix, y_predicted)
print("accuracy = ", accuracy)
conf_matrix = sklearn.metrics.confusion_matrix(y_test_matrix.argmax(axis=1), y_predicted.argmax(axis=1))
print("Confusion matrix : \n", conf_matrix)
