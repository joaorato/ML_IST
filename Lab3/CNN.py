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

#plt.imshow(x_train[1,:,:,0])
#plt.show()

print(X_train.shape)

model = Sequential()
model.add(layer.Conv2D(filters = 16, kernel_size = 3, input_shape=(28,28,1), activation='relu'))
model.add(layer.MaxPooling2D(pool_size = (2, 2)))
model.add(layer.Conv2D(filters = 32, kernel_size = 3, activation='relu'))
model.add(layer.MaxPooling2D(pool_size = (2, 2)))
model.add(layer.Flatten())
model.add(layer.Dense(64, activation='relu'))
model.add(layer.Dense(10, activation='softmax'))

print(model.summary())

earlyStopping = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
model.compile(loss='categorical_crossentropy', optimizer=optimizer.Adam(lr = 0.01, clipnorm = 1))
history = model.fit(X_train, Y_train, batch_size=300, epochs=400, callbacks=earlyStopping, validation_data=(X_val, Y_val))

#print(history.history.keys())
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

y_predicted = model.predict(x_test)

accuracy = sklearn.metrics.accuracy_score(y_test_matrix, y_predicted.round())
print("accuracy = ", accuracy)
conf_matrix = sklearn.metrics.confusion_matrix(y_test_matrix.argmax(axis=1), y_predicted.argmax(axis=1))
print("Confusion matrix : \n", conf_matrix)