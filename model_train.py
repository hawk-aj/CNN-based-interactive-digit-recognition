#imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

#loading and processing datasets
train_data = pd.read_csv("C:/Users/aj240/Downloads/Aarya Jha - mnist_train.csv")
test_data = pd.read_csv("C:/Users/aj240/Downloads/Aarya Jha - mnist_test.csv")

train = np.array(train_data)
test = np.array(test_data)

train_x = train[:,1:]
train_y = pd.get_dummies(train[:,0])

test_x = test[:,1:]
test_y = pd.get_dummies(test[:,0])

train_x = train_x.reshape(-1,28,28,1)
test_x = test_x.reshape(-1,28,28,1)

train_X,train_Y = shuffle(train_x, train_y)
test_X,test_Y = shuffle(test_x, test_y)

#creating the neural network
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_X,train_Y, epochs = 10, validation_data=(test_X,test_Y))

print("The model has successfully trained")
model.save('aj_mnist.h5')
print("Saving the model as aj_mnist.h5")
