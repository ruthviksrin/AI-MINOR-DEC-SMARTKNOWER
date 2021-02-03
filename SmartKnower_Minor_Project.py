#importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
#reading the data
df = pd.read_csv('heart.csv')

#Spliting Features and Target variables
y = df["target"]
X = df.drop("target", axis=1)

#Preprocessing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.30,random_state=42)

#Model Architecture
model = Sequential()
model.add(Dense(8, activation= 'relu',input_shape = (13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the data
results = model.fit(X, y, epochs=50, validation_data=(X_test, y_test))
plt.subplot(2,2,1)
_ = plt.plot(results.history['accuracy'], np.linspace(1,50))
_ = plt.xlabel('Accuracy')
_ = plt.ylabel('Epochs')
_ = plt.title('Training Accuracy')
plt.subplot(2,2,2)
_ = plt.plot(results.history['loss'], np.linspace(1,50))
_ = plt.xlabel('Loss')
_ = plt.ylabel('Epochs')
_ = plt.title('Training Loss')
plt.subplot(2,2,3)
_ = plt.plot(results.history['val_accuracy'], np.linspace(1,50))
_ = plt.xlabel('Accuracy')
_ = plt.ylabel('Epochs')
_ = plt.title('Test Accuracy')
plt.subplot(2,2,4)
_ = plt.plot(results.history['val_loss'], np.linspace(1,50))
_ = plt.xlabel('Loss')
_ = plt.ylabel('Epochs')
_ = plt.title('Test Loss')
