from MnistClassifierInterface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

class CNN(MnistClassifierInterface):
    def __init__(self, input_shape = (28,28,1), learning_rate = 0.001):
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(filters = 32, kernel_size = (7,7), activation ='relu'),
            layers.Conv2D(filters = 32, kernel_size = (7,7), activation ='relu'),
            layers.MaxPooling2D(pool_size=(2, 2), padding = 'same'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs, batch_size, **kwargs):
        print("training")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.1)

    def predict(self, X_test):
        print("predicting")
        result = self.model.predict(X_test)
        return result