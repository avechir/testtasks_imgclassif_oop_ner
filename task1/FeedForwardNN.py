from MnistClassifierInterface import MnistClassifierInterface
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam

class FeedForwardNN(MnistClassifierInterface):
    def __init__(self, input_size=784, hidden_size=256, output_size=10, learning_rate = 0.001):
        self.model = keras.Sequential([
            keras.Input(shape=input_size),
            layers.Dense(hidden_size, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(output_size, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate = learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs, batch_size, **kwargs):
        print("training")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.1)

    def predict(self, X_test):
        print("predicting")
        result = self.model.predict(X_test)
        return result