from MnistClassifierInterface import MnistClassifierInterface

class MnistClassifier:
    def __init__(self, algorithm: MnistClassifierInterface):
        self._algorithm = algorithm  

    @property
    def algorithm(self):
        return self._algorithm  

    @algorithm.setter
    def algorithm(self, algorithm: MnistClassifierInterface):
        self._algorithm = algorithm 

    def train(self, X_train, y_train, **kwargs):
        return self._algorithm.train(X_train, y_train, **kwargs)  
    
    def predict(self, X_test):
        return self._algorithm.predict(X_test)  
