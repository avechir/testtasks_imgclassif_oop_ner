from MnistClassifierInterface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier

class RandomForest(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        print("training")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        print("predicting")
        result = self.model.predict(X_test)
        return result