import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import math

class SVMModel():
    def __init__(self, X_train, y_train, kernel="linear", cache_size=1024):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.svm = SVC(kernel=kernel, decision_function_shape="ovo", cache_size=cache_size)
        self.svm.fit(X_train_scaled, y_train)

    def predict(self, X):
        return self.svm.predict(self.scaler.transform(X))
    
    def predict_with_adversarial_prediction(self, X, adv_prediction_function):
        """Uses the `adv_prediction_function` to decide if the input is
           adversarial; if so, outputs class -1. Otherwise, outputs the
           predicted class."""
        y_predict = self.predict(X)
        class_distances = self.get_predicted_class_decision_boundary_distances(X, y_predict)
        y_predict_is_adv = np.fromiter(map(adv_prediction_function, class_distances), dtype=np.bool)

        for i, x in enumerate(y_predict_is_adv):
            if x:
                y_predict[i] = -1
        
        return y_predict

    def get_decision_boundary_distances(self, X):
        return self.svm.decision_function(self.scaler.transform(X))

    def get_predicted_class_decision_boundary_distances(self, X, predictions, num_classes=10):
        distances = self.get_decision_boundary_distances(X)

        predicted_class_distances = []

        for k in range(len(X)):
            distances_for_k = distances[k]
            predicted_class_distances_for_k = []
            i = predictions[k]
            # When j < i
            for j in range(i):
                index = int((num_classes - 0.5) * j - math.pow(j, 2) / 2 + (i - j - 1))
                predicted_class_distances_for_k.append(-1 * distances_for_k[index])
            # When i < j
            base_index = int((num_classes - 0.5) * i - math.pow(i, 2) / 2)
            for j in range(i + 1, num_classes):
                index = base_index + (j - i - 1)
                predicted_class_distances_for_k.append(distances_for_k[index])
            predicted_class_distances.append(predicted_class_distances_for_k)

        return predicted_class_distances

    def test(self, X_test, y_test):
        """Calculates the accuracy of the model on some test data, without
           adversarial prediction."""

        return self.svm.score(self.scaler.transform(X_test), y_test)

    def test_with_adversarial_prediction(self, X_test, y_test, is_adversarial, adv_prediction_function):
        """Calculates the accuracy of the model on some test data when using
           adversarial prediction. Takes a boolean flag indicating if the test
           data is adversarial or not, and an adversarial prediction function"""

        y_predict = self.predict_with_adversarial_prediction(X_test, adv_prediction_function)
        
        # Number of inputs with correct class predicted
        num_correct = np.sum(y_predict == y_test)

        if is_adversarial:
            # If the examples are adversarial it's also correct to predict the
            # class -1 (which indicates adversarialness)
            num_correct += np.sum(y_predict == -1)

        return num_correct / len(X_test)

    def distances_linear_map(self, image_feature_dim, num_classes=10):
        start = 0
        end = 0
        class_support_vect = []

        class_dual_coef = []
        for k in self.svm.n_support_:
            end += k
            class_support_vect.append(self.svm.support_vectors_[start:end])
            class_dual_coef.append(self.svm.dual_coef_[:, start:end])
            start = end

        dists = np.zeros((int(num_classes * (num_classes - 1) / 2), image_feature_dim))
        k = 0
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                dists[k] += np.sum(class_dual_coef[i][j-1].reshape(1,-1).T * class_support_vect[i], axis=0)
                dists[k] += np.sum(class_dual_coef[j][i].reshape(1,-1).T * class_support_vect[j], axis=0)
                k += 1

        return ((dists / self.scaler.scale_).T, self.svm.intercept_ - ((dists / self.scaler.scale_) @ self.scaler.mean_))
