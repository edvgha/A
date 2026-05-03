from typing import List
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from data_gen import Dataset

class AdaBoostM1:
    def __init__(self, random_state: int = 42, n_estimators: int = 50) -> None:
        '''
        ESL second edition Algorithm 10.1 AdaBoost.M1
        '''
        self.n_estimators = n_estimators

        self.estimators: List = []
        self.estimator_weights: List[float] = []
        for _ in range(n_estimators):
            self.estimators.append(DecisionTreeClassifier(max_depth=1, random_state=random_state))
            self.estimator_weights.append(.0)
        
        self.history: List[float] = []

    def fit(self, data: Dataset) -> None:
        epoch = 0

        # init weights
        w = np.full(data.num_train_samples, 1/data.num_train_samples)

        while epoch < self.n_estimators:
            self.estimators[epoch].fit(data.X_train, data.y_train, sample_weight=w)
            
            indicator = (self.estimators[epoch].predict(data.X_train) != data.y_train)
            error = np.average(indicator, weights=w)

            estimated_weight = np.log((1.0 - error) / error)
            self.estimator_weights[epoch] = estimated_weight

            w = w * np.exp(estimated_weight * indicator)

            epoch += 1

            if error >= .5:
                print(f'WARNING: Epoch: {epoch} err: {error:.3f}')
            
            self.history.append(self._eval(data, epoch))
            
            
    def _predict(self, data: Dataset, num_predictors: int) -> np.ndarray:
        predictions_np = np.zeros((num_predictors, data.num_test_samples))
        for i in range(num_predictors):
            predictions_np[i] = self.estimator_weights[i] * self.estimators[i].predict(data.X_test)
        return np.sign(np.sum(predictions_np, axis=0, keepdims=True)).squeeze()

    def _eval(self, data: Dataset, num_predictors: int) -> float:
        predictions = self._predict(data, num_predictors=num_predictors)
        correct = (predictions == data.y_test)
        return correct.sum() / data.num_test_samples
    
    def predict(self, data: Dataset) -> np.ndarray:
        return self._predict(data, num_predictors=self.n_estimators)


