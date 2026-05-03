from typing import List
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from data_gen import Dataset


class GBDTL2Reg:
    def __init__(self, random_state: int = 42, n_estimators: int = 100, learning_rate: float = 0.1) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_prediction : float = float('inf')

        self.estimators: List = []
        for _ in range(n_estimators):
            self.estimators.append(DecisionTreeRegressor(max_depth=2, random_state=random_state))

        
        self.progress: List[np.ndarray] = []

    
    def fit(self, data: Dataset) -> None:
        self.initial_prediction = np.mean(data.y_train) # type: ignore
        current_predictions = np.full(shape=data.num_train_samples, fill_value=self.initial_prediction)
        
        for epoch in range(self.n_estimators):
            # Compute the "Pseudo-Residuals"
            # For L2 loss, this is simply (Actual - Predicted)
            residuals = data.y_train - current_predictions

            self.estimators[epoch].fit(data.X_train, residuals)

            # F_m(x) = F_{m-1}(x) + (nu * h_m(x))
            stump_predictions = self.estimators[epoch].predict(data.X_train)
            current_predictions += self.learning_rate * stump_predictions

            self._eval(data, epoch + 1)

    def _predict(self, data: Dataset, num_predictors: int) -> np.ndarray:
        predictions = np.full(shape=data.num_test_samples, fill_value=self.initial_prediction)
        
        for i in range(num_predictors):
            predictions += self.learning_rate * self.estimators[i].predict(data.X_test)
            
        return predictions

    def _eval(self, data: Dataset, num_predictors: int) -> None:
        predictions = self._predict(data, num_predictors=num_predictors)
        self.progress.append(predictions)
    
    def predict(self, data: Dataset) -> np.ndarray:
        return self._predict(data, num_predictors=self.n_estimators)
