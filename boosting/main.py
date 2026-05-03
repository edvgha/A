import numpy as np
import matplotlib.pyplot as plt

from data_gen import chi_square_10dof, noisy_1d_regression
from adaboost_m1 import AdaBoostM1
from gbdt_l2_reg import GBDTL2Reg

def run_adaboost():
    dataset = chi_square_10dof(num_samples=5000)
    # dataset = sklearn_binary_classification(num_samples=5000)
    ada_boost = AdaBoostM1(n_estimators=400)
    ada_boost.fit(dataset)

    # plt.plot(ada_boost.history, marker='o', linestyle='-', color='b')
    plt.scatter(range(len(ada_boost.history)), ada_boost.history, color='red')
    plt.title("Eval per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Test acc")
    plt.grid(True)
    plt.show()

def run_gbdt_l2_reg():
    dataset = noisy_1d_regression()
    gbdt = GBDTL2Reg(n_estimators=100, learning_rate=0.1)
    gbdt.fit(dataset)

    plt.figure(figsize=(15, 4))
    test_estimators = [0, 49, 99]
    
    for i, m in enumerate(test_estimators):
        plt.subplot(1, 3, i + 1)
        plt.scatter(dataset.X_test, dataset.y_test, color='lightgray', s=10) # type: ignore
        plt.plot(dataset.X_test, gbdt.progress[m], color='red', linewidth=2) # type: ignore
        plt.plot(dataset.X_true, dataset.y_true, color='blue', linestyle='--', label='True underlying function') # type: ignore
        plt.title(f'Predictions after {m} Trees')
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    run_adaboost()
    # run_gbdt_l2_reg()

    