import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from dataclasses import dataclass, field

@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None

    X_true: np.ndarray | None = None
    y_true: np.ndarray | None = None
    
    num_train_samples: int = field(init=False)
    num_test_samples: int = field(init=False)
    num_features: int = field(init=False)

    def __post_init__(self):
        self.num_train_samples = self.X_train.shape[0]
        self.num_features = self.X_train.shape[1] if self.X_train.ndim > 1 else 1
        
        if self.X_test is not None:
            self.num_test_samples = self.X_test.shape[0]
            # --- Feature Validation ---
            test_features = self.X_test.shape[1] if self.X_test.ndim > 1 else 1
            if self.num_features != test_features:
                raise ValueError(
                    f"Feature mismatch: X_train has {self.num_features} features, "
                    f"but X_test has {test_features} features."
                )
        else:
            self.num_test_samples = 0


def chi_square_10dof(num_samples: int = 2000, num_features: int = 10, test_size: float = 0.2, random_state: int = 42) -> Dataset:
    # 1. Generate 10 features, each from a standard normal distribution N(0, 1)
    # size=(num_samples, 10) creates a matrix with 'num_samples' rows and 10 columns
    X = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_features))
    
    # 2. Calculate the sum of squares for each row
    # axis=1 ensures we sum across the columns for each individual sample
    sum_of_squares = np.sum(X**2, axis=1)
    
    # 3. Apply the label condition
    chi_square_median = 10 * np.pow((1 - (2 / (9 * num_features))), 3)
    y = np.where(sum_of_squares > chi_square_median, 1, -1)
    
    if test_size > .0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state
            )
        return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return Dataset(X_train=X, y_train=y)


def sklearn_binary_classification(num_samples: int = 2000, num_features: int = 10, test_size: float = 0.2, random_state: int = 42):
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_classes=2,
        random_state=random_state
    )
    
    y = np.where(y == 0, -1, y)
    if test_size > .0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state
            )
        return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return Dataset(X_train=X, y_train=y)


def noisy_1d_regression():
    np.random.seed(42)
    X = np.linspace(-5, 5, 500).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

    return Dataset(X_train=X, y_train=y, X_test=X, y_test=y, X_true=X, y_true=np.sin(X))