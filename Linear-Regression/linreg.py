import numpy as np
from costs import rmse

class LinRegModel:
    def __init__(self, inputs: int, Theta: np.ndarray = None, costFunc: str = "MSE") -> None:
        costFunctions = {
            "MSE" : rmse
        }

        self.thetas = np.random.rand(inputs) if Theta is None else Theta
        self.cost = costFunctions[costFunc]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.thetas)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              maxIters: int = 1000, alpha: float = 0.1, convergenceThreshold: float = 0) -> list:
        X_scale = self.scaleFeatures(X_train)
        m = y_train.shape[0]
        J_Hist = []
        J_Hist.append(self.cost(self.predict(X_train), y_train))
        for i in range(1, maxIters):
            hx = self.predict(X_train)
            errors = (hx - y_train.T)
            step = (alpha * (1 / m) * np.dot(errors, X_scale))
            print("Step: ", step, "weights", self.thetas)
            self.thetas -= step[0].T

            J_Hist.append(self.cost(y_train.T, self.predict(X_train)))
            print("Iteration: %d Cost: %f" % (i, J_Hist[i]))

            if (np.abs(J_Hist[i - 1] - J_Hist[i]) < convergenceThreshold):
                print("Training converged at iteration: %d" % i)
                break

        return J_Hist
    
    def scaleFeatures(self, X: np.ndarray) -> np.ndarray:
        X_scale = X.copy()
        for i in range(X.shape[1]):
            X_scale[:, i] = (X_scale[:, i] - X_scale[:, i].mean()) / (X_scale[:, i].std())
        return X_scale
