import numpy as np
from costs import rmse

class LinRegModel:
    def __init__(self, inputs: int, Theta: np.ndarray = None, costFunc: str = "MSE") -> None:
        self.costFunctions = {
            "MSE" : rmse
        }

        self.thetas = np.random.rand(inputs) if Theta is None else Theta
        self.cost = self.costFunctions[costFunc]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.thetas)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, maxIters: int = 1000,
              alpha: float = 0.1, convergenceThreshold: float = 0,
              beta1: float = 0.999, beta2: float = 0.9, epsilon: float = 1e-6,
              lambda_l1: float = 0.5, lambda_l2: float = 0.3) -> list:
        X_scale = self.scaleFeatures(X_train)
        m = y_train.shape[0]
        J_Hist = [self.cost(self.predict(X_train), y_train) + 
                  (lambda_l1 / (2 * m)) * np.sum(np.abs(self.thetas)) +
                  (lambda_l2 / (2 * m)) * np.sum(self.thetas**2)]

        m_t = np.zeros_like(self.thetas)
        v_t = np.zeros_like(self.thetas)

        for i in range(1, maxIters):
            hx = self.predict(X_train)
            errors = hx - y_train.T
            regularized_term_l1 = (lambda_l1 / m) * np.sign(self.thetas)
            regularized_term_l2 = (lambda_l2 / m) * self.thetas
            gradient = (1 / m) * np.dot(errors, X_scale) + regularized_term_l1 + regularized_term_l2

            m_t = beta1 * m_t + (1 - beta1) * gradient
            v_t = beta2 * v_t + (1 - beta2) * (gradient**2)

            m_t_hat = m_t / (1 - beta1**i)
            v_t_hat = v_t / (1 - beta2**i)

            step = self.get_learning_rate(alpha, i) * (m_t_hat) / (np.sqrt(v_t_hat) + epsilon)
            print("Step: ", step, "weights", self.thetas)
            self.thetas -= step[0].T

            J_Hist.append(self.cost(y_train.T, self.predict(X_train)) +
                          (lambda_l1 / (2 * m)) * np.sum(np.abs(self.thetas)) +
                          (lambda_l2 / (2 * m)) * np.sum(self.thetas**2))
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

    def get_learning_rate(self, alpha: float, iteration: int) -> float:
            return alpha * 1 / (1 + alpha * np.sqrt(iteration))
