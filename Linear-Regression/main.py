import numpy as np
import matplotlib.pyplot as plt
from linreg import LinRegModel
import pandas as pd

def main() -> None:
    file_path = "data/kc_house_data_NaN.csv"
    data = pd.read_csv(file_path)

    X = data.iloc[:, np.r_[4, 6]].to_numpy()
    y = data.iloc[:, 3].to_numpy()

    model = LinRegModel(X.shape[1])

    J_Hist = model.train(X, y, maxIters = 5 * (10**5), 
                         alpha = 1 * (10**-6), convergenceThreshold = 1 * (10**-5))

    draw(model, X, y, J_Hist, bool3D=False)

def draw(model, X: np.ndarray, y: np.ndarray, J_Hist: list, bool3D: bool = False) -> None:
    fig = plt.figure()
    fig.suptitle('Linear Regression')
    fig.tight_layout(pad=2.5, w_pad=1.5, h_pad=0)

    costPlot = fig.add_subplot(121)
    drawCostHistory(J_Hist, costPlot)

    if bool3D:
        predPlot = fig.add_subplot(122, projection='3d')
        drawPrediction3D(model, X, y, predPlot)
    else:
        predPlot = fig.add_subplot(122)
        drawPrediction2D(model, X, y, predPlot)

    plt.show()

def drawCostHistory(J_Hist: list, plot) -> None:
    plot.plot(J_Hist)
    plot.set_ylabel('Cost')
    plot.set_xlabel('Iterations')
    plot.axis([0, len(J_Hist), 0, max(J_Hist)])
    plot.set_aspect(len(J_Hist)/max(J_Hist))

def drawPrediction2D(model, X: np.ndarray, y: np.ndarray, plot) -> None:
    X_values = X[:, 0]

    plot.scatter(X_values, y, s=10, c="blue", label="Real")
    plot.plot(X_values, model.predict(X), c="red", label="Predicted")
    plot.set(xlabel='X', ylabel='Y')
    plot.set_title('Prediction')
    plot.set_aspect(max(X_values)/max(y))

def drawPrediction3D(model, X: np.ndarray, y: np.ndarray, plot) -> None:
    plot.scatter(X[:, 0], X[:, 1], y, s=0.5, c="blue")
    plot.scatter(X[:, 0], X[:, 1], model.predict(X), s=5, c="red")
    plot.set(xlabel='X', ylabel='Y', zlabel='Z')
    plot.set_title('Prediction\nBlue = Real, Red = Predicted')

if __name__ == "__main__":
    main()
