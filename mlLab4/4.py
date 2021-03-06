import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/Users/danilalbutov/Data/Учеба/7 семак/mlLab/mlLab4/letter-recognition.data')
# Модель из документации изначально масштабирована от 0 до 15, поэтому не будем ее масштабировать

x_columns = data[['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']]
y_columns = data[['lettr']]

X_train = x_columns.iloc[:16000]
y_train = y_columns.iloc[:16000]

X_test = x_columns.iloc[4000:]
y_test = y_columns.iloc[4000:]



alphas = [0.00001, 0.0001, 0.001, 0.01]
#np.arange(0, 1, 0.01)

score_list = []

optimization = ['adam']
# lbfgs, sgd, adam
for opt in optimization:
    for alpha in alphas:
        mlp = MLPClassifier(alpha=alpha, solver=f'{opt}', max_iter=200)
        mlp.fit(X_train, y_train.values.ravel())
        mlp.predict(X_test)
        score_list.append(mlp.score(X_test, y_test))

print("Best score={r2} with alpha={alpha}".format(r2=max(score_list), alpha=alphas[score_list.index(max(score_list))]))

plt.axis([0, 0.011, 0, 1])
plt.plot(alphas, score_list)
plt.plot(alphas, score_list, 'bo')
plt.xlabel("alpha")
plt.ylabel("r2")
plt.savefig('mlp_adam.png')
plt.show()

