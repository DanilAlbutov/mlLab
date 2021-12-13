import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
import csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/Users/danilalbutov/Data/Учеба/7 семак/mlLab/mlLab4/letter-recognition.data')
# Модель из документации изначально масштабирована от 0 до 15, поэтому не будем ее масштабировать

x_columns = data[['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']].head()
y_columns = data[['lettr']]

X_train = x_columns.iloc[:16000]
y_train = y_columns.iloc[:16000]

X_test = x_columns.iloc[4000:]
y_test = y_columns.iloc[4000:]

alphas = [0.00001, 0.0001, 0.001, 0.01]
#np.arange(0, 1, 0.01)

score_list = []

penalty=['l2']
# l1, l2, elasticnet
for pen in penalty:
    for alpha in alphas:
        perc = Perceptron(penalty='l2', alpha=alpha)
        perc.fit(X_train, y_train.values.ravel())
        perc.predict(X_test)
        score_list.append(perc.score(X_test, y_test))

print("Best score={r2} with alpha={alpha}".format(r2=max(score_list), alpha=alphas[score_list.index(max(score_list))]))

plt.axis([0, 0.011, 0, 1])
plt.plot(alphas, score_list)
plt.plot(alphas, score_list, 'bo')
plt.xlabel("alpha")
plt.ylabel("r2")
plt.savefig('perceptron_l2.png')
plt.show()

