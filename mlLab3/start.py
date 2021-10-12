import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
# Загрузите набор данных о диабете


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Используйте только одну функцию
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Разделите исходные данные на наборы для обучения / тестирования
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Разделите цели на наборы для обучения / тестирования
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Создать объект линейной регрессии
regr = linear_model.LinearRegression()

# Обучите модель с помощью обучающих наборов
regr.fit(diabetes_X_train, diabetes_y_train)

# Делайте прогнозы с помощью набора для тестирования
diabetes_y_pred = regr.predict(diabetes_X_test)

# Коэффициенты
print('Coefficients: \n', regr.coef_)
# Среднеквадратичная ошибка
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# коэффициент детерминации: 1 - идеальное предсказание
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Выходные данные графика
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()