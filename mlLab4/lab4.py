from collections import defaultdict

from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, preprocessing, neural_network
from sklearn.metrics import accuracy_score
from joblib import dump, load

def get_training_set():
    training_set = read_csv('/Users/danilalbutov/Data/Учеба/7 семак/ЛАБЫ/laby/Машинное обучение/lab4/segmentation.training', header=0, index_col=False)
    return training_set.values[:, 1:].astype('float16'), training_set.values[:, 0].astype('U')

def get_test_set():
    test_set = read_csv('/Users/danilalbutov/Data/Учеба/7 семак/ЛАБЫ/laby/Машинное обучение/lab4/segmentation.test', header=0, index_col=False)
    return test_set.values[:, 1:].astype('float16'), test_set.values[:, 0].astype('U')

def create_scaler(training_set, copy=False):
    scaler = preprocessing.MinMaxScaler(copy=copy)
    scaler.fit(training_set)
    return scaler

def learn_Perceptron(learn_set_X, learn_set_Y, penalty, alpha):
    perceptron = linear_model.Perceptron(random_state=0,
        penalty=penalty, alpha=alpha)
    perceptron.fit(learn_set_X, learn_set_Y)
    return perceptron

def iter_Perceptron_params():
    for penalty in 'l1', 'l2', 'elasticnet':
        for alpha in 0.01, 0.001, 0.0001, 0.00001:
            yield penalty, alpha

def get_Perceptron_filename(penalty, alpha):
    return f'perceptron_penalty({penalty})_alpha({alpha}).joblib'

def learn_MLPClassifier(learn_set_X, learn_set_Y,
        hidden_layer_sizes, solver, alpha, learning_rate):
    MLPClassifier = neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=0, max_iter=2000, tol=0.001,
        solver=solver, alpha=alpha, learning_rate=learning_rate)
    MLPClassifier.fit(learn_set_X, learn_set_Y)
    return MLPClassifier

def iter_MLPClassifier_params():
    for hidden_layer_width in 1, 2, 3, 4:
        for hidden_layer_length in 50, 100, 150, 200:
            for solver in 'lbfgs', 'sgd', 'adam':
                for alpha in 0.001, 0.0001:
                    for learning_rate in 'constant', 'invscaling', 'adaptive':
                        yield ((hidden_layer_length, hidden_layer_width),
                            solver, alpha, learning_rate)

def get_MLPClassifier_filename(hidden_layer_sizes, solver, alpha, learning_rate):
    return f'MLPClassifier_hidden_layer_sizes:{hidden_layer_sizes}_solver:{solver}_alpha:{alpha}_learning_rate:{learning_rate}.joblib'

def test_model(model, test_set_X, test_set_Y, print_=False):
    y_pred = model.predict(test_set_X)
    a_s = accuracy_score(test_set_Y, y_pred)

    if print_:
        print('Accuracy score:', a_s)

    return a_s

if __name__ == '__main__':
    training_set_X, training_set_Y = get_training_set()
    test_set_X, test_set_Y = get_test_set()

    scaler = create_scaler(training_set_X)
    scaler.transform(training_set_X)
    scaler.transform(test_set_X)

    for penalty, alpha in iter_Perceptron_params():
        perceptron = learn_Perceptron(training_set_X, training_set_Y,
            penalty, alpha)
        print(f'Completed: {penalty}-{alpha}')
        dump(perceptron, get_Perceptron_filename(penalty, alpha))

    params = []
    results = []
    for penalty, alpha in iter_Perceptron_params():
        perceptron = load(get_Perceptron_filename(penalty, alpha))
        params.append(f'{penalty}\n{alpha}')
        results.append(test_model(perceptron, test_set_X, test_set_Y, print_=True))
    plt.plot(params, results)
    plt.savefig('perceptron.png')
    plt.close('all')

    for hidden_layer_sizes, solver, alpha, learning_rate in iter_MLPClassifier_params():
        MLPClassifier = learn_MLPClassifier(training_set_X, training_set_Y,
            hidden_layer_sizes, solver, alpha, learning_rate)
        print(f'Completed: {hidden_layer_sizes}-{solver}-{alpha}-{learning_rate}')
        dump(MLPClassifier, '/Users/danilalbutov/Data/Учеба/7 семак/mlLab/mlLab4/MLPClassifier/' + get_MLPClassifier_filename(
            hidden_layer_sizes, solver, alpha, learning_rate))

    for params in solver, solver in hidden_layer_sizes:
        params = defaultdict(lambda: defaultdict(list)) 
        results = defaultdict(lambda: defaultdict(list))
    for hidden_layer_sizes, solver, alpha, learning_rate in iter_MLPClassifier_params():
        MLPClassifier = load('/Users/danilalbutov/Data/Учеба/7 семак/mlLab/mlLab4/MLPClassifier/' + get_MLPClassifier_filename(
            hidden_layer_sizes,solver, alpha, learning_rate))

        params[hidden_layer_sizes][solver].append(f'{alpha}\n{learning_rate}')
        results[hidden_layer_sizes][solver].append(
            test_model(MLPClassifier, test_set_X, test_set_Y))

    for layer_sizes, solvers_and_params in params.items():
        plt.title('layer_sizes:' + str(layer_sizes))
        plt.xlabel('alpha and learning rate')
        plt.ylabel('Accuracy score')
        plt.grid(True)
        for solver, params in solvers_and_params.items():
            plt.plot(params, results[layer_sizes][solver])
        plt.legend(tuple(map(str, solvers_and_params.keys())), loc=2)
        plt.savefig(f'MLPClassifier_layer_sizes:{layer_sizes}.png')
    #     plt.close('all')
