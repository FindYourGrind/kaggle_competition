import age_prediction as ap
import cabin_prediction as cp
import titanic_preprocessor as tp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor
import json

test = False

titanic_train_data = pd.read_csv(r"data/train.csv").drop(['Ticket'], axis=1)
titanic_test_data = pd.read_csv(r"data/test.csv").drop(['Ticket'], axis=1)
all_data = titanic_train_data.append(titanic_test_data).drop(['Survived'], axis=1)
survived_data = titanic_train_data['Survived']

freq_port = all_data.Embarked.dropna().mode()[0]
all_data.loc[:, 'Embarked'] = all_data['Embarked'].fillna(freq_port)

all_data = ap.correct_age(all_data)
#all_data = cp.correct_cabin(all_data)

all_data.loc[:, 'Cabin'] = all_data['Cabin'].map(lambda x: 'U' if pd.isnull(x) else x[0])
all_data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], [0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True)

all_data = tp.preprocessing(all_data)


train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.2)


def read_parameters_json():
    with open('predictor_parameters.json') as json_file:
        data = json.load(json_file)
    return data


class PredictorModel:
    """
        Class with predictor models
    """
    predictor_models = {
        'DecisionTreeRegressor': DecisionTreeRegressor,
        'LogisticRegression': LogisticRegression,
        'SVC': SVC,
        'K-NN': KNeighborsClassifier,
        'GaussianNB': GaussianNB,
        'RandomForestClassifier': RandomForestClassifier,
        'Perceptron': Perceptron,
        'LinearSVC': LinearSVC,
        'SGDClassifier': SGDClassifier
    }
    predictor_parameters = read_parameters_json()

    @classmethod
    def get_model_by_name(cls, name):
        return cls.predictor_models[name]

    @classmethod
    def get_parameters_by_model_name(cls, name):
        return cls.predictor_parameters[name]


class Predictor:
    """
        Class for one predictor
    """
    def __init__(self, name):
        self.name = name
        self.model = PredictorModel.get_model_by_name(name)
        self.parameters = PredictorModel.get_parameters_by_model_name(name)
        self.best_parameters = {}
        self.best_accuracy = 0
        self.weight = 0

    def find_best_parameters(self, classes, teacher, target='accuracy'):
        grid_search_cv = GridSearchCV(self.model(), scoring=target, param_grid=self.parameters).fit(classes, teacher)
        self.best_parameters = grid_search_cv.best_params_
        self.best_accuracy = grid_search_cv.best_score_


class Ensemble:
    """
        Class for ensembling predictors
    """
    def __init__(self, train_data, teacher_data, test_data):
        self.predictors = []
        self.train_data = train_data
        self.teacher_data = teacher_data
        self.test_data = test_data
        self.threshold = 0.39

    def add_predictor(self, predictor):
        self.predictors.append(predictor)

    def find_best_parameters(self):
        for predictor in self.predictors:
            predictor.find_best_parameters(self.train_data, self.teacher_data)
            print(predictor.name, ' best parameters has been found')

    def calculate_weights(self):
        sum_accuracy = 0

        for predictor in self.predictors:
            sum_accuracy += predictor.best_accuracy

        for predictor in self.predictors:
            predictor.weight = predictor.best_accuracy / sum_accuracy

    def print_accuracies(self):
        for predictor in self.predictors:
            print(predictor.name, ': ', predictor.best_accuracy)

    def predict_test(self):
        train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.teacher_data, test_size=0.2)
        prediction = np.zeros(test_x.shape[0], np.uint32)

        for predictor in self.predictors:
            model = predictor.model(**predictor.best_parameters)
            model.fit(train_x, train_y)
            prediction = np.add(prediction, model.predict(test_x) * predictor.weight)

        result = np.asarray(list(map(lambda x: 1 if x > self.threshold else 0, prediction)))

        print('Prediction error: ', mean_absolute_error(test_y, result), ' Threshold: ', self.threshold)
        print('\n')


ensemble = Ensemble(all_data[0:survived_data.shape[0]], survived_data, [])

ensemble.add_predictor(Predictor("LogisticRegression"))
ensemble.add_predictor(Predictor("K-NN"))
ensemble.add_predictor(Predictor("GaussianNB"))
ensemble.add_predictor(Predictor("RandomForestClassifier"))
ensemble.add_predictor(Predictor("Perceptron"))
ensemble.add_predictor(Predictor("SGDClassifier"))

ensemble.find_best_parameters()
ensemble.calculate_weights()
ensemble.print_accuracies()

ensemble.threshold = 0.55
ensemble.predict_test()

