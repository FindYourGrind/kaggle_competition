import json
import numpy as np
import os

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


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

    def save_best_parameters(self, name, path):
        for predictor in self.predictors:
            with open(os.path.join(path, name + '_best_parameters.json', 'w+')) as json_file:
                json.dump(predictor.best_parameters, json_file)

    def fit_ensemble(self):
        for predictor in self.predictors:
            predictor.fitted_model = predictor.model(**predictor.best_parameters)
            predictor.fitted_model.fit(self.train_data, self.teacher_data)

    def predict_test(self):
        train_x, test_x, train_y, test_y = train_test_split(self.train_data, self.teacher_data, test_size=0.2)
        prediction = np.zeros(test_x.shape[0], np.uint32)

        for predictor in self.predictors:
            model = predictor.model(**predictor.best_parameters)
            model.fit(train_x, train_y)
            prediction = np.add(prediction, model.predict(test_x) * predictor.weight)

        result = np.asarray(list(map(lambda x: 1 if x > self.threshold else 0, prediction)))

        return mean_absolute_error(test_y, result)

    def predict(self):
        prediction = np.zeros(self.test_data.shape[0], np.uint32)

        self.fit_ensemble()

        for predictor in self.predictors:
            prediction = np.add(prediction, predictor.fitted_model.predict(self.test_data) * predictor.weight)

        return np.asarray(list(map(lambda x: 1 if x > self.threshold else 0, prediction)))

    def predict2(self, data):
        predictions = []

        self.fit_ensemble()

        for predictor in self.predictors:
            predictions.append(predictor.fitted_model.predict(data).reshape(data.shape[0], 1))

        return np.concatenate(predictions, axis=1)
