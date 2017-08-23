import json
import os

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_parameters_json():
    with open(os.path.join(__location__, 'predictor_parameters.json')) as json_file:
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
        'SGDClassifier': SGDClassifier,
        'SVR': SVR,
        'AdaBoostClassifier': AdaBoostClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'XGBClassifier': xgb.XGBClassifier
    }
    predictor_parameters = read_parameters_json()

    @classmethod
    def get_model_by_name(cls, name):
        return cls.predictor_models[name]

    @classmethod
    def get_parameters_by_model_name(cls, name):
        return cls.predictor_parameters[name]