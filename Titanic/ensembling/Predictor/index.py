from sklearn.model_selection import GridSearchCV

from Titanic.ensembling.PredictorModel.index import PredictorModel


class Predictor:
    """
        Class for one predictor
    """
    def __init__(self, name):
        self.name = name
        self.model = PredictorModel.get_model_by_name(name)
        self.fitted_model = None
        self.parameters = PredictorModel.get_parameters_by_model_name(name)
        self.best_parameters = {}
        self.best_accuracy = 0
        self.weight = 0

    def find_best_parameters(self, classes, teacher, target='accuracy'):
        grid_search_cv = GridSearchCV(self.model(), scoring=target, param_grid=self.parameters).fit(classes, teacher)
        self.best_parameters = grid_search_cv.best_params_
        self.best_accuracy = grid_search_cv.best_score_
