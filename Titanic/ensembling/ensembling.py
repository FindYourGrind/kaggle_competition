import age_prediction as ap
import cabin_prediction as cp
import titanic_preprocessor as tp
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from Ensemble.Ensemble import Ensemble
from Ensemble.Predictor import Predictor
import xgboost as xgb

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
all_data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], [0, 0, 0, 0, 0, 0, 0, 0, 1], inplace=True)

all_data = tp.preprocessing(all_data)
to_predict = all_data[survived_data.shape[0]:]

train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.4)

ensemble = Ensemble(train_x, train_y, test_x)

# ensemble.add_predictor(Predictor("DecisionTreeRegressor"))
# ensemble.add_predictor(Predictor("LogisticRegression"))
# ensemble.add_predictor(Predictor("SVC"))
# ensemble.add_predictor(Predictor("K-NN"))
ensemble.add_predictor(Predictor("GaussianNB"))
# ensemble.add_predictor(Predictor("RandomForestClassifier"))
# ensemble.add_predictor(Predictor("Perceptron"))
# ensemble.add_predictor(Predictor("LinearSVC"))
# ensemble.add_predictor(Predictor("SGDClassifier"))
# ensemble.add_predictor(Predictor("SVR"))
# ensemble.add_predictor(Predictor("AdaBoostClassifier"))
# ensemble.add_predictor(Predictor("GradientBoostingClassifier"))
# ensemble.add_predictor(Predictor("ExtraTreesClassifier"))
# ensemble.add_predictor(Predictor("XGBClassifier"))

ensemble.find_best_parameters()
ensemble.save_best_parameters('./best_parameters')

