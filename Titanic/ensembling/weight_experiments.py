from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import age_prediction as ap
import titanic_preprocessor as tp
import pandas as pd
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

seed = 7


def wrap_bagging(estimator, n_estimators=10):
    return BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=seed)


def preproc_predict(train_x, train_y, data):
    RandomForestClassifierModel = RandomForestClassifier(min_samples_leaf=3, max_depth=12, min_samples_split=2,
                                                         bootstrap=True, n_estimators=50, max_features='sqrt',
                                                         random_state=seed)
    XGBClassifierModel = xgb.XGBClassifier(max_depth=6, n_estimators=50, learning_rate=0.01, random_state=seed)
    AdaBoostClassifierModel = AdaBoostClassifier(**{"algorithm": "SAMME", "n_estimators": 50}, random_state=seed)

    predictors_to_drop = [None, 'Age', 'Sex']
    predictions = {}

    for pred in predictors_to_drop:
        if pred:
            tx = train_x.drop(pred, axis=1)
            ttx = data.drop(pred, axis=1)
            brf = RandomForestClassifierModel.fit(tx, train_y)
            prediction = brf.predict(ttx)
        else:
            brf = RandomForestClassifierModel.fit(train_x, train_y)
            prediction = brf.predict(data)
        predictions[str(pred) + '__'] = prediction

    for pred in predictors_to_drop:
        if pred:
            tx = train_x.drop(pred, axis=1)
            ttx = data.drop(pred, axis=1)
            brf = XGBClassifierModel.fit(tx, train_y)
            prediction = brf.predict(ttx)
        else:
            brf = XGBClassifierModel.fit(train_x, train_y)
            prediction = brf.predict(data)
        predictions[str(pred) + '__'] = prediction

    for pred in predictors_to_drop:
        if pred:
            tx = train_x.drop(pred, axis=1)
            ttx = data.drop(pred, axis=1)
            brf = AdaBoostClassifierModel.fit(tx, train_y)
            prediction = brf.predict(ttx)
        else:
            brf = AdaBoostClassifierModel.fit(train_x, train_y)
            prediction = brf.predict(data)
        predictions[str(pred) + '__'] = prediction

    return pd.concat([data, pd.DataFrame(data=predictions, index=data.index)], axis=1)



if __name__ == "__main__":
    titanic_train_data = pd.read_csv(r"data/train.csv").drop(['Ticket'], axis=1)
    titanic_test_data = pd.read_csv(r"data/test.csv").drop(['Ticket'], axis=1)
    all_data = titanic_train_data.append(titanic_test_data).drop(['Survived'], axis=1)
    survived_data = titanic_train_data['Survived']

    freq_port = all_data.Embarked.dropna().mode()[0]
    all_data.loc[:, 'Embarked'] = all_data['Embarked'].fillna(freq_port)

    # all_data = ap.correct_age(all_data)
    all_data.Cabin.fillna(0)

    all_data.loc[:, 'Cabin'] = all_data['Cabin'].map(lambda x: 'U' if pd.isnull(x) else x[0])
    all_data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], [0, 1, 2, 3, 4, 5, 6, 7, 8], inplace=True)

    all_data = tp.preprocessing(all_data)
    to_predict = all_data[survived_data.shape[0]:]

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data,
                                                        test_size=0.5)
    train_x2, test_x2, train_y2, test_y2 = train_test_split(train_x, train_y, test_size=0.5)

    pred = preproc_predict(train_x2, train_y2, test_x2)
    pred2 = preproc_predict(train_x, train_y, test_x)
    pred3 = preproc_predict(all_data[0:survived_data.shape[0]], survived_data, all_data[survived_data.shape[0]:])

    gbm2 = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=2000,
        max_depth=6,
        min_child_weight=2,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=1,
        random_state=seed)

    rr = RandomForestClassifier(min_samples_leaf=3, max_depth=12, min_samples_split=2, bootstrap=True,
                                n_estimators=50, max_features='sqrt', random_state=seed)

    ll = LogisticRegression(
        **{"tol": 0.0001, "fit_intercept": False, "C": 1, "solver": "newton-cg", "max_iter": 1000},
        random_state=seed)
    aa = AdaBoostClassifier(**{"algorithm": "SAMME", "n_estimators": 50}, random_state=seed)
    gg = wrap_bagging(GradientBoostingClassifier(criterion='friedman_mse', max_depth=5, loss='exponential', n_estimators=10, random_state=seed))

    ensemble3 = wrap_bagging(VotingClassifier([('1', gbm2), ('2', rr), ('3', ll), ('4', aa), ('5', gg)], voting='soft')).fit(pred, test_y2)

    print(mean_absolute_error(test_y, ensemble3.predict(pred2)))

    ppp = ensemble3.predict(pred3)
    evaluation = titanic_test_data[['PassengerId']].copy()
    evaluation["Survived"] = ppp.astype(int)
    evaluation.to_csv(r"prediction/voting2.csv", index=False)
