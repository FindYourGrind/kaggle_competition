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

seed = 7


def wrap_bagging(estimator, n_estimators=10):
    return BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators, random_state=seed)


if __name__ == "__main__":
    titanic_train_data = pd.read_csv(r"data/train.csv").drop(['Ticket'], axis=1)
    titanic_test_data = pd.read_csv(r"data/test.csv").drop(['Ticket'], axis=1)
    all_data = titanic_train_data.append(titanic_test_data).drop(['Survived'], axis=1)
    survived_data = titanic_train_data['Survived']

    freq_port = all_data.Embarked.dropna().mode()[0]
    all_data.loc[:, 'Embarked'] = all_data['Embarked'].fillna(freq_port)

    all_data = ap.correct_age(all_data)

    all_data.loc[:, 'Cabin'] = all_data['Cabin'].map(lambda x: 'U' if pd.isnull(x) else x[0])
    all_data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], [0, 0, 0, 0, 0, 0, 0, 0, 1], inplace=True)

    all_data = tp.preprocessing(all_data)
    to_predict = all_data[survived_data.shape[0]:]

    train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.2)

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    DecisionTreeClassifierModel = DecisionTreeClassifier(max_features=10, min_samples_leaf=6, criterion='gini', max_depth=6, max_leaf_nodes=50, splitter='best', min_samples_split=3, random_state=seed)
    RandomForestClassifierModel = RandomForestClassifier(min_samples_leaf=3, max_depth=12, min_samples_split=2, bootstrap=True, n_estimators=50, max_features='sqrt', random_state=seed)
    GradientBoostingClassifierModel = GradientBoostingClassifier(criterion='friedman_mse', max_depth=5, loss='exponential', n_estimators=10, random_state=seed)
    XGBClassifierModel = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=seed)
    LogisticRegressionModel = LogisticRegression(**{"tol": 0.0001, "fit_intercept": False, "C": 1, "solver": "newton-cg", "max_iter": 1000}, random_state=seed)
    AdaBoostClassifierModel = AdaBoostClassifier(**{"algorithm": "SAMME", "n_estimators": 50}, random_state=seed)
    ExtraTreesClassifierModel = ExtraTreesClassifier(**{"criterion": "entropy", "n_estimators": 20}, random_state=seed)

    estimators = [
        ('DecisionTreeClassifierModel', wrap_bagging(wrap_bagging(DecisionTreeClassifierModel))),
        ('AdaBoostClassifierModel', wrap_bagging(AdaBoostClassifierModel)),
        ('ExtraTreesClassifierModel', wrap_bagging(wrap_bagging(ExtraTreesClassifierModel))),
        ('RandomForestClassifierModel', wrap_bagging(RandomForestClassifierModel)),
        ('GradientBoostingClassifierModel', wrap_bagging(wrap_bagging(GradientBoostingClassifierModel))),
        ('LogisticRegressionModel', wrap_bagging(wrap_bagging(LogisticRegressionModel))),
    ]

    predictions = {}

    train_x2, test_x2, train_y2, test_y2 = train_test_split(train_x, train_y, test_size=0.2)

    for estimator in estimators:
        estimator[1].fit(train_x2, train_y2)
        predictions[estimator[0]] = estimator[1].predict(test_x2)

    pred = pd.concat([test_x2, pd.DataFrame(data=predictions, index=test_x2.index)], axis=1)

    gbm = xgb.XGBClassifier(
        learning_rate=0.01,
        n_estimators=2000,
        max_depth=6,
        min_child_weight=2,
        # gamma=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=-1,
        scale_pos_weight=1,
        random_state=seed).fit(pred, test_y2)



    DecisionTreeClassifierModel = DecisionTreeClassifier(max_features=10, min_samples_leaf=6, criterion='gini', max_depth=6, max_leaf_nodes=50, splitter='best', min_samples_split=3, random_state=seed)
    RandomForestClassifierModel = RandomForestClassifier(min_samples_leaf=3, max_depth=12, min_samples_split=2, bootstrap=True, n_estimators=50, max_features='sqrt', random_state=seed)
    GradientBoostingClassifierModel = GradientBoostingClassifier(criterion='friedman_mse', max_depth=5, loss='exponential', n_estimators=10, random_state=seed)
    XGBClassifierModel = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=seed)
    LogisticRegressionModel = LogisticRegression(**{"tol": 0.0001, "fit_intercept": False, "C": 1, "solver": "newton-cg", "max_iter": 1000}, random_state=seed)
    AdaBoostClassifierModel = AdaBoostClassifier(**{"algorithm": "SAMME", "n_estimators": 50}, random_state=seed)
    ExtraTreesClassifierModel = ExtraTreesClassifier(**{"criterion": "entropy", "n_estimators": 20}, random_state=seed)

    estimators = [
        ('DecisionTreeClassifierModel', wrap_bagging(wrap_bagging(DecisionTreeClassifierModel))),
        ('AdaBoostClassifierModel', wrap_bagging(AdaBoostClassifierModel)),
        ('ExtraTreesClassifierModel', wrap_bagging(wrap_bagging(ExtraTreesClassifierModel))),
        ('RandomForestClassifierModel', wrap_bagging(RandomForestClassifierModel)),
        ('GradientBoostingClassifierModel', wrap_bagging(wrap_bagging(GradientBoostingClassifierModel))),
        ('LogisticRegressionModel', wrap_bagging(wrap_bagging(LogisticRegressionModel))),
    ]

    predictions = {}

    for estimator in estimators:
        estimator[1].fit(all_data[0:survived_data.shape[0]], survived_data)
        predictions[estimator[0]] = estimator[1].predict(all_data[survived_data.shape[0]:])

    pred = pd.concat([all_data[survived_data.shape[0]:], pd.DataFrame(data=predictions, index=all_data[survived_data.shape[0]:].index)], axis=1)
    ppp = gbm.predict(pred)

    evaluation = titanic_test_data[['PassengerId']].copy()
    evaluation["Survived"] = ppp.astype(int)
    evaluation.to_csv(r"prediction/voting.csv", index=False)

    # results = model_selection.cross_val_score(gbm, pred, test_y, cv=kfold)
    # print(results.mean())

    #predictions = gbm.predict(pred)
    #print(mean_absolute_error(test_y, predictions))

    #
    # results = model_selection.cross_val_score(gbm, all_data[0:survived_data.shape[0]], survived_data, cv=kfold)
    # print(results.mean())

    # print(list(map(lambda estimator: model_selection.cross_val_score(estimator[1], all_data[0:survived_data.shape[0]], survived_data, cv=kfold).mean(), estimators)))
    # print(list(map(lambda estimator: model_selection.cross_val_score(wrap_bagging(estimator[1]), all_data[0:survived_data.shape[0]], survived_data, cv=kfold).mean(), estimators)))


    # ensemble3 = VotingClassifier(estimators3, voting='soft')
    #
    # results = model_selection.cross_val_score(ensemble3, all_data[0:survived_data.shape[0]], survived_data, cv=kfold)
    # print(results.mean())

    # BaggingEnsemble = BaggingClassifier(base_estimator=ensemble, n_estimators=10, random_state=seed)
    #
    # results = model_selection.cross_val_score(BaggingEnsemble, all_data[0:survived_data.shape[0]], survived_data, cv=kfold)
    # print(results.mean())

    # BaggingEnsemble.fit(all_data[0:survived_data.shape[0]], survived_data)
    # predictions = BaggingEnsemble.predict(all_data[survived_data.shape[0]:])
    #
    # evaluation = titanic_test_data[['PassengerId']].copy()
    # evaluation["Survived"] = predictions.astype(int)
    # evaluation.to_csv(r"prediction/voting.csv", index=False)

