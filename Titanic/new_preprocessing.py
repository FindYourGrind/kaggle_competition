import age_prediction as ap
import cabin_prediction as cp
import titanic_preprocessor as tp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

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

if test:
    train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.15)
    predictions = np.zeros(test_x.shape[0], np.uint32)
else:
    predictions = np.zeros(titanic_test_data.shape[0], np.uint32)
    test_x = all_data[survived_data.shape[0]:]


for i in range(25):
    if test:
        train_x_inner, test_x_inner, train_y_inner, test_y_inner = train_test_split(train_x, train_y, test_size=0.1 + (i / 2000))
        #train_x_inner, test_x_inner, train_y_inner, test_y_inner = train_test_split(train_x, train_y, test_size=0.15)
    else:
        train_x_inner, test_x_inner, train_y_inner, test_y_inner = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.1 + (i / 30000))

    titanic_logreg_model = LogisticRegression()
    titanic_logreg_model.fit(train_x_inner, train_y_inner)
    prediction1 = titanic_logreg_model.predict(test_x)

    titanic_logreg_model = SVC()
    titanic_logreg_model.fit(train_x_inner, train_y_inner)
    prediction2 = titanic_logreg_model.predict(test_x)

    parameters = {'bootstrap': False, 'min_samples_split': 10, 'max_features': 'auto', 'min_samples_leaf': 3, 'max_depth': 4, 'n_estimators': 10}
    titanic_logreg_model = RandomForestClassifier(**parameters)
    #titanic_logreg_model = RandomForestClassifier()
    titanic_logreg_model.fit(train_x_inner, train_y_inner)
    prediction3 = titanic_logreg_model.predict(test_x)

    titanic_knn_model = KNeighborsClassifier(n_neighbors=3)
    titanic_knn_model.fit(train_x_inner, train_y_inner)
    prediction4 = titanic_knn_model.predict(test_x)

    prediction1 = prediction1 * 0.3
    prediction2 = prediction2 * 0.15
    prediction3 = prediction3 * 0.4
    prediction4 = prediction4 * 0.15
    predictions = np.add(predictions, np.add(prediction4, np.add(prediction3, np.add(prediction2, prediction1))))

predictions = np.asarray(list(map(lambda x: 1 if x > 25 * 0.57 else 0, predictions)))

if test:
    print('error: ', mean_absolute_error(test_y, predictions))
    print('\n')
else:
    evaluation = titanic_test_data[['PassengerId']].copy()
    evaluation["Survived"] = predictions.astype(int)
    evaluation.to_csv(r"prediction/new_preproc.csv", index=False)