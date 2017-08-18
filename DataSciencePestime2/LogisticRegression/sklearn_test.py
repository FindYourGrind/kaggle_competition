import age_prediction as ap
import titanic_preprocessor as tp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# read data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

titanic_train_data = pd.read_csv(r"data/train.csv")
titanic_test_data = pd.read_csv(r"data/test.csv")

# correct age
titanic_train_data = ap.correct_age(titanic_train_data)
#titanic_train_data.Age.dropna(axis=0)
titanic_test_data = ap.correct_age(titanic_test_data)

# preprocess data
titanic_train_data = tp.preprocessing(titanic_train_data)
titanic_test_data = tp.preprocessing(titanic_test_data)

# split train data

predictions = np.zeros(titanic_test_data.shape[0], np.uint32)
counter = 0

for i in range(2000):
    counter += 1
    train_x, test_x, train_y, test_y = train_test_split(titanic_train_data.drop(["Survived"], axis=1),
                                                        titanic_train_data["Survived"], test_size=0.1 + (i / 3000))

    titanic_logreg_model = LogisticRegression()
    titanic_logreg_model.fit(train_x, train_y)
    #prediction = titanic_logreg_model.predict(test_x)

    # print('LogisticRegression accuracy: ', round(titanic_logreg_model.score(test_x, test_y) * 100, 2))
    # print('LogisticRegression error: ', mean_absolute_error(test_y, prediction))
    # print('\n')

    prediction = titanic_logreg_model.predict(titanic_test_data)
    predictions = np.add(predictions, prediction)

for i in range(2000):
    counter += 1
    train_x, test_x, train_y, test_y = train_test_split(titanic_train_data.drop(["Survived"], axis=1),
                                                        titanic_train_data["Survived"], test_size=0.1 + (i / 3000))

    titanic_logreg_model = RandomForestClassifier()
    titanic_logreg_model.fit(train_x, train_y)
    #prediction = titanic_logreg_model.predict(test_x)

    # print('LogisticRegression accuracy: ', round(titanic_logreg_model.score(test_x, test_y) * 100, 2))
    # print('LogisticRegression error: ', mean_absolute_error(test_y, prediction))
    # print('\n')

    prediction = titanic_logreg_model.predict(titanic_test_data)
    predictions = np.add(predictions, prediction)


print(predictions)
prediction = np.asarray(list(map(lambda x: 1 if x > (counter * 0.6262) else 0, predictions)))

evaluation = titanic_test_data[['PassengerId']].copy()
evaluation["Survived"] = prediction.astype(int)
evaluation.to_csv(r"prediction/dsp2_logistic_regression.csv", index=False)