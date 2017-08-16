import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

titanic_train_data = pd.read_csv(r"data/train.csv")
titanic_test_data = pd.read_csv(r"data/test.csv")

titanic_train_data.loc[:, 'SecondName'] = titanic_train_data.Name.str.extract('^([A-Za-z ]+),', expand=False)
print(titanic_train_data[['SecondName', 'Survived']].groupby(['SecondName'], as_index=False).agg(['mean', 'count']))
titanic_train_data = titanic_train_data.drop(['SecondName'], axis=1)

def preprocessing(data):
    processed_data = data

    def nan_padding(data, columns):
        for column in columns:
            imputer = Imputer()
            data.loc[:, column] = imputer.fit_transform(data[column].values.reshape(-1, 1))
        return data

    processed_data = nan_padding(processed_data, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])

    processed_data.loc[:, 'Title'] = processed_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    pd.crosstab(processed_data['Title'], processed_data['Sex'])
    processed_data.loc[:, 'Title'] = processed_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    processed_data.loc[:, 'Title'] = processed_data['Title'].replace('Mlle', 'Miss')
    processed_data.loc[:, 'Title'] = processed_data['Title'].replace('Ms', 'Miss')
    processed_data.loc[:, 'Title'] = processed_data['Title'].replace('Mme', 'Mrs')
    processed_data = processed_data.drop(['Name'], axis=1)

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    processed_data.loc[:, 'Title'] = processed_data['Title'].map(title_mapping)
    processed_data.loc[:, 'Title'] = processed_data['Title'].fillna(0)

    processed_data.loc[:, 'FamilySize'] = processed_data['SibSp'] + processed_data['Parch'] + 1
    processed_data.loc[:, 'IsAlone'] = 0
    processed_data.loc[processed_data['FamilySize'] == 1, 'IsAlone'] = 1
    processed_data = processed_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    processed_data['AgeBand'] = pd.cut(processed_data['Age'], 5)

    processed_data.loc[ processed_data['Age'] <= 16, 'Age'] = 0
    processed_data.loc[(processed_data['Age'] > 16) & (processed_data['Age'] <= 32), 'Age'] = 1
    processed_data.loc[(processed_data['Age'] > 32) & (processed_data['Age'] <= 48), 'Age'] = 2
    processed_data.loc[(processed_data['Age'] > 48) & (processed_data['Age'] <= 64), 'Age'] = 3
    processed_data.loc[ processed_data['Age'] > 64, 'Age']

    processed_data = processed_data.drop(['AgeBand'], axis=1)

    freq_port = processed_data.Embarked.dropna().mode()[0]
    processed_data.loc[:, 'Embarked'] = processed_data['Embarked'].fillna(freq_port)
    processed_data.loc[:, 'Embarked'] = processed_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    def dummy_data(data, columns):
        for column in columns:
            data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
            data = data.drop(column, axis=1)
        return data

    processed_data = dummy_data(processed_data, ["Pclass"])

    def sex_to_int(data):
        le = LabelEncoder()
        le.fit(["male", "female"])
        data.loc[:, "Sex"] = le.transform(data["Sex"])
        return data

    processed_data = sex_to_int(processed_data)

    def normalize_age(data):
        scaler = MinMaxScaler()
        data.loc[:, "Age"] = scaler.fit_transform(data["Age"].values.reshape(-1, 1))
        return data

    processed_data = normalize_age(processed_data)

    return processed_data


def split_valid_test_data(data, fraction=(1 - 0.85)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    return train_test_split(data_x, data_y, test_size=fraction)


processed_train_data = titanic_train_data[['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
input_x, input_y, output_x, output_y = split_valid_test_data(preprocessing(processed_train_data))
output_x = output_x.reshape((output_x.size,))

titanic_forestclass_model = RandomForestClassifier(n_estimators=100, max_leaf_nodes=1000, random_state=0)


input_x = np.concatenate((input_x, input_x))
output_x = np.concatenate((output_x, output_x))

titanic_forestclass_model.fit(input_x, output_x)


prediction = titanic_forestclass_model.predict(input_y)
print('RandomForestClassifier accuracy: ', round(titanic_forestclass_model.score(input_y, output_y) * 100, 2))
print('RandomForestClassifier error: ', mean_absolute_error(output_y, prediction))
print('\n')


processed_test_data = titanic_test_data[['Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
input_y = preprocessing(processed_test_data)

##
# titanic_tree_model = DecisionTreeRegressor()
# titanic_tree_model.fit(input_x, output_x)
# prediction = titanic_tree_model.predict(input_y)


#print('DecisionTreeRegressor accuracy: ', round(titanic_tree_model.score(input_y, output_y) * 100, 2))
# print('DecisionTreeRegressor error: ', mean_absolute_error(output_y, prediction))
# print('\n')

##
# titanic_logreg_model = LogisticRegression()
# titanic_logreg_model.fit(input_x, output_x)
# prediction = titanic_logreg_model.predict(input_y)
#
# evaluation = titanic_test_data[['PassengerId']].copy()
# evaluation["Survived"] = prediction.astype(int)
# evaluation.to_csv(r"prediction/titanic_evaluation_submission_logistic_regression.csv", index=False)

# print('LogisticRegression accuracy: ', round(titanic_logreg_model.score(input_y, output_y) * 100, 2))
# print('LogisticRegression error: ', mean_absolute_error(output_y, prediction))
# print('\n')

##
# titanic_svc_model = SVC()
# titanic_svc_model.fit(input_x, output_x)
# prediction = titanic_svc_model.predict(input_y)
#
# evaluation = titanic_test_data[['PassengerId']].copy()
# evaluation["Survived"] = prediction.astype(int)
# evaluation.to_csv(r"prediction/titanic_evaluation_submission_svc.csv", index=False)

# print('SVC accuracy: ', round(titanic_svc_model.score(input_y, output_y) * 100, 2))
# print('SVC error: ', mean_absolute_error(output_y, prediction))
# print('\n')

##
# titanic_knn_model = KNeighborsClassifier(n_neighbors=3)
# titanic_knn_model.fit(input_x, output_x)
# prediction = titanic_knn_model.predict(input_y)
#
# evaluation = titanic_test_data[['PassengerId']].copy()
# evaluation["Survived"] = prediction.astype(int)
# evaluation.to_csv(r"prediction/titanic_evaluation_submission_knn.csv", index=False)

# print('K-NN accuracy: ', round(titanic_knn_model.score(input_y, output_y) * 100, 2))
# print('K-NN error: ', mean_absolute_error(output_y, prediction))
# print('\n')


##
# titanic_forestclass_model = RandomForestClassifier(n_estimators=100)
# titanic_forestclass_model.fit(input_x, output_x)
prediction = titanic_forestclass_model.predict(input_y)

evaluation = titanic_test_data[['PassengerId']].copy()
evaluation["Survived"] = prediction.astype(int)

#print('RandomForestClassifier accuracy: ', round(titanic_forestclass_model.score(input_y, output_y) * 100, 2))
#print('RandomForestClassifier error: ', mean_absolute_error(output_y, prediction))
#print('\n')

evaluation.to_csv(r"prediction/titanic_evaluation_submission_random_forest_classifier.csv", index=False)
print('\n')
print('printed')
'''
# Tree
processed_train_data = titanic_train_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp']]
#processed_train_data = processed_train_data.dropna(axis=0, subset=['Survived', 'Pclass', 'Age', 'SibSp'])

data_to_train = preprocessing(processed_train_data)
input_data_x = data_to_train.drop(["Survived"], axis=1)
output_data_x = LabelBinarizer().fit_transform(data_to_train["Survived"])


processed_test_data = titanic_test_data[['Pclass', 'Sex', 'Age', 'SibSp']]
#processed_test_data = processed_test_data.dropna(axis=0, subset=['Pclass', 'Age', 'SibSp'])

data_to_test = preprocessing(processed_test_data)


titanic_tree_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=0)
titanic_tree_model.fit(input_data_x, output_data_x)
tree_prediction = titanic_tree_model.predict(data_to_test)

evaluation = titanic_test_data[['PassengerId']].copy()
evaluation["Survived"] = tree_prediction.astype(int)
print(evaluation)
evaluation.to_csv("./prediction/titanic_evaluation_submission.csv", index=False)
'''

'''
# Forest
titanic_forest_model = RandomForestRegressor(max_leaf_nodes=50, random_state=0)
titanic_forest_model.fit(input_x, output_x.reshape((712,)))
forest_prediction = titanic_forest_model.predict(input_y)

print(mean_absolute_error(output_y, forest_prediction))
'''

'''
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, input_x, input_y, output_x, output_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
'''
