import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Imputer

titanic_train_data = pd.read_csv(r"data/train.csv")
titanic_test_data = pd.read_csv(r"data/test.csv")


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

    processed_data.loc[:, 'FamilySizeD'] = 0
    processed_data.loc[processed_data['FamilySize'] > 1, 'FamilySizeD'] = 1
    processed_data.loc[processed_data['FamilySize'] > 4, 'FamilySizeD'] = 2

    processed_data = processed_data.drop(['FamilySize', 'PassengerId'], axis=1)

    freq_port = processed_data.Embarked.dropna().mode()[0]
    processed_data.loc[:, 'Embarked'] = processed_data['Embarked'].fillna(freq_port)
    processed_data.loc[:, 'Embarked'] = processed_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    def dummy_data(data, columns):
        for column in columns:
            data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
            data = data.drop(column, axis=1)
        return data

    processed_data = dummy_data(processed_data, ["Pclass", 'Embarked', 'Cabin', 'FamilySizeD', 'Title'])

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

    #processed_data = normalize_age(processed_data)

    return processed_data


def split_valid_test_data(data, fraction=(1 - 0.9)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    return train_test_split(data_x, data_y, test_size=fraction)


def get_train_data(fraction):
    return split_valid_test_data(preprocessing(titanic_train_data), fraction)


def get_test_data():
    return preprocessing(titanic_test_data)

