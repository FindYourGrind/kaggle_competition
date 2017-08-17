import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(data, drop_na=True):
    result = data.drop(['PassengerId', 'Cabin', 'Name', 'Ticket', 'Fare'], axis=1)

    if 'Survived' in result:
        result = data.drop(['Survived'], axis=1)

    if drop_na:
        result = result.dropna(axis=0)

    result.replace(["female", "male"], [0, 1], inplace=True)
    result.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)

    return result


def init (data):
    processed_data = preprocess_data(data)
    model = RandomForestRegressor(n_estimators=300, max_leaf_nodes=3000, random_state=0)
    model.fit(processed_data.drop(["Age"], axis=1), processed_data['Age'])

    return model


def correct_age(data):
    model = init(data)

    for index, row in data.iterrows():
        if pd.isnull(row['Age']):
            data.loc[index, 'Age'] = model.predict(preprocess_data(data.iloc[[index]].drop(["Age"], axis=1), False))

    return data

