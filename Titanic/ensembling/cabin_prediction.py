import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(data, drop_na=True):
    result = data.drop(['PassengerId', 'Name', 'Fare'], axis=1)

    if drop_na:
        result = result.dropna(axis=0)

    result.Sex.replace(["female", "male"], [0, 1], inplace=True)
    result.Embarked.replace(["Q", "C", "S"], [0, 1, 2], inplace=True)

    return result


def init(data):
    processed_data = preprocess_data(data)
    model = RandomForestRegressor(n_estimators=300, max_leaf_nodes=3000, random_state=0)
    model.fit(processed_data.drop(["Cabin"], axis=1), processed_data['Cabin'])

    return model


def correct_cabin(data):
    if "Cabin" in data:
        data.loc[:, 'Cabin'] = data['Cabin'].map(lambda x: x if pd.isnull(x) else x[0])
        data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'], [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

    model = init(data)

    for index, row in data.iterrows():
        if pd.isnull(row['Cabin']):
            data.loc[index, 'Cabin'] = model.predict(preprocess_data(data.iloc[[index]].drop(["Cabin"], axis=1), False)).round()

    return data

