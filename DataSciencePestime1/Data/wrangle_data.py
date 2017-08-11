import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

titanic_data = titanic_data.drop(['Ticket', 'Cabin'], axis=1)

print(titanic_data.head())
print('\n')

titanic_data['Title'] = titanic_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(titanic_data['Title'], titanic_data['Sex'])

titanic_data['Title'] = titanic_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
titanic_data['Title'] = titanic_data['Title'].replace('Mlle', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Ms', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')

print(titanic_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
print('\n')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
titanic_data['Title'] = titanic_data['Title'].fillna(0)

print(titanic_data.head())
print('\n')