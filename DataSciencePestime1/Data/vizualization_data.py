import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_data = pd.read_csv(r'data/titanic.csv')

print('### Pclass to Survived ###')
print(titanic_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

print('### Sex to Survived ###')
print(titanic_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

print('### SibSp to Survived ###')
print(titanic_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

print('### Parch to Survived ###')
print(titanic_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('\n')

''' Correlating numerical features '''
g = sns.FacetGrid(titanic_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)

''' Correlating numerical and ordinal features '''
grid1 = sns.FacetGrid(titanic_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid1.map(plt.hist, 'Age', alpha=.5, bins=20)
grid1.add_legend();

''' Correlating numerical and ordinal features '''
grid2 = sns.FacetGrid(titanic_data, row='Embarked', size=2.2, aspect=1.6)
grid2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid2.add_legend()

plt.show()