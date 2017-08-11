import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
These values classify the samples into sets of similar samples. 
Within categorical features are the values nominal, ordinal, ratio, or interval based? 
Among other things this helps us select the appropriate plots for visualization.
Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
'''
print('Categorical: ', titanic_data['Survived', 'Sex', 'Embarked'])
print('Ordinal: ', titanic_data['Pclass'])
