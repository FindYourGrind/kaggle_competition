import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
Which features are numerical? These values change from sample to sample. 
Within numerical features are the values discrete, continuous, or timeseries based? 
Among other things this helps us select the appropriate plots for visualization.
Continous: Age, Fare. Discrete: SibSp, Parch.
'''
print('Continous: ', titanic_data[['Age', 'Fare']])
print('Discrete: ', titanic_data[['SibSp', 'Parch']])
