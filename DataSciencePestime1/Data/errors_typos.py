import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
Name feature may contain errors or typos as there are several ways used to describe 
a name including titles, round brackets, and quotes used for alternative or short names.
These will require correcting.
    Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
    Cabin > Age are incomplete in case of test dataset.

'''
print('Tail: ', titanic_data.tail())
