import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
Numerical, alphanumeric data within same feature.
Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
'''
print('Mix of numeric and alphanumeric: ', titanic_data[['Ticket']])
print('Alphanumeric: ', titanic_data[['Cabin']])
