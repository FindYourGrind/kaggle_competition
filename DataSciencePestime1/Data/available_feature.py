import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

print(titanic_data.columns.values)
