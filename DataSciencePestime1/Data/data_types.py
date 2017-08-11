import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
Seven features are integer or floats. Six in case of test dataset.
Five features are strings (object).
'''
titanic_data.info()
