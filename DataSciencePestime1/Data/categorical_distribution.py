import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
Names are unique across the dataset (count=unique=891)
Sex variable as two possible values with 65% male (top=male, freq=577/count=891).
Cabin values have several dupicates across samples. Alternatively several passengers shared a cabin.
Embarked takes three possible values. S port used by most passengers (top=S)
Ticket feature has high ratio (22%) of duplicate values (unique=681).
'''
print(titanic_data.describe(include=['O']))
