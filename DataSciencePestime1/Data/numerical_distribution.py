import pandas as pd

titanic_data = pd.read_csv(r'data/titanic.csv')

'''
This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.
Total samples are 891 or 40% of the actual number of passengers on board the Titanic (2,224).
Survived is a categorical feature with 0 or 1 values.
Around 38% samples survived representative of the actual survival rate at 32%.
Most passengers (> 75%) did not travel with parents or children.
Nearly 30% of the passengers had siblings and/or spouse aboard.
Fares varied significantly with few passengers (<1%) paying as high as $512.
Few elderly passengers (<1%) within age range 65-80.
'''
print(titanic_data.describe())
