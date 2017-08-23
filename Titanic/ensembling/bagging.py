from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import age_prediction as ap
import titanic_preprocessor as tp
import pandas as pd

titanic_train_data = pd.read_csv(r"data/train.csv").drop(['Ticket'], axis=1)
titanic_test_data = pd.read_csv(r"data/test.csv").drop(['Ticket'], axis=1)
all_data = titanic_train_data.append(titanic_test_data).drop(['Survived'], axis=1)
survived_data = titanic_train_data['Survived']

freq_port = all_data.Embarked.dropna().mode()[0]
all_data.loc[:, 'Embarked'] = all_data['Embarked'].fillna(freq_port)

all_data = ap.correct_age(all_data)

all_data.loc[:, 'Cabin'] = all_data['Cabin'].map(lambda x: 'U' if pd.isnull(x) else x[0])
all_data.Cabin.replace(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'], [0, 0, 0, 0, 0, 0, 0, 0, 1], inplace=True)

all_data = tp.preprocessing(all_data)
to_predict = all_data[survived_data.shape[0]:]

train_x, test_x, train_y, test_y = train_test_split(all_data[0:survived_data.shape[0]], survived_data, test_size=0.4)


seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
parameters = {
    "criterion": ("gini", "entropy"),
    "splitter": ("best", "random"),
    "max_depth": [None, 1, 2, 4, 6, 7, 8],
    "min_samples_split": [2, 3, 4, 6],
    "min_samples_leaf": [1, 2, 6, 10, 12],
    "max_features": [1, 3, 5, 10, 15, "auto", "sqrt", "log2", None],
    "max_leaf_nodes": [None, 25, 50, 100, 500, 2000, 5000]
}
#cart = DecisionTreeClassifier()

#clf = GridSearchCV(cart, parameters, verbose=20)
#clf.fit(all_data[0:survived_data.shape[0]], survived_data)

cart = DecisionTreeClassifier(max_features=10, min_samples_leaf=6, criterion='gini', max_depth=6, max_leaf_nodes=50, splitter='best', min_samples_split=3)

num_trees = 300
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, all_data[0:survived_data.shape[0]], survived_data, cv=kfold)
print(results.mean())

model.fit(all_data[0:survived_data.shape[0]], survived_data)
predictions = model.predict(all_data[survived_data.shape[0]:])
evaluation = titanic_test_data[['PassengerId']].copy()
evaluation["Survived"] = predictions.astype(int)
evaluation.to_csv(r"prediction/bagging.csv", index=False)