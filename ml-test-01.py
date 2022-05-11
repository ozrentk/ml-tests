import numpy as np 
import pandas as pd 

from sklearn import linear_model
from sklearn import tree
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

from joblib import Memory
import time

import matplotlib.pyplot as plt 
import seaborn as sns

memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)
bunch = fetch_openml_cached("titanic", version=1, as_frame=True)

# print(f"DataFrame shape : {bunch.frame.shape}\n=================================")
# print(f"DataFrame info : {bunch.frame.info()}\n=================================")
# print(f"DataFrame columns : {bunch.frame.columns}\n=================================")
# print(f"The type of each column : {bunch.frame.dtypes}\n=================================")
# print(f"How much missing value in every column : {bunch.frame.isna().sum()}\n=================================")

# Drop rows with missing values
bunch.frame = bunch.frame.dropna(subset=['age','sibsp','parch','fare'])
#print(bunch.frame.to_string())

# Form data (X) and target (y) frame
X = bunch.frame[['age','sibsp','parch','fare']]
y = bunch.frame[['survived']]
# print(X.to_string())
# print(y.to_string())

# Split training / validation
X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=19761210)
# print(X_t.to_string())
# print(y_t.to_string())
# print(X_v.to_string())
# print(y_v.to_string())

# Create classifier
#model = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
model = RandomForestClassifier(random_state=10197612)

# Train model
model.fit(X, y)

# See if prediction fits
score = accuracy_score(y_t, model.predict(X_t[['age','sibsp','parch','fare']]))
print("Accuracy score: ", score)

#kfold = StratifiedKFold(n_splits=10, random_state=10121976, shuffle=True)
# model = RandomForestClassifier(random_state=10197612)

# for train_index, test_index in kfold.split(X_, y):
#     model.fit(X_t[], y_t)
#     print("TRAIN:", train_index, "TEST:", test_index)
