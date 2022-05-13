'''
-----------------------------------------------------------------------------------------------
Features
-----------------------------------------------------------------------------------------------
Survived: Outcome of survival (0 = No; 1 = Yes)
Pclass: Socio-economic class (1 = Upper class; 2 = Middle class; 3 = Lower class)
Name: Name of passenger
Sex: Sex of the passenger
Age: Age of the passenger (Some entries contain NaN)
SibSp: Number of siblings and spouses of the passenger aboard
Parch: Number of parents and children of the passenger aboard
Ticket: Ticket number of the passenger
Fare: Fare paid by the passenger
Cabin Cabin number of the passenger (Some entries contain NaN)
Embarked: Port of embarkation of the passenger (C = Cherbourg; Q = Queenstown; S = Southampton)
-----------------------------------------------------------------------------------------------
'''

import numpy as np 
import pandas as pd 

from sklearn import linear_model
from sklearn import tree
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
# from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from joblib import Memory
import time

import matplotlib.pyplot as plt 
import seaborn as sns

memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)
bunch = fetch_openml_cached("titanic", version=1, as_frame=True)
df = bunch.frame

# print(f"DataFrame shape : {bunch.frame.shape}\n=================================")
# print(f"DataFrame info : {bunch.frame.info()}\n=================================")
# print(f"DataFrame columns : {bunch.frame.columns}\n=================================")
# print(f"The type of each column : {bunch.frame.dtypes}\n=================================")
# print(f"How much missing value in every column : {bunch.frame.isna().sum()}\n=================================")

# Drop rows with missing values
# df = df.dropna(subset=['pclass','age','sibsp','parch','fare','sex'])
#print(bunch.frame.to_string())

# Form data (X) and target (y) frame
# Column 'sex' will break down to 'sex_male' and 'sex_female'
X = pd.get_dummies(df[['pclass','age','sibsp','parch','fare','sex']])
# print(X.to_string())
X.to_csv(path_or_buf='C:/Programiranje/ml-tests/titanic-not-imputed.csv')

# Impute missing values
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# No early stopping: sample_posterior=True
imp = IterativeImputer(
    max_iter=15, 
    random_state=0, 
    verbose=1, 
    sample_posterior=True)
#print(X.dtypes)
df_imputed_arr=imp.fit_transform(df[['age','sibsp','parch','fare','survived']])
#X_arr = imp.fit_transform(X)
df_imputed = pd.DataFrame(df_imputed_arr, columns=['age','sibsp','parch','fare','survived'])
X['age']=df_imputed['age']
X['fare']=df_imputed['fare']
X['sex_female'] = X['sex_female'].astype('int8')
X['sex_male'] = X['sex_male'].astype('int8')

X['age'] = np.select(
    [df_imputed.age < 0, df_imputed.age <= 1, df_imputed.age > 1],
    [0, df_imputed.age, round(df_imputed.age*2)/2]
)

#X['age'] = X['age'].round(decimals=0)
#print(X.dtypes)

# Drop other NaN's
#X = X.dropna(subset=['pclass','age','sibsp','parch','fare','sex_female','sex_male'])

# Validate!
if np.any(np.isnan(X)):
    print("There are NaN's")
    print(X.isna().sum())
    exit()
elif np.any(np.isinf(X)):
    print("There are infinite vaues")
    exit()


X.to_csv(path_or_buf='C:/Programiranje/ml-tests/titanic-imputed.csv')

y = df[['survived']]
# print(y.to_string())

#bunch.frame.to_csv(path_or_buf='C:/Work/Projects/ml-tests/titanic.csv')

# Analisys by male/female
lbd_sex = lambda df, sex: df.sex == sex
lbd_survived = lambda df, survived: df.survived == survived
women_list = df.loc[lbd_sex(df, 'female')]
women_survived_list = women_list.loc[lbd_survived(df, '1')]
women_rate = len(women_survived_list)/len(women_list)
men_list = df.loc[lbd_sex(df, 'male')]
men_survived_list = men_list.loc[lbd_survived(df, '1')]
men_rate = len(men_survived_list)/len(men_list)
print(women_rate, men_rate)

# Split training / validation
X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=19761210)
# print(X_t.to_string())
# print(y_t.to_string())
# print(X_v.to_string())
# print(y_v.to_string())

# Create classifier
#model = tree.DecisionTreeClassifier(random_state=1, max_depth=2)
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=10197612)

# Train model
model.fit(X_t, y_t.values.ravel())

# See if prediction fits
score = accuracy_score(y_v, model.predict(X_v[['pclass','age','sibsp','parch','fare','sex_female','sex_male']]))
print("Accuracy score: ", score)

total_folds = 3
kfold = StratifiedKFold(n_splits=total_folds, random_state=10121976, shuffle=True)
model_skf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=10197612)
for train_indices, test_indices in kfold.split(X, y):
    X_t = X.iloc[train_indices,:]
    X_v = X.iloc[test_indices,:]
    y_t = y.iloc[train_indices,:]
    y_v = y.iloc[test_indices,:]

    model_skf.fit(X_t, y_t.values.ravel())
    score = model_skf.score(X_v, y_v)
    total_folds = total_folds - 1
    print(f'score = {score}, {total_folds} to go...')
