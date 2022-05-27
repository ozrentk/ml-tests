import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

def to_timestamp(col):
    col.loc

def delete_fraction(col, **kwargs):
    fraction = kwargs['fraction']
    columns = kwargs['columns']
    if(col.name in columns):
        col.loc[col.sample(frac=fraction).index] = np.nan
    return col

input_filename = "titanic.csv"
start_filename = "titanic-start.csv"
nullified_filename = "titanic-nullified.csv"
imputed_filename = "titanic-imputed.csv"

cwd = os.getcwd()
start_path = os.path.join(cwd, start_filename)
nullified_path = os.path.join(cwd, nullified_filename)
imputed_path = os.path.join(cwd, imputed_filename)

# Import the data
columns = ['age','sex','sibsp','parch','fare']
columns_encoded = ['age','sex_female','sex_male','sibsp','parch','fare']
columns_target = ['survived']
df = pd.read_csv(input_filename, header=0)
print(list(df.columns.values))
#df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# df['Date'] = df['Date'].apply(
#     lambda date: 
#         pd.Timestamp(datetime.strptime(date, "%d/%m/%Y")))

X = pd.get_dummies(df[columns]).astype(float)
X = X[columns_encoded]
X.to_csv(path_or_buf=start_filename)
#print(df.dtypes)
print(list(X.columns.values))

y = df[columns_target]

X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.2, random_state=19761210)

delete_fraction_num = 0

if delete_fraction_num > 0:
    # Delete values at random
    X_t.apply(delete_fraction, axis=0, fraction=delete_fraction_num, columns=columns_encoded)
    #df.apply(delete_fraction, axis=0, fraction=0.02, columns=['Date'])
    X_t.to_csv(path_or_buf=nullified_path)
    #print(df.dtypes)
    print(list(df.columns.values))

# Impute
imp = IterativeImputer(
    max_iter=15, 
    random_state=0, 
    verbose=1, 
    sample_posterior=True)
imputed_array = imp.fit_transform(X_t[columns_encoded])
X_t = pd.DataFrame(imputed_array, columns=columns_encoded)
X_t.to_csv(path_or_buf=imputed_path)
#print(df.dtypes)
print(list(X_t.columns.values))

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=10197612)

model.fit(X_t, y_t.values.ravel())

score = accuracy_score(y_v, model.predict(X_v[columns_encoded]))
print("Accuracy score: ", score)
