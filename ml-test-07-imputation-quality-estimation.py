'''
1. Napraviš cijeli model nad datasetom koji imaš - zapiši performansu na testu
2. Progresivno iz dataseta bacaš podatke, gradiš model i gledaš performansu na testu
3. Nacrtati graf, osi su performansa i količina šuma (ili udio ne-null ćelija)
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def delete_fraction(col, **kwargs):
    fraction = kwargs['fraction']
    columns = kwargs['columns']
    if(col.name in columns):
        col.loc[col.sample(frac=fraction).index] = np.nan
    return col

# config
n_seed = 823594
f_split = 0.2
input_filename = "synthetic-data.csv"
columns_feat = ['ft-1','ft-2','ft-3','ft-6']
columns_tgt = ['tgt']
nullified_filename = "synthetic-data-nullified.csv"
imputed_filename = "synthetic-data-imputed.csv"
accuracy_filename = "synthetic-data-imputation-accuracy.png"

# construct paths
cwd = os.getcwd()
nullified_path = os.path.join(cwd, nullified_filename)
imputed_path = os.path.join(cwd, imputed_filename)
accuracy_path = os.path.join(cwd, accuracy_filename)

# import data - we imply there are no dummies!
df = pd.read_csv(input_filename, header=0)
print(df.dtypes)

X = df[columns_feat]
y = df[columns_tgt]

# split data, train + validation
X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=f_split, random_state=n_seed)

accuracy_X = []
accuracy_Y = []
for f_del_fraction in np.arange(0, 0.55, 0.05):
    f_del_fraction = round(f_del_fraction, 2)
    print(f"Deleting + imputing {f_del_fraction * 100}% of data")

    # random delete
    if f_del_fraction > 0:
        X_t.apply(delete_fraction, axis=0, fraction=f_del_fraction, columns=columns_feat)
        X_t.to_csv(path_or_buf=nullified_path, index=True)

    # impute
    imp = IterativeImputer(
        max_iter=15, 
        random_state=0, 
        verbose=1, 
        sample_posterior=True)
    imputed_array = imp.fit_transform(X_t[columns_feat])
    X_t = pd.DataFrame(imputed_array, columns=columns_feat)
    #X_t.to_csv(path_or_buf=imputed_path, index=True)
    #print(list(X_t.columns.values))

    # build model
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=n_seed)
    model.fit(X_t, y_t.values.ravel())

    # validate model
    score = model.score(X_v, y_v)
    print("Accuracy score: ", score)
    accuracy_X.append(f_del_fraction)
    accuracy_Y.append(score)

plt.xlim([0, 0.5])
plt.ylim([-1.0, 1.0])
plt.plot(accuracy_X, accuracy_Y)
plt.savefig(accuracy_path)