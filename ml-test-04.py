import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def to_timestamp(col):
    col.loc

def delete_fraction(col, **kwargs):
    fraction = kwargs['fraction']
    columns = kwargs['columns']
    if(col.name in columns):
        col.loc[col.sample(frac=fraction).index] = np.nan
    return col

input_filename = "customers.csv"
start_filename = "customers-start.csv"
nullified_filename = "customers-nullified.csv"
imputed_filename = "customers-imputed.csv"

cwd = os.getcwd()
start_path = os.path.join(cwd, start_filename)
nullified_path = os.path.join(cwd, nullified_filename)
imputed_path = os.path.join(cwd, imputed_filename)

# Import the data
columns = ['Loyalty Reward Points','Segment','Fraction']
columns_encoded = ['Loyalty Reward Points','Segment_Consumer','Segment_Corporate','Segment_Home Office','Fraction']
df = pd.read_csv(input_filename, header=0)
print(list(df.columns.values))
#df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# df['Date'] = df['Date'].apply(
#     lambda date: 
#         pd.Timestamp(datetime.strptime(date, "%d/%m/%Y")))

df = pd.get_dummies(df[columns]).astype(float)
df = df[columns_encoded]
df.to_csv(path_or_buf=start_filename)
#print(df.dtypes)
print(list(df.columns.values))

# Delete values at random
df.apply(delete_fraction, axis=0, fraction=0.05, columns=['Loyalty Reward Points', 'Fraction'])
#df.apply(delete_fraction, axis=0, fraction=0.02, columns=['Date'])
df.to_csv(path_or_buf=nullified_path)
#print(df.dtypes)
print(list(df.columns.values))

# Impute
imp = IterativeImputer(
    max_iter=15, 
    random_state=0, 
    verbose=1, 
    sample_posterior=True)
imputed_array = imp.fit_transform(df[columns_encoded])
df = pd.DataFrame(imputed_array, columns=columns_encoded)
df.to_csv(path_or_buf=imputed_path)
#print(df.dtypes)
print(list(df.columns.values))
