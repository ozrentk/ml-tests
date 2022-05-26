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
nullified_filename = "customers-nullified.csv"

cwd = os.getcwd()
nullified_path = os.path.join(cwd, nullified_filename)

# Import the data
columns = ['Loyalty Reward Points','Segment','Date','Fraction']
columns_dummies = ['Loyalty Reward Points','Segment_Consumer','Segment_Corporate','Segment_Home Office','Date','Fraction']
df = pd.read_csv(input_filename, header=0)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
# df['Date'] = df['Date'].apply(
#     lambda date: 
#         pd.Timestamp(datetime.strptime(date, "%d/%m/%Y")))

df_dum = pd.get_dummies(df[columns])
#print(df_dum.dtypes)

# Delete values at random
df_dum.apply(delete_fraction, axis=0, fraction=0.05, columns=['Loyalty Reward Points', 'Fraction'])
df_dum.apply(delete_fraction, axis=0, fraction=0.02, columns=['Date'])

# Check deleted values
df_dum.to_csv(path_or_buf=nullified_path)

# Impute
imp = IterativeImputer(
    max_iter=15, 
    random_state=0, 
    verbose=1, 
    sample_posterior=True)
df_imp_arr = imp.fit_transform(df_dum[columns_dummies])
df_imp = pd.DataFrame(df_imp_arr, columns=columns_dummies)
