import os
import math
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn import datasets

# config
n_feat = 8
n_info = 3
n_samples = 100
f_noise = 0
n_seed = 27455
b_show_corr = True
s_output_data_file = "synthetic-data"

# data synth
X, y = datasets.make_regression(
    n_samples=n_samples,
    n_features=n_feat, 
    n_informative=n_info, 
    noise=f_noise,
    bias=100,
    random_state=n_seed)

# data: features
reg_df = pd.DataFrame(X, columns=['ft-%i' % i for i in range(n_feat)])
print(reg_df.columns.values)

# data: target
tgt_df = pd.DataFrame(y, columns=['tgt'])
print(tgt_df.columns.values)

# data: dataframe concat
df = pd.concat([reg_df, tgt_df], axis=1)
print(df.columns.values)

# output to csv?
if s_output_data_file:
    cwd = os.getcwd()
    out_path = os.path.join(cwd, f"{s_output_data_file}.csv")
    df.to_csv(out_path, index=False)

# plot setup
n_cell = n_feat
if b_show_corr:
    n_cell = n_cell + 1
n_grid = math.ceil(n_cell ** 0.5)
n_grid_y = n_grid - 1 if n_cell <= n_grid * (n_grid - 1) else n_grid
fig, ax = plt.subplots(n_grid_y, n_grid)

# plot
y_list = df['tgt'].tolist()
for i in range(n_cell):
    cell_x = i // n_grid
    cell_y = i % n_grid
    cell = ax[cell_x][cell_y]
    if b_show_corr and i >=  n_feat:
        tgt_corr_vct = df.corrwith(df['tgt'])
        tgt_corr_vct = np.expand_dims(tgt_corr_vct, axis=0)
        cell.matshow(tgt_corr_vct, cmap='hot')
        cell.title.set_text('corr matrix')
    else:        
        ft_name = f'ft-{i}'
        x_list = df[ft_name].tolist()
        cell.plot(x_list, y_list, 'bo')
        cell.title.set_text(ft_name)

# show numeric corr mtx
print(df.corr())

# save plot
#plt.show()
img_path = os.path.join(cwd, f"{s_output_data_file}-stats.png")
plt.savefig(img_path)