import os
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import datasets

#fig = plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(2, 2)

n_samples = 100
n_feat = 4
n_info = 4
f_noise = 0
f_output_data = False

X, y = datasets.make_regression(
    n_samples=n_samples,
    n_features=n_feat, 
    n_informative=n_info, 
    noise=f_noise,
    bias=100,
    random_state=27455)
reg_df = pd.DataFrame(X, columns=['ft-%i' % i for i in range(n_feat)])
print(reg_df.columns.values)

tgt_df = pd.DataFrame(y, columns=['tgt'])
print(tgt_df.columns.values)

df = pd.concat([reg_df, tgt_df], axis=1)
print(df.columns.values)

if f_output_data:
    cwd = os.getcwd()
    out_path = os.path.join(cwd, "synthetic_data.csv")
    df.to_csv(out_path)

x_c0 = df['ft-0'].tolist()
x_c1 = df['ft-1'].tolist()
x_c2 = df['ft-2'].tolist()
x_c3 = df['ft-3'].tolist()
y_c = df['tgt'].tolist()

print(df.corr())

ax[0][0].plot(x_c0, y_c, 'bo')
ax[0][0].title.set_text('ft-0')
ax[0][1].plot(x_c1, y_c, 'bo')
ax[0][1].title.set_text('ft-1')
ax[1][0].plot(x_c2, y_c, 'bo')
ax[1][0].title.set_text('ft-2')
ax[1][1].plot(x_c3, y_c, 'bo')
ax[1][1].title.set_text('ft-3')

plt.show()

exit()


reg_df['y'] = y


plt.matshow( reg_df.corr(), fignum=fig.number )

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);

plt.show()