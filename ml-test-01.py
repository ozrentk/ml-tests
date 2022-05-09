from sklearn import datasets
from sklearn.datasets import fetch_openml
from joblib import Memory
import time

time0 = time.time()
memory = Memory('./tmp')
fetch_openml_cached = memory.cache(fetch_openml)
X, y = fetch_openml_cached("titanic", version=1, as_frame=True, return_X_y=True)
print('Time: {:.0f}s'.format(time.time() - time0))

print(X.to_string())