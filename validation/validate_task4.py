import numpy as np
import pandas as pd

DIST_PATH = "../distance_matrix.csv"

df = pd.read_csv(DIST_PATH)

if not np.issubdtype(df.dtypes.iloc[0], np.number):
    df = df.drop(df.columns[0], axis=1)

M = df.to_numpy(dtype=float)

n = M.shape[0]
diag_max = float(np.max(np.abs(np.diag(M))))
sym_max = float(np.max(np.abs(M - M.T)))
min_val = float(np.min(M))
max_val = float(np.max(M))
nan_count = int(np.isnan(M).sum())
inf_count = int(np.isinf(M).sum())

print("N (products):", n)
print("Diagonal max abs:", diag_max)
print("Symmetry max abs:", sym_max)
print("Min value:", min_val)
print("Max value:", max_val)
print("NaN count:", nan_count)
print("Inf count:", inf_count)