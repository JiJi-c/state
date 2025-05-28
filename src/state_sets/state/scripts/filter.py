import os
from pathlib import Path
import pandas as pd
import h5py as h5
data_path = '/large_experiments/goodarzilab/mohsen/cellxgene/processed'
files = [f.name for f in Path(data_path).iterdir() if f.is_file()]
df = pd.read_csv('/checkpoint/ctc/ML/uce/h5ad_train_dataset.csv')
for file in df['path'].tolist():
    h5f = h5.File(os.path.join(data_path, file))
    try:
        indptrs = h5f["/X/indptr"]
        if ((df.loc[df['path'] == file]['num_cells']).values[0]) != (indptrs.shape[0] - 1):
            print(file, (indptrs.shape[0] - 1), (df.loc[df['path'] == file]['num_cells']).values[0])
    except KeyError as kex:
        # print(kex, file)
        pass