from pathlib import Path
import jpeg4py
import os
from joblib import Parallel, delayed

data_path = Path('data')
train_path = data_path / 'new_flickr'

train_file_names = list(train_path.glob('**/*.*'))


def helper(file_name):
    try:
        x = jpeg4py.JPEG(str(file_name)).decode()
    except:
        os.remove(str(file_name))


Parallel(n_jobs=20)(delayed(helper)(x) for x in train_file_names)

