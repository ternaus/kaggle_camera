from pathlib import Path
import jpeg4py
import os

from tqdm import tqdm

data_path = Path('data')
train_path = data_path / 'new_flickr'

train_file_names = list(train_path.glob('**/*.*'))

for file_name in tqdm(train_file_names):
    try:
        x = jpeg4py.JPEG(str(file_name)).decode()
    except:
        print(file_name)
        os.remove(str(file_name))
