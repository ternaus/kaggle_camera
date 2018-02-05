from pathlib import Path
import cv2
from tqdm import tqdm

from joblib import Parallel, delayed


def helper(file_name):
    img = cv2.imread(str(file_name))

    h, w, c = img.shape
    h_step = int(h / 3)
    w_step = int(w / 3)

    for h in range(0, h - h_step, h_step):
        for w in range(0, w - w_step, w_step):
            im = img[h:h + h_step, w:w + w_step, :]

            cv2.imwrite(
                str(new_flickr_path / class_name.name / (str(h) + '_' + str(w) + '_' + file_name.stem + '.jpg')), im,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


data_path = Path('data')

flickr_path = data_path / 'new_flickr'

new_flickr_path = data_path / 'new_flickr2'
new_flickr_path.mkdir(exist_ok=True)


train_path = data_path / 'train'

new_train_path = data_path / 'train2'
new_train_path.mkdir(exist_ok=True)


for class_name in tqdm(list(train_path.glob('*'))):
    (new_train_path / class_name.name).mkdir(exist_ok=True)
    Parallel(n_jobs=12)(delayed(helper)(file_name) for file_name in list(class_name.glob('*')))


