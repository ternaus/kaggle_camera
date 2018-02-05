from pathlib import Path
import cv2
from tqdm import tqdm

data_path = Path('data')

flickr_path = data_path / 'new_flickr'

new_flickr_path = data_path / 'new_flick2'
new_flickr_path.mkdir(exist_ok=True)

from joblib import Parallel, delayed


def helper(file_name):
    img = cv2.imread(str(file_name))

    h, w, c = img.shape
    h_step = int(h / 3)
    w_step = int(w / 3)

    for h in range(0, h, h_step):
        for w in range(0, w, w_step):
            im = img[h:h + h_step, w:w + w_step, :]

            cv2.imwrite(
                str(new_flickr_path / class_name.name / (str(h) + '_' + str(w) + '_' + file_name.stem + '.jpg')), im,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


for class_name in tqdm(list(flickr_path.glob('*'))):
    (new_flickr_path / class_name.name).mkdir(exist_ok=True)
    Parallel(n_jobs=12)(delayed(helper)(file_name) for file_name in list(class_name.glob('*')))