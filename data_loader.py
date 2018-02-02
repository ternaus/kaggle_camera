import numpy as np
import torch.utils.data as data
import torch
import utils
from functools import partial
from PIL import Image
from io import BytesIO
import cv2

num_classes = 10


def jpg_compress(x, quality=None):
    if quality is None:
        quality = np.random.randint(70, 91)
    x = Image.fromarray(x)
    out = BytesIO()
    x.save(out, format='jpeg', quality=quality)
    x = Image.open(out)
    return np.array(x)


def gamma_correction(x, gamma=None):
    if gamma is None:
        gamma = np.random.randint(80, 121) / 100.
    x = x.astype('float32') / 255.
    x = np.power(x, gamma)
    return x * 255


def rescale(img, scale=None):
    result = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if result.shape[0] < 512:
        return img


def augment(x, safe=False):
    if safe:
        augs = (np.fliplr,
                np.flipud,
                partial(np.rot90, k=1),
                partial(np.rot90, k=3),
                None)
    else:
        augs = (
            jpg_compress,
            gamma_correction,
            np.fliplr,
            np.flipud,
            partial(np.rot90, k=1),
            partial(np.rot90, k=3),
            partial(rescale, scale=0.5),
            partial(rescale, scale=0.8),
            partial(rescale, scale=1.5),
            partial(rescale, scale=2.0),
            None)
    f = np.random.choice(augs)

    if f is not None:
        return f(x)
    return x


class CSVDataset(data.Dataset):
    def __init__(self, df, transform=None, mode='train'):
        self.df = df
        self.path = df['file_name'].values.astype(str)
        self.target = df['class_id'].values.astype(np.int64)
        self.is_manip = df['is_manip'].values.astype(int)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = utils.load_image(self.path[idx])

        if X.shape[0] < 512 or X.shape[1] < 512:
            print(self.path[idx])

        if self.mode == 'train':
            # self.X = augment(X, safe=False)
            if self.is_manip[idx] == 1:
                self.X = augment(X, safe=True)
            else:
                self.X = augment(X, safe=False)

        y = self.target[idx]

        return self.transform(X), y


def get_loaders(batch_size,
                args,
                train_df=None,
                valid_df=None,
                train_transform=None,
                val_transform=None):
    train_dataset = CSVDataset(train_df, transform=train_transform)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    valid_dataset = CSVDataset(valid_df, transform=val_transform)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    return train_loader, valid_loader
