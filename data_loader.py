import numpy as np
import torch.utils.data as data
import torch
import utils
from functools import partial
from PIL import Image
from io import BytesIO
import cv2
import transforms as albu_trans
import train

num_classes = 10

target_size = train.target_size


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
    if scale is None:
        scale = np.random.choice([0.5, 0.8, 1.5, 2.0])

    result = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if result.shape[0] < target_size:
        return img


def rot90(img, k=None):
    if k is None:
        k = np.random.choice([1, 2, 3])
    return np.rot90(img, k=k)


def augment(x, safe=False):
    if safe:
        augs = (np.fliplr,
                np.flipud,
                partial(rot90),
                None)

        f = np.random.choice(augs)

        if f is not None:
            return f(x), 0
        return x, 0

    else:
        augs = (
            jpg_compress,
            gamma_correction,
            partial(rescale),
            np.fliplr,
            np.flipud,
            partial(rot90),
            None)

        num_aug = len(augs)

        aug_index = np.random.randint(0, num_aug)

        f = augs[aug_index]

        if f is not None:
            return f(x), 0
        return x, int(aug_index < 3)


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

        if X.shape[0] < target_size or X.shape[1] < target_size:
            print(self.path[idx])

        is_manipulated = self.is_manip[idx]

        if self.mode == 'train':
            self.X = albu_trans.RandomCrop(2 * target_size)

            self.X, manipulated = augment(X, is_manipulated == 1)
        else:
            manipulated = 0

        y = self.target[idx]

        if is_manipulated == 1:
            manipulated = 1

        return (self.transform(X), torch.from_numpy(np.array([manipulated])).float()), y


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
