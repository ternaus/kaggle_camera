import numpy as np
import torch.utils.data as data
import torch
import utils
import transforms as albu_trans
from torchvision.transforms import ToTensor, Normalize, Compose

num_classes = 10

img_transform = Compose([
    albu_trans.CenterCrop(512),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CSVDataset(data.Dataset):
    def __init__(self, df):
        self.df = df
        self.path = df['file_name'].values.astype(str)
        self.target = df['class_id'].values.astype(np.int64)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = utils.load_image(self.path[idx])
        y = self.target[idx]
        return img_transform(X), y


def get_loaders(batch_size,
                args,
                train_df=None,
                valid_df=None):
    train_dataset = CSVDataset(train_df)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    valid_dataset = CSVDataset(valid_df)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=torch.cuda.is_available())

    return train_loader, valid_loader
