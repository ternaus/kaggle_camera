import argparse
from torchvision import transforms
import utils
import data_loader
from tqdm import tqdm
import models
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
from scipy.stats.mstats import gmean
import train
import data_loader

import transforms as albu_trans


img_transform = transforms.Compose([
    # albu_trans.CenterCrop(train.target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class PredictionDataset:
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = utils.load_image(path)

        if self.transform == 1:
            image = np.rot90(image, 1)
        elif self.transform == 2:
            image = np.rot90(image, 1)
        elif self.transform == 3:
            image = np.rot90(image, 3)
        elif self.transform == 4:
            image = np.fliplr(image)
        elif self.transform == 5:
            image = np.rot90(np.fliplr(image), 1)
        elif self.transform == 6:
            image = np.rot90(np.fliplr(image), 2)
        elif self.transform == 7:
            image = np.rot90(np.fliplr(image), 3)

        return img_transform(image.copy()), path.stem


def predict(model, from_paths, batch_size: int, transform):
    loader = DataLoader(
        dataset=PredictionDataset(from_paths, transform),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    result = []

    for batch_num, (inputs, stems) in enumerate(tqdm(loader, desc='Predict')):
        inputs = utils.variable(inputs, volatile=True)
        outputs = F.softmax(model(inputs), dim=1)
        result += [outputs.data.cpu().numpy()]

    return np.vstack(result)


def get_model():
    num_classes = data_loader.num_classes

    model = models.DenseNetFinetune(num_classes, net_cls=models.M.densenet201, two_layer=True)
    # model = models.DenseNetFinetune(num_classes, net_cls=models.M.densenet201, two_layer=True)
    # model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet34, dropout=True)
    model = utils.cuda(model)

    if utils.cuda_is_available:
        model = nn.DataParallel(model, device_ids=[0]).cuda()

    state = torch.load(
        str(Path(args.root) / 'best-model.pt'))
        # str(Path(args.root) / 'model.pt'))

    model.load_state_dict(state['model'])
    model.eval()

    return model


def add_args(parser):
    arg = parser.add_argument
    # arg('--root', default='data/models/densenet201m_460', help='model path')
    arg('--root', default='data/models', help='model path')
    arg('--batch-size', type=int, default=20)
    arg('--workers', type=int, default=12)


if __name__ == '__main__':
    random_state = 2016

    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    add_args(parser)
    args = parser.parse_args()

    data_path = Path('data')

    # test_images = sorted(list((data_path / 'test').glob('*.tif')))
    test_images = sorted(list((data_path / 'test').glob('*')))

    result = []

    model = get_model()

    for transform in range(8):
        preds = predict(model, test_images, args.batch_size, transform)

        result += [preds]

    pred_probs = gmean(np.dstack(result), axis=2)

    row_sums = pred_probs.sum(axis=1)

    pred_probs = pred_probs / row_sums[:, np.newaxis]

    max_ind = np.argmax(pred_probs, axis=1)

    class_name, class_id = zip(*train.class_map.items())

    class_map_inv = dict(zip(class_id, class_name))

    preds = [class_map_inv[x] for x in max_ind]
    columns = [class_map_inv[x] for x in range(10)]

    df = pd.DataFrame(pred_probs, columns=columns)
    df['camera'] = preds
    df['fname'] = [x.name for x in test_images]
    df['fname'] = df['fname'].str.replace('jpg', 'tif')

    # df = pd.DataFrame({'fname': [x.name for x in test_images], 'camera': preds})
    df[['fname', 'camera']].to_csv(str(data_path / 'ternaus_x1.csv'), index=False)

    # df = df.sort_values(by='fname').reset_index(drop=True)
    #
    # vote7 = pd.read_csv('data/Voting_stats_v7.csv').reset_index(drop=True)
    #
    # submit988 = pd.read_csv('data/submit.csv').sort_values(by='fname').reset_index(drop=True)

    # print('total_mean vote 7 = ', np.mean(df['best_camera'].values == vote7['best_camera'].values))

    # ind = df['fname'].str.contains('manip')
    #
    # print('manip_mean vote 7 = ', np.mean(df.loc[ind, 'best_camera'].values == vote7.loc[ind, 'best_camera'].values))
    #
    # print('unmanip_mean vote 7 = ', np.mean(df.loc[~ind, 'best_camera'].values == vote7.loc[~ind, 'best_camera'].values))
    #
    # print('988 total = ', np.mean(df['best_camera'].values == submit988['camera'].values))
    #
    # ind = df['fname'].str.contains('manip')
    #
    # print('988 manip = ', np.mean(df.loc[ind, 'best_camera'].values == submit988.loc[ind, 'camera'].values))
    #
    # print('988 unmanip = ',
    #       np.mean(df.loc[~ind, 'best_camera'].values == submit988.loc[~ind, 'camera'].values))
    #
