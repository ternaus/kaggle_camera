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

# img_transform = transforms.Compose([
#     # transforms.RandomCrop(512),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


class PredictionDataset:
    def __init__(self, paths, k_angle=None, to_flip=False):
        self.paths = paths
        self.k_angle = k_angle
        self.to_flip = to_flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = utils.load_image(path)

        if self.k_angle > 0:
            image = np.rot90(image, self.k_angle)
        if self.to_flip:
            image = np.fliplr(image)

        return data_loader.img_transform(image.copy()), path.stem


def predict(model, from_paths, batch_size: int, k_angle: int, to_flip: bool):
    loader = DataLoader(
        dataset=PredictionDataset(from_paths, k_angle, to_flip),
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

    # model = models.DenseNetFinetune(num_classes, net_cls=models.M.densenet121)
    model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet34, dropout=True)
    model = utils.cuda(model)

    if utils.cuda_is_available:
        model = nn.DataParallel(model, device_ids=[0]).cuda()

    state = torch.load(
        # str(Path(args.root) / 'best-model_{fold}.pt'.format(fold=fold)))
        str(Path(args.root) / 'model.pt'))

    model.load_state_dict(state['model'])
    model.eval()

    return model


def add_args(parser):
    arg = parser.add_argument
    arg('--root', default='data/models/resnet34_147', help='model path')
    arg('--batch-size', type=int, default=20)
    arg('--workers', type=int, default=12)


if __name__ == '__main__':
    random_state = 2016

    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    add_args(parser)
    args = parser.parse_args()

    data_path = Path('data')

    test_images = sorted(list((data_path / 'test').glob('*.tif')))

    result = []

    model = get_model()
    for k_angle in [0, 1, 2, 3]:
        for to_flip in [False]:
            preds = predict(model, test_images, args.batch_size, k_angle, to_flip)

            result += [preds]

    pred_probs = gmean(np.dstack(result), axis=2)

    # max_ind = np.argmax(pred_probs, axis=1)

    class_name, class_id = zip(*train.class_map.items())

    class_map_inv = dict(zip(class_id, class_name))

    # preds = [class_map_inv[x] for x in max_ind]
    columns = [class_map_inv[x] for x in range(10)]

    df = pd.DataFrame(pred_probs, columns=columns)
    df['fname'] = [x.name for x in test_images]

    # df = pd.DataFrame({'fname': [x.name for x in test_images], 'camera': preds})
    df.to_csv(str(data_path / '18.csv'), index=False)
