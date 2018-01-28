"""
Experiments with pytorch
"""

import argparse

import data_loader
import models
import numpy as np
import utils
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from pathlib import Path

import pandas as pd

data_path = Path('data')

img_transform = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_map = {'HTC-1-M7': 0,
             'LG-Nexus-5x': 1,
             'Motorola-Droid-Maxx': 2,
             'Motorola-Nexus-6': 3,
             'Motorola-X': 4,
             'Samsung-Galaxy-Note3': 5,
             'Samsung-Galaxy-S4': 6,
             'Sony-NEX-7': 7,
             'iPhone-4s': 8,
             'iPhone-6': 9}


def validation(model, criterion, valid_loader):
    model.eval()
    losses = []
    accuracy_scores = []
    for inputs, targets in valid_loader:
        inputs = utils.variable(inputs, volatile=True)
        targets = utils.variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
        accuracy_scores += list(targets.data.cpu().numpy() == np.argmax(outputs.data.cpu().numpy(), axis=1))

    valid_loss = np.mean(losses)  # type: float
    valid_accuracy = np.mean(accuracy_scores)  # type: float
    print('Valid loss: {:.4f}, accuracy: {:.4f}'.format(valid_loss, valid_accuracy))
    return {'valid_loss': valid_loss, 'accuracy': valid_accuracy}


def get_df(mode=None):
    if mode == 'train':
        main_df = pd.read_csv(str(data_path / 'train_df.csv'))
        flickr_df = pd.read_csv(str(data_path / 'flickr_train.csv'))
        df = pd.concat([main_df, flickr_df])
        df['class_id'] = df['target'].map(class_map)
        return df

    elif mode == 'val':
        main_df = pd.read_csv(str(data_path / 'val_df.csv'))
        flickr_df = pd.read_csv(str(data_path / 'flickr_val.csv'))
        pseudo_df = pd.read_csv(str(data_path / 'test_preds_trunc.csv'))
        df = pd.concat([main_df, flickr_df, pseudo_df])
        df['class_id'] = df['target'].map(class_map)

        return df

    return None


def add_args(parser):
    arg = parser.add_argument
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=4)
    arg('--n-epochs', type=int, default=30)
    arg('--lr', type=float, default=0.0001)
    arg('--workers', type=int, default=12)
    arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--model', type=str)


if __name__ == '__main__':

    random_state = 2016

    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    add_args(parser)
    args = parser.parse_args()
    model_name = args.model

    Path(args.root).mkdir(exist_ok=True, parents=True)

    batch_size = args.batch_size

    train_df = get_df('train')
    val_df = get_df('val')

    train_loader, valid_loader = data_loader.get_loaders(batch_size, args, train_df=train_df, valid_df=val_df)

    num_classes = data_loader.num_classes

    # model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet50)
    model = models.DenseNetFinetune(num_classes, net_cls=models.M.densenet121)
    model = utils.cuda(model)

    if utils.cuda_is_available:
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None

        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    criterion = CrossEntropyLoss()

    train_kwargs = dict(
        args=args,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=validation,
        patience=10,
    )

    if getattr(model, 'finetune', None):
        utils.train(
            init_optimizer=lambda lr: SGD(model.net.fc.parameters(), lr=lr, momentum=0.9),
            n_epochs=1,
            **train_kwargs)

        utils.train(
            init_optimizer=lambda lr: SGD(model.parameters(), lr=lr, momentum=0.9),
            **train_kwargs)
    else:
        utils.train(
            init_optimizer=lambda lr: SGD(model.parameters(), lr=lr, momentum=0.9),
            **train_kwargs)
