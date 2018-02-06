"""
Train for manipulate files only
"""

import argparse

import data_loader
import models
import numpy as np
import utils
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from pathlib import Path
import transforms as albu_trans
from torchvision.transforms import ToTensor, Normalize, Compose

import pandas as pd

data_path = Path('data')

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

target_size = 256


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
        train_path = data_path / 'train'
        train_file_names = list(train_path.glob('**/*.*'))
        train_file_names = [x.absolute() for x in train_file_names]
        main_df = pd.DataFrame({'file_name': train_file_names})
        main_df['fname'] = main_df['file_name'].apply(lambda x: x.name, 1)

        main_df = main_df[main_df['fname'] != '(MotoNex6)8.jpg']

        main_df['target'] = main_df['file_name'].apply(lambda x: x.parent.name, 1)
        main_df['is_manip'] = 0

        flickr_path = data_path / 'new_flickr'

        flickr_file_names = list(flickr_path.glob('**/*.*'))
        flickr_file_names = [x.absolute() for x in flickr_file_names]

        flickr_df = pd.DataFrame({'file_name': flickr_file_names})
        flickr_df['fname'] = flickr_df['file_name'].apply(lambda x: x.name, 1)

        flickr_df['target'] = flickr_df['file_name'].apply(lambda x: x.parent.name, 1)
        flickr_df['is_manip'] = 0

        test_preds = pd.read_csv(str(data_path / 'Voting_stats_v5.csv'))

        test_preds['file_name'] = test_preds['fname'].apply(
            lambda x: (data_path / 'test' / x.replace('tif', 'jpg')).absolute(), 1)
        test_preds = test_preds.rename(columns={'best_camera': 'target'})
        test_preds = test_preds[test_preds['votes'] >= 4]
        test_preds['is_manip'] = test_preds['fname'].astype(str).str.contains('manip').astype(int)

        test_preds = test_preds[test_preds['is_manip'] == 0]

        df = pd.concat([main_df, flickr_df, test_preds])

        df['class_id'] = df['target'].map(class_map)

        # df = df[df['target'].notnull()]

        df['is_manip'] = 1
        return df

    elif mode == 'val':
        main_df = pd.read_csv(str(data_path / 'val_df.csv'))
        df = main_df

        df['class_id'] = df['target'].map(class_map)

        df['is_manip'] = 0
        df = df[df['target'].notnull()]
        df['is_manip'] = 1
        return df

    return None


train_transform = Compose([
    albu_trans.RandomCrop(target_size),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = Compose([
    albu_trans.CenterCrop(target_size),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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

    print(train_df.shape, val_df.shape)

    train_loader, valid_loader = data_loader.get_loaders(batch_size, args, train_df=train_df, valid_df=val_df,
                                                         train_transform=train_transform, val_transform=val_transform)

    num_classes = data_loader.num_classes

    # model = models.ResNetFinetune(num_classes, net_cls=models.M.resnet34, dropout=True)
    model = models.DenseNetFinetune(num_classes, net_cls=models.M.densenet201, two_layer=True)
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
        patience=1000,
    )

    utils.train(
        init_optimizer=lambda lr: SGD(model.parameters(), lr=lr, momentum=0.9),
        **train_kwargs)
    #
    # if getattr(model, 'finetune', None):
    #     utils.train(
    #         init_optimizer=lambda lr: SGD(model.net.fc.parameters(), lr=lr, momentum=0.9),
    #         n_epochs=1,
    #         **train_kwargs)
    #
    #     utils.train(
    #         init_optimizer=lambda lr: SGD(model.parameters(), lr=lr, momentum=0.9),
    #         **train_kwargs)
    # else:
    #     utils.train(
    #         init_optimizer=lambda lr: SGD(model.parameters(), lr=lr, momentum=0.9),
    #         **train_kwargs)
