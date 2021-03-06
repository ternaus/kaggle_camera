import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from torch import nn
from torch.autograd import Variable
import jpeg4py


DATA_ROOT = Path(__file__).absolute() / 'data'

cuda_is_available = torch.cuda.is_available()


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x.cuda(async=True), volatile=volatile))


def cuda(x):
    return x.cuda() if cuda_is_available else x


def load_image(path: Path) -> np.array:
    img = cv2.imread(str(path))
    # try:
    # img = jpeg4py.JPEG(str(path)).decode()
    # except:
    # img = cv2.imread(str(path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(path)

    # img = cv2.imread(str(path))
    # if img.shape[0] == 512 and img.shape[1] == 512:
    #     return img

    # height, width = 1024, 1024
    #
    # h, w, c = img.shape
    # dy = (h - height) // 2
    # dx = (w - width) // 2
    #
    # y1 = dy
    # y2 = y1 + height
    # x1 = dx
    # x2 = x1 + width
    #
    # img = img[y1:y2, x1:x2, :]

    # if img.shape != (1024, 1024, 3):
    #      print(path)

    # img = np.array(Image.open(str(path)))
    return img


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def train(args,
          model: nn.Module,
          criterion, *, train_loader,
          valid_loader,
          validation,
          init_optimizer,
          save_predictions=None,
          n_epochs=None,
          patience=2):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)

    root = Path(args.root)
    model_path = root / 'model.pt'
    best_model_path = root / 'best-model.pt'
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss
    }, str(model_path))

    report_each = 10
    save_prediction_each = report_each * 20
    log = root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = variable(inputs), variable(targets)

                outputs = model(inputs)

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
                    if save_predictions and i % save_prediction_each == 0:
                        p_i = (i // save_prediction_each) % 5
                        save_predictions(root, p_i, inputs, targets, outputs)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(best_model_path))
            elif (patience and epoch - lr_reset_epoch > patience and
                          min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr /= 5
                lr_reset_epoch = epoch
                optimizer = init_optimizer(lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


# def load_best_model(model: nn.Module, root: Path, fold) -> None:
#     state = torch.load(str(root / 'best-model_{fold}.pt'.format(fold=fold.fold)))
#     model.load_state_dict(state['model'])
#     print('Loaded model from epoch {epoch}, step {step:,}'.format(**state))
