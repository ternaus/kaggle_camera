#!/bin/bash

python train_unmanip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.00001 --n-epochs 140
python train_unmanip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.000001 --n-epochs 160
python train_unmanip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.0000001 --n-epochs 180