#!/bin/bash

python train_manip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.0001 --n-epochs 20
python train_manip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.00001 --n-epochs 40
python train_manip.py --device-ids 0,1 --batch-size 32 --workers 12 --lr 0.000001 --n-epochs 60