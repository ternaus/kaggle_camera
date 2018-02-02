#!/bin/bash

python train.py --device-ids 0,1,2,3 --batch-size 32 --workers 8 --lr 0.0001 --n-epochs 100
python train.py --device-ids 0,1,2,3 --batch-size 32 --workers 8 --lr 0.00001 --n-epochs 200
python train.py --device-ids 0,1,2,3 --batch-size 32 --workers 8 --lr 0.000001 --n-epochs 300