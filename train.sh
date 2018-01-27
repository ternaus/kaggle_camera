#!/bin/bash

for i in 0 1 2 3 4
do
   python train.py --device-ids 0,1 --batch-size 24 --fold $i --workers 8 --lr 0.0001 --n-epochs 100
done
