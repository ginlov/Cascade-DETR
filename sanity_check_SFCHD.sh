#!/bin/bash

python train.py --data-path data/miniSFCHD/yolo --image-folder data/miniSFCHD/images \\
--N 10 --num_classes 7 --batch-size 16 --epochs 100 --lr 0.001