#!/bin/bash

python train.py --data-path ../sanity_check_data/yolo --image-folder ../sanity_check_data/images \
    --N 10 --num-classes 7 --batch-size 1 --num-epochs 5000 --lr 0.001 \
    --cost-class 1.0 --cost-bbox 1.0 --cost-giou 1.5 --output-dir sanity_check \
    --pretrained --batch-norm-freeze --bbox-loss-coef 1.0 --giou-loss-coef 1.5 \
    --backbone-name resnet101 --save-freq 100\