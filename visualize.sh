#!/bin/bash

python test.py --backbone-name resnet101 --checkpoint sanity_check/detr_epoch_4900.pth \
	--data-path ../sanity_check_data/yolo --image-folder ../sanity_check_data/images --N 10 \
	--num-classes 7 --batch-size 1 --output-dir ../outputs --batch-norm-freeze
