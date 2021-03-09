#!/bin/bash

# The ImageNet dataset should be named  ImageNet and place under "data" directory

#python make_hdf5.py --dataset I128 --batch_size 256 --data_root ./data/
#python calculate_inception_moments.py --dataset I128_hdf5 --data_root ./data/

python calculate_inception_moments.py --dataset I128 --data_root ./data/

