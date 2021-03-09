#!/bin/bash

# The CIFAR dataset folder should be named "cifar" and placed in the "data" directory
python calculate_inception_moments.py --dataset C100_ImageFolder --data_root ./data
