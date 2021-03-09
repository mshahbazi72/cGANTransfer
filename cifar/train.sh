#!/bin/bash
python train.py \
--experiment_name  CIFAR_l0.01_res0.1 \
--base_root ./ \
--batch_size 50 --stage BN \
--dataset C100_ImageFolder --shuffle  --parallel \
--n_classes 20 --n_pretrain_classes 80 \
--comb_l1_scale 0.01 --res_l2_scale 0.01 \
--test_every 100 --save_every 100 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 5000 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \

# Use to continues the training from the last checkpoint. Otherwise, model starts from the pre-training weights
# --resume
# Specify the suffix for the checkpoint to load an specific checkpoint.
# --load_weights best0

# Use if you also want to calculate KMMD when evaluating the model.
# In that case, "num_inception_images" is decreased to avoid a memory error
# --kmmd --num_inception_images 25000

#--ema --use_ema --ema_start 1000 \
