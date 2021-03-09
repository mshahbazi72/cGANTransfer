#!/bin/bash
python train.py \
--experiment_name  ImageNet2Places_l0.01_res0.1 \
--base_root ./ \
--batch_size 256  --stage BN \
--n_classes 5 --n_pretrain_classes 1000 \
--dataset I128 --parallel --shuffle  --num_workers 8 \
--res_l2_scale 0.1 --comb_l1_scale 0.01 \
--test_every 100 --save_every 100 --num_best_copies 5 --num_save_copies 2 --seed 0 \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--use_multiepoch_sampler \

# Use to continues the training from the last checkpoint. Otherwise, model starts from the pre-training weights
# --resume
# Specify the suffix for the checkpoint to load an specific checkpoint.
# --load_weights best0

# Use if you also want to calculate KMMD when evaluating the model.
# In that case, "num_inception_images" is decreased to avoid a memory error
# --kmmd --num_inception_images 25000

