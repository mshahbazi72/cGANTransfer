# use z_var to change the variance of z for all the sampling
# use --mybn --accumulate_stats --num_standing_accumulations 32 to
# use running stats
python sample.py \
--experiment_name  CIFAR_l0.01_res0.1 \
--load_weights best2 \
--base_root ./ \
--dataset C100_ImageFolder --parallel --shuffle  --batch_size 50  \
--n_class 20 --n_pretrain_classes 80 \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 500 \
--num_D_steps 4 --G_lr 2e-4 --D_lr 2e-4 \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \

--test_every 5000 --save_every 2000 --num_best_copies 5 --num_save_copies 2 --seed 0


# Use if you also want to calculate KMMD when evaluating the model.
# In that case, "num_inception_images" is decreased to avoid a memory error
#--kmmd --num_inception_images 25000

# --ema --use_ema --ema_start 1000
