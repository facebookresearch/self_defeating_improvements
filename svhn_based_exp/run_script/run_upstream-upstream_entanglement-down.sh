#num_seed=10
for seed in {0..9}; do
    for para_to_vary_model_1 in 10 20 30 40 50 60 70 80 90 100; do
        for para_to_vary_model_2 in 10 20 30 40 50 60 70 80 90 100; do
            upstream_setting="uu-ent-up1_${para_to_vary_model_1}.0_resnet18_${seed}_preds_last_uu-ent-up2_${para_to_vary_model_2}.0_resnet18_${seed}_preds_last"
            python train_svhn.py --upstream_setting $upstream_setting --model "linear" --epoch 100 --lr 1e-2 --seed $seed --task "uu-ent-down"
done
done
done