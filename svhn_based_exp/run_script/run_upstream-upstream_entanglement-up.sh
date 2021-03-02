#num_seed=10
for seed in {0..9}; do
    for para_to_vary_model in 10 20 30 40 50 60 70 80 90 100; do
        python train_svhn.py --task "uu-ent-up1" --model "resnet18" --epoch 200 --seed $seed --para_to_vary_model $para_to_vary_model 
        python train_svhn.py --task "uu-ent-up2" --model "resnet18" --epoch 200 --seed $seed --para_to_vary_model $para_to_vary_model 
done
done