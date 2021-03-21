#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for seed in {0..9}; do
    for para_to_vary_model in 10 20 30 40 50 60 70 80 90 100; do
        python train_svhn.py --task "uu-ent-up1" --model "resnet18" --epoch 200 --seed $seed --para_to_vary_model $para_to_vary_model
        python train_svhn.py --task "uu-ent-up2" --model "resnet18" --epoch 200 --seed $seed --para_to_vary_model $para_to_vary_model
done
done
