#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for seed in {0..9}; do
    for para_to_vary_model in 10 20 30 40 50 60 70 80 90 100; do
        python train_svhn.py --task "dd-mis-up" --model "resnet18" --epoch 200 --seed $seed --para_to_vary_model $para_to_vary_model
        upstream_setting="dd-mis-up_${para_to_vary_model}.0_resnet18_${seed}_feat_last"
        python train_svhn.py --upstream_setting $upstream_setting --model "linear" --epoch 100 --lr 1e-2 --seed $seed --task "dd-mis-down"
done
done
