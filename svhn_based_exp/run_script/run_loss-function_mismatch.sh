#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for seed in {0..9}; do
    for para_to_vary_model in {0..9}; do
        python train_svhn.py --task "lf-mis-up" --model "resnet18" --epoch 500 --seed $seed --para_to_vary_model $para_to_vary_model
        upstream_setting="lf-mis-up_${para_to_vary_model}.0_resnet18_${seed}_logits_best"
        python train_svhn.py --upstream_setting $upstream_setting --model "MLP-128" --epoch 100 --lr 1e-2 --seed $seed --task "lf-mis-down"
done
done
