#!/usr/bin/env bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for seed in {0..9}; do
    for model in "convnet" "convnet-512" "convnet-512-256" "convnet-512-256-128" "convnet-512-256-128-64"; do
        python train_svhn.py --task "hs-mis-up" --model $model --epoch 200 --seed $seed
        upstream_setting="hs-mis-up_None_${model}_${seed}_feat_last"
        python train_svhn.py --upstream_setting $upstream_setting --model "linear" --epoch 100 --lr 1e-2 --seed $seed --task "hs-mis-down"
done
done
