#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torchvision.transforms as transforms

svhn_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

svhn_transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

awa_transform_train = transforms.Compose([
transforms.RandomRotation(15),
transforms.RandomHorizontalFlip(),
transforms.ColorJitter(brightness=0.3, contrast=0.3),
transforms.Resize((224,224)),
transforms.ToTensor()
])
awa_transform_test = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor()
])
