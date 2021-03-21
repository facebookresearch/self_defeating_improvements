#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Installs dependencies for open-source users.
"""

import os
import requests


#original repo
PL_LINK = "https://github.com/mileyan/pseudo_lidar.git"
PL_FOLDER = "pseudo_lidar"
PL_COMMIT = "032c7a0d73c3fdf84e934af3f57f8eb489a52906"
PRCNN_LINK = "https://github.com/sshaoshuai/PointRCNN.git"
PRCNN_FOLDER = "PointRCNN"
PRCNN_COMMIT = "1d0dee91262b970f460135252049112d80259ca0"
FPTNT_LINK = "https://github.com/charlesq34/frustum-pointnets.git"
FPTNT_FOLDER = "frustum-pointnets"
FPTNT_COMMIT = "2ffdd345e1fce4775ecb508d207e0ad465bcca80"
EVAL_LINK = "https://github.com/traveller59/kitti-object-eval-python.git"
EVAL_FOLDER = "kitti-object-eval-python"
EVAL_COMMIT = "9f385f8fd40c195a6370ae3682889d8d5dddf42b"

links = [PL_LINK, PRCNN_LINK, FPTNT_LINK, EVAL_LINK]
folders = [PL_FOLDER, PRCNN_FOLDER, FPTNT_FOLDER, EVAL_FOLDER]
commits = [PL_COMMIT, PRCNN_COMMIT, FPTNT_COMMIT, EVAL_COMMIT]

#install the above repos
for (link, folder, commit) in zip(links, folders, commits):
    os.system(f"git clone {link}")
    os.system(f"git --git-dir={folder}/.git --work-tree={folder} checkout {commit}")
    os.system(f"rm -rf {folder}/.git")
    #os.system(f"diff -ruN {folder} ../pseudo_lidar_exp/{folder} > patches/{folder}.patch")
    os.system(f"patch -s -p0 < patches/{folder}.patch")
    if os.path.exists(f"{folder}_diff"):
        os.system(f"cp -r {folder}_diff/* {folder}")
        os.system(f"rm -r {folder}_diff/")
