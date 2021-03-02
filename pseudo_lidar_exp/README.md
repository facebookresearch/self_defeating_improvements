# Pseudo-LiDAR Experiments

## 1. Data Preparation
#### Download KITTI dataset
Follow the instruction at [https://github.com/mileyan/pseudo_lidar](https://github.com/mileyan/pseudo_lidar) and organize the data in the following way
```angular2html
./Kitti/object/
    
    train.txt
    val.txt
    test.txt 
    
    training/
        calib/
        image_2/ #left image
        image_3/ #right image
        label_2/
        velodyne/ 

    testing/
        calib/
        image_2/
        image_3/
        velodyne/
```

Download `label_2_close` [here](https://drive.google.com/file/d/1pUQPUtJHXQUhfCWKSruo6dF6uEwYRDDq/view?usp=sharing) and put it under the folder `./Kitti/object/training`


## 2. Pseudo-LiDAR Training and Generation
We mainly follow the code at [https://github.com/mileyan/pseudo_lidar](https://github.com/mileyan/pseudo_lidar) and use the same environment set-up. After building the environment, follow the steps below to generate the pseudo lidar.


### 2.1 Generate ground-truth image disparities
Use the following script in the root of `./pseudo_lidar/` to generate the ground-truth disparities.
```
python generate_disp.py --data_path ./KITTI/object/training/ --split_file ./KITTI/object/train.txt 
```

### 2.1 Train the disparity predictor
We provide the checkpoints `./pseudo_lidar/psmnet/kitti_3d/finetune_300.tar` trained under disparity L1 loss and `./pseudo_lidar/psmnet/kitti_3d/finetune_300.tar` trained under depth L1 loss. You can also use following scripts to train the model by yourself.

#### Train the model under disparity L1 loss
Run the following script in the root of `./pseudo_lidar/`.
```
mkdir ./psmnet/kitti_3d
python ./psmnet/finetune_3d.py --maxdisp 192 --model stackhourglass --datapath ../Kitti/object/training/ --split_file ../Kitti/object/train.txt  --epochs 300 --lr_scale 50 --loadmodel ./pretrained_sceneflow.tar --savemodel ./psmnet/kitti_3d/  --btrain 12
```

#### Train the model under depth L1 loss
Run the following script in the root of `./pseudo_lidar/`.
```
mkdir ./psmnet/kitti_3d_dl
python ./psmnet/finetune_3d.py --maxdisp 192 --model stackhourglass --datapath ../Kitti/object/training/ --split_file ../Kitti/object/train.txt  --epochs 300 --lr_scale 50 --loadmodel ./pretrained_sceneflow.tar --savemodel ./psmnet/kitti_3d_dl/  --btrain 12 --data_type depth
```

### 2.2 Generate the Pseudo-LiDAR

#### 2.2.1 Predict the cloud points using the trained models

##### Generate the cloud points from the model trained under disparity L1 loss
```
#Predict the disparities
python ./psmnet/submission.py \
    --loadmodel ./psmnet/kitti_3d/finetune_300.tar \
    --datapath ../KITTI/object/training/ \
    --save_path ../KITTI/object/training/predict_disparity   
#Convert the disparities to point clouds
python ./preprocessing/generate_lidar.py  \
    --calib_dir ../KITTI/object/training/calib/ \
    --save_dir ../KITTI/object/training/pseudo-lidar_velodyne/ \
    --disparity_dir ../KITTI/object/training/predict_disparity \
    --max_high 1
#Convert the disparities to point clouds and remove points farther than 20 meters
python ./preprocessing/generate_lidar.py  \
    --calib_dir ../KITTI/object/training/calib/ \
    --save_dir ../KITTI/object/training/pseudo-lidar_velodyne_close/ \
    --disparity_dir ../KITTI/object/training/predict_disparity \
    --max_high 1 \
    --max_depth 20
```

##### Generate the cloud points from the model trained under depth L1 loss
```
#Predict the disparities
python ./psmnet/submission.py \
    --loadmodel ./psmnet/kitti_3d_dl/finetune_300.tar \
    --datapath ../KITTI/object/training/ \
    --save_path ../KITTI/object/training/predict_disparity_dl  
#Convert the disparities to point clouds
python ./preprocessing/generate_lidar.py  \
    --calib_dir ../KITTI/object/training/calib/ \
    --save_dir ../KITTI/object/training/pseudo-lidar_velodyne_dl/ \
    --disparity_dir ../KITTI/object/training/predict_disparity_dl \
    --max_high 1
#Convert the disparities to point clouds and remove points farther than 20 meters
python ./preprocessing/generate_lidar.py  \
    --calib_dir ../KITTI/object/training/calib/ \
    --save_dir ../KITTI/object/training/pseudo-lidar_velodyne_dl_close/ \
    --disparity_dir ../KITTI/object/training/predict_disparity_dl \
    --max_high 1 \
    --max_depth 20
```

##### Generate the ground-truth LiDAR cloud points but remove points farther than 20 meters
```
python ./preprocessing/generate_lidar.py  \
    --calib_dir ../KITTI/object/training/calib/ \
    --save_dir ../KITTI/object/training/velodyne_close/ \
    --disparity_dir ../KITTI/object/training/disparity/ \
    --max_high 1 \
    --max_depth 20
```

#### 2.2.2 Sparsify Pseudo-LiDAR
PointRCNN requires sparse point clouds. Pseudo-Lidar ++ [https://github.com/mileyan/Pseudo_Lidar_V2](https://github.com/mileyan/Pseudo_Lidar_V2) provides an script to downsample the dense Pseudo-LiDAR clouds. We copy the code to `./preprocessing` for convenience. Run the following code by set `velodyne=pseudo-lidar_velodyne, pseudo-lidar_velodyne_dl, pseudo-lidar_velodyne_close, pseudo-lidar_velodyne_dl_close`.
```
python ./preprocessing/kitti_sparsify.py --pl_path  ../Kitti/object/training/{velodyne} --sparse_pl_path  ~/Kitti/object/training/{velodyne}_sparse/
```

So far, we should have following folders of different `velodyne` under `../KITTI/object/training` for the downstream detection tasks.
`velodyne`:
```
velodyne/
velodyne_close/
pseudo-lidar_velodyne/
pseudo-lidar_velodyne_dl/
pseudo-lidar_velodyne_close/
pseudo-lidar_velodyne_dl_close/
pseudo-lidar_velodyne_sparse/
pseudo-lidar_velodyne_dl_sparse/
pseudo-lidar_velodyne_close_sparse/
pseudo-lidar_velodyne_dl_close_sparse/
```


## 3. PointRCNN Training
We follow [https://github.com/sshaoshuai/PointRCNN](https://github.com/sshaoshuai/PointRCNN) to build up the environment. We provide `virtual_env/env_prcnn.yml` that we used for the PointRCNN related experiment running.

`lidar_mode` is an argument to specify which lidar source we'd like to use. The dictionary is
```
0 - original lidar
1 - original lidar with far points removal
2 - pseudo-lidar trained under disparity loss
3 - pseudo-lidar trained under disparity loss with far points removal
4 - pseudo-lidar trained under depth loss
5 - pseudo-lidar trained under depth loss with far points removal
```

The original provide
```
python generate_gt_database.py --class_name 'Car' --split train
```
to generate the ground truth database for class car. We can similarly run
```
python generate_gt_database.py --class_name 'Pedestrian' --split train
```
to generate the ground truth database for class pedestrian.

Here is a script example to train a car detection given the original lidar source.
```
cd ./PointRCNN/tools
python train_rcnn.py --cfg_file cfgs/{task}.yaml --batch_size 16 --train_mode rpn --epochs 200 --output_dir ../output/rpn_{task}_{lidar_mode} --lidar_mode {lidar_mode} --ckpt_save_interval 5 --gt_database gt_database/train_gt_database_3level_{task}.pkl --data_root ../../Kitti
python train_rcnn.py --cfg_file cfgs/{task}.yaml --batch_size 4 --train_mode rcnn --epochs 70 --ckpt_save_interval 2 --rpn_ckpt ../output/rpn_{task}_{lidar_mode}/ckpt/checkpoint_epoch_200.pth --lidar_mode {lidar_mode} --output_dir ../output/rcnn_{task}_{lidar_mode} --gt_database gt_database/train_gt_database_3level_{task}.pkl --data_root ../../Kitti
python eval_rcnn.py --cfg_file cfgs/{task}.yaml --ckpt ../output/rcnn_{task}_{lidar_mode}/ckpt/checkpoint_epoch_70.pth --batch_size 4 --eval_mode rcnn --output_dir ../output/rcnn_{task}_{lidar_mode} --lidar_mode {lidar_mode} --data_root ../../Kitti
```
We can feed different lidar source by specifying different `lidar_mode` and `task=car / pedestrian`.

## 4. Frustum PointNets Training
We follow [https://github.com/charlesq34/frustum-pointnets](https://github.com/charlesq34/frustum-pointnets) to build up the environment. We provide `virtual_env/env_fpn.yml` that we used for the Frustum PointNet related experiment running.

`lidar_mode` is an argument to specify which lidar source we'd like to use. The dictionary is
```
0 - original lidar
1 - original lidar with far points removal
2 - pseudo-lidar trained under disparity loss
3 - pseudo-lidar trained under disparity loss with far points removal
4 - pseudo-lidar trained under depth loss
5 - pseudo-lidar trained under depth loss with far points removal
```

#### Data preparation
Run the following script for `lidar_mode=0, 1, ..., 5`
```
sh scripts/command_prep_data_pedestrian.sh {lidar_mode}
sh scripts/command_prep_data_car.sh {lidar_mode}
```

#### Run the training
Run the following script for `lidar_mode=0, 1, ..., 5`, `task=pedestrian / car`
```
python train/train.py --model frustum_pointnets_v2 --log_dir train/log_{task}_v2_{lidar_mode} --num_point 1024 --max_epoch 201 --batch_size 24 --decay_step 800000 --decay_rate 0.5 --lidar_mode {lidar_mode} --task {task}
```

#### Run the testing
Run the following script for `lidar_mode=0, 1, ..., 5`, `task=pedestrian / car`
```
python train/test.py --num_point 1024 --model frustum_pointnets_v2 --model_path train/log_{task}_v2_{lidar_mode}/model.ckpt --output train/detection_results_{task}_v2_{lidar_mode} --idx_path kitti/image_sets/val.txt --from_rgb_detection --lidar_mode {lidar_mode} --task {task}
```

## 5. Results Evaluation
Generally, set proper paths for `label_path=/Kitti/object/training/{filename}` (`filename="label_2" / "label_2_close"`), `result_path`, `current_class` and `eval_mode`, and run the following script
```
python ./kitti_object_eval_python_range_noocctrunc/evaluate.py evaluate --label_path={label_path}  --result_path={result_path} --label_split_file=/private/home/ruihan/Kitti/object/val.txt --current_class={current_class} --eval_mode={eval_mode}
```

`label_path` should be set as `./Kitti/object/training/{filename}`, where `filename="label_2" / "label_2_close"`

As for `result_path`, for PointRCNN, it should be set as `/PointRCNN/output/rcnn_{task}_{lidar_mode}/eval/epoch_70/val/final_result/data`; for  Frustum PointNet, it should be set as `./frustum-pointnets/train/detection_results_{task}_v2_{lidar_mode}/data`.

`current_class=0` when `task=car`; `current_class=1` when `task=pedestrian`.

`eval_mode=0` if we want to get the official numbers condition on different difficulty level (easy, medium, hard); `eval_mode=1` if we want to get the evaluation for objects in a certain range. The default range is set as [0, 20]. See the argument `ranges` in the function `get_range_eval_result` in function `evaluate` in `evaluate.py`
