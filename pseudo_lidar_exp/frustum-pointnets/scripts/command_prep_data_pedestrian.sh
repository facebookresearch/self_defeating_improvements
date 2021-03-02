#/bin/bash
python kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --lidar_mode $1 --pedestrian_only
