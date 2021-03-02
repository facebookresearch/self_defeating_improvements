import time
import fire
import sys
sys.path.insert(0, '..')
import kitti_common as kitti
from eval import get_range_eval_result, get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             score_thresh=-1,
             printstr=False,
             eval_mode=0):
    print(score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    #rst = get_range_eval_result(gt_annos, dt_annos, current_class)
    if eval_mode == 0:
        rst = get_official_eval_result(gt_annos, dt_annos, current_class)
    else:
        rst = get_range_eval_result(gt_annos, dt_annos, current_class, ranges=[0, 20])
    if printstr:
        print(rst[0])
    return rst


def evaluate_coarse(label_path,
                    result_path,
                    label_split_file,
                    current_class=0,
                    score_thresh=-1,
                    printstr=False):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    rst = get_range_eval_result(
        gt_annos, dt_annos, current_class, ranges=[0, 30, 50, 80])
    if printstr:
        print(rst[0])
    return rst


def evaluate_very_coarse(label_path,
                         result_path,
                         label_split_file,
                         current_class=0,
                         score_thresh=-1,
                         printstr=False):
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    rst = get_range_eval_result(gt_annos, dt_annos, current_class, ranges=[0, 80])
    if printstr:
        print(rst[0])
    return rst

if __name__ == '__main__':
    fire.Fire()
