import h5py
import os


target_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets/'
special_target = os.path.join(target_dir, 'special_videos.h5')
special_h5f = h5py.File(special_target, 'r')
# print(special_h5f.keys())
# print(special_h5f['video59_right']['video59_right_1']['start'][()])
# print(special_h5f['video59_right']['video59_right_1']['end'][()])
# print(special_h5f['video59_right']['video59_right_1']['length'][()])

ori_target = os.path.join(target_dir, 'val_original_all_targets.h5')
ori_h5f = h5py.File(ori_target, 'r')
print(len(ori_h5f.keys()))