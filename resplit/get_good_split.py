# import h5py
# import numpy as np
# import os
# from tqdm import tqdm
# import random
# import collections
# import json

# targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
# resplit_dir = os.path.join(targets_dir, 'resplit')
# info_path = os.path.join(resplit_dir, 'original_train_info.h5')
# info_h5f = h5py.File(info_path, 'r')

# # print(len(info_h5f.keys())) # 训练集共341个视频，划分为5份就是[68, 68, 68, 68, 69]
# video_ids = list(info_h5f.keys())

# info_dict = {}
# total_frame_nbs = 0
# for video in video_ids:
#     info_dict[video] = {}
#     info_dict[video]['frame_nb'] = info_h5f[video]['frame_nb'][()]
#     info_dict[video]['valence'] = info_h5f[video]['valence'][()]
#     info_dict[video]['arousal'] = info_h5f[video]['arousal'][()]
#     total_frame_nbs += info_h5f[video]['frame_nb'][()]
# print('total frame numbers:\t{}'.format(total_frame_nbs))

# def get_random_split(video_ids):
#     random.shuffle(video_ids)
#     assert len(video_ids[4*68:]) == 69
#     return [video_ids[:68], video_ids[68: 2*68], video_ids[2*68: 3*68], video_ids[3*68: 4*68], video_ids[4*68:]]

# def get_split_frame_nbs(video_splits, score=True):
#     '''
#     video_splits: [[vid, vid, ...], [...], [...], [...], [...]]
#     return:
#     - [xx, xx, xx, xx, xx]
#     '''
#     result_list = []
#     for i in range(5):
#         res = 0
#         video_split = video_splits[i]
#         for video in video_split:
#             res += info_dict[video]['frame_nb']
#         result_list.append(res)
#     if score == False:
#         return result_list
#     else:
#         return max(result_list) - min(result_list)

# def get_label_sec(video_splits, score=True):
#     '''
#     video_splits: [[vid, vid, ...], [...], [...], [...], [...]]
#     return: score
#         5个划分中，0以下区间找个最大的找个最小的取个差值，0.4以上的也取个差值。两个差值相加（valence和arousal再相加）作为排序指标
#     '''
#     valence_sec_1_list = []
#     valence_sec_2_list = []
#     valence_sec_3_list = []
#     arousal_sec_1_list = []
#     arousal_sec_2_list = []
#     arousal_sec_3_list = []
#     for i in range(5):
#         valence_sec_1 = 0
#         valence_sec_2 = 0
#         valence_sec_3 = 0
#         arousal_sec_1 = 0
#         arousal_sec_2 = 0
#         arousal_sec_3 = 0
#         video_split = video_splits[i]
#         for video in video_split:
#             valence_sec_1 += info_dict[video]['valence'][()][0]
#             valence_sec_2 += info_dict[video]['valence'][()][1]
#             valence_sec_3 += info_dict[video]['valence'][()][2]
#             arousal_sec_1 += info_dict[video]['arousal'][()][0]
#             arousal_sec_2 += info_dict[video]['arousal'][()][1]
#             arousal_sec_3 += info_dict[video]['arousal'][()][2]
#         valence_sec_1_list.append(valence_sec_1)
#         valence_sec_2_list.append(valence_sec_2)
#         valence_sec_3_list.append(valence_sec_3)
#         arousal_sec_1_list.append(arousal_sec_1)
#         arousal_sec_2_list.append(arousal_sec_2)
#         arousal_sec_3_list.append(arousal_sec_3)
#     if score == False:
#         return valence_sec_1_list, valence_sec_2_list, valence_sec_3_list, arousal_sec_1_list, arousal_sec_2_list, arousal_sec_3_list
#     else:
#         valence_score = (max(valence_sec_1_list) - min(valence_sec_1_list)) + (max(valence_sec_2_list) - min(valence_sec_2_list)) + (max(valence_sec_3_list) - min(valence_sec_3_list))
#         arousal_score = (max(arousal_sec_1_list) - min(arousal_sec_1_list)) + (max(arousal_sec_2_list) - min(arousal_sec_2_list)) + (max(arousal_sec_3_list) - min(arousal_sec_3_list))
#         return valence_score + arousal_score

# total_dict = {}
# for i in tqdm(range(1, 10001)):
#     total_dict[i] = {}
#     total_dict[i]['split'] = get_random_split(video_ids)
#     total_dict[i]['frame_score'] = get_split_frame_nbs(total_dict[i]['split'], score=True)
# sorted_total = sorted(total_dict.items(), key=lambda x: x[1]['frame_score'], reverse=False)

# # for i in range(100):
# #     print(sorted_total[i][1]['frame_score']) # 第100个为24217
# # print(sorted_total[-1][1]['frame_score']) # 最后一个为197200

# bal_frames_dict = {}
# for i in tqdm(range(1, 101)):
#     bal_frames_dict[i] = {}
#     bal_frames_dict[i]['split'] = sorted_total[i-1][1]['split']
#     bal_frames_dict[i]['frame_score'] = sorted_total[i-1][1]['frame_score']
#     bal_frames_dict[i]['label_score'] = get_label_sec(bal_frames_dict[i]['split'], score=True)
# sorted_bal_frames = sorted(bal_frames_dict.items(), key=lambda x: x[1]['label_score'], reverse=False)

# # save_file_path = os.path.join(resplit_dir, 'train_split_result.h5')
# # save_h5f = h5py.File(save_file_path, 'w')
# json_file = os.path.join(resplit_dir, 'train_split.json')
# record_file = os.path.join(resplit_dir, 'train_split_result.txt')
# res_split = sorted_bal_frames[0]

# res_dict = {}
# for i in range(5):
#     res_dict['split_{}'.format(i+1)] = res_split[1]['split'][i]
# json.dump(res_dict, open(json_file, 'w'))

# frame_nums = np.array(get_split_frame_nbs(res_split[1]['split'], score=False))
# label_sec_res = get_label_sec(res_split[1]['split'], score=False)
# context = ''
# context += 'frame_nums:\t{}\t{}\t{}\t{}\t{}\n'.format(frame_nums[0], frame_nums[1], frame_nums[2], frame_nums[3], frame_nums[4])
# context += 'total_frames:\t{}\n'.format(total_frame_nbs)
# context += 'valence_-1~0:\t{}\n'.format(label_sec_res[0])
# context += 'valence_0~0.4:\t{}\n'.format(label_sec_res[1])
# context += 'valence_0.4~1:\t{}\n'.format(label_sec_res[2])
# context += 'arousal_-1~0:\t{}\n'.format(label_sec_res[3])
# context += 'arousal_0~0.4:\t{}\n'.format(label_sec_res[4])
# context += 'arousal_0.4~1:\t{}'.format(label_sec_res[5])

# with open(record_file, 'w') as f:
#     f.writelines(context)
    
