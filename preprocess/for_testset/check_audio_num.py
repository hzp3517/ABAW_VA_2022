'''
确认一下处理后的audio文件的数量和cropped img的目录数量一致，且名称对齐。
'''
import os

audio_dir = '/data2/hzp/ABAW_VA_2022/processed_data/audios/'
face_root = '/data2/hzp/Aff-Wild2/cropped_aligned/'

audio_lst = [i.split('.')[0] for i in os.listdir(audio_dir)]
visual_lst = [i for i in os.listdir(face_root)]

assert len(audio_lst) == len(visual_lst)

for i in audio_lst:
    assert i in visual_lst

for i in visual_lst:
    assert i in audio_lst

print('------done----------')