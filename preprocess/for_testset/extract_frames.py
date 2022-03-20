import subprocess
import os
from tqdm import tqdm

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

video_dir = '/data2/hzp/Aff-Wild2/videos/'
save_root = '/data2/hzp/ABAW_VA_2022/processed_data/frames'
video_path_lst = os.listdir(video_dir)
mkdir(save_root)

for video in tqdm(video_path_lst):
    video_path = os.path.join(video_dir, video)
    video_id = get_basename(video_path)
    save_path = os.path.join(save_root, video_id)
    mkdir(save_path)
    fps_check = subprocess.check_output('ffprobe '+ video_path +' 2>&1 | grep fps',shell=True)
    fps_check = str(fps_check)
    fps = float(fps_check.split(' fps')[0].split(',')[-1][1:])
    subprocess.call('ffmpeg -loglevel panic -i '+ video_path +' -vf fps='+ str(fps)+' '+ save_path +'/%05d.jpg',shell=True)    ## download_location is the location to where you want the extracted videoframes to be downloaded
