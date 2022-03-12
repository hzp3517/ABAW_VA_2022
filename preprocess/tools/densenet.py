'''
得到 DenseNet(Affectnet, FER+) 特征
'''

import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms
import pickle

sys.path.append('/data8/hzp/ABAW_VA_2022/code/preprocess')
from tools.base_worker import BaseWorker
from tools.denseface.model.dense_net import DenseNetEncoder

class DensenetExtractor(BaseWorker):
    def __init__(self, model_path, gpu_id=0):
        self.device = torch.device("cuda:{}".format(gpu_id))
        # model_path = '/data9/datasets/AffectNetDataset/combine_with_fer/results/densenet100_adam0.0002_0.0/ckpts/model_step_12.pt'
        self.mean = 101.63449 # 在VA任务的训练集上计算得到
        self.std = 59.74126
        self.model = DenseNetEncoder(gpu_id=gpu_id).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        
    def preprocess(self, img):
        '''
        直接从保存好的h5文件中读出64*64的图像数据。

        input:
        - img: [bs x 64 x 64]

        return:
        - img: [bs x 1 x 64 x 64]
        '''
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(1)
        return img
    
    
    def chunk(self, lst, chunk_size):
        '''
        如果一个目录下的人脸图片过多，超过`chunk_size`张，就每`chunk_size`张截断一次
        '''
        index = 0
        while chunk_size * index < len(lst):
            yield lst[index*chunk_size: (index+1)*chunk_size]
            index += 1


    def __call__(self, img_list, chunk_size=64):
        '''
        input:
        - img_list: [bs x 224 x 224 x 3] (BGR)
        - chunk_size: batch size when extract imgs
        '''
        img_list = self.preprocess(img_list)
        video_feat = []
        for faces_bs in self.chunk(img_list, chunk_size):
            faces_bs = faces_bs.to(self.device)
            with torch.no_grad():
                # feats, _ = self.model(faces_bs)
                feats = self.model(faces_bs)
            feats = feats.detach().cpu().numpy()
            video_feat.append(feats)
        video_feat = np.concatenate(video_feat, axis=0) # 把截断的每一段再接回来
        return video_feat


if __name__ == '__main__':
    # img_path = '/data9/datasets/Aff-Wild2/cropped_aligned/1-30-1280x720/00001.jpg'
    # img = cv2.imread(img_path)
    # img_list = np.expand_dims(img, 0).astype(np.float32)

    model_path = '/data8/hzp/ABAW_VA_2022/pretrained_models/denseface_run1_1_net_dense_mse1239.pth'
    img_list = (np.random.random((5, 64, 64)) * 255).astype(np.int32).astype(np.float32) # 生成伪图像数据
    affectnet_extractor = DensenetExtractor(model_path, gpu_id=1)
    fts = affectnet_extractor(img_list)
    print(fts.shape) # (5, 342)
    print(fts)