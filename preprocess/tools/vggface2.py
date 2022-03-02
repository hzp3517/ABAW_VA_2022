'''
得到VGGFace2特征
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
from tools.vggface2_pytorch.models.resnet import ResNet, Bottleneck, resnet50
from tools.vggface2_pytorch.models.senet import SENet, senet50

N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained. Pretrained weights fc layer has 8631 outputs.

class Vggface2Extractor(BaseWorker):
    def __init__(self, arch_type='resnet50_ft', gpu_id=0):
        '''
        - arch_type: network architecture type (default: resnet50_ft)
            - resnet50_ft ResNet-50 which are first pre-trained on MS1M, and then fine-tuned on VGGFace2
            - senet50_ft SE-ResNet-50 trained like resnet50_ft
            - resnet50_scratch ResNet-50 trained from scratch on VGGFace2
            - senet50_scratch SE-ResNet-50 trained like resnet50_scratch
        '''
        weight_file = '/data8/hzp/models/vggface2-pytorch/resnet50_ft_weight.pkl'
        self.device = torch.device("cuda:{}".format(gpu_id))

        if 'resnet' in arch_type:
            # self.model = resnet50(num_classes=N_IDENTITY, include_top=False)
            self.model = self.load_resnet50_weight(weight_file, num_classes=N_IDENTITY, include_top=False)
        else:
            self.model = senet50(num_classes=N_IDENTITY, include_top=False)

        # 权重文件比较古老，使用了 pickle 存储并且编码使用了 latin1，无法使用torch.load()加载
        # ckpt = torch.load(weight_file)
        # self.model.load_state_dict(ckpt['model_state_dict'])
        # assert ckpt['arch'] == arch_type

        self.model.to(self.device) # 将网络放到指定的设备上
        self.model.eval()


    def load_resnet50_weight(self, weights_path=None, **kwargs):
        """Constructs a ResNet-50 model.
        """
        model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if weights_path:
            import pickle
            with open(weights_path, 'rb') as f:
                obj = f.read()
            weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
            model.load_state_dict(weights)
        return model


    def preprocess_official(self, img_path, mean_bgr=[91.4953, 103.8827, 131.0912]):
        '''
        按照： https://github.com/cydonia999/VGGFace2-pytorch/blob/master/datasets/vgg_face2.py 中的预处理过程

        input:
        - img_path: original face image path
        - mean_bgr: from resnet50_ft.prototxt

        return:
        - img: [3 x 224 x 224]
        '''
        img = Image.open(img_path)
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img


    def preprocess_hzp(self, img, mean_bgr=[86.19, 92.31, 120.105]):
        '''
        直接从保存好的h5文件中读出224*224*3的图像数据。

        input:
        - img: [bs x 224 x 224 x 3] (BGR)
        - mean_bgr: cal on Aff-Wild2 subset (5%)

        return:
        - img: [bs x 3 x 224 x 224]
        '''
        img -= mean_bgr
        img = img.transpose(0, 3, 1, 2)  # bs x C x H x W
        img = torch.from_numpy(img).float()
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
        img_list = self.preprocess_hzp(img_list)
        video_feat = []
        for faces_bs in self.chunk(img_list, chunk_size):
            faces_bs = faces_bs.to(self.device)
            feats = self.model(faces_bs)
            feats = feats.view(feats.size(0), -1)
            feats = feats.data.cpu().numpy()
            video_feat.append(feats)
        video_feat = np.concatenate(video_feat, axis=0) # 把截断的每一段再接回来
        return video_feat


if __name__ == '__main__':
    img_path = '/data9/datasets/Aff-Wild2/cropped_aligned/1-30-1280x720/00001.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_list = np.expand_dims(img, 0).astype(np.float32)

    # img_list = (np.random.random((5, 224, 224, 3)) * 255).astype(np.int32).astype(np.float32) # 生成伪图像数据
    vggface2_extractor = Vggface2Extractor(gpu_id=1)
    fts = vggface2_extractor(img_list)
    print(fts.shape) # (5, 2048)
    print(fts)