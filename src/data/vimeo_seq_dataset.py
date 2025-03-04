from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils.registry import DATASET_REGISTRY

import data.core_bicubic  as core_bicubic
import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
@DATASET_REGISTRY.register()
class Vimeo_SepTuplet(Dataset):  # load train dataset
    def __init__(self, opt):
        super(Vimeo_SepTuplet, self).__init__()

        # opt['image_dir']
        # opt['file_list']
        # opt['image_dir']
        # opt['data_augmentation']
        # opt['image_dir']
        self.opt = opt
        self.load_mode = opt['load_mode']
        self.gt_size = opt['gt_size']
        alist = [line.rstrip() for line in open(os.path.join( opt['image_dir'],
                                                                  opt['file_list']))]  # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
        self.image_filenames = [os.path.join( opt['image_dir'], x) for x in alist]  # get image path list

        self.data_augmentation =  opt['data_augmentation']  # flip and rotate
        self.lr_idx = opt['T_index']
        self.down_size = opt['S_scale']

    def train_process(self, GT, flip_h=True, rot=True, flip_v=True, converse=True):  # input:list, target:PIL.Image
        if random.random() < 0.5 and flip_v:
            GT = [LR[::-1, :, :].copy() for LR in GT]
        if random.random() < 0.5 and flip_h:
            GT = [LR[:, ::-1, :].copy() for LR in GT]
        if rot and random.random() < 0.5:
            GT = [LR.transpose(1, 0, 2).copy() for LR in GT]
        if converse and random.random() < 0.5:
            GT = GT.copy()[::-1]
        return GT

    def load_img(self, image_path):

        def crop_256(img_list, H, W):
            crop_size = self.gt_size
            rnd_h = random.randint(0, H - crop_size)
            rnd_w = random.randint(0, W - crop_size)

            img_gt_crop = [v[rnd_h:rnd_h + crop_size, rnd_w:rnd_w + crop_size, :] for v in img_list]
            return img_gt_crop

        HR = []
        for img_num in range(7):
            img_gt = Image.open(os.path.join(image_path, 'im{}.png'.format(img_num + 1))).convert('RGB')

            HR.append(img_gt)
        HR = [np.asarray(HR) for HR in HR]
        H, W = HR[0].shape[:2]
        if self.load_mode == 'train':
            HR = crop_256(HR, H, W)
    
        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        GT = self.load_img(self.image_filenames[index])
        name = self.image_filenames[index]
        if self.load_mode == 'train':
            if self.data_augmentation:
                GT = self.train_process(GT)

        GT = np.asarray(GT).astype('float32')/255.0  # numpy, [T,H,W,C], stack with temporal dimension

        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]


        # t,h,w,c -> c,t,h,w
        

        GT = torch.from_numpy(GT).permute(3,0,1,2).contiguous()
        
        # idx = [0,2,4,6]
        LR = GT[:,self.lr_idx ,:,:]
        # print('LR shape',LR.shape)
        if self.down_size!=1:
            LR = core_bicubic.imresize(LR.permute(1,0,2,3).contiguous(),sizes = (h//self.down_size,w//self.down_size))
            LR = LR.permute(1,0,2,3).contiguous()
        return {"imgs":GT,"LR":LR,"path":name}
      
    def __len__(self):
        return len(self.image_filenames)  # total video number. not image number