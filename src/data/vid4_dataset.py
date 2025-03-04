from os import path as osp
from torch.utils import data as data
from utils.registry import DATASET_REGISTRY
import numpy as np
import cv2
import os
import torch
import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.model_utils import get_model_total_params

@DATASET_REGISTRY.register()
class Vid4(Dataset):  # load train dataset
    def __init__(self, data_dir, transform,gop_size = 1,overlay = True):
        super(Vid4, self).__init__()
     
        self.vid4_dir = data_dir
        self.total_len = len(os.listdir(self.vid4_dir))
        self.scenes = ('calendar','city','foliage','walk')
        self.gop_size = gop_size
        self.alist = []
       
        for s in self.scenes:
            s_ls =  sorted(glob.glob(self.vid4_dir+'/'+s+'/*.png'))
            if self.gop_size==1:
                seq_len = len(s_ls)
            else:
                seq_len = self.gop_size
            # 
            # reverse_num = 0 if len(s_ls)%seq_len==0 else seq_len - len(s_ls)%seq_len
            # s_ls = s_ls + list(reversed(s_ls))[1:1+reverse_num] 
            
              # load image_name from image name list, note: label list of vimo90k is video name list, not image name list.
            for i in range(0, len(s_ls), seq_len-1):
                if (i+seq_len)>=len(s_ls):
                    break
                self.alist.append( s_ls[i:i+seq_len])
        self.transform = transform  # To_tensor
        # print(self.alist)
    def load_img(self, image_path):

        HR = []
        for each  in image_path:
            img_gt = Image.open(each).convert('RGB')

            img_gt = self.transform(img_gt)
            HR.append(img_gt)
        # HR = [np.asarray(HR) for HR in HR]
        
        return HR

    def __getitem__(self, index):

        # GT shape 长度为7的list
        # return self.alist[index]
        GT = self.load_img(self.alist[index])
        this_scene = [each.split('/')[-2]+'/'+each .split('/')[-1] for each in self.alist[index]]
        # print(this_scene)
       
        GT = torch.stack(GT,dim=0)  # numpy, [T,H,W,C], stack with temporal dimension
        return this_scene, GT.permute(1,0,2,3)

    def __len__(self):
        return len(self.alist)  # total video number. not image number