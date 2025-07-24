
from module.general_module import make_layer,ResidualBlock3D_NoBN
import torch
import torch.nn as nn
from module.general_module import make_layer
from module.sample_module import ModuleNet3D
from module.Quantization import Quantize_ste
from module.inv_module import D2DTInput
from module.my_3D_module import BasicLayer
from module.general_module import SpaceTimePixelShuffle,SpaceTimePixelUnShuffle
import os
from basicsr.utils.registry import ARCH_REGISTRY


class STREV_down(nn.Module):

    def __init__(self,opt):
        super(STREV_down, self).__init__()
     
        self.mid_channels = opt['mid_channels']

        three_D_res_num = opt['three_D_res_num']
        kernel_size =  opt['kernel_size']
        self.first_block = nn.Sequential(nn.Conv3d(3,  self.mid_channels, 
                              kernel_size=kernel_size, 
                              padding=(1, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
    
        self.feat_extractor = make_layer(ResidualBlock3D_NoBN,three_D_res_num,num_feat=self.mid_channels)
        self.down_head = ModuleNet3D(self.mid_channels)

    def forward(self, imgs,target_size):
        feats = self.first_block(imgs)
        feats_3d =  self.feat_extractor(feats)
        down_feat = self.down_head(feats_3d,target_size)
        return down_feat


class STREV_up(nn.Module):

    def __init__(self,opt):
        super(STREV_up, self).__init__()      
        input_dim = opt['in_channels']
        dim = opt['dim']
        three_D_res_num = opt['three_D_res_num']
        num_heads =  opt['num_heads']
        window_size = opt['window_size']
        depths =  opt['depths']
        fuse_channels = opt['fuse_channels']
        self.use_shortcut =  opt['use_shortcut']
        self.use_shuffle = opt['use_shuffle']
        total_block_num = len(depths)

        if self.use_shuffle:
            self.STPS = SpaceTimePixelShuffle(r=1,s=2)
            self.UNSTPS  = SpaceTimePixelUnShuffle(r=1,s=2)
            self.first_block = nn.Sequential(nn.Conv3d(input_dim,  dim//4, 
                                kernel_size=3, 
                                padding=(1, 1, 1), 
                                stride=(1,1,1)),
                                nn.PReLU()) 
            self.fuse_channel_out = nn.Sequential(nn.Conv3d(dim,  fuse_channels*4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
            if self.use_shortcut:
                self.fuse_channel_in = nn.Sequential(nn.Conv3d(dim,  fuse_channels*4, 
                                    kernel_size=3, 
                                    padding=(1, 1, 1), 
                                    stride=(1,1,1)),
                                    nn.PReLU()) 
        else:
            self.first_block = nn.Sequential(nn.Conv3d(input_dim,  dim, 
                                kernel_size=3, 
                                padding=(1, 1, 1), 
                                stride=(1,1,1)),
                                nn.PReLU())
            self.fuse_channel_out = nn.Sequential(nn.Conv3d(dim,  fuse_channels, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU())
            if self.use_shortcut:
                self.fuse_channel_in = nn.Sequential(nn.Conv3d(dim,  fuse_channels, 
                                    kernel_size=3, 
                                    padding=(1, 1, 1), 
                                    stride=(1,1,1)),
                                    nn.PReLU()) 
        self.relu = nn.PReLU()
        self.layers = nn.ModuleList()
        if total_block_num>0:
            for i in range(total_block_num):
                three_d_block = make_layer( D2DTInput,three_D_res_num,channel_in=dim,channel_out=dim,shortcut = self.use_shortcut)
                self.layers.append(three_d_block)
                swin_block =  BasicLayer(dim=dim,window_size=window_size,depth=depths[i],num_heads=num_heads[i],)
                self.layers.append(swin_block)
        else:
            three_d_block = make_layer( D2DTInput,three_D_res_num,channel_in=dim,channel_out=dim,shortcut = self.use_shortcut)
            self.layers.append(three_d_block)
      
        self.head = ModuleNet3D(in_channel=fuse_channels)
       

    def forward(self, imgs,target_size):
        
        x = self.first_block(imgs)

        if self.use_shuffle:
         
            x = self.UNSTPS(x)
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        if self.use_shortcut:
            x = self.fuse_channel_out(x)+self.fuse_channel_in(shortcut)
        else:
            x = self.fuse_channel_out(x)
        if self.use_shuffle:
        
            x = self.STPS(x)

        x_up = self.head(x,target_size)
        return x_up

@ARCH_REGISTRY.register()
class Rescaler_MixNet(nn.Module):

    def __init__(self,opt):
        super(Rescaler_MixNet, self).__init__()
        down_opt = opt['down_opt']
        up_opt = opt['up_opt']
        self.quan_type = opt['quan_type']
        self.control_rate = opt['control_rate']
        self.downet = STREV_down(down_opt)
        self.upnet = STREV_up(up_opt)
        self.quan_layer = Quantize_ste(min_val=0.0,max_val=1.0)

        
    def forward(self, imgs,down_size):
        B,C,T,H,W = imgs.shape
        down_feat = self.downet(imgs,target_size = down_size)
        down_feat = self.quan_layer(down_feat)
        up_imgs = self.upnet(down_feat,target_size =(T,H,W)).contiguous()
        return down_feat,up_imgs


    @torch.no_grad()
    def inference_down(self,imgs,down_size):

        return self.downet(imgs,target_size = down_size)
    @torch.no_grad()
    def inference_up(self,rev_back,size):
        B,C,T,H,W = rev_back.shape
        up_imgs = self.upnet(rev_back,target_size =size)
        return up_imgs
        
