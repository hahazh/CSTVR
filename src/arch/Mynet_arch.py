# 输入部分先用独立卷积，再用 depth-wise 3d 卷积
from module.general_module import DepthwiseSeparableConv3d
from module.general_module import make_layer,ResidualBlock3D_NoBN,DepthwiseTransSeparableConv3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.general_module import ResBlock,make_layer
from module.sample_module import ModuleNet3D
from module.Quantization import Quantize_ste
from module.inv_module import D2DTInput
from module.general_module import SpaceTimePixelShuffle,SpaceTimePixelUnShuffle
from utils.registry import ARCH_REGISTRY

import os
class Unet_3d(nn.Module):
    """Residual blocks with a convolution in front.

    """

    def __init__(self):
        super(Unet_3d, self).__init__()
        channels = [64,72,96]

        self.conv1 = nn.Sequential(
            ResidualBlock3D_NoBN(channels[0]),
           DepthwiseSeparableConv3d(channels[0],channels[0],stride=(1,1,1)),
        )

        self.Down1 = nn.Sequential(
            ResidualBlock3D_NoBN(channels[0]),
           DepthwiseSeparableConv3d(channels[0],channels[1],stride=(1,2,2)),
        )

        self.Down2 = nn.Sequential(
            ResidualBlock3D_NoBN(channels[1]),
             DepthwiseSeparableConv3d(channels[1],channels[2],stride=(1,2,2)),
        )

        self.bottle = nn.Sequential(
            ResidualBlock3D_NoBN(channels[2]),
             DepthwiseSeparableConv3d(channels[2],channels[2],stride=(1,1,1)),
        )

        self.up1 = nn.Sequential(
            ResidualBlock3D_NoBN(channels[2]),
           DepthwiseTransSeparableConv3d(channels[2],channels[1],stride=(1,2,2)),
        )

        self.up2 = nn.Sequential(
            ResidualBlock3D_NoBN(channels[1]*2),
           DepthwiseTransSeparableConv3d(channels[1]*2,channels[0],stride=(1,2,2)),
        )
        self.conv_last = nn.Sequential(
            ResidualBlock3D_NoBN(channels[0]*2),
           DepthwiseSeparableConv3d(channels[0]*2,channels[0],stride=(1,1,1)),
        )

    def forward(self, feats):
        raw1 = self.conv1(feats)
        down1 = self.Down1(raw1)
        down2 = self.Down2(down1)
        bottle = self.bottle(down2)
        up1 = self.up1(bottle)
        up2 = self.up2(torch.cat([up1,down1],dim=1))
        back = self.conv_last(torch.cat([up2,raw1],dim=1))
        
        return back




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
       
        # self.unet_3d = Unet_3d()
        self.feat_extractor = make_layer(ResidualBlock3D_NoBN,three_D_res_num,num_feat=self.mid_channels)
        self.down_head = ModuleNet3D(self.mid_channels)

    def forward(self, imgs,target_size):

        b,c,t,h,w = imgs.shape

        # b,3,t,h,w -> b*t,3,h,w
        # imgs_permute = imgs.permute(0,2,1,3,4).contiguous().view(b*t,c,h,w)
        # b*t,3,h,w -> b*t,c,h,w
        feats = self.first_block(imgs)
        # feats = self.feat_extract_2D(imgs_permute)
        # b*t,3,h,w -> b,c,t,h,w
        # feats_permute = feats.view(b,t,self.mid_channels,h,w).permute(0,2,1,3,4).contiguous()
        # b,c,t,h,w -> b,c,t,h,w extract 3d features
        feats_3d =  self.feat_extractor(feats)

        down_feat = self.down_head(feats_3d,target_size)

        return down_feat


class STREV_up(nn.Module):

    def __init__(self,opt):
        super(STREV_up, self).__init__()
   
        self.mid_channels =  opt['mid_channels']
        three_D_res_num =  opt['three_D_res_num']
        self.first_block = nn.Sequential(nn.Conv3d(opt['in_channels'],  self.mid_channels//4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        self.relu = nn.PReLU()
        # self.feat_extract_2D = ResidualBlocksWithInputConv(3,64,2)
        # flow_weight_p = '/home/zhangyuantong/code/ST_rescale/src/weight/spynet.pth'
        # 上采样进来的是特征，还用光流对齐，这是非常奇怪的，感觉不如纯3D卷积
        # self.bi_rnn = BiRNN_encoder(flow_weight_p=flow_weight_p,out_c=self.mid_channels)
        self.dense3d_backbone = make_layer( D2DTInput,three_D_res_num,channel_in=opt['mid_channels'],channel_out=opt['mid_channels'])
        self.fuse_channel_in = nn.Sequential(nn.Conv3d(opt['mid_channels'],  opt['fuse_channels']*4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        self.fuse_channel_out = nn.Sequential(nn.Conv3d(opt['mid_channels'],  opt['fuse_channels']*4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        self.head = ModuleNet3D(in_channel=opt['fuse_channels'])
        self.STPS = SpaceTimePixelShuffle(r=1,s=2)
        self.UNSTPS  = SpaceTimePixelUnShuffle(r=1,s=2)
        # self.feat_extract_3D = make_layer(ResidualBlock3D_NoBN,three_D_res_num,num_feat=self.mid_channels)

    def forward(self, imgs,target_size):

        feats = self.first_block(imgs)
        # print('before unshuffle feats shape',feats.shape)
        feats = self.UNSTPS(feats)
        # print('after unshuffle feats shape',feats.shape)
        feats = self.fuse_channel_out(self.dense3d_backbone(feats))+self.fuse_channel_in(feats)
        # print('before shuffle feats shape',feats.shape)
        feats = self.STPS(feats)
        # print('after shuffle feats shape',feats.shape)
        # feats = self.feat_extract_3D(feats)
        x_up = self.head(feats,target_size)


        return x_up

@ARCH_REGISTRY.register()
class RescalerNet(nn.Module):

    def __init__(self,opt):
        super(RescalerNet, self).__init__()
        down_opt = opt['down_opt']
        up_opt = opt['up_opt']


        self.quan_type = opt['quan_type']

        self.control_rate = opt['control_rate']
        self.downet = STREV_down(down_opt)
        self.upnet = STREV_up(up_opt)
        self.quan_layer = Quantize_ste(min_val=0.0,max_val=1.0)
      
    def forward(self, imgs,down_size,inference = False):
        B,C,T,H,W = imgs.shape
        if self.quan_type =='h265':
            down_feat = self.inference_down_w_compress(imgs,down_size = down_size,qp  =None)
        else:
            down_feat = self.downet(imgs,target_size = down_size)
            down_feat = torch.clamp(down_feat, 0, 1)
            # print('down_feat shape',down_feat.shape)
            if inference:
                down_feat = down_feat*255.0
                down_feat= down_feat.to(torch.uint8)
                down_feat = down_feat.to(torch.float32)/255.0
            else:
                # print('use round !!!!')
                down_feat = self.quan_layer(down_feat)
        # shape (1,3,5,64,64)
        # down_feat_permute = down_feat.permute(0,2,1,3,4).view(B,down_size[0],3,down_size[1],down_size[2])
        up_imgs = self.upnet(down_feat,target_size =(T,H,W)).contiguous()
        # print('up_imgs shape',up_imgs.shape)
        return down_feat,up_imgs
    @torch.no_grad()
    def inference_down(self,imgs,down_size):
        out = self.downet(imgs,target_size = down_size)
        out = torch.clamp(out, 0, 1)
        return out
    
    def inference_compress(self,imgs,qp):
       
        # print('down_feat shape',down_feat.shape)
        batch,channel,t_len,h,w = imgs.shape
        # print(f'qp is {qp}')
        if self.control_rate:
            this_stream = self.Quantization_H265_Stream.open_writer(imgs.device,w,h,qp)
        else:
            this_stream = self.Quantization_H265_Stream.open_writer(imgs.device,w,h,qp)
       
        # down_feat = down_feat.squeeze(0)
        for b in range(batch):
            this_ten = imgs[b].permute(1,0,2,3)
            # print('this_ten shape',this_ten.shape)
            self.Quantization_H265_Stream.write_multi_frames(this_ten)

        
        bpp = self.Quantization_H265_Stream.close_writer()
        self.Quantization_H265_Stream.open_reader()
        outs = []
        for b in range(batch):
            v_seg = self.Quantization_H265_Stream.read_multi_frames(t_len)
            outs+=[v_seg]
        out = torch.cat(outs,dim=0)
        h,w = out.size(-2),out.size(-1)
        
        out = out.reshape(batch,t_len,3,h,w).permute(0,2,1,3,4).cuda()
        # print(down_feat.shape)
        # assert down_feat.shape == out.shape
        # print('this stream!!!!!',this_stream)
        # os.remove(this_stream)
        return out,bpp

    def inference_down_w_compress(self,imgs,down_size,qp):
        down_feat =  self.downet(imgs,target_size = down_size)
        # print('down_feat shape',down_feat.shape)
        batch,channel,t_len,down_h,down_w = down_feat.shape
        if self.control_rate:
            this_stream = self.Quantization_H265_Stream.open_writer(down_feat.device,down_w,down_h,qp)
        else:
            this_stream = self.Quantization_H265_Stream.open_writer(down_feat.device,down_w,down_h,None)
        # down_feat = down_feat.squeeze(0)
        for b in range(batch):
            this_ten = down_feat[b].permute(1,0,2,3)
            # print('this_ten shape',this_ten.shape)
            self.Quantization_H265_Stream.write_multi_frames(this_ten)

        # print('use h265 !!!!')
        img_distri = self.Quantization_H265_Stream.close_writer()
        self.Quantization_H265_Stream.open_reader()
        outs = []
        for b in range(batch):
            v_seg = self.Quantization_H265_Stream.read_multi_frames(t_len)
            outs+=[v_seg]
        out = torch.cat(outs,dim=0)
        h,w = out.size(-2),out.size(-1)
        
        out = out.reshape(batch,t_len,3,h,w).permute(0,2,1,3,4).cuda()
        # print(down_feat.shape)
        # assert down_feat.shape == out.shape
        # print('this stream!!!!!',this_stream)
        os.remove(this_stream)
        return out
    @torch.no_grad()
    def inference_up(self,rev_back,size):
        B,C,T,H,W = rev_back.shape
        # rev_back = rev_back.permute(0,2,1,3,4)
        up_imgs = self.upnet(rev_back,target_size =size)
        up_imgs = torch.clamp(up_imgs, 0, 1)
        # up_imgs = up_imgs.permute(0,2,1,3,4).contiguous().view(B,C,size[0],size[1],size[2])
        return up_imgs
    @torch.no_grad()
    def inference_down_w_cpu_cache(self,imgs,down_size):
        down_feat = self.downet(imgs,target_size = down_size)
        return down_feat
    @torch.no_grad()
    def inference_up_w_cpu_cache(self,rev_back,size):
        B,C= rev_back.shape[:2]
        T,H,W = size
        y_grid = 4
        x_grid = 4
        y_size = H//y_grid
        x_size = W//x_grid
        patch_list = []
        for i in range(y_grid):
            for j in range(x_grid):
                patch = [i*y_size,(i+1)*y_size,j*x_size,(j+1)*x_size]
                # print(patch)
                patch_list.append(patch)
     
        rev_back = rev_back.permute(0,2,1,3,4)
        propa_feats = self.upnet.bi_rnn(rev_back)
        propa_feats = propa_feats.permute(0,2,1,3,4).contiguous()
        propa_feats = self.upnet.relu(self.upnet.feat_extract_3D(propa_feats))
        self.upnet.head.query_rgb_3d(propa_feats,size)
   
        