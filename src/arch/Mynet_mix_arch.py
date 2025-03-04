# 输入部分先用独立卷积，再用 depth-wise 3d 卷积


from module.general_module import make_layer,ResidualBlock3D_NoBN
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # total_block_num 表示整个block的数量
        # swin_block_num 表示每个block里面的swin数量
        # 3d_dense_num 表示每个block里面3d dense的数量

       
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
       
        # self.feat_extract_3D = make_layer(ResidualBlock3D_NoBN,three_D_res_num,num_feat=self.mid_channels)

    def forward(self, imgs,target_size):
        
        x = self.first_block(imgs)

        if self.use_shuffle:
            # print(f'before unshuffle {x.shape}')
            x = self.UNSTPS(x)
            # print(f'before shuffle {x.shape}')
        shortcut = x
        for layer in self.layers:
            x = layer(x)
        if self.use_shortcut:
            x = self.fuse_channel_out(x)+self.fuse_channel_in(shortcut)
        else:
            x = self.fuse_channel_out(x)
            # print('no shortcut')
        # feats = self.feat_extract_3D(feats)
        if self.use_shuffle:
            # print(f'before shuffle {x.shape}')
            x = self.STPS(x)
            # print(f'after shuffle {x.shape}')
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
        h265_q = 9
        h265_keyint = -1
        scale = 2
        print('Rescaler_MixNet')
        
    def forward(self, imgs,down_size):
        B,C,T,H,W = imgs.shape
        if self.quan_type =='h265':
            down_feat,_ = self.inference_down_w_compress(imgs,down_size = down_size,qp  =None)
        elif self.quan_type =='H265_Suggrogate':
            down_feat = self.downet(imgs,target_size = down_size)
            down_feat,mimick_loss = self.Quantization_H265_Stream(down_feat)
        else:
            down_feat = self.downet(imgs,target_size = down_size)
            down_feat = self.quan_layer(down_feat)
            # shape (1,3,5,64,64)
        # down_feat_permute = down_feat.permute(0,2,1,3,4).view(B,down_size[0],3,down_size[1],down_size[2])

        up_imgs = self.upnet(down_feat,target_size =(T,H,W)).contiguous()

        if self.quan_type =='h265':
            return down_feat,up_imgs
        elif self.quan_type =='H265_Suggrogate':
            return down_feat,up_imgs,mimick_loss
        else:
            return down_feat,up_imgs



    @torch.no_grad()
    def inference_down(self,imgs,down_size):
        return self.downet(imgs,target_size = down_size)
    def inference_compress(self,imgs,qp):
       
        # print('down_feat shape',down_feat.shape)
        batch,channel,t_len,h,w = imgs.shape
        print(f'qp is {qp}')
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

    # @torch.no_grad()
    def inference_down_w_compress(self,imgs,down_size,qp):
        down_feat =  self.downet(imgs,target_size = down_size)
        # print('down_feat shape',down_feat.shape)
        batch,channel,t_len,down_h,down_w = down_feat.shape
        print(f'qp is {qp}')
        if self.control_rate:
            this_stream = self.Quantization_H265_Stream.open_writer(down_feat.device,down_w,down_h,qp)
        else:
            this_stream = self.Quantization_H265_Stream.open_writer(down_feat.device,down_w,down_h,None)
       
        # down_feat = down_feat.squeeze(0)
        for b in range(batch):
            this_ten = down_feat[b].permute(1,0,2,3)
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
    
    @torch.no_grad()
    def inference_up(self,rev_back,size):
        B,C,T,H,W = rev_back.shape
        # rev_back = rev_back.permute(0,2,1,3,4)
        up_imgs = self.upnet(rev_back,target_size =size)
        # up_imgs = up_imgs.permute(0,2,1,3,4).contiguous().view(B,C,size[0],size[1],size[2])
        return up_imgs
    @torch.no_grad()
    def inference_down_w_cpu_cache(self,imgs,down_size):
        down_feat = self.downet(imgs,target_size = down_size)
        return down_feat
        