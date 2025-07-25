from utils.registry import ARCH_REGISTRY
import torch
import torch.nn as nn
from arch.Mynet_arch import STREV_down
from arch.InvNN_3D_arch import my_InvNN
from module.inv_module import D2DTInput_dense
from module.Quantization import Quantize_ste
from module.general_module import SpaceTimePixelShuffle,SpaceTimePixelUnShuffle


@ARCH_REGISTRY.register()
class IND_inv3D(nn.Module):
    def __init__(self,opt):
        super(IND_inv3D, self).__init__()
        self.opt = opt
        self.downsample = STREV_down(opt['down_opt'])
        self.quant_type = opt['down_opt']['quant_type']
        self.inv_block =  my_InvNN(mode = 'train',channel_in=3,subnet_constructor =D2DTInput_dense ,block_num=opt['block_num'])
        self.quan_layer = Quantize_ste(min_val=0.0,max_val=1.0)
    
        self.st_shuffle = SpaceTimePixelShuffle(r=1,s=2)
        self.st_unshuffle = SpaceTimePixelUnShuffle(r=1,s=2)
    @torch.no_grad()
    def inference_down(self,imgs,down_size):
        x_down = self.downsample(imgs,down_size)
        return x_down
    
    @torch.no_grad()
    def inference_latent2RGB(self,x_down):
        LR_img_stack,jac = self.inv_block(x_down,cal_jacobian = True)
        LR_img = self.st_shuffle(self.st_shuffle(LR_img_stack))
        LR_img = torch.clamp(LR_img, 0, 1)
        return LR_img
    @torch.no_grad()
    def inference_RGB2latent(self,LR_img):
        LR_latent = self.st_unshuffle(self.st_unshuffle(LR_img))
        rev_back = self.inv_block(LR_latent,rev = True)
        rev_back = torch.clamp(rev_back, 0, 1)
        return rev_back

    def forward(self,x,down_size):
        x_down = self.inference_down(x,down_size)
        LR_img_stack,jac = self.inv_block(x_down,cal_jacobian = True)
        LR_img = self.st_shuffle(self.st_shuffle(LR_img_stack))
        LR_img_quan = self.quan_layer(LR_img)
        LR_latent = self.st_unshuffle(self.st_unshuffle(LR_img_quan))
        rev_back = self.inv_block(LR_latent,rev = True)
            
        return LR_img_quan,rev_back,x_down