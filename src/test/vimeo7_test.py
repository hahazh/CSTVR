import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from os import path as osp
import numpy as np
from utils.model_utils import get_model_total_params
from skimage.metrics import structural_similarity as compare_ssim
import data.core_bicubic  as core_bicubic


# 考虑中间的量化
def test_vimeo_w_quan():
    from data.vimeo_seq_dataset import Vimeo_SepTuplet
    from torchvision import transforms
    from arch.surrogate_arch import IND_inv3D
    from utils.options import yaml_load
    device = torch.device('cuda')
    weight_base_p = '/home/zhangyuantong/code/ST_rescale_open_source/CVRS/archieved/'
    base_out_p = '/home/zhangyuantong/code/ST_rescale_open_source/CVRS/output'
    Time_factor = 2
    Scale_factor = 1
    if Time_factor==2 and Scale_factor==1:
        from arch.Mynet_arch import RescalerNet
    else:
        from arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
    train_dataset_name = 'vimeo'
    test_dataset_name = 'vimeo'
    weight_p = f'{weight_base_p}/Tx{Time_factor}_Sx{Scale_factor}_{train_dataset_name}'
    out_p =  f'{base_out_p}/Tx{Time_factor}_Sx{Scale_factor}/{test_dataset_name}'
    inv_root_p = f"{weight_p}/inverter/config.yml"
    inv_opt = yaml_load(inv_root_p)['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = f"{weight_p}/inverter/model.pth"
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'],strict=True)
    
    # load rescaling model
    model_root_path = f"{weight_p}/rescaler/config.yml"
    rescale_opt = yaml_load(model_root_path)
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device) 
    rescale_weight_p =  f"{weight_p}/rescaler/model.pth"
    weight = torch.load(rescale_weight_p)
    rescale_model.load_state_dict(weight['params'],strict=True)
    rescale_model.eval()
    param = get_model_total_params(rescale_model)
    print(f'param :{param}')


    # data config
    data_opt = rescale_opt['datasets']['val']
    descibe = f'inv_model yaml {inv_root_p} \n inv_model weight {inv_weight_p} \n rescale_model yaml {model_root_path} \n rescale_model weight {weight_p} \n'
    print(descibe)
    dataset = Vimeo_SepTuplet(data_opt)
    if rescale_opt['down_type']['T']==1:
        down_t = 7
    else:
        down_t = 4
    down_size = (down_t,256//rescale_opt['down_type']['S'],448//rescale_opt['down_type']['S'])
    for ix,data in enumerate( dataset):
        imgs = data['imgs'].unsqueeze(0).cuda()
        img_p = data['path'].split('/')[-2]+'/'+data['path'].split('/')[-1]
        print(img_p)
        this_p = osp.join(out_p,img_p)

        if not os.path.exists(this_p):
            os.makedirs(this_p)
        with torch.no_grad():
            b,c,t,h,w = imgs.shape
            #quantization
            x_down = rescale_model.inference_down(imgs,down_size)
            # latent2RGB
            LR_img = model.inference_latent2RGB(x_down)
            # quant
            LR_img = LR_img.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            LR_img = LR_img.astype(np.uint8)
            LR_img_ten = torch.from_numpy(LR_img).unsqueeze(0).permute(0,4,1,2,3).cuda()/255.0
            torch.cuda.empty_cache()
            # transform back
            rev_back = model.inference_RGB2latent(LR_img_ten)
            #upsample
            out = rescale_model.inference_up(rev_back,(t,int(h),int(w)))
            x_down = x_down.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            out = out.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            rev_back = rev_back.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            quan_p = this_p+'/quant'
            if not os.path.exists(quan_p):
                os.makedirs(quan_p)
            latent_p = this_p+'/latent'
            if not os.path.exists(latent_p):
                os.makedirs(latent_p)
            rev_p = this_p+'/rev'
            if not os.path.exists(rev_p):
                os.makedirs(rev_p)
            sr_p = this_p+'/sr'
            if not os.path.exists(sr_p):
                os.makedirs(sr_p)
           
            for i in range(down_t):
                cv2.imwrite(quan_p+'/im'+str(2*i+1)+'.png',LR_img[i][:,:,::-1])
                cv2.imwrite(latent_p+'/im'+str(2*i+1)+'.png',x_down[i][:,:,::-1])
                cv2.imwrite(rev_p+'/im'+str(2*i+1)+'.png',rev_back[i][:,:,::-1])

            for i in range(7):
                cv2.imwrite(sr_p+'/im'+str(i+1)+'.png',out[i][:,:,::-1])
          
            
if __name__=='__main__':
    test_vimeo_w_quan()
    # CUDA_VISIBLE_DEVICES=4 python vimeo7_test.py
    