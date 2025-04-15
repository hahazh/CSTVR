import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data.SPMCS import SPMCS_arb
from torch.utils.data import DataLoader
import numpy as np
from os import path as osp
device = torch.device('cuda')

def init_model(weight_base_p, base_out_p, test_dataset_name):
    from arch.Mynet_arch import RescalerNet
    from utils.options import yaml_load
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    from utils.model_utils import get_model_total_params
    from arch.IMSM import IND_inv3D

    out_p = os.path.join(base_out_p, test_dataset_name)

    # Load inversion model
    inv_root_p = os.path.join(weight_base_p, 'inverter/config.yml')
    inv_opt = yaml_load(inv_root_p)['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = os.path.join(weight_base_p, 'inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)

    # Load rescaling model
    model_root_path = os.path.join(weight_base_p, 'rescaler/config.yml')
    rescale_opt = yaml_load(model_root_path)
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device)
    rescale_weight_p = os.path.join(weight_base_p, 'rescaler/model.pth')
    weight = torch.load(rescale_weight_p)
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
    param = get_model_total_params(rescale_model)
    print(f'param: {param}')
    return model, rescale_model

def single_inference(model,rescale_model,imgs,down_shape):

    T,H,W = imgs.shape[2:]
    y_grid = 2
    x_grid = 2
    y_size = H//y_grid
    x_size = W//x_grid
    y_d_size = down_shape[1]//y_grid
    x_d_size = down_shape[2]//x_grid
    B = 1
    place_holder_la = torch.zeros(B,3,down_shape[0],down_shape[1],down_shape[2]).cuda()
    place_holder_back = torch.zeros(B,3,down_shape[0],down_shape[1],down_shape[2]).cuda()
    place_holder_quan = torch.zeros(B,3,down_shape[0],down_shape[1],down_shape[2]).cuda()
    place_holder = torch.zeros(B,3,T,H,W).cuda()
    with torch.no_grad():
        for i in range(y_grid):
            for j in range(x_grid):
                patch = [i*y_size,(i+1)*y_size,j*x_size,(j+1)*x_size]
                patch_d = [i*y_d_size,(i+1)*y_d_size,j*x_d_size,(j+1)*x_d_size]
                img_pa = imgs[:,:,:,patch[0]:patch[1],patch[2]:patch[3]]
                down_size_p = (down_shape[0],down_shape[1]//y_grid,down_shape[2]//x_grid)

                
                x_down = rescale_model.inference_down(img_pa,down_size_p)
                LR_img = model.inference_latent2RGB(x_down)
                # quant
                LR_img = LR_img.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
                LR_img = LR_img.astype(np.uint8)
                
            
                LR_img_ten = torch.from_numpy(LR_img).unsqueeze(0).permute(0,4,1,2,3).to(device)/255.0
       
                # reverse back
                rev_back = model.inference_RGB2latent(LR_img_ten)
                # upsample
                out = rescale_model.inference_up(rev_back,(T,H//y_grid,W//x_grid))

                place_holder_la[:,:,:,patch_d[0]:patch_d[1],patch_d[2]:patch_d[3]] = x_down.detach().cpu()
                place_holder_back[:,:,:,patch_d[0]:patch_d[1],patch_d[2]:patch_d[3]] = rev_back.detach().cpu()
                place_holder_quan[:,:,:,patch_d[0]:patch_d[1],patch_d[2]:patch_d[3]] = LR_img_ten.detach().cpu()
                place_holder[:,:,:,patch[0]:patch[1],patch[2]:patch[3]] = out.detach().cpu()


            LR_img = place_holder_quan.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            latent = place_holder_la.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            out = place_holder.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
            rev_back = place_holder_back.squeeze(0).permute(1,2,3,0).detach().cpu().numpy()*255.0
    return LR_img,latent,rev_back,out

def SPMCS_test(data_dir, base_out_p, weight_base_p):
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    model, rescale_model = init_model(weight_base_p, base_out_p, 'SPMCS')

    scale_list = [4.0, 3.6, 3.2, 2.8, 2.4, 2.0]
    time_list = [2]
    input_two = False
    for scale in scale_list:
        for tempo in time_list:

            if input_two:
                modulate_factor = 'two_'+(str(scale)+'_'+str(tempo)).replace('.','p')
            else:
                modulate_factor = 'mul_'+(str(scale)+'_'+str(tempo)).replace('.','p')
            print(modulate_factor)

            out_p = base_out_p+modulate_factor

            vid_data = SPMCS_arb(data_dir,scale,tempo)
            Vid_ST_dl = DataLoader(vid_data,batch_size=1,shuffle=False)

            time_idx_list,downsize_t =  gen_seq_index(tempo)
            print(time_idx_list)
            for ix,(GT,LR,name,crop_shape,output_list) in enumerate(Vid_ST_dl):
                # if name[0]!='LDVTG_009':
                #     continue
                print(f'GT shape {GT.shape}')
                for time_idx in time_idx_list:
                    print(f'time_idx {time_idx}')
                    input_ten = GT[:,time_idx,:,:,:].permute(0,2,1,3,4)
                    
                    input_ten = input_ten.cuda()
                    
                    # s_scale = 2
                    down_size = (downsize_t,(int(input_ten.shape[3]//scale)//8)*8,(int(input_ten.shape[4]//scale)//8)*8)
                    print(f'input shape {input_ten.shape[2:]} down shape {down_size}')
                    # print(f'down size is {down_size}')
                    LR_img,latent,rev_back,out = single_inference(model,rescale_model,input_ten,down_size)
                    # print(name)
                    for jx in range(down_size[0]):
                        sub_p = osp.join(out_p,name[0],'in')
                        if not os.path.exists( sub_p):
                            os.makedirs(sub_p)
                        img_name = str(time_idx[jx*tempo]).zfill(3)+'.png'
                        cv2.imwrite(f'{sub_p}/{img_name}',latent[jx][:,:,::-1])
                    for jx in range(down_size[0]):
                        sub_p = osp.join(out_p,name[0],'quan')
                        if not os.path.exists( sub_p):
                            os.makedirs(sub_p)
                        img_name = str(time_idx[jx*tempo]).zfill(3)+'.png'
                        cv2.imwrite(f'{sub_p}/{img_name}',LR_img[jx][:,:,::-1])
                    for jx in range(down_size[0]):
                        sub_p = osp.join(out_p,name[0],'rev')
                        if not os.path.exists( sub_p):
                            os.makedirs(sub_p)
                        img_name = str(time_idx[jx*tempo]).zfill(3)+'.png'
                        cv2.imwrite(f'{sub_p}/{img_name}',rev_back[jx][:,:,::-1])
                    for jx in range(input_ten.shape[2]):
                        sub_p = osp.join(out_p,name[0],'out')
                        if not os.path.exists( sub_p):
                            os.makedirs(sub_p)
                        img_name = str(time_idx[jx]).zfill(3)+'.png'
                        cv2.imwrite(f'{sub_p}/{img_name}',out[jx][:,:,::-1])

                        img_restore = cv2.imread(f'{sub_p}/{img_name}')
    
                        img_gt = input_ten[:,:,jx,:,:].detach().cpu().squeeze(0).permute(1,2,0).numpy()*255.0
                        img_gt = img_gt[:,:,::-1]
                      
                        psnr = calculate_psnr(img_gt,img_restore,crop_border=0,test_y_channel=True)
                        ssim = calculate_ssim(img_gt,img_restore,crop_border=0,test_y_channel=True)
                        print(f'{name[0]}/{img_name} psnr {psnr} ssim  {ssim}')

    # ... (rest of the function remains unchanged)

def gen_seq_index(t_scale):
    res = []
    time_interval= 4
    if t_scale==2:
        
        for each in list(range(0,30,time_interval)):
            if each+time_interval>31:
                break
            tmp = [each +i for i in range(time_interval+1)]

            res.append(tmp)
        down_size_t = 3
    return res,down_size_t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/home/zhangyuantong/dataset/SPMCS/',
                        help='Path to dataset directory')
    parser.add_argument('--base_out_p', type=str,
                        default='/home/zhangyuantong/code/ST_rescale_open_source/CSTVR/output/contin/',
                        help='Base output directory')
    parser.add_argument('--weight_base_p', type=str,
                        default='/home/zhangyuantong/code/ST_rescale_open_source/CSTVR/archieved/Contin',
                        help='Base path for model weights')
    parser.add_argument('--test_dataset_name', type=str,
                        default='adobe',
                        help='Name of test dataset')
    
    args = parser.parse_args()

    SPMCS_test(args.data_dir, args.base_out_p, args.weight_base_p)