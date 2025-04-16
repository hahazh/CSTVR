
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from arch.IMSM import IND_inv3D
from utils.options import yaml_load
import torch
import os
from utils.model_utils import get_model_total_params
import cv2
import numpy as np

def init_model():
    
    train_dataset_name = 'vimeo'
    weight_base_p = '../../archieved'
    
    # Construct paths with parameters
    weight_p = os.path.join(weight_base_p, f'Tx{time_factor}_Sx{scale_factor}_{train_dataset_name}')

    # Load inversion model
    inv_root_p = os.path.join(weight_p, 'inverter/config.yml')
    inv_opt = yaml_load(inv_root_p)['network_g']['opt']
    model = IND_inv3D(inv_opt).to(device)
    inv_weight_p = os.path.join(weight_p, 'inverter/model.pth')
    inv_weight = torch.load(inv_weight_p)
    model.load_state_dict(inv_weight['params'], strict=True)

    # Load rescaling model
    model_root_path = os.path.join(weight_p, 'rescaler/config.yml')
    rescale_opt = yaml_load(model_root_path)
    if time_factor == 2 and scale_factor == 1:
        from arch.Mynet_arch import RescalerNet
    else:
        from arch.Mynet_mix_arch import Rescaler_MixNet as RescalerNet
    rescale_model = RescalerNet(rescale_opt['network_g']['opt']).to(device)
    rescale_weight_p = os.path.join(weight_p, 'rescaler/model.pth')
    weight = torch.load(rescale_weight_p)
    rescale_model.load_state_dict(weight['params'], strict=True)
    rescale_model.eval()
    return rescale_model,model
def load_imgs():
    root_p = '../../test_input/'
    dir1 = sorted( os.listdir(root_p))
    imgs_list = []
    for each in dir1:
        this_img_list = []
        this_seq_p = os.path.join(root_p,each)
        for i in range(1,8):
            this_p = os.path.join(this_seq_p,f"im{i}.png")
            this_img = cv2.imread(this_p).astype('float32')/255.0
            this_img = torch.from_numpy(this_img).permute(2,0,1)
            this_img_list.append(this_img)
        this_img_list = torch.stack(this_img_list).permute(1,0,2,3).unsqueeze(0).to(device)
        imgs_list.append((each,this_img_list))
    return imgs_list
def infer():
    for seqs in test_imgs:
        name,imgs = seqs
        this_p = os.path.join(output_dir, name)
        print(name)
         # Create output directories
        quan_p = os.path.join(this_p, 'stegan')
        latent_p = os.path.join(this_p, 'latent')
        rev_p = os.path.join(this_p, 'rev')
        sr_p = os.path.join(this_p, 'sr')
        os.makedirs(quan_p, exist_ok=True)
        os.makedirs(latent_p, exist_ok=True)
        os.makedirs(rev_p, exist_ok=True)
        os.makedirs(sr_p, exist_ok=True)
        with torch.no_grad():
            x_down = rescale_model.inference_down(imgs, down_size)
            LR_img = model.inference_latent2RGB(x_down)
            LR_img = LR_img.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            LR_img = LR_img.astype(np.uint8)
            LR_img_ten = torch.from_numpy(LR_img).unsqueeze(0).permute(0, 4, 1, 2, 3).cuda() / 255.0

            # Reconstruction process
            rev_back = model.inference_RGB2latent(LR_img_ten)
            out = rescale_model.inference_up(rev_back, (imgs.shape[2], imgs.shape[3], imgs.shape[4]))

            # Convert tensors to numpy arrays
            x_down = x_down.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            out = out.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            rev_back = rev_back.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() * 255.0
            for i in range(down_t):
                cv2.imwrite(quan_p+'/im'+str(2*i+1)+'_stegan.png',LR_img[i])
                cv2.imwrite(latent_p+'/im'+str(2*i+1)+'_latent.png',x_down[i])
                cv2.imwrite(rev_p+'/im'+str(2*i+1)+'_back.png',rev_back[i])
            for i in range(7):
                cv2.imwrite(sr_p+'/im'+str(i+1)+'_out.png',out[i])
if __name__=='__main__':
    time_factor = 2
    scale_factor = 4
    device = torch.device('cuda')
    test_imgs = load_imgs()
    rescale_model,model = init_model()

    down_t = 7//time_factor+1 if time_factor!=1 else 7
    down_h = 256//scale_factor
    down_w = 448//scale_factor
    down_size = (down_t,down_h,down_w)
    output_dir = '../../output/demo'
    
    infer()
 
            
        

