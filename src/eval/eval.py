import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from metrics.psnr_ssim import calculate_psnr,calculate_ssim
from skimage.metrics import structural_similarity as compare_ssim
import cv2
import os

def test_vid():
    pred_dir = '/home/zhangyuantong/code/ST_rescale_open_source/CSTVR/output/Tx1_Sx4/vid4'
    base_p = '/home/zhangyuantong/dataset/Vid4/GT/'
    seq_list = sorted( [each for each in os.listdir(base_p) if  each.split('.')[-1]!='txt'])
    print(seq_list)
    avg_psnr = []
    avg_ssim = []
    avg_psnr_Y = []
    avg_ssim_Y = []
    ty = 'float32'
    for seq in seq_list:
        sub_seq = sorted( os.listdir(pred_dir+'/'+seq+'/sr'))
        print(seq)
        seq_psnr = []
        seq_ssim = []
        seq_psnr_Y = []
        seq_ssim_Y = []
        for ix,sub in enumerate( sub_seq):
            pred_p = os.path.join(pred_dir,seq,'sr',sub)
            gt_p = os.path.join(base_p,seq,sub)
            pred = cv2.imread(pred_p).astype(ty)
            gt = cv2.imread(gt_p).astype(ty)
            ssim_y = calculate_ssim(gt,pred,crop_border=0,test_y_channel=True)
            psnr_y = calculate_psnr(gt,pred,crop_border=0,test_y_channel=True)
            ssim = calculate_ssim(gt,pred,crop_border=0,test_y_channel=False)
            psnr = calculate_psnr(gt,pred,crop_border=0,test_y_channel=False)
            if psnr<90:
                avg_psnr.append(psnr)
                seq_psnr.append(psnr)
                
                avg_ssim.append(ssim)
                seq_ssim.append(ssim)  

                avg_psnr_Y.append(psnr_y)
                seq_psnr_Y.append(psnr_y)
                
                avg_ssim_Y.append(ssim_y)
                seq_ssim_Y.append(ssim_y)  
                print(f'ix {ix+1 } psnr {psnr} ssim {ssim} psnr Y {psnr_y} ssim Y {ssim_y}')
    
        print(f'seq psnr {sum(seq_psnr)/len(seq_psnr)} ssim {sum(seq_ssim)/len(seq_ssim)} seq psnr Y {sum(seq_psnr_Y)/len(seq_psnr_Y)} ssim Y {sum(seq_ssim_Y)/len(seq_ssim_Y)}')
        print(f'now avg psnr {sum(avg_psnr)/len(avg_psnr)} ssim {sum(avg_ssim)/len(avg_ssim)}  psnr Y {sum(avg_psnr_Y)/len(avg_psnr_Y)} ssim Y {sum(avg_ssim_Y)/len(avg_ssim_Y)} ')

def com_vimeo():
    file = open('/home/zhangyuantong/dataset/vimeo_septuplet/sequences/sep_testlist.txt','r').readlines()
    gt_root = '/home/zhangyuantong/dataset/vimeo_septuplet/sequences/'
    pred_root = '/home/zhangyuantong/code/ST_rescale_open_source/CSTVR/output/Tx1_Sx4/vimeo'
    ty = 'uint8'
    avg_psnr = []
    avg_ssim = []
    avg_psnr_Y = []
    avg_ssim_Y = []
    for each in file:
        each = each.strip('\n')
        this_gt_p = os.path.join(gt_root,each)
        this_pred_p = os.path.join(pred_root,each)
        print(each)
        seq_psnr = []
        seq_ssim = []
        seq_psnr_Y = []
        seq_ssim_Y = []
        for i in range(1,8):
            gt = cv2.imread(f'{this_gt_p}/im{i}.png').astype(ty)
            pred = cv2.imread(f'{this_pred_p}/im{i}.png').astype(ty)
            ssim_y = calculate_ssim(gt,pred,crop_border=0,test_y_channel=True)
            psnr_y = calculate_psnr(gt,pred,crop_border=0,test_y_channel=True)
            ssim = calculate_ssim(gt,pred,crop_border=0,test_y_channel=False)
            psnr = calculate_psnr(gt,pred,crop_border=0,test_y_channel=False)
            if psnr<90:
                avg_psnr.append(psnr)
                seq_psnr.append(psnr)
                
                avg_ssim.append(ssim)
                seq_ssim.append(ssim)  

                avg_psnr_Y.append(psnr_y)
                seq_psnr_Y.append(psnr_y)
                
                avg_ssim_Y.append(ssim_y)
                seq_ssim_Y.append(ssim_y)  
                print(f'ix {i+1 } psnr {psnr} ssim {ssim} psnr Y {psnr_y} ssim Y {ssim_y}')
    
        print(f'seq psnr {sum(seq_psnr)/len(seq_psnr)} ssim {sum(seq_ssim)/len(seq_ssim)} seq psnr Y {sum(seq_psnr_Y)/len(seq_psnr_Y)} ssim Y {sum(seq_ssim_Y)/len(seq_ssim_Y)}')
        print(f'now avg psnr {sum(avg_psnr)/len(avg_psnr)} ssim {sum(avg_ssim)/len(avg_ssim)}  psnr Y {sum(avg_psnr_Y)/len(avg_psnr_Y)} ssim Y {sum(avg_ssim_Y)/len(avg_ssim_Y)} ')
def com_SPMCS():
    pred_dir = '/home/zhangyuantong/code/ST_rescale/out/continuous/SPMCS_test_tmp/'
    base_p = '/home/zhangyuantong/dataset/SPMCS/'
    scale_list = [4.0,3.6,3.2,2.8,2.4,2.0]
    time_list = [2]
    seq_list = sorted( [each for each in os.listdir(base_p) if  each.split('.')[-1]!='txt'])
    print(seq_list)
    psnr_list = []
    ssim_list = []
    psnr_list_Y = []
    ssim_list_Y = []
    for scale in scale_list:
        for tempo in time_list:
            modulate_factor = 'mul_'+(str(scale)+'_'+str(tempo)).replace('.','p')
            f = open('/home/zhangyuantong/code/ST_rescale/out/continuous/SPMCS_cri_new/'+modulate_factor+'.txt','w')
            for seq in seq_list:
                f.write("scene: %s "%(seq)+'\n')
                sub_seq = sorted( os.listdir(pred_dir+'/'+modulate_factor+'/'+seq+'/out'))
                print(seq)
                tmp_psnr_list = []
                tmp_ssim_list = []
                tmp_psnr_list_Y = []
                tmp_ssim_list_Y = []
                for ix,sub in enumerate( sub_seq):
    
                    pred_p = os.path.join(pred_dir,modulate_factor,seq,'out',sub)
                    gt_p = os.path.join(base_p,seq,'HR',sub)
                    pred = cv2.imread(pred_p)
                    gt = cv2.imread(gt_p)
                    psnr_y = calculate_psnr(gt,pred,crop_border=0,test_y_channel=True)
                    ssim_y = calculate_ssim(gt,pred,crop_border=0,test_y_channel=True)
                    psnr = calculate_psnr(gt,pred,crop_border=0,test_y_channel=False)
                    ssim = calculate_ssim(gt,pred,crop_border=0,test_y_channel=False)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    tmp_psnr_list.append(psnr)
                    tmp_ssim_list.append(ssim)

                    psnr_list_Y.append(psnr_y)
                    ssim_list_Y.append(ssim_y)
                    tmp_psnr_list_Y.append(psnr_y)
                    tmp_ssim_list_Y.append(ssim_y)
                    print(f' scene {seq}  scale {scale} ix {sub}  psnr {psnr} ssim {ssim} psnr_y {psnr_y} ssim_y {ssim_y}')
                    f.write(f' scene {seq}  scale {scale} ix {sub}  psnr {psnr} ssim {ssim} psnr_y {psnr_y} ssim_y {ssim_y}'+'\n')
                print(f'for scene {seq} avg psnr {sum(tmp_psnr_list)/len(tmp_psnr_list)} avg ssim  {sum(tmp_ssim_list)/len(tmp_ssim_list)}')
                print(f'for scene {seq} avg psnr Y {sum(tmp_psnr_list_Y)/len(tmp_psnr_list_Y)} avg ssim Y  {sum(tmp_ssim_list_Y)/len(tmp_ssim_list_Y)}')
                f.write(f'for scene {seq} avg psnr {sum(tmp_psnr_list)/len(tmp_psnr_list)} avg ssim  {sum(tmp_ssim_list)/len(tmp_ssim_list)}'+'\n')
                f.write(f'for scene {seq} avg psnr Y {sum(tmp_psnr_list_Y)/len(tmp_psnr_list_Y)} avg ssim Y  {sum(tmp_ssim_list_Y)/len(tmp_ssim_list_Y)}'+'\n')

            print(f' total {len(psnr_list)} psnr {sum(psnr_list)/len(psnr_list)} ssim {sum(ssim_list)/len(ssim_list)}')
            print(f' total {len(psnr_list_Y)} psnr_y {sum(psnr_list_Y)/len(psnr_list_Y)} ssim_y {sum(ssim_list_Y)/len(ssim_list_Y)}')
            
            f.write(f' total {len(psnr_list)} psnr {sum(psnr_list)/len(psnr_list)} ssim {sum(ssim_list)/len(ssim_list)}'+'\n')
            f.write(f' total {len(psnr_list_Y)} psnr_y {sum(psnr_list_Y)/len(psnr_list_Y)} ssim_y {sum(ssim_list_Y)/len(ssim_list_Y)}'+'\n')
if __name__=='__main__':
    # test_vid()
    com_vimeo()
    # test_vid()
    # com_SPMCS()