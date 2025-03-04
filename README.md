# CVRS 
This is the code of the paper "Continuous Space-Time Video Resampling with Invertible Motion Steganography"


overview
<div align="center">
  <img src="pic/overview.png" alt="Description of the image" width="500"/>
</div>

# test code

### test fix-scale space-time video super-resolution
```
cd src/test_script

python test_vid4.py --datapath REDSPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

### test continuous space-time video super-resolution
```
cd src/test_script

python test_contin.py --datapath REDSPATH --outputpath  OUTPUTPATH --weight PATHTOWEIGHT
```

### pretrained weight
[pretrained model]( https://pan.baidu.com/s/1PA7IoclyZsDXA7EhNlGQjA?pwd=8n5e)
password: 8n5e 



# Acknowledgment
Our code is built on

 [Zooming-Slow-Mo-CVPR-2020](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020)

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

 [IFRNet](https://github.com/ltkong218/IFRNet)

 [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI)
 
 [Galerkin Transformer](https://github.com/scaomath/galerkin-transformer)
 
 We thank the authors for sharing their codes!