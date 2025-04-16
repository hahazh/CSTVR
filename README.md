# CSTVR 
This is the code of the paper "Continuous Space-Time Video Resampling with Invertible Motion Steganography"

## Introduction

<div style="text-align: justify; word-break: keep-all; hyphens: none;">
Space-time video resampling aims to conduct both spatial-temporal downsampling and upsampling processes to achieve high-quality video reconstruction. Although there has been much progress, some major challenges still exist, such as how to preserve motion information during temporal resampling while avoiding blurring artifacts, and how to achieve flexible temporal and spatial resampling factors.  In this paper, we introduce an Invertible Motion Steganography Module (IMSM), designed to preserve motion information in high-frame-rate videos. This module embeds motion information from high-frame-rate videos into downsampled frames with lower frame rates in a visually imperceptible manner. Its reversible nature allows the motion information to be recovered, facilitating the reconstruction of high-frame-rate videos. Furthermore, we propose a 3D implicit feature modulation technique that enables continuous spatiotemporal resampling. With tailored training strategies, our method supports flexible frame rate conversions, including non-integer changes like 30 FPS to 24 FPS and vice versa.  Extensive experiments show that our method significantly outperforms existing solutions across multiple datasets in various video resampling tasks with high flexibility. 

</div>
 
## Overview
<div align="center">
  <img src="pic/overview.png" alt="Description of the image" width="800"/>
</div>

## Performance
<div align="center">
  <img src="pic/performance.png" alt="Description of the image" width="800"/>
</div>

# test code


### environment

Our code runs well under Python 3.8.18 with torch==2.0.0, torchvision==0.15.1, and numpy==1.22.3

### pretrained weight

You should first download the pretrained model and place it in the directory CSTVR/archived

[pretrained model]( https://pan.baidu.com/s/16L1WyclbxvkRSIJDImIjWQ?pwd=43x5)

password: 43x5 

### quick start

You can run a demo in a few seconds, which performs temporal 2x and spatial 4x resampling on the input. 
```
cd src/test

python demo.py
```
The results will be stored in CSTVR/output/demo.
The generated output directory structure and meanings are as follows:
```
├── latent: Latent features  (just for visualization, not available to users)
├── rev: Reverse reconstruction  
├── sr: Super-resolved output  
└── stegan: Steganography-related images  
```


### test fix-scale space-time video resampling
```
cd src/test

# for vid4
python vid4_test.py 
# for vimeo7
python vimeo7_test.py
```

### test continuous space-time video resampling
```
cd src/test

python SPMCS_contin_test.py
```





# Acknowledgment
Our code is built on

 [open-mmlab](https://github.com/open-mmlab)

 [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)

 [Video Swin Transformers](https://github.com/shoaib6174/GSOC-22-Video-Swin-Transformers/tree/574dc10bdc12b47bea31917dfbd45d216855b033)

 [SelfC](https://github.com/tianyuan168326/SelfC)

 We thank the authors for sharing their codes!