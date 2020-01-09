auto-face-parsing
===========================
This project aims to implement Face parsing with AutoML. Feel free to contact me if you have any idea.

|Author|Zhe Lin|
|---|---
|E-mail|chirplam@foxmail.com
|LinkedIn| [Zhe Lin](https://www.linkedin.com/in/zhe-lin-65a9a8142/)


## Updates
- Training with HRNet, without any data augmentation([demo](https://youtu.be/Yd077zcvm1A))

## Installation
    python3.7 -m pip install requirements.txt
## Usage
### Train HRNet
Download [Relabeled-HELEN Dataset](https://github.com/JPlin/Relabeled-HELEN-Dataset) and uncompress. 
Then modify scripts/train.py line 34-35 & 43-44 to your dataset path. Cd to ./scripts, run 

    python3.7 train.py
### Test on video
Download demo video & trained model on [Google Drive](https://drive.google.com/open?id=1V2597ckYCb1EMyX9nQ9U4wVSRa0CObFT).
Put the video on ./data/ folder and checkpoint on ./outputs/hrnet/model. Cd to folder scripts, run

    python3.7 video_demo.py

## TODO
- [RoI Tanh-warping](https://arxiv.org/abs/1906.01342)
- [Guided filtering](http://kaiminghe.com/publications/eccv10guidedfilter.pdf)
- [HLNet](https://arxiv.org/abs/1912.12888)
- Augmentation searching
