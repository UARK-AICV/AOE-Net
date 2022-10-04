# AOE-Net

Source code of paper:
 "AOE-Net: Entities Interactions Modeling with Adaptive Attention Mechanism for Temporal Action Proposals Generation",
  which is accepted for publication in [International Journal of Computer Vision](https://www.springer.com/journal/11263).

## Installation Guide

1. conda create -n AOENet python=3.8
2. conda activate AOENet
3. git clone https://github.com/UARK-AICV/AOE-Net.git
4. cd AOE-Net
5. conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
6. pip install tqdm pandas tensorboard matplotlib fvcore scipy


## Download Features
3D Resnet-50 features extracted from rescaled videos of ActivityNet-1.3 can be downloaded below:
* Environment features are [here](https://drive.google.com/file/d/1hPhcQ7EzyCh0A3SyZfgZScFVFZMEvVhe/view?usp=sharing).
* Actor features are [here](https://drive.google.com/file/d/1lOQG1FgDseRKDs3RNgpKd000OOZiag1s/view?usp=sharing).
* Object features are [here](https://uark-my.sharepoint.com/:u:/g/personal/sangt_uark_edu/EW1wAz-z955HuZUD49yxAaQB8Rc3eOPMpkWZQ0b2LURqnA?e=jj9zSU).
* Annotations of [Activitynet-1.3](http://ec2-52-25-205-214.us-west-2.compute.amazonaws.com/files/activity_net.v1-3.min.json) can be downloaded from the [official website](http://activity-net.org/download.html).

## Training and Testing  of AOE-Net
Default configurations of AOE-Net are stored in config/defaults.py.
The modified configurations are stored in config/*.yaml for training and testing of AOE on different datasets (ActivityNet-1.3 and THUMOS-14).
We can also modify configurations through commandline arguments.

1. To train AOE-Net on TAPG task of ActivityNet-1.3 with 1 GPU:
```
python main.py --cfg-file config/anet_proposal_CLIP_v1.yaml MODE 'training' GPU_IDS [0]
```

2. To evaluate AOE-Net on validation set of ActivityNet-1.3 with 1 GPU:
```
python main.py --cfg-file config/anet_proposals_CLIP_v1.yaml MODE 'validation' GPU_IDS [0]
```

## Reference

This implementation is partly based on this [pytorch-implementation of BMN](https://github.com/JJBOY/BMN-Boundary-Matching-Network.git) for the boundary matching module and our previous work, named [AEI](https://github.com/UARK-AICV/TAPG-AgentEnvInteration).

paper: [AOE-Net: Entities Interactions Modeling with Adaptive Attention Mechanism for Temporal Action Proposals Generation](https://github.com/UARK-AICV/AOE-Net) (link will be updated soon).


## Q&A
1q. "UserWarning: This DataLoader will create 12 worker processes in total. Our suggested max number of worker in current system is #, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary."

1a. Change num_workers to # in line 171 of root>main.py>inference function

## Citation
If you find AEI useful for your research, please consider citing:
```
@article{vo2021aei,
 author = {Vo, Khoa and Joo, Hyekang and Yamazaki, Kashu and Truong, Sang and Kitani, Kris and Tran, Minh-Triet and Le, Ngan},
 journal = {BMVC},
 title = {{{AEI}: Actors-Environment Interaction with Adaptive Attention for Temporal Action Proposals Generation}},
 year = {2021}
}
```

and 
```
@article{vo2022aoe,
 author = {Vo, Khoa and Truong, Sang and Yamazaki, Kashu and Raj, Bhiksha and Tran, Minh-Triet and Le, Ngan},
 journal = {International Journal of Computer Vision},
 title = {{AOE-Net}: Entities Interactions Modeling with Adaptive Attention Mechanism for Temporal Action Proposals Generation},
 year = {2022}
}
```

## Contact
Khoa Vo:
```
khoavoho@uark.edu
```
