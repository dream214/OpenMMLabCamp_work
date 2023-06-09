#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# Author：Sakura time:2023/6/8
from mmpretrain import ImageClassificationInferencer
inferencer = ImageClassificationInferencer(
        model='/root/autodl-tmp/mmpretrain/work_dirs/resnet_config/resnet_config.py',
        pretrained='/root/autodl-tmp/mmpretrain/work_dirs/resnet_config/best_accuracy/top1_epoch_90.pth',
        device='cuda')
inferencer(['/root/autodl-tmp/mmpretrain/data/myPic/1.jpeg', '/root/autodl-tmp/mmpretrain/data/myPic/2.jpeg','/root/autodl-tmp/mmpretrain/data/myPic/3.jpg'], show_dir="./visualize/")