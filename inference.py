#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 上午11:05
# @Author  : Aliang

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
from util.add_watermark import watermark_image

opt = TestOptions()
opt.initialize()
opt.parser.add_argument('-f')                                           # dummy arg to avoid bug
opt = opt.parse()
opt.pic_a_path = './demo_file/multispecific/DST_03.jpg'       # or replace it with image from your own google drive
opt.video_path = './demo_file/03.10V13.mp4'                   # or replace it with video from your own google drive
opt.output_path = './output/demo.mp4'
opt.temp_path = './tmp'
opt.Arc_path = './arcface_model/arcface_checkpoint.tar'
opt.isTrain = False
opt.use_mask = True  # new feature up-to-date

crop_size = opt.crop_size

torch.nn.Module.dump_patches = True
model = create_model(opt)
model.eval()

# 裁剪模型
app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

# 设置图片的转换器
transformer = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 设置arcface转换器
transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 设置detransformer的转换器
detransformer = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
])

with torch.no_grad():
    pic_a = opt.pic_a_path
    # img_a = Image.open(pic_a).convert('RGB')
    img_a_whole = cv2.imread(pic_a)
    # 对图片上的人脸进行裁剪、仿射变换等
    img_a_align_crop, _ = app.get(img_a_whole, crop_size)
    img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
    img_a = transformer_Arcface(img_a_align_crop_pil)
    img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

    # convert numpy to tensor
    img_id = img_id.cuda()

    # create latent id（人脸特征向量）
    img_id_downsample = F.interpolate(img_id, size=(112, 112))
    latend_id = model.netArc(img_id_downsample)
    latend_id = latend_id.detach().to('cpu')
    latend_id = latend_id / np.linalg.norm(latend_id, axis=1, keepdims=True)
    latend_id = latend_id.to('cuda')

    video_swap(opt.video_path, latend_id, model, app, opt.output_path, temp_results_dir=opt.temp_path,
               use_mask=opt.use_mask)
