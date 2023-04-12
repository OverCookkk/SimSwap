'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:45:41
Description: 
'''
from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface_func.utils import face_align_ffhqandnewarc as face_align

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, max_num=0):
        # （1）bboxes 表示从图像中检测到的人脸框（bounding boxes），是一个二维的数组，形状为 (N, 4)，其中 N 是检测得到的人脸框的数量，
        # 4 表示每个人脸框是一个四元组 (x1,y1,x2,y2)，(x1,y1) 表示左上角坐标，(x2,y2) 表示右下角坐标；
        # （2）kpss 表示从图像中检测到的关键点位置（keypoints），每个关键点是一个二元组 (x,y)，表示关键点在图像中的坐标位置。
        # 这些结果可以被用于进一步的人脸识别、面部表情识别、人脸年龄、性别等属性分析。
        bboxes, kpss = self.det_model.detect(img,
                                             threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        # 第0维为0，表示检测到的人脸框数量为0
        if bboxes.shape[0] == 0:
            return None
        ret = []
        # for i in range(bboxes.shape[0]):
        #     bbox = bboxes[i, 0:4]
        #     det_score = bboxes[i, 4]
        #     kps = None
        #     if kpss is not None:
        #         kps = kpss[i]
        #     M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        #     align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        # 由于人脸的差异，需要保证人脸在三个方面：大小、姿态和位置 保持一致，所以需要对人脸进行仿射变换。
        align_img_list = []
        M_list = []
        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            # estimate_norm() 函数是用来进行人脸对齐的计算，使用keypoints计算出仿射变换矩阵；其输出结果包含两个变量: M 表示可用于对齐的仿射变换矩阵
            M, _ = face_align.estimate_norm(kps, crop_size, mode=self.mode)  # crop_size 是要对齐后生成的人脸图像大小
            # warpAffine() 函数是OpenCV中用于进行仿射变换的函数，它通过变换矩阵将原始图像中的像素位置变换为目标位置。
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
            align_img_list.append(align_img)
            M_list.append(M)

        # det_score = bboxes[..., 4]

        # best_index = np.argmax(det_score)

        # kps = None
        # if kpss is not None:
        #     kps = kpss[best_index]
        # M, _ = face_align.estimate_norm(kps, crop_size, mode ='None') 
        # align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
        
        return align_img_list, M_list
