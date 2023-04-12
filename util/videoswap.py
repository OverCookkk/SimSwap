'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:19:52
Description: 
'''
import os
import cv2
import glob
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import time
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet


# 转成tensor
def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def video_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results',
               crop_size=224, no_simswaplogo=False, use_mask=False):
    # 返回对象包含了视频文件的帧数据以及一些视频相关的属性，如帧率、宽度、高度等信息
    video_forcheck = VideoFileClip(video_path)
    # 判定是否有声音
    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    # 然后有声音
    if not no_audio:
        video_audio_clip = AudioFileClip(video_path)

    # 图片
    # 返回对象可以用来读取视频中的每一帧，并提供了许多方法用于访问视频文件的各种属性，如帧率、分辨率、时长等等
    video = cv2.VideoCapture(video_path)

    # 增加logo图片
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    ret = True
    frame_index = 0

    # 帧数统计
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频fps
    fps = video.get(cv2.CAP_PROP_FPS)
    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)

    # 正则
    spNorm = SpecificNorm()
    if use_mask:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
        net.load_state_dict(torch.load(save_pth))
        net.eval()
    else:
        net = None

    # while ret:
    for frame_index in tqdm(range(frame_count)):
        # 读取视频帧
        ret, frame = video.read()
        if ret:
            # 对该帧进行裁剪和仿射变换
            detect_results = detect_model.get(frame, crop_size)

            if detect_results is not None:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)

                # 对齐裁剪后的图片列表
                frame_align_crop_list = detect_results[0]

                # 仿射变换矩阵列表
                frame_mat_list = detect_results[1]

                # 换脸结果列表
                swap_result_list = []

                # 帧数对齐
                frame_align_crop_tenor_list = []
                for frame_align_crop in frame_align_crop_list:
                    # BGR TO RGB
                    # frame_align_crop_RGB = frame_align_crop[...,::-1]

                    frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB))[
                        None, ...].cuda()

                    # 换脸推理
                    swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                    cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
                    swap_result_list.append(swap_result)
                    frame_align_crop_tenor_list.append(frame_align_crop_tenor)

                reverse2wholeimage(frame_align_crop_tenor_list,
                                   swap_result_list,
                                   frame_mat_list,
                                   crop_size,
                                   frame,
                                   logoclass,
                                   os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)),
                                   no_simswaplogo, pasring_model=net, use_mask=use_mask, norm=spNorm)

            else:
                if not os.path.exists(temp_results_dir):
                    os.mkdir(temp_results_dir)
                frame = frame.astype(np.uint8)
                if not no_simswaplogo:
                    frame = logoclass.apply_frames(frame)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame)
        else:
            break

    # 视频释放
    video.release()

    # image_filename_list = []
    path = os.path.join(temp_results_dir, '*.jpg')
    image_filenames = sorted(glob.glob(path))

    # 序列
    clips = ImageSequenceClip(image_filenames, fps=fps)

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    # 写入视频
    clips.write_videofile(save_path, audio_codec='aac')
