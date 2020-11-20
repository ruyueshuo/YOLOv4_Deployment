#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/13 16:26
# @Author  : ruyueshuo
# @File    : split_datasets.py
# @Software: PyCharm
import os
import random


def split_datasets(image_path, save_path, ratio=0.8, seed=41, name=None):
    """
    Split dataset to train and validation sets.
    :param image_path: str or Path. raw image data path.
    :param save_path: str or Path. train and validation text file saving path.
    :param ratios: float. ratio of train set.
    :param seed: int. random seed.
    :return:
    """
    # set random seed
    random.seed(seed)

    # get raw image list
    image_list = os.listdir(image_path)
    image_list = [image for image in image_list if image.endswith('.jpg')]
    image_list = [os.path.join(image_path, file) for file in image_list]

    # # get label list
    # label_list = os.listdir(label_path)
    # label_list = [os.path.join(label_path, file) for file in label_list]

    file_num = len(image_list)

    # split dataset
    train_list = random.sample(image_list, int(file_num*ratio))
    valid_list = list(set(image_list).difference(set(train_list)))
    print("length of train dataset :{}".format(len(train_list)))
    print("length of valid dataset :{}".format(len(valid_list)))

    # save results
    save_txt(train_list, os.path.join(save_path, 'train-{}.txt'.format(name)))
    save_txt(valid_list, os.path.join(save_path, 'valid-{}.txt'.format(name)))


def save_txt(data, file):
    """save data to text file."""
    with open(file, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')
        # f.write(str(data))
        f.close()


def joint_video(video1, video2, video_out):
    """将视频拼接"""
    import cv2
    import numpy as np
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    suc1 = cap1.isOpened()  # 是否成功打开
    suc2 = cap2.isOpened()  # 是否成功打开
    frame_count = 0

    # 获得码率及尺寸
    video_fps = cap1.get(cv2.CAP_PROP_FPS)
    video_fps2 = cap2.get(cv2.CAP_PROP_FPS)

    size1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    size2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    size = (size1[0]+size2[0], size1[1])

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(video_out, fourcc, video_fps, size)

    while suc1 and suc2:
      frame_count += 1

      suc1, frame1 = cap1.read()
      suc2, frame2 = cap2.read()
      if frame_count > 1000:
          break
      if frame1 is not None and frame2 is not None:
          frame1 = cv2.putText(frame1, "DeepStream-YOLOv4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          frame2 = cv2.putText(frame2, "Darknet-YOLOv4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          new_frame = np.concatenate((frame1, frame2),axis=1)
          vw.write(new_frame)
    cap1.release()
    cap2.release()
    vw.release()
    print('Unlock video, total image number: {}.'.format(frame_count))


if __name__ == '__main__':
    # video2video('results/deepstream_yolov4.mp4', 'results/test_result.mp4', 'results/test.avi')
    image_path = "/home/ubuntu/Datasets/reflective/VOC2021/JPEGImages"
    save_path = "/home/ubuntu/Projects/DeepStream-YOLOv4/data"
    split_datasets(image_path, save_path, ratio=0.8, seed=41, name='reflective')
