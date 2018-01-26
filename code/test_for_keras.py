#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# test_for_keras.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-1-24 15:04:51
# @Explanation  : 测试tensorflow中keras结果
"""

import os
import math
import time
import numpy as np
import scipy.io as scio
import cv2
from resnet_tf import resnet_model


if __name__ == '__main__':

    # 生成模型
    model = resnet_model(480, 480, color_type=3, num_classes=3)
    # 导入训练好的模型
    model.load_weights('./model/resnet_101.h5')

    # batch size
    batch_size = 512

    # 数据路径
    digestive_dir = u'/home/docker/code/app/tensorflow/digestive_change'

    data_dirs = []
    for data_dir in os.listdir(digestive_dir):
        data_dirs.append(os.path.join(digestive_dir, data_dir))

    for data_dir in data_dirs:
        imagelists = [img for img in os.listdir(data_dir) if img.endswith('jpg')]
        imagelists.sort()

        data_num = len(imagelists)
        # data_num = 10
        batch_num = int(math.ceil(float(data_num) / float(batch_size)))

        predictions = []
        start_time = time.time()
        for batch_idx in range(batch_num):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, data_num)

            imgs = []
            for idx in range(start, end):
                img = cv2.imread(os.path.join(data_dir, imagelists[idx]).encode('utf8'))
                img = cv2.resize(img, (480, 480))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)
            imgs = np.array(imgs, dtype='float32') / 255.0
            prediction = np.argmax(model.predict(np.array(imgs)), 1).tolist()
            predictions.extend(prediction)

        end_time = time.time()
        with open(os.path.join('./results/', 'time_for_keras.txt'), 'a') as ff:
            ff.write('%s: %f \n' % (os.path.basename(data_dir), end_time - start_time))
        scio.savemat(os.path.join('./results/results_for_keras', os.path.basename(data_dir) + '.mat'), {'data': predictions})
