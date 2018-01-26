#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# test_for_fp16.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-1-24 18:11:28
# @Explanation  : 测试tensorRT优化后效果
"""

import os
import math
import time
import numpy as np
import scipy.io as scio
import cv2

from tensorrt.lite import Engine
import tensorrt as trt

if __name__ == '__main__':

    # 模型
    # engine_fp16 = Engine(PLAN='./model/resnet_tf_b256_fp16.engine', log_sev=4, max_batch_size=256)
    # 屏蔽信息输出，log_sev应换成logger_severity
    engine_fp16 = Engine(PLAN='./model/resnet_tf_b256_fp16.engine', logger_severity=trt.infer.LogSeverity.ERROR, max_batch_size=256)


    # batch size
    batch_size = 256

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

            # imgs = []
            # for idx in range(start, end):
            #     img = cv2.imread(os.path.join(data_dir, imagelists[idx]).encode('utf8'))
            #     img = cv2.resize(img, (480, 480))
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #     img = img.transpose([2, 0, 1])
            #     imgs.append(img)
            imgs = [cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(data_dir, imagelists[idx]).encode('utf8')), (480, 480)),
                                 cv2.COLOR_BGR2RGB).transpose([2, 0, 1]) for idx in range(start, end)]
            imgs = np.array(imgs, dtype='float32') / 255.0
            prediction = np.argmax(engine_fp16.infer(imgs)[0], 1).tolist()
            pre = [ff[0][0] for ff in prediction]
            predictions.extend(pre)

        end_time = time.time()
        with open(os.path.join('./results/', 'time_for_fp16.txt'), 'a') as ff:
            ff.write('%s: %f \n' % (os.path.basename(data_dir), end_time - start_time))
        scio.savemat(os.path.join('./results/results_for_fp16', os.path.basename(data_dir) + '.mat'), {'data': predictions})
