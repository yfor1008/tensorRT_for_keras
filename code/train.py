#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# train.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-1-17 11:11:06
# @Explanation  : 说明
"""

import tensorflow as tf
from resnet_tf import resnet_model

from tensorflow.python.keras._impl.keras.optimizers import SGD
from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras._impl.keras.callbacks import EarlyStopping
from tensorflow.python.keras._impl.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras._impl.keras.callbacks import TensorBoard

from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

if __name__ == '__main__':

    config = {
        # Training params
        "train_data_dir": "/home/docker/code/app/get_samples/samples_train/train",  # training data
        "val_data_dir": "/home/docker/code/app/get_samples/samples_train/valid",  # validation data
        "train_batch_size": 8,  # training batch size
        "epochs": 30,  # number of training epochs
        "num_train_samples" : 217728,  # number of training examples
        "num_val_samples" : 55872,  # number of test examples
        # "num_train_samples" : 500,  # number of training examples
        # "num_val_samples" : 300,  # number of test examples

        # Where to save models (Tensorflow + TensorRT)
        "graphdef_file": "./model/resnet_tf_graphdef.pb",
        "frozen_model_file": "./model/resnet_tf_frozen_model.pb",
        "snapshot_dir": "./model/snapshot",
        "engine_save_dir": "./model/",

        # Needed for TensorRT
        "image_dim": 480,  # the image size (square images)
        "inference_batch_size": 1,  # inference batch size
        "input_layer": "data",  # name of the input tensor in the TF computational graph
        "out_layer": "fc3/Softmax",  # name of the output tensorf in the TF conputational graph
        "output_size" : 3,  # number of classes in output (3)
        "precision": "fp32",  # desired precision (fp32, fp16)

        "test_image_path" : "/home/data/val/roses"
    }

    model = resnet_model(480, 480, color_type=3, num_classes=3)
    model.summary()

    # 导入预先训练好的模型，进行finetune
    model.load_weights('./model/resnet_101.h5')

    # 数据保存
    cbks = [TensorBoard(log_dir='./logs', histogram_freq=0, write_images=True, write_graph=True, write_grads=True),
            ModelCheckpoint('./model/resnet_101_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True),
           ]

    # 优化参数
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    data_gen = ImageDataGenerator(rescale=1./255,
                                  shear_range=10,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1)
    train_generator = data_gen.flow_from_directory(directory=config['train_data_dir'],
                                                   target_size=(config['image_dim'], config['image_dim']),
                                                   batch_size=config['train_batch_size'],
                                                   class_mode='categorical')
    val_generator = data_gen.flow_from_directory(directory=config['val_data_dir'],
                                                 target_size=(config['image_dim'], config['image_dim']),
                                                 batch_size=config['train_batch_size'],
                                                 class_mode='categorical')

    # train the model on the new data for a few epochs
    model.fit_generator(train_generator,
                        steps_per_epoch=config['num_train_samples']//config['train_batch_size'],
                        epochs=config['epochs'],
                        # verbose=1,
                        # callbacks=cbks,
                        validation_data=val_generator,
                        validation_steps=config['num_val_samples']//config['train_batch_size'],
                        callbacks=cbks,
                        verbose=1)

    model.save_weights('./model/resnet_101_last.h5')

    # Now, let's use the Tensorflow backend to get the TF graphdef and frozen graph
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    # save model weights in TF checkpoint
    checkpoint_path = saver.save(sess, config['snapshot_dir'], global_step=0, latest_filename='checkpoint_state')
    # checkpoint_path = saver.restore(sess, config['snapshot_dir'])

    # remove nodes not needed for inference from graph def
    train_graph = sess.graph
    inference_graph = tf.graph_util.remove_training_nodes(train_graph.as_graph_def())

    # write the graph definition to a file.
    # You can view this file to see your network structure and
    # to determine the names of your network's input/output layers.
    graph_io.write_graph(inference_graph, '.', config['graphdef_file'])

    # specify which layer is the output layer for your graph.
    # In this case, we want to specify the softmax layer after our
    # last dense (fully connected) layer.
    out_names = config['out_layer']

    # freeze your inference graph and save it for later! (Tensorflow)
    freeze_graph.freeze_graph(config['graphdef_file'],
                              '',
                              False,
                              checkpoint_path,
                              out_names,
                              "save/restore_all",
                              "save/Const:0",
                              config['frozen_model_file'],
                              False,
                              ""
                             )
