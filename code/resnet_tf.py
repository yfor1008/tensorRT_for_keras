#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# resnet_tf.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-1-17 10:29:01
# @Explanation  : 使用tensorflow中的keras实现
"""

from tensorflow.python.keras._impl.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras._impl.keras.layers import ZeroPadding2D, add, Activation
# from tensorflow.python.keras._impl.keras.layers.normalization import BatchNormalization
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras._impl.keras.models import Model

from scale_layer_tf import Scale

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    ### 说明:
        - The identity_block is the block that has no conv layer at shortcut

    ### Arguments:
        - input_tensor: input tensor
        - kernel_size: defualt 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names
    """
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    tensor_x = Conv2D(nb_filter1, (1, 1), use_bias=False, name=conv_name_base + '2a')(input_tensor)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2a')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2a')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2a_relu')(tensor_x)

    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2b')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2b')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2b_relu')(tensor_x)

    tensor_x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2c')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2c')(tensor_x)

    tensor_x = add([tensor_x, input_tensor], name='res' + str(stage) + block)
    tensor_x = Activation('relu', name='res' + str(stage) + block + '_relu')(tensor_x)
    return tensor_x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''
    ### 说明
        - conv_block is the block that has a conv layer at shortcut

    ### Arguments
        - input_tensor: input tensor
        - kernel_size: defualt 3, the kernel size of middle conv layer at main path
        - filters: list of integers, the nb_filters of 3 conv layer at main path
        - stage: integer, current stage label, used for generating layer names
        - block: 'a','b'..., current block label, used for generating layer names

    ### 注意
        - Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        - And the shortcut should have strides=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    tensor_x = Conv2D(nb_filter1, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '2a')(input_tensor)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2a')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2a')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2a_relu')(tensor_x)

    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter2, (kernel_size, kernel_size), use_bias=False, name=conv_name_base + '2b')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2b')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2b')(tensor_x)
    tensor_x = Activation('relu', name=conv_name_base + '2b_relu')(tensor_x)

    tensor_x = Conv2D(nb_filter3, (1, 1), use_bias=False, name=conv_name_base + '2c')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '2c')(tensor_x)
    tensor_x = Scale(axis=3, name=scale_name_base + '2c')(tensor_x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=3, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=3, name=scale_name_base + '1')(shortcut)

    tensor_x = add([tensor_x, shortcut], name='res' + str(stage) + block)
    tensor_x = Activation('relu', name='res' + str(stage) + block + '_relu')(tensor_x)

    return tensor_x

def resnet_model(img_rows, img_cols, color_type=3, num_classes=3):
    '''
    ### 说明:
        - DenseNet Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights

    ### Arguments:
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - nb_layers: (list), number of filters in each block
    ### Returns
        - A Keras model instance.
    '''

    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    img_input = Input(shape=(img_rows, img_cols, color_type), name='data')

    # conv1
    tensor_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    tensor_x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name='bn_conv1')(tensor_x)
    tensor_x = Scale(axis=3, name='scale_conv1')(tensor_x)
    tensor_x = Activation('relu', name='conv1_relu')(tensor_x)
    tensor_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(tensor_x)

    # conv2_x
    tensor_x = conv_block(tensor_x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # for i in range(1, nb_layers[0]):
    #     tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='b' + str(i))
    tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='b')
    tensor_x = identity_block(tensor_x, 3, [64, 64, 256], stage=2, block='c')

    # conv3_x
    tensor_x = conv_block(tensor_x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 4):
        tensor_x = identity_block(tensor_x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    # conv4_x
    tensor_x = conv_block(tensor_x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        tensor_x = identity_block(tensor_x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    # conv5_x
    tensor_x = conv_block(tensor_x, 3, [512, 512, 2048], stage=5, block='a')
    # for i in range(1, nb_layers[3]):
    #     tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='b' + str(i))
    tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='b')
    tensor_x = identity_block(tensor_x, 3, [512, 512, 2048], stage=5, block='c')

    # fc
    x_fc = GlobalAveragePooling2D(name='avg_pool')(tensor_x)
    x_fc = Dense(num_classes, activation='softmax', name='fc3')(x_fc)

    model = Model(img_input, x_fc)

    # model.load_weights('resnet_101_finetune_weights_92%.h5')

    return model

if __name__ == '__main__':

    model = resnet_model(480, 480, color_type=3, num_classes=3)
    model.summary()
