#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# densenet_tf.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-3-1 16:23:53
# @Explanation  : 使用tensorflow中的keras实现，方便tensorRT加速，这里去除了dropout层
"""

from tensorflow.python.keras._impl.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras._impl.keras.layers import ZeroPadding2D, add, Activation, concatenate
# from tensorflow.python.keras._impl.keras.layers.normalization import BatchNormalization
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras._impl.keras.models import Model

from scale_layer_tf import Scale

def conv_block(tensor_x, stage, branch, nb_filter):
    '''
    ### 说明:
        - Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D

    ### Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - branch: layer index within each dense block
        - nb_filter: number of filters
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base + '_x1_bn')(tensor_x)
    tensor_x = Scale(axis=3, name=conv_name_base + '_x1_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base + '_x1')(tensor_x)
    tensor_x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False)(tensor_x)

    # 3x3 Convolution
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_x2_bn')(tensor_x)
    tensor_x = Scale(axis=3, name=conv_name_base + '_x2_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base + '_x2')(tensor_x)
    tensor_x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(tensor_x)
    tensor_x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False)(tensor_x)

    return tensor_x

def transition_block(tensor_x, stage, nb_filter, compression=1.0):
    '''
    # 说明:
        Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression
    # Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - nb_filter: number of filters
        - compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    tensor_x = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_bn')(tensor_x)
    tensor_x = Scale(axis=3, name=conv_name_base+'_scale')(tensor_x)
    tensor_x = Activation('relu', name=relu_name_base)(tensor_x)
    tensor_x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(tensor_x)

    tensor_x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(tensor_x)

    return tensor_x

def dense_block(tensor_x, stage, nb_layers, nb_filter, growth_rate, grow_nb_filters=True):
    '''
    # 说明:
        Build a dense_block where the output of each conv_block is fed to subsequent ones
    # Arguments
        - tensor_x: input tensor
        - stage: index for dense block
        - nb_layers: the number of layers of conv_block to append to the model.
        - nb_filter: number of filters
        - growth_rate: growth rate
        - grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    # eps = 1.1e-5
    concat_feat = tensor_x

    for i in range(nb_layers):
        branch = i + 1
        tensor_x = conv_block(concat_feat, stage, branch, growth_rate)
        concat_feat = concatenate([concat_feat, tensor_x], axis=3, \
                                    name='concat_' + str(stage) + '_' + str(branch))
        # concat_feat = add([concat_feat, tensor_x])

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def densenet_model(img_rows, img_cols, color_type=3, num_classes=3,
                   nb_dense_block=4, growth_rate=None,
                   nb_filter=None, nb_layers=None,
                   reduction=0.5,
                   model_size=None):
    '''
    ### 说明:
        - DenseNet Model for Keras
        - Model Schema is based on https://github.com/flyyufelix/DenseNet-Keras
        - ImageNet Pretrained Weights

    ### Arguments:
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - nb_dense_block: (int), number of dense blocks to add to end
        - growth_rate: (int), number of filters to add per dense block
        - nb_filter: (int), initial number of filters
        - nb_layers: (list), number of filters in each block
        - reduction: (float), reduction factor of transition blocks.
        - model_size: (string), layer number of model
    ### Returns
        - A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    img_input = Input(shape=(img_rows, img_cols, color_type), name='data')

    # Initial convolution
    tensor_x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    tensor_x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(tensor_x)
    tensor_x = BatchNormalization(epsilon=eps, axis=3, name='conv1_bn')(tensor_x)
    tensor_x = Scale(axis=3, name='conv1_scale')(tensor_x)
    tensor_x = Activation('relu', name='relu1')(tensor_x)
    tensor_x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(tensor_x)
    tensor_x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(tensor_x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        tensor_x, nb_filter = dense_block(tensor_x, stage, nb_layers[block_idx], nb_filter, growth_rate)

        # Add transition_block
        tensor_x = transition_block(tensor_x, stage, nb_filter, compression=compression)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    tensor_x, nb_filter = dense_block(tensor_x, final_stage, nb_layers[-1], nb_filter, growth_rate)

    tensor_x = BatchNormalization(epsilon=eps, axis=3, name='conv' + str(final_stage) + '_blk_bn')(tensor_x)
    tensor_x = Scale(axis=3, name='conv' + str(final_stage) + '_blk_scale')(tensor_x)
    tensor_x = Activation('relu', name='relu' + str(final_stage) + '_blk')(tensor_x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(tensor_x)
    x_fc = Dense(num_classes, activation='softmax', name='fc6')(x_fc)
    # x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    return model

def generate_densenet(img_rows, img_cols, color_type,
                      reduction=0.5,
                      model_size=None, num_classes=None):
    '''
    ### 说明：
        - 生成需要的模型

    ### 参数：
        - img_rows: (int), image height
        - img_cols: (int), image width
        - color_type: (int), image channel
        - reduction: (float), reduction factor of transition blocks.
        - model_size: (int), layer number of model
        - num_classes: (int), classes

    ### 返回：
        - MODEL: 生成好的模型
    '''

    # dense模型参数: 121/169/161
    if model_size == 121:
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 24, 16]
    elif model_size == 169:
        nb_dense_block = 4
        growth_rate = 32
        nb_filter = 64
        nb_layers = [6, 12, 32, 32]
    elif model_size == 161:
        nb_dense_block = 4
        growth_rate = 48
        nb_filter = 96
        nb_layers = [6, 12, 36, 24]

    model = densenet_model(img_rows=img_rows, img_cols=img_cols, color_type=color_type, num_classes=num_classes,
                           nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                           nb_filter=nb_filter, nb_layers=nb_layers,
                           reduction=reduction, 
                           model_size=model_size)

    return model

if __name__ == '__main__':

    model = generate_densenet(480, 480, 3, model_size=121, num_classes=3)
    model.summary()
