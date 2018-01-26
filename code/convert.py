#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# convert.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-1-18 11:51:59
# @Explanation  : 转RT
"""

import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

if __name__ == '__main__':

    config = {
        # Training params
        "train_data_dir": "/home/docker/code/app/get_samples/samples_train/train",  # training data
        "val_data_dir": "/home/docker/code/app/get_samples/samples_train/valid",  # validation data
        "train_batch_size": 8,  # training batch size
        "epochs": 3,  # number of training epochs
        # "num_train_samples" : 217728,  # number of training examples
        # "num_val_samples" : 55872,  # number of test examples
        "num_train_samples" : 500,  # number of training examples
        "num_val_samples" : 300,  # number of test examples

        # Where to save models (Tensorflow + TensorRT)
        "graphdef_file": "./model/resnet_tf_graphdef.pb",
        "frozen_model_file": "./model/resnet_tf_frozen_model.pb",
        "snapshot_dir": "./model/snapshot",
        "engine_save_dir": "./model/",

        # Needed for TensorRT
        "image_dim": 480,  # the image size (square images)
        "inference_batch_size": 256,  # inference batch size
        "input_layer": "data",  # name of the input tensor in the TF computational graph
        "out_layer": "fc3/Softmax",  # name of the output tensorf in the TF conputational graph
        "output_size" : 3,  # number of classes in output (3)
        "precision": "fp16",  # desired precision (fp32, fp16)

        "test_image_path" : "/home/data/val/roses"
    }

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

    # Define network parameters, including inference batch size, name & dimensionality of input/output layers
    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['out_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = 3
    INPUT_H = config['image_dim']
    INPUT_W = config['image_dim']

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(config['frozen_model_file'], OUTPUT_LAYERS)

    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(INPUT_LAYERS[0], (INPUT_C, INPUT_H, INPUT_W), 0)
    parser.register_output(OUTPUT_LAYERS[0])

    # Build your TensorRT inference engine
    if config['precision'] == 'fp32':
        engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                             uff_model,
                                             parser,
                                             INFERENCE_BATCH_SIZE,
                                             1<<20,
                                             trt.infer.DataType.FLOAT
                                            )

    elif config['precision'] == 'fp16':
        engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                             uff_model,
                                             parser,
                                             INFERENCE_BATCH_SIZE,
                                             1<<20,
                                             trt.infer.DataType.HALF
                                            )

    # Serialize TensorRT engine to a file for when you are ready to deploy your model.
    save_path = str(config['engine_save_dir']) + "resnet_tf_b" \
        + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

    trt.utils.write_engine_to_file(save_path, engine.serialize())

    print("Saved TRT engine to {}".format(save_path))
