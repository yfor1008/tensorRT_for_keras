#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# convert_densenet.py
# @Author       : yuanwenjin
# @Mail         : yfor1008@gmail.com
# @Date         : 2018-3-1 18:17:55
# @Explanation  : 转RT，densenet
"""

import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

if __name__ == '__main__':

    config = {
        # Where to save models (Tensorflow + TensorRT)
        "frozen_model_file": "./model_densenet/densenet_tf_frozen_model.pb",
        "engine_save_dir": "./model_densenet/",

        # Needed for TensorRT
        "image_dim": 480,  # the image size (square images)
        "inference_batch_size": 16,  # inference batch size
        "input_layer": "data",  # name of the input tensor in the TF computational graph
        "out_layer": "fc6/Softmax",  # name of the output tensorf in the TF conputational graph
        "output_size" : 3,  # number of classes in output (3)
        "precision": "fp32",  # desired precision (fp32, fp16)

    }

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    # G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    # G_LOGGER = trt.infer.ConsoleLogger(0)

    # Define network parameters, including inference batch size, name & dimensionality of input/output layers
    INPUT_LAYERS = [config['input_layer']]
    OUTPUT_LAYERS = [config['out_layer']]
    INFERENCE_BATCH_SIZE = config['inference_batch_size']

    INPUT_C = config['output_size']
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
    save_path = str(config['engine_save_dir']) + "densenet_tf_b" \
        + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

    trt.utils.write_engine_to_file(save_path, engine.serialize())

    print("Saved TRT engine to {}".format(save_path))