# tensorRT_for_keras
使用tensorRT来加速keras代码



这里实现`resnet` ：

`keras` 实现地址为：[https://github.com/flyyufelix/cnn_finetune](https://github.com/flyyufelix/cnn_finetune)

`tensorRT `使用可以参考[https://github.com/parallel-forall/code-samples/tree/master/posts/TensorRT-3.0](https://github.com/parallel-forall/code-samples/tree/master/posts/TensorRT-3.0)



**相关说明：** 

1. `resnet_tf.py` 为调整后的`keras` 实现，使用了`tensorflow` 中的`keras` ；
2. `train.py` 训练模型，并保存为`freeze_graph` ；
3. `convert.py` 将`freeze_graph` 转换成`tensorRT` 中的`engine` ；
4. `test_*.py` 为测试代码，`*keras` 为没有使用`tensorRT` 加速，`*fp16` 、`fp32` 为使用了`tensorRT` 加速；



为了实现`tensorRT` 对`keras` 代码加速，需进行一些修改(可参考已修改的代码，`resnet_tf.py` )：

1. `import` 库需修改：

   ```python
   from keras.models import Sequential
   from keras.optimizers import SGD
   from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
   from keras.layers.normalization import BatchNormalization
   from keras.models import Model
   from keras import backend as K
   ```

   更改为

   ```python
   from tensorflow.python.keras._impl.keras.optimizers import SGD
   from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.python.keras._impl.keras import backend as K
   from tensorflow.python.keras._impl.keras.callbacks import ModelCheckpoint
   from tensorflow.python.keras._impl.keras.callbacks import EarlyStopping
   from tensorflow.python.keras._impl.keras.callbacks import LearningRateScheduler
   from tensorflow.python.keras._impl.keras.callbacks import TensorBoard
   ```

2. `BatchNormalization` 修改：

   这里不能使用

   ```python
   from tensorflow.python.keras._impl.keras.layers.normalization import BatchNormalization
   ```

   主要是由于`tensorRT` 只支持`tensorflow` 中的`fused batch normalization` ，因此，应使用：

   ```python
   from tensorflow.python.layers.normalization import BatchNormalization
   ```

   **注意：这里修改了`batchnormalization` ，预先训练好的模型导不进来，需要寻找方法解决这个问题，或者重新训练模型(从0开始训练)**



**使用中注意事项：**

1. `tensorboard` 使用：在`callbacks` 中添加`tensorboard` 时，其参数`histogram_freq=0` ，必须设置为0，否则会报错，参考：[https://github.com/tensorflow/tensorflow/pull/9787#issuecomment-360039499](https://github.com/tensorflow/tensorflow/pull/9787#issuecomment-360039499)
2. 转换成`freeze_graph` 时，`freeze_graph.freeze_graph` 需要指定输出层的名字`output_node_names` ，这里`output_node_names=指定名字/功能` ，如`output_node_names=fc3/Softmax` ，`fc3` 为人工指定名字，`Softmax` 为实现功能；
3. 转换成`engine` 时，`trt.utils.uff_to_trt_engine` 有个参数`max_batch_size` ，需根据显存设置，不能太大；
4. 官网帮助文档可能还没有更新，使用时需注意；
5. 屏蔽`log` 输出，可以将`log_sev` 替换成`logger_severity=trt.infer.LogSeverity.ERROR` ，详见：[http://note.youdao.com/noteshare?id=b3fdec4fc9e5861c753987c0196675ef&sub=F20AB86FBA0B4547B0FD3D7930DE4988](http://note.youdao.com/noteshare?id=b3fdec4fc9e5861c753987c0196675ef&sub=F20AB86FBA0B4547B0FD3D7930DE4988) ，也可以参考代码中使用方法；


