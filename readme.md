
# Custom cifar-10 conv net with Caffe

**Here, I train a custom convnet on the cifar-10 dataset. I did not try to implement any specific known architecture, but to design a new one quickly for learning purposes. It is inspired from the official caffe .ipynb examples available at: https://github.com/BVLC/caffe/tree/master/examples.**

## Dynamically download and convert the cifar-10 dataset to Caffe's HDF5 format using code of another git repo of mine.
More info on the dataset can be found at http://www.cs.toronto.edu/~kriz/cifar.html.


```python
%%time

!rm download-and-convert-cifar-10.py
print("Getting the download script...")
!wget https://raw.githubusercontent.com/guillaume-chevalier/Caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py
print("Downloaded script. Will execute to download and convert the cifar-10 dataset:")
!python download-and-convert-cifar-10.py
```

    rm: cannot remove ‘download-and-convert-cifar-10.py’: No such file or directory
    Getting the download script...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-26 03:49:12--  https://raw.githubusercontent.com/guillaume-chevalier/Caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 23.235.39.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|23.235.39.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3336 (3.3K) [text/plain]
    Saving to: ‘download-and-convert-cifar-10.py’

    100%[======================================>] 3,336       --.-K/s   in 0s      

    2015-12-26 03:49:12 (928 MB/s) - ‘download-and-convert-cifar-10.py’ saved [3336/3336]

    Downloaded script. Will execute to download and convert the cifar-10 dataset:

    Downloading...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-26 03:49:12--  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
    Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 170498071 (163M) [application/x-gzip]
    Saving to: ‘cifar-10-python.tar.gz’

    100%[======================================>] 170,498,071 1.23MB/s   in 2m 13s

    2015-12-26 03:51:25 (1.22 MB/s) - ‘cifar-10-python.tar.gz’ saved [170498071/170498071]

    Downloading done.

    Extracting...
    cifar-10-batches-py/
    cifar-10-batches-py/data_batch_4
    cifar-10-batches-py/readme.html
    cifar-10-batches-py/test_batch
    cifar-10-batches-py/data_batch_3
    cifar-10-batches-py/batches.meta
    cifar-10-batches-py/data_batch_2
    cifar-10-batches-py/data_batch_5
    cifar-10-batches-py/data_batch_1
    Extracting successfully done to /home/gui/Documents/custom-cifar-10/cifar-10-batches-py.
    Converting...
    INFO: each dataset's element are of shape 3*32*32:
    "print(X.shape)" --> "(50000, 3, 32, 32)"

    From the Caffe documentation:
    The conventional blob dimensions for batches of image data are number N x channel K x height H x width W.

    Data is fully loaded, now truly converting.
    Conversion successfully done to "/home/gui/Documents/custom-cifar-10/cifar_10_caffe_hdf5".

    CPU times: user 752 ms, sys: 156 ms, total: 908 ms
    Wall time: 2min 21s


## Build the model with Caffe.


```python
import numpy as np

import caffe
from caffe import layers as L
from caffe import params as P
```


```python
def cnn(hdf5, batch_size):
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=12, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=3, num_output=32, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu2 = L.ReLU(n.pool2, in_place=True)
    n.conv3 = L.Convolution(n.relu2, kernel_size=5, num_output=64, weight_filler=dict(type='xavier'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)

    n.ip1 = L.InnerProduct(n.relu3, num_output=512, weight_filler=dict(type='xavier'))
    n.relu4 = L.ReLU(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.relu4, num_output=10, weight_filler=dict(type='xavier'))

    n.accuracy = L.Accuracy(n.ip2, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open('cnn_train.prototxt', 'w') as f:
    f.write(str(cnn('cifar_10_caffe_hdf5/train.txt', 100)))

with open('cnn_test.prototxt', 'w') as f:
    f.write(str(cnn('cifar_10_caffe_hdf5/test.txt', 120)))
```

## Load and visualise the untrained network's internal structure and shape
The network visualisation tool of caffe is broken in the current release. We will simply print here the data shapes.


```python
caffe.set_mode_gpu()
solver = caffe.get_solver('cnn_solver_rms.prototxt')
```


```python
print("Layers' features:")
[(k, v.data.shape) for k, v in solver.net.blobs.items()]
```

    Layers' features:

    [('data', (100, 3, 32, 32)),
     ('label', (100,)),
     ('label_data_1_split_0', (100,)),
     ('label_data_1_split_1', (100,)),
     ('conv1', (100, 12, 30, 30)),
     ('pool1', (100, 12, 15, 15)),
     ('conv2', (100, 32, 13, 13)),
     ('pool2', (100, 32, 7, 7)),
     ('conv3', (100, 64, 3, 3)),
     ('ip1', (100, 512)),
     ('ip2', (100, 10)),
     ('ip2_ip2_0_split_0', (100, 10)),
     ('ip2_ip2_0_split_1', (100, 10)),
     ('accuracy', ()),
     ('loss', ())]


```python
print("Parameters and shape:")
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
```

    Parameters and shape:

    [('conv1', (12, 3, 3, 3)),
     ('conv2', (32, 12, 3, 3)),
     ('conv3', (64, 32, 5, 5)),
     ('ip1', (512, 576)),
     ('ip2', (10, 512))]


## Solver's params
The solver's params for the created net are defined in a `.prototxt` file.

Notice that because `max_iter: 150000`, the training will loop 3 times on the 50000 training data. Because we train data by minibatches of 100 as defined above upon creating the net, there will be a total of `150000*100/50000 = 300` epochs.

We will test the net on `test_iter: 1000` test data at each `test_interval: 5000` images trained.

The learning rate has been adjusted so that the learning does not diverge but performs rapidly.


```python
!cat cnn_solver.prototxt
```

    cat: cnn_solver.prototxt: No such file or directory


## Alternative way to train directly in Python
Since a recent update, there is no output in python by default, which is bad for debugging.
Skip this cell and train with the second method shown below if needed. It is commented out in case you just chain some `shift+enter` ipython shortcuts.


```python
# %%time
# solver.solve()
```

## Train by calling caffe in command line
Just set the parameters correctly. Be sure that the notebook is at the root of the ipython notebook server.
You can run this in an external terminal if you open it in the notebook's directory.

It is also possible to finetune an existing net with a different solver or different data. Here I do it, because I feel the net could better fit the data.


```python
%%time
!$CAFFE_ROOT/build/tools/caffe train -solver cnn_solver_rms.prototxt
```

    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1226 14:34:18.170622 25521 caffe.cpp:184] Using GPUs 0
    I1226 14:34:18.328018 25521 solver.cpp:48] Initializing solver from parameters:
    train_net: "cnn_train.prototxt"
    test_net: "cnn_test.prototxt"
    test_iter: 100
    test_interval: 1000
    base_lr: 0.001
    display: 100
    max_iter: 100000
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    momentum: 0
    weight_decay: 0.0005
    snapshot: 25000
    snapshot_prefix: "cnn_snapshot"
    solver_mode: GPU
    device_id: 0
    rms_decay: 0.98
    type: "RMSProp"
    I1226 14:34:18.328227 25521 solver.cpp:81] Creating training net from train_net file: cnn_train.prototxt
    I1226 14:34:18.328565 25521 net.cpp:49] Initializing net from parameters:
    state {
      phase: TRAIN
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label"
      hdf5_data_param {
        source: "cifar_10_caffe_hdf5/train.txt"
        batch_size: 100
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 12
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "pool1"
      top: "pool1"
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      convolution_param {
        num_output: 32
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu2"
      type: "ReLU"
      bottom: "pool2"
      top: "pool2"
    }
    layer {
      name: "conv3"
      type: "Convolution"
      bottom: "pool2"
      top: "conv3"
      convolution_param {
        num_output: 64
        kernel_size: 5
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "conv3"
      top: "conv3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "conv3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip2"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip2"
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy"
      type: "Accuracy"
      bottom: "ip2"
      bottom: "label"
      top: "accuracy"
    }
    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
      top: "loss"
    }
    I1226 14:34:18.328951 25521 layer_factory.hpp:77] Creating layer data
    I1226 14:34:18.328968 25521 net.cpp:106] Creating Layer data
    I1226 14:34:18.328974 25521 net.cpp:411] data -> data
    I1226 14:34:18.328994 25521 net.cpp:411] data -> label
    I1226 14:34:18.329005 25521 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/train.txt
    I1226 14:34:18.329030 25521 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1226 14:34:18.329920 25521 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1226 14:34:20.432296 25521 net.cpp:150] Setting up data
    I1226 14:34:20.432355 25521 net.cpp:157] Top shape: 100 3 32 32 (307200)
    I1226 14:34:20.432364 25521 net.cpp:157] Top shape: 100 (100)
    I1226 14:34:20.432370 25521 net.cpp:165] Memory required for data: 1229200
    I1226 14:34:20.432381 25521 layer_factory.hpp:77] Creating layer label_data_1_split
    I1226 14:34:20.432402 25521 net.cpp:106] Creating Layer label_data_1_split
    I1226 14:34:20.432410 25521 net.cpp:454] label_data_1_split <- label
    I1226 14:34:20.432425 25521 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1226 14:34:20.432436 25521 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1226 14:34:20.432479 25521 net.cpp:150] Setting up label_data_1_split
    I1226 14:34:20.432488 25521 net.cpp:157] Top shape: 100 (100)
    I1226 14:34:20.432494 25521 net.cpp:157] Top shape: 100 (100)
    I1226 14:34:20.432500 25521 net.cpp:165] Memory required for data: 1230000
    I1226 14:34:20.432505 25521 layer_factory.hpp:77] Creating layer conv1
    I1226 14:34:20.432518 25521 net.cpp:106] Creating Layer conv1
    I1226 14:34:20.432524 25521 net.cpp:454] conv1 <- data
    I1226 14:34:20.432569 25521 net.cpp:411] conv1 -> conv1
    I1226 14:34:20.433496 25521 net.cpp:150] Setting up conv1
    I1226 14:34:20.433523 25521 net.cpp:157] Top shape: 100 12 30 30 (1080000)
    I1226 14:34:20.433529 25521 net.cpp:165] Memory required for data: 5550000
    I1226 14:34:20.433543 25521 layer_factory.hpp:77] Creating layer pool1
    I1226 14:34:20.433554 25521 net.cpp:106] Creating Layer pool1
    I1226 14:34:20.433569 25521 net.cpp:454] pool1 <- conv1
    I1226 14:34:20.433576 25521 net.cpp:411] pool1 -> pool1
    I1226 14:34:20.433614 25521 net.cpp:150] Setting up pool1
    I1226 14:34:20.433622 25521 net.cpp:157] Top shape: 100 12 15 15 (270000)
    I1226 14:34:20.433627 25521 net.cpp:165] Memory required for data: 6630000
    I1226 14:34:20.433632 25521 layer_factory.hpp:77] Creating layer relu1
    I1226 14:34:20.433640 25521 net.cpp:106] Creating Layer relu1
    I1226 14:34:20.433645 25521 net.cpp:454] relu1 <- pool1
    I1226 14:34:20.433651 25521 net.cpp:397] relu1 -> pool1 (in-place)
    I1226 14:34:20.433670 25521 net.cpp:150] Setting up relu1
    I1226 14:34:20.433676 25521 net.cpp:157] Top shape: 100 12 15 15 (270000)
    I1226 14:34:20.433682 25521 net.cpp:165] Memory required for data: 7710000
    I1226 14:34:20.433687 25521 layer_factory.hpp:77] Creating layer conv2
    I1226 14:34:20.433706 25521 net.cpp:106] Creating Layer conv2
    I1226 14:34:20.433712 25521 net.cpp:454] conv2 <- pool1
    I1226 14:34:20.433720 25521 net.cpp:411] conv2 -> conv2
    I1226 14:34:20.434329 25521 net.cpp:150] Setting up conv2
    I1226 14:34:20.434355 25521 net.cpp:157] Top shape: 100 32 13 13 (540800)
    I1226 14:34:20.434361 25521 net.cpp:165] Memory required for data: 9873200
    I1226 14:34:20.434371 25521 layer_factory.hpp:77] Creating layer pool2
    I1226 14:34:20.434379 25521 net.cpp:106] Creating Layer pool2
    I1226 14:34:20.434386 25521 net.cpp:454] pool2 <- conv2
    I1226 14:34:20.434402 25521 net.cpp:411] pool2 -> pool2
    I1226 14:34:20.434429 25521 net.cpp:150] Setting up pool2
    I1226 14:34:20.434437 25521 net.cpp:157] Top shape: 100 32 7 7 (156800)
    I1226 14:34:20.434442 25521 net.cpp:165] Memory required for data: 10500400
    I1226 14:34:20.434447 25521 layer_factory.hpp:77] Creating layer relu2
    I1226 14:34:20.434453 25521 net.cpp:106] Creating Layer relu2
    I1226 14:34:20.434458 25521 net.cpp:454] relu2 <- pool2
    I1226 14:34:20.434463 25521 net.cpp:397] relu2 -> pool2 (in-place)
    I1226 14:34:20.434470 25521 net.cpp:150] Setting up relu2
    I1226 14:34:20.434476 25521 net.cpp:157] Top shape: 100 32 7 7 (156800)
    I1226 14:34:20.434480 25521 net.cpp:165] Memory required for data: 11127600
    I1226 14:34:20.434485 25521 layer_factory.hpp:77] Creating layer conv3
    I1226 14:34:20.434494 25521 net.cpp:106] Creating Layer conv3
    I1226 14:34:20.434499 25521 net.cpp:454] conv3 <- pool2
    I1226 14:34:20.434505 25521 net.cpp:411] conv3 -> conv3
    I1226 14:34:20.434983 25521 net.cpp:150] Setting up conv3
    I1226 14:34:20.435004 25521 net.cpp:157] Top shape: 100 64 3 3 (57600)
    I1226 14:34:20.435010 25521 net.cpp:165] Memory required for data: 11358000
    I1226 14:34:20.435019 25521 layer_factory.hpp:77] Creating layer relu3
    I1226 14:34:20.435027 25521 net.cpp:106] Creating Layer relu3
    I1226 14:34:20.435034 25521 net.cpp:454] relu3 <- conv3
    I1226 14:34:20.435050 25521 net.cpp:397] relu3 -> conv3 (in-place)
    I1226 14:34:20.435057 25521 net.cpp:150] Setting up relu3
    I1226 14:34:20.435062 25521 net.cpp:157] Top shape: 100 64 3 3 (57600)
    I1226 14:34:20.435067 25521 net.cpp:165] Memory required for data: 11588400
    I1226 14:34:20.435072 25521 layer_factory.hpp:77] Creating layer ip1
    I1226 14:34:20.435084 25521 net.cpp:106] Creating Layer ip1
    I1226 14:34:20.435089 25521 net.cpp:454] ip1 <- conv3
    I1226 14:34:20.435096 25521 net.cpp:411] ip1 -> ip1
    I1226 14:34:20.437358 25521 net.cpp:150] Setting up ip1
    I1226 14:34:20.437383 25521 net.cpp:157] Top shape: 100 512 (51200)
    I1226 14:34:20.437389 25521 net.cpp:165] Memory required for data: 11793200
    I1226 14:34:20.437398 25521 layer_factory.hpp:77] Creating layer relu4
    I1226 14:34:20.437407 25521 net.cpp:106] Creating Layer relu4
    I1226 14:34:20.437412 25521 net.cpp:454] relu4 <- ip1
    I1226 14:34:20.437419 25521 net.cpp:397] relu4 -> ip1 (in-place)
    I1226 14:34:20.437441 25521 net.cpp:150] Setting up relu4
    I1226 14:34:20.437448 25521 net.cpp:157] Top shape: 100 512 (51200)
    I1226 14:34:20.437453 25521 net.cpp:165] Memory required for data: 11998000
    I1226 14:34:20.437459 25521 layer_factory.hpp:77] Creating layer ip2
    I1226 14:34:20.437466 25521 net.cpp:106] Creating Layer ip2
    I1226 14:34:20.437472 25521 net.cpp:454] ip2 <- ip1
    I1226 14:34:20.437479 25521 net.cpp:411] ip2 -> ip2
    I1226 14:34:20.437999 25521 net.cpp:150] Setting up ip2
    I1226 14:34:20.438022 25521 net.cpp:157] Top shape: 100 10 (1000)
    I1226 14:34:20.438029 25521 net.cpp:165] Memory required for data: 12002000
    I1226 14:34:20.438040 25521 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1226 14:34:20.438048 25521 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1226 14:34:20.438055 25521 net.cpp:454] ip2_ip2_0_split <- ip2
    I1226 14:34:20.438071 25521 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1226 14:34:20.438079 25521 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1226 14:34:20.438105 25521 net.cpp:150] Setting up ip2_ip2_0_split
    I1226 14:34:20.438112 25521 net.cpp:157] Top shape: 100 10 (1000)
    I1226 14:34:20.438118 25521 net.cpp:157] Top shape: 100 10 (1000)
    I1226 14:34:20.438123 25521 net.cpp:165] Memory required for data: 12010000
    I1226 14:34:20.438128 25521 layer_factory.hpp:77] Creating layer accuracy
    I1226 14:34:20.438135 25521 net.cpp:106] Creating Layer accuracy
    I1226 14:34:20.438140 25521 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1226 14:34:20.438146 25521 net.cpp:454] accuracy <- label_data_1_split_0
    I1226 14:34:20.438153 25521 net.cpp:411] accuracy -> accuracy
    I1226 14:34:20.438161 25521 net.cpp:150] Setting up accuracy
    I1226 14:34:20.438168 25521 net.cpp:157] Top shape: (1)
    I1226 14:34:20.438172 25521 net.cpp:165] Memory required for data: 12010004
    I1226 14:34:20.438177 25521 layer_factory.hpp:77] Creating layer loss
    I1226 14:34:20.438186 25521 net.cpp:106] Creating Layer loss
    I1226 14:34:20.438192 25521 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1226 14:34:20.438208 25521 net.cpp:454] loss <- label_data_1_split_1
    I1226 14:34:20.438215 25521 net.cpp:411] loss -> loss
    I1226 14:34:20.438227 25521 layer_factory.hpp:77] Creating layer loss
    I1226 14:34:20.438303 25521 net.cpp:150] Setting up loss
    I1226 14:34:20.438311 25521 net.cpp:157] Top shape: (1)
    I1226 14:34:20.438316 25521 net.cpp:160]     with loss weight 1
    I1226 14:34:20.438336 25521 net.cpp:165] Memory required for data: 12010008
    I1226 14:34:20.438341 25521 net.cpp:226] loss needs backward computation.
    I1226 14:34:20.438347 25521 net.cpp:228] accuracy does not need backward computation.
    I1226 14:34:20.438352 25521 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1226 14:34:20.438357 25521 net.cpp:226] ip2 needs backward computation.
    I1226 14:34:20.438362 25521 net.cpp:226] relu4 needs backward computation.
    I1226 14:34:20.438367 25521 net.cpp:226] ip1 needs backward computation.
    I1226 14:34:20.438372 25521 net.cpp:226] relu3 needs backward computation.
    I1226 14:34:20.438377 25521 net.cpp:226] conv3 needs backward computation.
    I1226 14:34:20.438382 25521 net.cpp:226] relu2 needs backward computation.
    I1226 14:34:20.438397 25521 net.cpp:226] pool2 needs backward computation.
    I1226 14:34:20.438403 25521 net.cpp:226] conv2 needs backward computation.
    I1226 14:34:20.438408 25521 net.cpp:226] relu1 needs backward computation.
    I1226 14:34:20.438415 25521 net.cpp:226] pool1 needs backward computation.
    I1226 14:34:20.438419 25521 net.cpp:226] conv1 needs backward computation.
    I1226 14:34:20.438426 25521 net.cpp:228] label_data_1_split does not need backward computation.
    I1226 14:34:20.438441 25521 net.cpp:228] data does not need backward computation.
    I1226 14:34:20.438446 25521 net.cpp:270] This network produces output accuracy
    I1226 14:34:20.438452 25521 net.cpp:270] This network produces output loss
    I1226 14:34:20.438464 25521 net.cpp:283] Network initialization done.
    I1226 14:34:20.438694 25521 solver.cpp:181] Creating test net (#0) specified by test_net file: cnn_test.prototxt
    I1226 14:34:20.438807 25521 net.cpp:49] Initializing net from parameters:
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label"
      hdf5_data_param {
        source: "cifar_10_caffe_hdf5/test.txt"
        batch_size: 120
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 12
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "pool1"
      top: "pool1"
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      convolution_param {
        num_output: 32
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu2"
      type: "ReLU"
      bottom: "pool2"
      top: "pool2"
    }
    layer {
      name: "conv3"
      type: "Convolution"
      bottom: "pool2"
      top: "conv3"
      convolution_param {
        num_output: 64
        kernel_size: 5
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "conv3"
      top: "conv3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "conv3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip2"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip2"
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy"
      type: "Accuracy"
      bottom: "ip2"
      bottom: "label"
      top: "accuracy"
    }
    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
      top: "loss"
    }
    I1226 14:34:20.439190 25521 layer_factory.hpp:77] Creating layer data
    I1226 14:34:20.439200 25521 net.cpp:106] Creating Layer data
    I1226 14:34:20.439206 25521 net.cpp:411] data -> data
    I1226 14:34:20.439214 25521 net.cpp:411] data -> label
    I1226 14:34:20.439223 25521 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1226 14:34:20.439242 25521 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1226 14:34:20.794638 25521 net.cpp:150] Setting up data
    I1226 14:34:20.794672 25521 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1226 14:34:20.794679 25521 net.cpp:157] Top shape: 120 (120)
    I1226 14:34:20.794684 25521 net.cpp:165] Memory required for data: 1475040
    I1226 14:34:20.794692 25521 layer_factory.hpp:77] Creating layer label_data_1_split
    I1226 14:34:20.794708 25521 net.cpp:106] Creating Layer label_data_1_split
    I1226 14:34:20.794713 25521 net.cpp:454] label_data_1_split <- label
    I1226 14:34:20.794723 25521 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1226 14:34:20.794733 25521 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1226 14:34:20.794762 25521 net.cpp:150] Setting up label_data_1_split
    I1226 14:34:20.794770 25521 net.cpp:157] Top shape: 120 (120)
    I1226 14:34:20.794776 25521 net.cpp:157] Top shape: 120 (120)
    I1226 14:34:20.794781 25521 net.cpp:165] Memory required for data: 1476000
    I1226 14:34:20.794786 25521 layer_factory.hpp:77] Creating layer conv1
    I1226 14:34:20.794797 25521 net.cpp:106] Creating Layer conv1
    I1226 14:34:20.794802 25521 net.cpp:454] conv1 <- data
    I1226 14:34:20.794809 25521 net.cpp:411] conv1 -> conv1
    I1226 14:34:20.794981 25521 net.cpp:150] Setting up conv1
    I1226 14:34:20.794991 25521 net.cpp:157] Top shape: 120 12 30 30 (1296000)
    I1226 14:34:20.794996 25521 net.cpp:165] Memory required for data: 6660000
    I1226 14:34:20.795006 25521 layer_factory.hpp:77] Creating layer pool1
    I1226 14:34:20.795014 25521 net.cpp:106] Creating Layer pool1
    I1226 14:34:20.795019 25521 net.cpp:454] pool1 <- conv1
    I1226 14:34:20.795027 25521 net.cpp:411] pool1 -> pool1
    I1226 14:34:20.795055 25521 net.cpp:150] Setting up pool1
    I1226 14:34:20.795063 25521 net.cpp:157] Top shape: 120 12 15 15 (324000)
    I1226 14:34:20.795104 25521 net.cpp:165] Memory required for data: 7956000
    I1226 14:34:20.795110 25521 layer_factory.hpp:77] Creating layer relu1
    I1226 14:34:20.795120 25521 net.cpp:106] Creating Layer relu1
    I1226 14:34:20.795126 25521 net.cpp:454] relu1 <- pool1
    I1226 14:34:20.795133 25521 net.cpp:397] relu1 -> pool1 (in-place)
    I1226 14:34:20.795142 25521 net.cpp:150] Setting up relu1
    I1226 14:34:20.795150 25521 net.cpp:157] Top shape: 120 12 15 15 (324000)
    I1226 14:34:20.795156 25521 net.cpp:165] Memory required for data: 9252000
    I1226 14:34:20.795171 25521 layer_factory.hpp:77] Creating layer conv2
    I1226 14:34:20.795177 25521 net.cpp:106] Creating Layer conv2
    I1226 14:34:20.795183 25521 net.cpp:454] conv2 <- pool1
    I1226 14:34:20.795189 25521 net.cpp:411] conv2 -> conv2
    I1226 14:34:20.795356 25521 net.cpp:150] Setting up conv2
    I1226 14:34:20.795366 25521 net.cpp:157] Top shape: 120 32 13 13 (648960)
    I1226 14:34:20.795370 25521 net.cpp:165] Memory required for data: 11847840
    I1226 14:34:20.795379 25521 layer_factory.hpp:77] Creating layer pool2
    I1226 14:34:20.795387 25521 net.cpp:106] Creating Layer pool2
    I1226 14:34:20.795392 25521 net.cpp:454] pool2 <- conv2
    I1226 14:34:20.795398 25521 net.cpp:411] pool2 -> pool2
    I1226 14:34:20.795426 25521 net.cpp:150] Setting up pool2
    I1226 14:34:20.795433 25521 net.cpp:157] Top shape: 120 32 7 7 (188160)
    I1226 14:34:20.795439 25521 net.cpp:165] Memory required for data: 12600480
    I1226 14:34:20.795444 25521 layer_factory.hpp:77] Creating layer relu2
    I1226 14:34:20.795450 25521 net.cpp:106] Creating Layer relu2
    I1226 14:34:20.795455 25521 net.cpp:454] relu2 <- pool2
    I1226 14:34:20.795461 25521 net.cpp:397] relu2 -> pool2 (in-place)
    I1226 14:34:20.795469 25521 net.cpp:150] Setting up relu2
    I1226 14:34:20.795475 25521 net.cpp:157] Top shape: 120 32 7 7 (188160)
    I1226 14:34:20.795480 25521 net.cpp:165] Memory required for data: 13353120
    I1226 14:34:20.795485 25521 layer_factory.hpp:77] Creating layer conv3
    I1226 14:34:20.795493 25521 net.cpp:106] Creating Layer conv3
    I1226 14:34:20.795498 25521 net.cpp:454] conv3 <- pool2
    I1226 14:34:20.795506 25521 net.cpp:411] conv3 -> conv3
    I1226 14:34:20.796016 25521 net.cpp:150] Setting up conv3
    I1226 14:34:20.796030 25521 net.cpp:157] Top shape: 120 64 3 3 (69120)
    I1226 14:34:20.796036 25521 net.cpp:165] Memory required for data: 13629600
    I1226 14:34:20.796046 25521 layer_factory.hpp:77] Creating layer relu3
    I1226 14:34:20.796053 25521 net.cpp:106] Creating Layer relu3
    I1226 14:34:20.796059 25521 net.cpp:454] relu3 <- conv3
    I1226 14:34:20.796067 25521 net.cpp:397] relu3 -> conv3 (in-place)
    I1226 14:34:20.796075 25521 net.cpp:150] Setting up relu3
    I1226 14:34:20.796082 25521 net.cpp:157] Top shape: 120 64 3 3 (69120)
    I1226 14:34:20.796087 25521 net.cpp:165] Memory required for data: 13906080
    I1226 14:34:20.796093 25521 layer_factory.hpp:77] Creating layer ip1
    I1226 14:34:20.796103 25521 net.cpp:106] Creating Layer ip1
    I1226 14:34:20.796108 25521 net.cpp:454] ip1 <- conv3
    I1226 14:34:20.796116 25521 net.cpp:411] ip1 -> ip1
    I1226 14:34:20.798964 25521 net.cpp:150] Setting up ip1
    I1226 14:34:20.798985 25521 net.cpp:157] Top shape: 120 512 (61440)
    I1226 14:34:20.798991 25521 net.cpp:165] Memory required for data: 14151840
    I1226 14:34:20.799001 25521 layer_factory.hpp:77] Creating layer relu4
    I1226 14:34:20.799010 25521 net.cpp:106] Creating Layer relu4
    I1226 14:34:20.799015 25521 net.cpp:454] relu4 <- ip1
    I1226 14:34:20.799023 25521 net.cpp:397] relu4 -> ip1 (in-place)
    I1226 14:34:20.799032 25521 net.cpp:150] Setting up relu4
    I1226 14:34:20.799039 25521 net.cpp:157] Top shape: 120 512 (61440)
    I1226 14:34:20.799044 25521 net.cpp:165] Memory required for data: 14397600
    I1226 14:34:20.799051 25521 layer_factory.hpp:77] Creating layer ip2
    I1226 14:34:20.799058 25521 net.cpp:106] Creating Layer ip2
    I1226 14:34:20.799064 25521 net.cpp:454] ip2 <- ip1
    I1226 14:34:20.799072 25521 net.cpp:411] ip2 -> ip2
    I1226 14:34:20.799201 25521 net.cpp:150] Setting up ip2
    I1226 14:34:20.799209 25521 net.cpp:157] Top shape: 120 10 (1200)
    I1226 14:34:20.799229 25521 net.cpp:165] Memory required for data: 14402400
    I1226 14:34:20.799242 25521 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1226 14:34:20.799250 25521 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1226 14:34:20.799257 25521 net.cpp:454] ip2_ip2_0_split <- ip2
    I1226 14:34:20.799263 25521 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1226 14:34:20.799273 25521 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1226 14:34:20.799304 25521 net.cpp:150] Setting up ip2_ip2_0_split
    I1226 14:34:20.799321 25521 net.cpp:157] Top shape: 120 10 (1200)
    I1226 14:34:20.799329 25521 net.cpp:157] Top shape: 120 10 (1200)
    I1226 14:34:20.799335 25521 net.cpp:165] Memory required for data: 14412000
    I1226 14:34:20.799340 25521 layer_factory.hpp:77] Creating layer accuracy
    I1226 14:34:20.799347 25521 net.cpp:106] Creating Layer accuracy
    I1226 14:34:20.799353 25521 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1226 14:34:20.799360 25521 net.cpp:454] accuracy <- label_data_1_split_0
    I1226 14:34:20.799367 25521 net.cpp:411] accuracy -> accuracy
    I1226 14:34:20.799376 25521 net.cpp:150] Setting up accuracy
    I1226 14:34:20.799383 25521 net.cpp:157] Top shape: (1)
    I1226 14:34:20.799388 25521 net.cpp:165] Memory required for data: 14412004
    I1226 14:34:20.799394 25521 layer_factory.hpp:77] Creating layer loss
    I1226 14:34:20.799401 25521 net.cpp:106] Creating Layer loss
    I1226 14:34:20.799407 25521 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1226 14:34:20.799413 25521 net.cpp:454] loss <- label_data_1_split_1
    I1226 14:34:20.799420 25521 net.cpp:411] loss -> loss
    I1226 14:34:20.799429 25521 layer_factory.hpp:77] Creating layer loss
    I1226 14:34:20.799499 25521 net.cpp:150] Setting up loss
    I1226 14:34:20.799507 25521 net.cpp:157] Top shape: (1)
    I1226 14:34:20.799513 25521 net.cpp:160]     with loss weight 1
    I1226 14:34:20.799525 25521 net.cpp:165] Memory required for data: 14412008
    I1226 14:34:20.799531 25521 net.cpp:226] loss needs backward computation.
    I1226 14:34:20.799537 25521 net.cpp:228] accuracy does not need backward computation.
    I1226 14:34:20.799545 25521 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1226 14:34:20.799551 25521 net.cpp:226] ip2 needs backward computation.
    I1226 14:34:20.799556 25521 net.cpp:226] relu4 needs backward computation.
    I1226 14:34:20.799561 25521 net.cpp:226] ip1 needs backward computation.
    I1226 14:34:20.799566 25521 net.cpp:226] relu3 needs backward computation.
    I1226 14:34:20.799572 25521 net.cpp:226] conv3 needs backward computation.
    I1226 14:34:20.799578 25521 net.cpp:226] relu2 needs backward computation.
    I1226 14:34:20.799583 25521 net.cpp:226] pool2 needs backward computation.
    I1226 14:34:20.799589 25521 net.cpp:226] conv2 needs backward computation.
    I1226 14:34:20.799594 25521 net.cpp:226] relu1 needs backward computation.
    I1226 14:34:20.799600 25521 net.cpp:226] pool1 needs backward computation.
    I1226 14:34:20.799607 25521 net.cpp:226] conv1 needs backward computation.
    I1226 14:34:20.799612 25521 net.cpp:228] label_data_1_split does not need backward computation.
    I1226 14:34:20.799618 25521 net.cpp:228] data does not need backward computation.
    I1226 14:34:20.799624 25521 net.cpp:270] This network produces output accuracy
    I1226 14:34:20.799630 25521 net.cpp:270] This network produces output loss
    I1226 14:34:20.799643 25521 net.cpp:283] Network initialization done.
    I1226 14:34:20.799697 25521 solver.cpp:60] Solver scaffolding done.
    I1226 14:34:20.799978 25521 caffe.cpp:212] Starting Optimization
    I1226 14:34:20.799990 25521 solver.cpp:288] Solving
    I1226 14:34:20.799996 25521 solver.cpp:289] Learning Rate Policy: inv
    I1226 14:34:20.800971 25521 solver.cpp:341] Iteration 0, Testing net (#0)
    I1226 14:34:23.628445 25521 solver.cpp:409]     Test net output #0: accuracy = 0.0998333
    I1226 14:34:23.628509 25521 solver.cpp:409]     Test net output #1: loss = 53.4034 (* 1 = 53.4034 loss)
    I1226 14:34:23.663450 25521 solver.cpp:237] Iteration 0, loss = 55.1967
    I1226 14:34:23.663491 25521 solver.cpp:253]     Train net output #0: accuracy = 0.07
    I1226 14:34:23.663506 25521 solver.cpp:253]     Train net output #1: loss = 55.1967 (* 1 = 55.1967 loss)
    I1226 14:34:23.663568 25521 sgd_solver.cpp:106] Iteration 0, lr = 0.001
    I1226 14:34:30.216686 25521 solver.cpp:237] Iteration 100, loss = 2.29982
    I1226 14:34:30.216753 25521 solver.cpp:253]     Train net output #0: accuracy = 0.1
    I1226 14:34:30.216764 25521 solver.cpp:253]     Train net output #1: loss = 2.29982 (* 1 = 2.29982 loss)
    I1226 14:34:30.216774 25521 sgd_solver.cpp:106] Iteration 100, lr = 0.000992565
    I1226 14:34:36.774407 25521 solver.cpp:237] Iteration 200, loss = 2.30387
    I1226 14:34:36.774453 25521 solver.cpp:253]     Train net output #0: accuracy = 0.11
    I1226 14:34:36.774468 25521 solver.cpp:253]     Train net output #1: loss = 2.30387 (* 1 = 2.30387 loss)
    I1226 14:34:36.774478 25521 sgd_solver.cpp:106] Iteration 200, lr = 0.000985258
    I1226 14:34:43.277602 25521 solver.cpp:237] Iteration 300, loss = 2.26831
    I1226 14:34:43.277647 25521 solver.cpp:253]     Train net output #0: accuracy = 0.08
    I1226 14:34:43.277658 25521 solver.cpp:253]     Train net output #1: loss = 2.26831 (* 1 = 2.26831 loss)
    I1226 14:34:43.277668 25521 sgd_solver.cpp:106] Iteration 300, lr = 0.000978075
    I1226 14:34:50.998688 25521 solver.cpp:237] Iteration 400, loss = 2.25397
    I1226 14:34:50.998905 25521 solver.cpp:253]     Train net output #0: accuracy = 0.14
    I1226 14:34:50.998944 25521 solver.cpp:253]     Train net output #1: loss = 2.25397 (* 1 = 2.25397 loss)
    I1226 14:34:50.998953 25521 sgd_solver.cpp:106] Iteration 400, lr = 0.000971013
    ...
    I1226 16:53:58.854172 25521 sgd_solver.cpp:106] Iteration 99600, lr = 0.000166013
    I1226 16:54:05.705972 25521 solver.cpp:237] Iteration 99700, loss = 0.00198227
    I1226 16:54:05.706137 25521 solver.cpp:253]     Train net output #0: accuracy = 1
    I1226 16:54:05.706152 25521 solver.cpp:253]     Train net output #1: loss = 0.00198172 (* 1 = 0.00198172 loss)
    I1226 16:54:05.706162 25521 sgd_solver.cpp:106] Iteration 99700, lr = 0.0001659
    I1226 16:54:12.529934 25521 solver.cpp:237] Iteration 99800, loss = 0.00312694
    I1226 16:54:12.529984 25521 solver.cpp:253]     Train net output #0: accuracy = 1
    I1226 16:54:12.529999 25521 solver.cpp:253]     Train net output #1: loss = 0.00312638 (* 1 = 0.00312638 loss)
    I1226 16:54:12.530010 25521 sgd_solver.cpp:106] Iteration 99800, lr = 0.000165786
    I1226 16:54:19.412245 25521 solver.cpp:237] Iteration 99900, loss = 0.00573747
    I1226 16:54:19.412289 25521 solver.cpp:253]     Train net output #0: accuracy = 1
    I1226 16:54:19.412302 25521 solver.cpp:253]     Train net output #1: loss = 0.00573691 (* 1 = 0.00573691 loss)
    I1226 16:54:19.412312 25521 sgd_solver.cpp:106] Iteration 99900, lr = 0.000165673
    I1226 16:54:26.147887 25521 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_100000.caffemodel
    I1226 16:54:26.181751 25521 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_100000.solverstate
    I1226 16:54:26.205761 25521 solver.cpp:321] Iteration 100000, loss = 0.0115375
    I1226 16:54:26.205813 25521 solver.cpp:341] Iteration 100000, Testing net (#0)
    I1226 16:54:29.107939 25521 solver.cpp:409]     Test net output #0: accuracy = 0.635917
    I1226 16:54:29.107987 25521 solver.cpp:409]     Test net output #1: loss = 2.67888 (* 1 = 2.67888 loss)
    I1226 16:54:29.107995 25521 solver.cpp:326] Optimization Done.
    I1226 16:54:29.108368 25521 caffe.cpp:215] Optimization Done.
    CPU times: user 30.4 s, sys: 3.96 s, total: 34.4 s
    Wall time: 2h 20min 11s

## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_100000.caffemodel -iterations 83
```


    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1226 16:54:29.476748 29707 caffe.cpp:234] Use CPU.
    I1226 16:54:29.757699 29707 net.cpp:49] Initializing net from parameters:
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label"
      hdf5_data_param {
        source: "cifar_10_caffe_hdf5/test.txt"
        batch_size: 120
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 12
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "conv1"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu1"
      type: "ReLU"
      bottom: "pool1"
      top: "pool1"
    }
    layer {
      name: "conv2"
      type: "Convolution"
      bottom: "pool1"
      top: "conv2"
      convolution_param {
        num_output: 32
        kernel_size: 3
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool2"
      type: "Pooling"
      bottom: "conv2"
      top: "pool2"
      pooling_param {
        pool: MAX
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu2"
      type: "ReLU"
      bottom: "pool2"
      top: "pool2"
    }
    layer {
      name: "conv3"
      type: "Convolution"
      bottom: "pool2"
      top: "conv3"
      convolution_param {
        num_output: 64
        kernel_size: 5
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "conv3"
      top: "conv3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "conv3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu4"
      type: "ReLU"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip2"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip2"
      inner_product_param {
        num_output: 10
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy"
      type: "Accuracy"
      bottom: "ip2"
      bottom: "label"
      top: "accuracy"
    }
    layer {
      name: "loss"
      type: "SoftmaxWithLoss"
      bottom: "ip2"
      bottom: "label"
      top: "loss"
    }
    I1226 16:54:29.758110 29707 layer_factory.hpp:77] Creating layer data
    I1226 16:54:29.758126 29707 net.cpp:106] Creating Layer data
    I1226 16:54:29.758134 29707 net.cpp:411] data -> data
    I1226 16:54:29.758153 29707 net.cpp:411] data -> label
    I1226 16:54:29.758318 29707 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1226 16:54:29.759383 29707 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1226 16:54:29.761139 29707 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1226 16:54:36.311343 29707 net.cpp:150] Setting up data
    I1226 16:54:36.311393 29707 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1226 16:54:36.311401 29707 net.cpp:157] Top shape: 120 (120)
    I1226 16:54:36.311408 29707 net.cpp:165] Memory required for data: 1475040
    I1226 16:54:36.311417 29707 layer_factory.hpp:77] Creating layer label_data_1_split
    I1226 16:54:36.311990 29707 net.cpp:106] Creating Layer label_data_1_split
    I1226 16:54:36.312011 29707 net.cpp:454] label_data_1_split <- label
    I1226 16:54:36.312023 29707 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1226 16:54:36.312034 29707 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1226 16:54:36.312047 29707 net.cpp:150] Setting up label_data_1_split
    I1226 16:54:36.312063 29707 net.cpp:157] Top shape: 120 (120)
    I1226 16:54:36.312069 29707 net.cpp:157] Top shape: 120 (120)
    I1226 16:54:36.312073 29707 net.cpp:165] Memory required for data: 1476000
    I1226 16:54:36.312079 29707 layer_factory.hpp:77] Creating layer conv1
    I1226 16:54:36.312089 29707 net.cpp:106] Creating Layer conv1
    I1226 16:54:36.312094 29707 net.cpp:454] conv1 <- data
    I1226 16:54:36.312101 29707 net.cpp:411] conv1 -> conv1
    I1226 16:54:36.312518 29707 net.cpp:150] Setting up conv1
    I1226 16:54:36.312541 29707 net.cpp:157] Top shape: 120 12 30 30 (1296000)
    I1226 16:54:36.312546 29707 net.cpp:165] Memory required for data: 6660000
    I1226 16:54:36.312561 29707 layer_factory.hpp:77] Creating layer pool1
    I1226 16:54:36.312569 29707 net.cpp:106] Creating Layer pool1
    I1226 16:54:36.312575 29707 net.cpp:454] pool1 <- conv1
    I1226 16:54:36.312592 29707 net.cpp:411] pool1 -> pool1
    I1226 16:54:36.313184 29707 net.cpp:150] Setting up pool1
    I1226 16:54:36.313226 29707 net.cpp:157] Top shape: 120 12 15 15 (324000)
    I1226 16:54:36.313243 29707 net.cpp:165] Memory required for data: 7956000
    I1226 16:54:36.313249 29707 layer_factory.hpp:77] Creating layer relu1
    I1226 16:54:36.313259 29707 net.cpp:106] Creating Layer relu1
    I1226 16:54:36.313266 29707 net.cpp:454] relu1 <- pool1
    I1226 16:54:36.313274 29707 net.cpp:397] relu1 -> pool1 (in-place)
    I1226 16:54:36.313295 29707 net.cpp:150] Setting up relu1
    I1226 16:54:36.313311 29707 net.cpp:157] Top shape: 120 12 15 15 (324000)
    I1226 16:54:36.313316 29707 net.cpp:165] Memory required for data: 9252000
    I1226 16:54:36.313320 29707 layer_factory.hpp:77] Creating layer conv2
    I1226 16:54:36.313328 29707 net.cpp:106] Creating Layer conv2
    I1226 16:54:36.313333 29707 net.cpp:454] conv2 <- pool1
    I1226 16:54:36.313339 29707 net.cpp:411] conv2 -> conv2
    I1226 16:54:36.313376 29707 net.cpp:150] Setting up conv2
    I1226 16:54:36.313382 29707 net.cpp:157] Top shape: 120 32 13 13 (648960)
    I1226 16:54:36.313387 29707 net.cpp:165] Memory required for data: 11847840
    I1226 16:54:36.313395 29707 layer_factory.hpp:77] Creating layer pool2
    I1226 16:54:36.313402 29707 net.cpp:106] Creating Layer pool2
    I1226 16:54:36.313407 29707 net.cpp:454] pool2 <- conv2
    I1226 16:54:36.313413 29707 net.cpp:411] pool2 -> pool2
    I1226 16:54:36.313431 29707 net.cpp:150] Setting up pool2
    I1226 16:54:36.313438 29707 net.cpp:157] Top shape: 120 32 7 7 (188160)
    I1226 16:54:36.313442 29707 net.cpp:165] Memory required for data: 12600480
    I1226 16:54:36.313447 29707 layer_factory.hpp:77] Creating layer relu2
    I1226 16:54:36.313454 29707 net.cpp:106] Creating Layer relu2
    I1226 16:54:36.313459 29707 net.cpp:454] relu2 <- pool2
    I1226 16:54:36.313475 29707 net.cpp:397] relu2 -> pool2 (in-place)
    I1226 16:54:36.313482 29707 net.cpp:150] Setting up relu2
    I1226 16:54:36.313488 29707 net.cpp:157] Top shape: 120 32 7 7 (188160)
    I1226 16:54:36.313493 29707 net.cpp:165] Memory required for data: 13353120
    I1226 16:54:36.313496 29707 layer_factory.hpp:77] Creating layer conv3
    I1226 16:54:36.313504 29707 net.cpp:106] Creating Layer conv3
    I1226 16:54:36.313508 29707 net.cpp:454] conv3 <- pool2
    I1226 16:54:36.313515 29707 net.cpp:411] conv3 -> conv3
    I1226 16:54:36.313858 29707 net.cpp:150] Setting up conv3
    I1226 16:54:36.313877 29707 net.cpp:157] Top shape: 120 64 3 3 (69120)
    I1226 16:54:36.313882 29707 net.cpp:165] Memory required for data: 13629600
    I1226 16:54:36.313890 29707 layer_factory.hpp:77] Creating layer relu3
    I1226 16:54:36.313897 29707 net.cpp:106] Creating Layer relu3
    I1226 16:54:36.313912 29707 net.cpp:454] relu3 <- conv3
    I1226 16:54:36.313918 29707 net.cpp:397] relu3 -> conv3 (in-place)
    I1226 16:54:36.313925 29707 net.cpp:150] Setting up relu3
    I1226 16:54:36.313930 29707 net.cpp:157] Top shape: 120 64 3 3 (69120)
    I1226 16:54:36.313935 29707 net.cpp:165] Memory required for data: 13906080
    I1226 16:54:36.313940 29707 layer_factory.hpp:77] Creating layer ip1
    I1226 16:54:36.313951 29707 net.cpp:106] Creating Layer ip1
    I1226 16:54:36.313956 29707 net.cpp:454] ip1 <- conv3
    I1226 16:54:36.313961 29707 net.cpp:411] ip1 -> ip1
    I1226 16:54:36.315733 29707 net.cpp:150] Setting up ip1
    I1226 16:54:36.315742 29707 net.cpp:157] Top shape: 120 512 (61440)
    I1226 16:54:36.315757 29707 net.cpp:165] Memory required for data: 14151840
    I1226 16:54:36.315765 29707 layer_factory.hpp:77] Creating layer relu4
    I1226 16:54:36.315773 29707 net.cpp:106] Creating Layer relu4
    I1226 16:54:36.315788 29707 net.cpp:454] relu4 <- ip1
    I1226 16:54:36.315793 29707 net.cpp:397] relu4 -> ip1 (in-place)
    I1226 16:54:36.315799 29707 net.cpp:150] Setting up relu4
    I1226 16:54:36.315805 29707 net.cpp:157] Top shape: 120 512 (61440)
    I1226 16:54:36.315809 29707 net.cpp:165] Memory required for data: 14397600
    I1226 16:54:36.315814 29707 layer_factory.hpp:77] Creating layer ip2
    I1226 16:54:36.315820 29707 net.cpp:106] Creating Layer ip2
    I1226 16:54:36.315825 29707 net.cpp:454] ip2 <- ip1
    I1226 16:54:36.315831 29707 net.cpp:411] ip2 -> ip2
    I1226 16:54:36.315871 29707 net.cpp:150] Setting up ip2
    I1226 16:54:36.315886 29707 net.cpp:157] Top shape: 120 10 (1200)
    I1226 16:54:36.315891 29707 net.cpp:165] Memory required for data: 14402400
    I1226 16:54:36.315899 29707 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1226 16:54:36.315907 29707 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1226 16:54:36.315912 29707 net.cpp:454] ip2_ip2_0_split <- ip2
    I1226 16:54:36.315927 29707 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1226 16:54:36.315934 29707 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1226 16:54:36.315943 29707 net.cpp:150] Setting up ip2_ip2_0_split
    I1226 16:54:36.315949 29707 net.cpp:157] Top shape: 120 10 (1200)
    I1226 16:54:36.315965 29707 net.cpp:157] Top shape: 120 10 (1200)
    I1226 16:54:36.315970 29707 net.cpp:165] Memory required for data: 14412000
    I1226 16:54:36.315974 29707 layer_factory.hpp:77] Creating layer accuracy
    I1226 16:54:36.315981 29707 net.cpp:106] Creating Layer accuracy
    I1226 16:54:36.315986 29707 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1226 16:54:36.315992 29707 net.cpp:454] accuracy <- label_data_1_split_0
    I1226 16:54:36.315997 29707 net.cpp:411] accuracy -> accuracy
    I1226 16:54:36.316009 29707 net.cpp:150] Setting up accuracy
    I1226 16:54:36.316015 29707 net.cpp:157] Top shape: (1)
    I1226 16:54:36.316018 29707 net.cpp:165] Memory required for data: 14412004
    I1226 16:54:36.316023 29707 layer_factory.hpp:77] Creating layer loss
    I1226 16:54:36.316030 29707 net.cpp:106] Creating Layer loss
    I1226 16:54:36.316035 29707 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1226 16:54:36.316040 29707 net.cpp:454] loss <- label_data_1_split_1
    I1226 16:54:36.316046 29707 net.cpp:411] loss -> loss
    I1226 16:54:36.316443 29707 layer_factory.hpp:77] Creating layer loss
    I1226 16:54:36.316488 29707 net.cpp:150] Setting up loss
    I1226 16:54:36.316506 29707 net.cpp:157] Top shape: (1)
    I1226 16:54:36.316514 29707 net.cpp:160]     with loss weight 1
    I1226 16:54:36.316570 29707 net.cpp:165] Memory required for data: 14412008
    I1226 16:54:36.316587 29707 net.cpp:226] loss needs backward computation.
    I1226 16:54:36.316594 29707 net.cpp:228] accuracy does not need backward computation.
    I1226 16:54:36.316612 29707 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1226 16:54:36.316617 29707 net.cpp:226] ip2 needs backward computation.
    I1226 16:54:36.316632 29707 net.cpp:226] relu4 needs backward computation.
    I1226 16:54:36.316635 29707 net.cpp:226] ip1 needs backward computation.
    I1226 16:54:36.316640 29707 net.cpp:226] relu3 needs backward computation.
    I1226 16:54:36.316645 29707 net.cpp:226] conv3 needs backward computation.
    I1226 16:54:36.316650 29707 net.cpp:226] relu2 needs backward computation.
    I1226 16:54:36.316655 29707 net.cpp:226] pool2 needs backward computation.
    I1226 16:54:36.316659 29707 net.cpp:226] conv2 needs backward computation.
    I1226 16:54:36.316675 29707 net.cpp:226] relu1 needs backward computation.
    I1226 16:54:36.316680 29707 net.cpp:226] pool1 needs backward computation.
    I1226 16:54:36.316685 29707 net.cpp:226] conv1 needs backward computation.
    I1226 16:54:36.316691 29707 net.cpp:228] label_data_1_split does not need backward computation.
    I1226 16:54:36.316697 29707 net.cpp:228] data does not need backward computation.
    I1226 16:54:36.316702 29707 net.cpp:270] This network produces output accuracy
    I1226 16:54:36.316709 29707 net.cpp:270] This network produces output loss
    I1226 16:54:36.316731 29707 net.cpp:283] Network initialization done.
    I1226 16:54:36.319108 29707 caffe.cpp:240] Running for 83 iterations.
    I1226 16:54:37.158118 29707 caffe.cpp:264] Batch 0, accuracy = 0.633333
    I1226 16:54:37.158164 29707 caffe.cpp:264] Batch 0, loss = 2.84635
    I1226 16:54:37.488252 29707 caffe.cpp:264] Batch 1, accuracy = 0.558333
    I1226 16:54:37.488301 29707 caffe.cpp:264] Batch 1, loss = 3.10272
    I1226 16:54:37.815809 29707 caffe.cpp:264] Batch 2, accuracy = 0.633333
    I1226 16:54:37.815852 29707 caffe.cpp:264] Batch 2, loss = 3.12302
    I1226 16:54:38.134845 29707 caffe.cpp:264] Batch 3, accuracy = 0.6
    I1226 16:54:38.134889 29707 caffe.cpp:264] Batch 3, loss = 3.20926
    I1226 16:54:38.454558 29707 caffe.cpp:264] Batch 4, accuracy = 0.683333
    I1226 16:54:38.454637 29707 caffe.cpp:264] Batch 4, loss = 2.89272
    I1226 16:54:38.792207 29707 caffe.cpp:264] Batch 5, accuracy = 0.558333
    I1226 16:54:38.792253 29707 caffe.cpp:264] Batch 5, loss = 3.12153
    I1226 16:54:39.121701 29707 caffe.cpp:264] Batch 6, accuracy = 0.575
    I1226 16:54:39.121743 29707 caffe.cpp:264] Batch 6, loss = 2.76126
    I1226 16:54:39.449329 29707 caffe.cpp:264] Batch 7, accuracy = 0.683333
    I1226 16:54:39.449374 29707 caffe.cpp:264] Batch 7, loss = 1.84268
    I1226 16:54:39.776589 29707 caffe.cpp:264] Batch 8, accuracy = 0.558333
    I1226 16:54:39.776630 29707 caffe.cpp:264] Batch 8, loss = 2.89075
    I1226 16:54:40.100194 29707 caffe.cpp:264] Batch 9, accuracy = 0.666667
    I1226 16:54:40.100239 29707 caffe.cpp:264] Batch 9, loss = 2.87031
    I1226 16:54:40.431360 29707 caffe.cpp:264] Batch 10, accuracy = 0.633333
    I1226 16:54:40.431402 29707 caffe.cpp:264] Batch 10, loss = 2.95756
    I1226 16:54:40.763556 29707 caffe.cpp:264] Batch 11, accuracy = 0.716667
    I1226 16:54:40.763597 29707 caffe.cpp:264] Batch 11, loss = 2.2198
    I1226 16:54:41.091974 29707 caffe.cpp:264] Batch 12, accuracy = 0.683333
    I1226 16:54:41.092025 29707 caffe.cpp:264] Batch 12, loss = 2.88751
    I1226 16:54:41.412300 29707 caffe.cpp:264] Batch 13, accuracy = 0.666667
    I1226 16:54:41.412343 29707 caffe.cpp:264] Batch 13, loss = 2.21505
    I1226 16:54:41.745707 29707 caffe.cpp:264] Batch 14, accuracy = 0.658333
    I1226 16:54:41.745751 29707 caffe.cpp:264] Batch 14, loss = 2.48203
    I1226 16:54:42.074924 29707 caffe.cpp:264] Batch 15, accuracy = 0.591667
    I1226 16:54:42.074970 29707 caffe.cpp:264] Batch 15, loss = 2.81148
    I1226 16:54:42.401322 29707 caffe.cpp:264] Batch 16, accuracy = 0.625
    I1226 16:54:42.401367 29707 caffe.cpp:264] Batch 16, loss = 2.54792
    I1226 16:54:42.729477 29707 caffe.cpp:264] Batch 17, accuracy = 0.65
    I1226 16:54:42.729516 29707 caffe.cpp:264] Batch 17, loss = 2.65833
    I1226 16:54:43.048812 29707 caffe.cpp:264] Batch 18, accuracy = 0.625
    I1226 16:54:43.048856 29707 caffe.cpp:264] Batch 18, loss = 2.74056
    I1226 16:54:43.375797 29707 caffe.cpp:264] Batch 19, accuracy = 0.65
    I1226 16:54:43.375843 29707 caffe.cpp:264] Batch 19, loss = 2.16904
    I1226 16:54:43.712318 29707 caffe.cpp:264] Batch 20, accuracy = 0.6
    I1226 16:54:43.712359 29707 caffe.cpp:264] Batch 20, loss = 2.74504
    I1226 16:54:44.027467 29707 caffe.cpp:264] Batch 21, accuracy = 0.666667
    I1226 16:54:44.027508 29707 caffe.cpp:264] Batch 21, loss = 2.02957
    I1226 16:54:44.355728 29707 caffe.cpp:264] Batch 22, accuracy = 0.641667
    I1226 16:54:44.355770 29707 caffe.cpp:264] Batch 22, loss = 3.17693
    I1226 16:54:44.684545 29707 caffe.cpp:264] Batch 23, accuracy = 0.65
    I1226 16:54:44.684584 29707 caffe.cpp:264] Batch 23, loss = 2.58016
    I1226 16:54:45.011737 29707 caffe.cpp:264] Batch 24, accuracy = 0.708333
    I1226 16:54:45.011782 29707 caffe.cpp:264] Batch 24, loss = 2.0133
    I1226 16:54:45.332301 29707 caffe.cpp:264] Batch 25, accuracy = 0.608333
    I1226 16:54:45.332342 29707 caffe.cpp:264] Batch 25, loss = 2.19429
    I1226 16:54:45.668545 29707 caffe.cpp:264] Batch 26, accuracy = 0.666667
    I1226 16:54:45.668586 29707 caffe.cpp:264] Batch 26, loss = 2.42378
    I1226 16:54:45.991729 29707 caffe.cpp:264] Batch 27, accuracy = 0.641667
    I1226 16:54:45.991771 29707 caffe.cpp:264] Batch 27, loss = 2.91304
    I1226 16:54:46.313104 29707 caffe.cpp:264] Batch 28, accuracy = 0.633333
    I1226 16:54:46.313144 29707 caffe.cpp:264] Batch 28, loss = 2.55022
    I1226 16:54:46.654511 29707 caffe.cpp:264] Batch 29, accuracy = 0.6
    I1226 16:54:46.654553 29707 caffe.cpp:264] Batch 29, loss = 2.97972
    I1226 16:54:46.970875 29707 caffe.cpp:264] Batch 30, accuracy = 0.533333
    I1226 16:54:46.970918 29707 caffe.cpp:264] Batch 30, loss = 2.88744
    I1226 16:54:47.293874 29707 caffe.cpp:264] Batch 31, accuracy = 0.641667
    I1226 16:54:47.293915 29707 caffe.cpp:264] Batch 31, loss = 3.12828
    I1226 16:54:47.626991 29707 caffe.cpp:264] Batch 32, accuracy = 0.533333
    I1226 16:54:47.627030 29707 caffe.cpp:264] Batch 32, loss = 3.24035
    I1226 16:54:47.943071 29707 caffe.cpp:264] Batch 33, accuracy = 0.65
    I1226 16:54:47.943135 29707 caffe.cpp:264] Batch 33, loss = 2.9852
    I1226 16:54:48.270411 29707 caffe.cpp:264] Batch 34, accuracy = 0.658333
    I1226 16:54:48.270462 29707 caffe.cpp:264] Batch 34, loss = 2.41814
    I1226 16:54:48.606137 29707 caffe.cpp:264] Batch 35, accuracy = 0.675
    I1226 16:54:48.606169 29707 caffe.cpp:264] Batch 35, loss = 2.74079
    I1226 16:54:48.920976 29707 caffe.cpp:264] Batch 36, accuracy = 0.683333
    I1226 16:54:48.921017 29707 caffe.cpp:264] Batch 36, loss = 2.19909
    I1226 16:54:49.242262 29707 caffe.cpp:264] Batch 37, accuracy = 0.683333
    I1226 16:54:49.242301 29707 caffe.cpp:264] Batch 37, loss = 2.17926
    I1226 16:54:49.583950 29707 caffe.cpp:264] Batch 38, accuracy = 0.641667
    I1226 16:54:49.583993 29707 caffe.cpp:264] Batch 38, loss = 2.60697
    I1226 16:54:49.905948 29707 caffe.cpp:264] Batch 39, accuracy = 0.675
    I1226 16:54:49.905990 29707 caffe.cpp:264] Batch 39, loss = 2.35934
    I1226 16:54:50.228498 29707 caffe.cpp:264] Batch 40, accuracy = 0.641667
    I1226 16:54:50.228538 29707 caffe.cpp:264] Batch 40, loss = 2.17921
    I1226 16:54:50.566402 29707 caffe.cpp:264] Batch 41, accuracy = 0.608333
    I1226 16:54:50.566431 29707 caffe.cpp:264] Batch 41, loss = 3.01902
    I1226 16:54:50.885550 29707 caffe.cpp:264] Batch 42, accuracy = 0.625
    I1226 16:54:50.885591 29707 caffe.cpp:264] Batch 42, loss = 2.92101
    I1226 16:54:51.215281 29707 caffe.cpp:264] Batch 43, accuracy = 0.683333
    I1226 16:54:51.215327 29707 caffe.cpp:264] Batch 43, loss = 2.49911
    I1226 16:54:51.552450 29707 caffe.cpp:264] Batch 44, accuracy = 0.558333
    I1226 16:54:51.552500 29707 caffe.cpp:264] Batch 44, loss = 3.47568
    I1226 16:54:51.876461 29707 caffe.cpp:264] Batch 45, accuracy = 0.666667
    I1226 16:54:51.876504 29707 caffe.cpp:264] Batch 45, loss = 2.56784
    I1226 16:54:52.199576 29707 caffe.cpp:264] Batch 46, accuracy = 0.683333
    I1226 16:54:52.199617 29707 caffe.cpp:264] Batch 46, loss = 2.28866
    I1226 16:54:52.540351 29707 caffe.cpp:264] Batch 47, accuracy = 0.716667
    I1226 16:54:52.540398 29707 caffe.cpp:264] Batch 47, loss = 2.08816
    I1226 16:54:52.867516 29707 caffe.cpp:264] Batch 48, accuracy = 0.741667
    I1226 16:54:52.867557 29707 caffe.cpp:264] Batch 48, loss = 1.52492
    I1226 16:54:53.193859 29707 caffe.cpp:264] Batch 49, accuracy = 0.625
    I1226 16:54:53.193900 29707 caffe.cpp:264] Batch 49, loss = 2.42564
    I1226 16:54:53.521078 29707 caffe.cpp:264] Batch 50, accuracy = 0.625
    I1226 16:54:53.521116 29707 caffe.cpp:264] Batch 50, loss = 2.18164
    I1226 16:54:53.860987 29707 caffe.cpp:264] Batch 51, accuracy = 0.591667
    I1226 16:54:53.861032 29707 caffe.cpp:264] Batch 51, loss = 2.92343
    I1226 16:54:54.189173 29707 caffe.cpp:264] Batch 52, accuracy = 0.675
    I1226 16:54:54.189216 29707 caffe.cpp:264] Batch 52, loss = 2.35836
    I1226 16:54:54.523001 29707 caffe.cpp:264] Batch 53, accuracy = 0.616667
    I1226 16:54:54.523056 29707 caffe.cpp:264] Batch 53, loss = 3.5686
    I1226 16:54:54.856695 29707 caffe.cpp:264] Batch 54, accuracy = 0.558333
    I1226 16:54:54.856737 29707 caffe.cpp:264] Batch 54, loss = 3.17011
    I1226 16:54:55.180647 29707 caffe.cpp:264] Batch 55, accuracy = 0.691667
    I1226 16:54:55.180696 29707 caffe.cpp:264] Batch 55, loss = 2.20688
    I1226 16:54:55.513947 29707 caffe.cpp:264] Batch 56, accuracy = 0.608333
    I1226 16:54:55.514004 29707 caffe.cpp:264] Batch 56, loss = 3.27846
    I1226 16:54:55.860539 29707 caffe.cpp:264] Batch 57, accuracy = 0.641667
    I1226 16:54:55.860581 29707 caffe.cpp:264] Batch 57, loss = 2.861
    I1226 16:54:56.182423 29707 caffe.cpp:264] Batch 58, accuracy = 0.633333
    I1226 16:54:56.182466 29707 caffe.cpp:264] Batch 58, loss = 2.81197
    I1226 16:54:56.519351 29707 caffe.cpp:264] Batch 59, accuracy = 0.616667
    I1226 16:54:56.519402 29707 caffe.cpp:264] Batch 59, loss = 2.95132
    I1226 16:54:56.850371 29707 caffe.cpp:264] Batch 60, accuracy = 0.591667
    I1226 16:54:56.850422 29707 caffe.cpp:264] Batch 60, loss = 3.22941
    I1226 16:54:57.173645 29707 caffe.cpp:264] Batch 61, accuracy = 0.708333
    I1226 16:54:57.173679 29707 caffe.cpp:264] Batch 61, loss = 1.90564
    I1226 16:54:57.515416 29707 caffe.cpp:264] Batch 62, accuracy = 0.625
    I1226 16:54:57.515462 29707 caffe.cpp:264] Batch 62, loss = 2.79696
    I1226 16:54:57.855909 29707 caffe.cpp:264] Batch 63, accuracy = 0.675
    I1226 16:54:57.855944 29707 caffe.cpp:264] Batch 63, loss = 2.53986
    I1226 16:54:58.184424 29707 caffe.cpp:264] Batch 64, accuracy = 0.625
    I1226 16:54:58.184470 29707 caffe.cpp:264] Batch 64, loss = 2.60521
    I1226 16:54:58.510110 29707 caffe.cpp:264] Batch 65, accuracy = 0.658333
    I1226 16:54:58.510169 29707 caffe.cpp:264] Batch 65, loss = 2.80084
    I1226 16:54:58.846294 29707 caffe.cpp:264] Batch 66, accuracy = 0.641667
    I1226 16:54:58.846335 29707 caffe.cpp:264] Batch 66, loss = 2.82834
    I1226 16:54:59.176218 29707 caffe.cpp:264] Batch 67, accuracy = 0.575
    I1226 16:54:59.176261 29707 caffe.cpp:264] Batch 67, loss = 2.90428
    I1226 16:54:59.500803 29707 caffe.cpp:264] Batch 68, accuracy = 0.616667
    I1226 16:54:59.500984 29707 caffe.cpp:264] Batch 68, loss = 2.63688
    I1226 16:54:59.839062 29707 caffe.cpp:264] Batch 69, accuracy = 0.608333
    I1226 16:54:59.839105 29707 caffe.cpp:264] Batch 69, loss = 3.00692
    I1226 16:55:00.157397 29707 caffe.cpp:264] Batch 70, accuracy = 0.666667
    I1226 16:55:00.157440 29707 caffe.cpp:264] Batch 70, loss = 1.88866
    I1226 16:55:00.500213 29707 caffe.cpp:264] Batch 71, accuracy = 0.608333
    I1226 16:55:00.500259 29707 caffe.cpp:264] Batch 71, loss = 2.97418
    I1226 16:55:00.836802 29707 caffe.cpp:264] Batch 72, accuracy = 0.608333
    I1226 16:55:00.836843 29707 caffe.cpp:264] Batch 72, loss = 2.51856
    I1226 16:55:01.158614 29707 caffe.cpp:264] Batch 73, accuracy = 0.708333
    I1226 16:55:01.158660 29707 caffe.cpp:264] Batch 73, loss = 2.13115
    I1226 16:55:01.490747 29707 caffe.cpp:264] Batch 74, accuracy = 0.691667
    I1226 16:55:01.490792 29707 caffe.cpp:264] Batch 74, loss = 1.88087
    I1226 16:55:01.841701 29707 caffe.cpp:264] Batch 75, accuracy = 0.666667
    I1226 16:55:01.841744 29707 caffe.cpp:264] Batch 75, loss = 2.74797
    I1226 16:55:02.166350 29707 caffe.cpp:264] Batch 76, accuracy = 0.616667
    I1226 16:55:02.166393 29707 caffe.cpp:264] Batch 76, loss = 2.34307
    I1226 16:55:02.491039 29707 caffe.cpp:264] Batch 77, accuracy = 0.708333
    I1226 16:55:02.491080 29707 caffe.cpp:264] Batch 77, loss = 2.1946
    I1226 16:55:02.838986 29707 caffe.cpp:264] Batch 78, accuracy = 0.608333
    I1226 16:55:02.839030 29707 caffe.cpp:264] Batch 78, loss = 2.72383
    I1226 16:55:03.158402 29707 caffe.cpp:264] Batch 79, accuracy = 0.525
    I1226 16:55:03.158443 29707 caffe.cpp:264] Batch 79, loss = 3.62278
    I1226 16:55:03.490020 29707 caffe.cpp:264] Batch 80, accuracy = 0.675
    I1226 16:55:03.490062 29707 caffe.cpp:264] Batch 80, loss = 2.13124
    I1226 16:55:03.822732 29707 caffe.cpp:264] Batch 81, accuracy = 0.633333
    I1226 16:55:03.822775 29707 caffe.cpp:264] Batch 81, loss = 2.79303
    I1226 16:55:04.139462 29707 caffe.cpp:264] Batch 82, accuracy = 0.558333
    I1226 16:55:04.139503 29707 caffe.cpp:264] Batch 82, loss = 4.19764
    I1226 16:55:04.139510 29707 caffe.cpp:269] Loss: 2.66752
    I1226 16:55:04.139523 29707 caffe.cpp:281] accuracy = 0.636747
    I1226 16:55:04.139534 29707 caffe.cpp:281] loss = 2.66752 (* 1 = 2.66752 loss)
    CPU times: user 156 ms, sys: 32 ms, total: 188 ms
    Wall time: 34.9 s

### 63% accuracy
Coffe brewed. Let's convert the notebook to github markdown as a readme:


```python
!jupyter nbconvert --to markdown custom-cifar-10.ipynb
!mv custom-cifar-10.md README.md
```
