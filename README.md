
# Custom cifar-100 convolutional neural network with Caffe in Python (Pycaffe)

Here, I train a custom convnet on the cifar-100 dataset. I will try to build a new convolutional neural network architecture. It is a bit based on the NIN (Network In Network) architecture detailed in this paper: http://arxiv.org/pdf/1312.4400v3.pdf.

I mainly use some convolution layers, cccp layers, pooling layers, dropout, fully connected layers, relu layers, as well ass sigmoid layers and softmax with loss on top of the neural network.

My code, other than the neural network architecture, is inspired from the official caffe python ".ipynb" examples available at: https://github.com/BVLC/caffe/tree/master/examples.

Please refer to https://www.cs.toronto.edu/~kriz/cifar.html for more information on the nature of the task and of the dataset on which the convolutional neural network is trained on.

## Dynamically download and convert the cifar-100 dataset to Caffe's HDF5 format using code of another git repo of mine.
More info on the dataset can be found at http://www.cs.toronto.edu/~kriz/cifar.html.


```python
%%time

!rm download-and-convert-cifar-100.py
print("Getting the download script...")
!wget https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-100.py
print("Downloaded script. Will execute to download and convert the cifar-100 dataset:")
!python download-and-convert-cifar-100.py
```

    rm: cannot remove ‘download-and-convert-cifar-100.py’: No such file or directory
    Getting the download script...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-30 18:23:26--  https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-100.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 23.235.39.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|23.235.39.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3526 (3.4K) [text/plain]
    Saving to: ‘download-and-convert-cifar-100.py’

    100%[======================================>] 3,526       --.-K/s   in 0s      

    2015-12-30 18:23:26 (1.06 GB/s) - ‘download-and-convert-cifar-100.py’ saved [3526/3526]

    Downloaded script. Will execute to download and convert the cifar-100 dataset:

    Downloading...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-30 18:23:26--  http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
    Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 169001437 (161M) [application/x-gzip]
    Saving to: ‘cifar-100-python.tar.gz’

    100%[======================================>] 169,001,437 1.22MB/s   in 2m 13s

    2015-12-30 18:25:39 (1.21 MB/s) - ‘cifar-100-python.tar.gz’ saved [169001437/169001437]

    Downloading done.

    Extracting...
    cifar-100-python/
    cifar-100-python/file.txt~
    cifar-100-python/train
    cifar-100-python/test
    cifar-100-python/meta
    Extracting successfully done to /home/gui/Documents/python-caffe-custom-cifar-100-conv-net/cifar-100-python.
    Converting...
    INFO: each dataset's element are of shape 3*32*32:
    "print(X.shape)" --> "(50000, 3, 32, 32)"

    From the Caffe documentation:
    The conventional blob dimensions for batches of image data are number N x channel K x height H x width W.

    Data is fully loaded, now truly converting.
    Conversion successfully done to "/home/gui/Documents/python-caffe-custom-cifar-100-conv-net/cifar_100_caffe_hdf5".

    CPU times: user 916 ms, sys: 88 ms, total: 1 s
    Wall time: 2min 23s


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
    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3)

    n.conv1 = L.Convolution(n.data, kernel_size=4, num_output=64, weight_filler=dict(type='xavier'))
    n.cccp1 = L.Convolution(n.conv1, kernel_size=1, num_output=42, weight_filler=dict(type='xavier'))
    n.cccp2 = L.Convolution(n.cccp1, kernel_size=1, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.cccp2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, in_place=True)
    n.relu1 = L.ReLU(n.drop1, in_place=True)

    n.conv2 = L.Convolution(n.relu1, kernel_size=4, num_output=42, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, in_place=True)
    n.relu2 = L.ReLU(n.drop2, in_place=True)

    n.conv3 = L.Convolution(n.relu2, kernel_size=2, num_output=64, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    n.relu3 = L.ReLU(n.pool3, in_place=True)

    n.ip1 = L.InnerProduct(n.relu3, num_output=512, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.ip1, in_place=True)

    n.ip_c = L.InnerProduct(n.sig1, num_output=20, weight_filler=dict(type='xavier'))
    n.accuracy_c = L.Accuracy(n.ip_c, n.label_coarse)
    n.loss_c = L.SoftmaxWithLoss(n.ip_c, n.label_coarse)

    n.ip_f = L.InnerProduct(n.sig1, num_output=100, weight_filler=dict(type='xavier'))
    n.accuracy_f = L.Accuracy(n.ip_f, n.label_fine)
    n.loss_f = L.SoftmaxWithLoss(n.ip_f, n.label_fine)

    return n.to_proto()

with open('cnn_train.prototxt', 'w') as f:
    f.write(str(cnn('cifar_100_caffe_hdf5/train.txt', 100)))

with open('cnn_test.prototxt', 'w') as f:
    f.write(str(cnn('cifar_100_caffe_hdf5/test.txt', 120)))
```

## Load and visualise the untrained network's internal structure and shape
The network's structure (graph) visualisation tool of caffe is broken in the current release. We will simply print here the data shapes.


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
     ('label_coarse', (100,)),
     ('label_fine', (100,)),
     ('label_coarse_data_1_split_0', (100,)),
     ('label_coarse_data_1_split_1', (100,)),
     ('label_fine_data_2_split_0', (100,)),
     ('label_fine_data_2_split_1', (100,)),
     ('conv1', (100, 64, 29, 29)),
     ('cccp1', (100, 42, 29, 29)),
     ('cccp2', (100, 32, 29, 29)),
     ('pool1', (100, 32, 14, 14)),
     ('conv2', (100, 42, 11, 11)),
     ('pool2', (100, 42, 5, 5)),
     ('conv3', (100, 64, 4, 4)),
     ('pool3', (100, 64, 2, 2)),
     ('ip1', (100, 512)),
     ('ip1_sig1_0_split_0', (100, 512)),
     ('ip1_sig1_0_split_1', (100, 512)),
     ('ip_c', (100, 20)),
     ('ip_c_ip_c_0_split_0', (100, 20)),
     ('ip_c_ip_c_0_split_1', (100, 20)),
     ('accuracy_c', ()),
     ('loss_c', ()),
     ('ip_f', (100, 100)),
     ('ip_f_ip_f_0_split_0', (100, 100)),
     ('ip_f_ip_f_0_split_1', (100, 100)),
     ('accuracy_f', ()),
     ('loss_f', ())]


```python
print("Parameters and shape:")
[(k, v[0].data.shape) for k, v in solver.net.params.items()]
```

    Parameters and shape:

    [('conv1', (64, 3, 4, 4)),
     ('cccp1', (42, 64, 1, 1)),
     ('cccp2', (32, 42, 1, 1)),
     ('conv2', (42, 32, 4, 4)),
     ('conv3', (64, 42, 2, 2)),
     ('ip1', (512, 256)),
     ('ip_c', (20, 512)),
     ('ip_f', (100, 512))]


## Solver's params

The solver's params for the created net are defined in a `.prototxt` file.

Notice that because `max_iter: 100000`, the training will loop 2 times on the 50000 training data. Because we train data by minibatches of 100 as defined above when creating the net, there will be a total of `100000*100/50000 = 200` epochs on some of those pre-shuffled 100 images minibatches.

We will test the net on `test_iter: 100` different test images at each `test_interval: 1000` images trained.
____

Here, **RMSProp** is used, it is SDG-based, it converges faster than a pure SGD and it is robust.
____


```python
!cat cnn_solver_rms.prototxt
```

    train_net: "cnn_train.prototxt"
    test_net: "cnn_test.prototxt"

    test_iter: 100
    test_interval: 1000

    base_lr: 0.0007
    momentum: 0.0
    weight_decay: 0.001

    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75

    display: 100

    max_iter: 100000

    snapshot: 50000
    snapshot_prefix: "cnn_snapshot"
    solver_mode: GPU

    type: "RMSProp"
    rms_decay: 0.98


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
    I1230 18:38:58.538300 23363 caffe.cpp:184] Using GPUs 0
    I1230 18:38:58.743958 23363 solver.cpp:48] Initializing solver from parameters:
    train_net: "cnn_train.prototxt"
    test_net: "cnn_test.prototxt"
    test_iter: 100
    test_interval: 1000
    base_lr: 0.0007
    display: 100
    max_iter: 100000
    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75
    momentum: 0
    weight_decay: 0.001
    snapshot: 50000
    snapshot_prefix: "cnn_snapshot"
    solver_mode: GPU
    device_id: 0
    rms_decay: 0.98
    type: "RMSProp"
    I1230 18:38:58.744199 23363 solver.cpp:81] Creating training net from train_net file: cnn_train.prototxt
    I1230 18:38:58.744943 23363 net.cpp:49] Initializing net from parameters:
    state {
      phase: TRAIN
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label_coarse"
      top: "label_fine"
      hdf5_data_param {
        source: "cifar_100_caffe_hdf5/train.txt"
        batch_size: 100
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 4
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp2"
      type: "Convolution"
      bottom: "cccp1"
      top: "cccp2"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp2"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
        num_output: 42
        kernel_size: 4
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
        kernel_size: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool3"
      type: "Pooling"
      bottom: "conv3"
      top: "pool3"
      pooling_param {
        pool: AVE
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "pool3"
      top: "pool3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "sig1"
      type: "Sigmoid"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip_c"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_c"
      inner_product_param {
        num_output: 20
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_c"
      type: "Accuracy"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "accuracy_c"
    }
    layer {
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "loss_c"
    }
    layer {
      name: "ip_f"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_f"
      inner_product_param {
        num_output: 100
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_f"
      type: "Accuracy"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "accuracy_f"
    }
    layer {
      name: "loss_f"
      type: "SoftmaxWithLoss"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "loss_f"
    }
    I1230 18:38:58.746045 23363 layer_factory.hpp:77] Creating layer data
    I1230 18:38:58.746070 23363 net.cpp:106] Creating Layer data
    I1230 18:38:58.746083 23363 net.cpp:411] data -> data
    I1230 18:38:58.746107 23363 net.cpp:411] data -> label_coarse
    I1230 18:38:58.746122 23363 net.cpp:411] data -> label_fine
    I1230 18:38:58.746139 23363 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/train.txt
    I1230 18:38:58.746170 23363 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1230 18:38:58.748234 23363 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1230 18:39:01.146251 23363 net.cpp:150] Setting up data
    I1230 18:39:01.146332 23363 net.cpp:157] Top shape: 100 3 32 32 (307200)
    I1230 18:39:01.146344 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146353 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146361 23363 net.cpp:165] Memory required for data: 1229600
    I1230 18:39:01.146375 23363 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1230 18:39:01.146409 23363 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1230 18:39:01.146420 23363 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1230 18:39:01.146437 23363 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1230 18:39:01.146456 23363 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1230 18:39:01.146517 23363 net.cpp:150] Setting up label_coarse_data_1_split
    I1230 18:39:01.146539 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146556 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146567 23363 net.cpp:165] Memory required for data: 1230400
    I1230 18:39:01.146579 23363 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1230 18:39:01.146597 23363 net.cpp:106] Creating Layer label_fine_data_2_split
    I1230 18:39:01.146611 23363 net.cpp:454] label_fine_data_2_split <- label_fine
    I1230 18:39:01.146630 23363 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1230 18:39:01.146648 23363 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1230 18:39:01.146709 23363 net.cpp:150] Setting up label_fine_data_2_split
    I1230 18:39:01.146730 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146744 23363 net.cpp:157] Top shape: 100 (100)
    I1230 18:39:01.146756 23363 net.cpp:165] Memory required for data: 1231200
    I1230 18:39:01.146769 23363 layer_factory.hpp:77] Creating layer conv1
    I1230 18:39:01.146795 23363 net.cpp:106] Creating Layer conv1
    I1230 18:39:01.146811 23363 net.cpp:454] conv1 <- data
    I1230 18:39:01.146831 23363 net.cpp:411] conv1 -> conv1
    I1230 18:39:01.148977 23363 net.cpp:150] Setting up conv1
    I1230 18:39:01.149040 23363 net.cpp:157] Top shape: 100 64 29 29 (5382400)
    I1230 18:39:01.149056 23363 net.cpp:165] Memory required for data: 22760800
    I1230 18:39:01.149091 23363 layer_factory.hpp:77] Creating layer cccp1
    I1230 18:39:01.149119 23363 net.cpp:106] Creating Layer cccp1
    I1230 18:39:01.149132 23363 net.cpp:454] cccp1 <- conv1
    I1230 18:39:01.149147 23363 net.cpp:411] cccp1 -> cccp1
    I1230 18:39:01.149446 23363 net.cpp:150] Setting up cccp1
    I1230 18:39:01.149463 23363 net.cpp:157] Top shape: 100 42 29 29 (3532200)
    I1230 18:39:01.149471 23363 net.cpp:165] Memory required for data: 36889600
    I1230 18:39:01.149497 23363 layer_factory.hpp:77] Creating layer cccp2
    I1230 18:39:01.149512 23363 net.cpp:106] Creating Layer cccp2
    I1230 18:39:01.149520 23363 net.cpp:454] cccp2 <- cccp1
    I1230 18:39:01.149533 23363 net.cpp:411] cccp2 -> cccp2
    I1230 18:39:01.149796 23363 net.cpp:150] Setting up cccp2
    I1230 18:39:01.149808 23363 net.cpp:157] Top shape: 100 32 29 29 (2691200)
    I1230 18:39:01.149816 23363 net.cpp:165] Memory required for data: 47654400
    I1230 18:39:01.149829 23363 layer_factory.hpp:77] Creating layer pool1
    I1230 18:39:01.149842 23363 net.cpp:106] Creating Layer pool1
    I1230 18:39:01.149849 23363 net.cpp:454] pool1 <- cccp2
    I1230 18:39:01.149859 23363 net.cpp:411] pool1 -> pool1
    I1230 18:39:01.149947 23363 net.cpp:150] Setting up pool1
    I1230 18:39:01.149961 23363 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 18:39:01.149981 23363 net.cpp:165] Memory required for data: 50163200
    I1230 18:39:01.149988 23363 layer_factory.hpp:77] Creating layer drop1
    I1230 18:39:01.150003 23363 net.cpp:106] Creating Layer drop1
    I1230 18:39:01.150012 23363 net.cpp:454] drop1 <- pool1
    I1230 18:39:01.150022 23363 net.cpp:397] drop1 -> pool1 (in-place)
    I1230 18:39:01.150055 23363 net.cpp:150] Setting up drop1
    I1230 18:39:01.150065 23363 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 18:39:01.150073 23363 net.cpp:165] Memory required for data: 52672000
    I1230 18:39:01.150118 23363 layer_factory.hpp:77] Creating layer relu1
    I1230 18:39:01.150133 23363 net.cpp:106] Creating Layer relu1
    I1230 18:39:01.150142 23363 net.cpp:454] relu1 <- pool1
    I1230 18:39:01.150152 23363 net.cpp:397] relu1 -> pool1 (in-place)
    I1230 18:39:01.150163 23363 net.cpp:150] Setting up relu1
    I1230 18:39:01.150172 23363 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 18:39:01.150192 23363 net.cpp:165] Memory required for data: 55180800
    I1230 18:39:01.150198 23363 layer_factory.hpp:77] Creating layer conv2
    I1230 18:39:01.150209 23363 net.cpp:106] Creating Layer conv2
    I1230 18:39:01.150216 23363 net.cpp:454] conv2 <- pool1
    I1230 18:39:01.150226 23363 net.cpp:411] conv2 -> conv2
    I1230 18:39:01.150645 23363 net.cpp:150] Setting up conv2
    I1230 18:39:01.150671 23363 net.cpp:157] Top shape: 100 42 11 11 (508200)
    I1230 18:39:01.150677 23363 net.cpp:165] Memory required for data: 57213600
    I1230 18:39:01.150687 23363 layer_factory.hpp:77] Creating layer pool2
    I1230 18:39:01.150699 23363 net.cpp:106] Creating Layer pool2
    I1230 18:39:01.150707 23363 net.cpp:454] pool2 <- conv2
    I1230 18:39:01.150717 23363 net.cpp:411] pool2 -> pool2
    I1230 18:39:01.150763 23363 net.cpp:150] Setting up pool2
    I1230 18:39:01.150774 23363 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 18:39:01.150779 23363 net.cpp:165] Memory required for data: 57633600
    I1230 18:39:01.150784 23363 layer_factory.hpp:77] Creating layer drop2
    I1230 18:39:01.150794 23363 net.cpp:106] Creating Layer drop2
    I1230 18:39:01.150801 23363 net.cpp:454] drop2 <- pool2
    I1230 18:39:01.150810 23363 net.cpp:397] drop2 -> pool2 (in-place)
    I1230 18:39:01.150832 23363 net.cpp:150] Setting up drop2
    I1230 18:39:01.150841 23363 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 18:39:01.150846 23363 net.cpp:165] Memory required for data: 58053600
    I1230 18:39:01.150852 23363 layer_factory.hpp:77] Creating layer relu2
    I1230 18:39:01.150862 23363 net.cpp:106] Creating Layer relu2
    I1230 18:39:01.150879 23363 net.cpp:454] relu2 <- pool2
    I1230 18:39:01.150887 23363 net.cpp:397] relu2 -> pool2 (in-place)
    I1230 18:39:01.150894 23363 net.cpp:150] Setting up relu2
    I1230 18:39:01.150902 23363 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 18:39:01.150908 23363 net.cpp:165] Memory required for data: 58473600
    I1230 18:39:01.150917 23363 layer_factory.hpp:77] Creating layer conv3
    I1230 18:39:01.150926 23363 net.cpp:106] Creating Layer conv3
    I1230 18:39:01.150933 23363 net.cpp:454] conv3 <- pool2
    I1230 18:39:01.150952 23363 net.cpp:411] conv3 -> conv3
    I1230 18:39:01.151917 23363 net.cpp:150] Setting up conv3
    I1230 18:39:01.151955 23363 net.cpp:157] Top shape: 100 64 4 4 (102400)
    I1230 18:39:01.151963 23363 net.cpp:165] Memory required for data: 58883200
    I1230 18:39:01.151983 23363 layer_factory.hpp:77] Creating layer pool3
    I1230 18:39:01.151998 23363 net.cpp:106] Creating Layer pool3
    I1230 18:39:01.152016 23363 net.cpp:454] pool3 <- conv3
    I1230 18:39:01.152027 23363 net.cpp:411] pool3 -> pool3
    I1230 18:39:01.152060 23363 net.cpp:150] Setting up pool3
    I1230 18:39:01.152073 23363 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1230 18:39:01.152083 23363 net.cpp:165] Memory required for data: 58985600
    I1230 18:39:01.152092 23363 layer_factory.hpp:77] Creating layer relu3
    I1230 18:39:01.152104 23363 net.cpp:106] Creating Layer relu3
    I1230 18:39:01.152112 23363 net.cpp:454] relu3 <- pool3
    I1230 18:39:01.152135 23363 net.cpp:397] relu3 -> pool3 (in-place)
    I1230 18:39:01.152163 23363 net.cpp:150] Setting up relu3
    I1230 18:39:01.152179 23363 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1230 18:39:01.152187 23363 net.cpp:165] Memory required for data: 59088000
    I1230 18:39:01.152194 23363 layer_factory.hpp:77] Creating layer ip1
    I1230 18:39:01.152230 23363 net.cpp:106] Creating Layer ip1
    I1230 18:39:01.152238 23363 net.cpp:454] ip1 <- pool3
    I1230 18:39:01.152248 23363 net.cpp:411] ip1 -> ip1
    I1230 18:39:01.153889 23363 net.cpp:150] Setting up ip1
    I1230 18:39:01.153921 23363 net.cpp:157] Top shape: 100 512 (51200)
    I1230 18:39:01.153929 23363 net.cpp:165] Memory required for data: 59292800
    I1230 18:39:01.153962 23363 layer_factory.hpp:77] Creating layer sig1
    I1230 18:39:01.153985 23363 net.cpp:106] Creating Layer sig1
    I1230 18:39:01.153992 23363 net.cpp:454] sig1 <- ip1
    I1230 18:39:01.154000 23363 net.cpp:397] sig1 -> ip1 (in-place)
    I1230 18:39:01.154011 23363 net.cpp:150] Setting up sig1
    I1230 18:39:01.154018 23363 net.cpp:157] Top shape: 100 512 (51200)
    I1230 18:39:01.154023 23363 net.cpp:165] Memory required for data: 59497600
    I1230 18:39:01.154029 23363 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1230 18:39:01.154038 23363 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1230 18:39:01.154044 23363 net.cpp:454] ip1_sig1_0_split <- ip1
    I1230 18:39:01.154067 23363 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1230 18:39:01.154079 23363 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1230 18:39:01.154158 23363 net.cpp:150] Setting up ip1_sig1_0_split
    I1230 18:39:01.154170 23363 net.cpp:157] Top shape: 100 512 (51200)
    I1230 18:39:01.154178 23363 net.cpp:157] Top shape: 100 512 (51200)
    I1230 18:39:01.154186 23363 net.cpp:165] Memory required for data: 59907200
    I1230 18:39:01.154191 23363 layer_factory.hpp:77] Creating layer ip_c
    I1230 18:39:01.154213 23363 net.cpp:106] Creating Layer ip_c
    I1230 18:39:01.154220 23363 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1230 18:39:01.154228 23363 net.cpp:411] ip_c -> ip_c
    I1230 18:39:01.154465 23363 net.cpp:150] Setting up ip_c
    I1230 18:39:01.154475 23363 net.cpp:157] Top shape: 100 20 (2000)
    I1230 18:39:01.154481 23363 net.cpp:165] Memory required for data: 59915200
    I1230 18:39:01.154490 23363 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1230 18:39:01.154500 23363 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1230 18:39:01.154506 23363 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1230 18:39:01.154515 23363 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1230 18:39:01.154523 23363 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1230 18:39:01.154577 23363 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1230 18:39:01.154594 23363 net.cpp:157] Top shape: 100 20 (2000)
    I1230 18:39:01.154618 23363 net.cpp:157] Top shape: 100 20 (2000)
    I1230 18:39:01.154644 23363 net.cpp:165] Memory required for data: 59931200
    I1230 18:39:01.154655 23363 layer_factory.hpp:77] Creating layer accuracy_c
    I1230 18:39:01.154665 23363 net.cpp:106] Creating Layer accuracy_c
    I1230 18:39:01.154674 23363 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1230 18:39:01.154682 23363 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1230 18:39:01.154708 23363 net.cpp:411] accuracy_c -> accuracy_c
    I1230 18:39:01.154729 23363 net.cpp:150] Setting up accuracy_c
    I1230 18:39:01.154757 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.154767 23363 net.cpp:165] Memory required for data: 59931204
    I1230 18:39:01.154779 23363 layer_factory.hpp:77] Creating layer loss_c
    I1230 18:39:01.154793 23363 net.cpp:106] Creating Layer loss_c
    I1230 18:39:01.154805 23363 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1230 18:39:01.154829 23363 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1230 18:39:01.154841 23363 net.cpp:411] loss_c -> loss_c
    I1230 18:39:01.154876 23363 layer_factory.hpp:77] Creating layer loss_c
    I1230 18:39:01.155086 23363 net.cpp:150] Setting up loss_c
    I1230 18:39:01.155104 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.155112 23363 net.cpp:160]     with loss weight 1
    I1230 18:39:01.155139 23363 net.cpp:165] Memory required for data: 59931208
    I1230 18:39:01.155148 23363 layer_factory.hpp:77] Creating layer ip_f
    I1230 18:39:01.155158 23363 net.cpp:106] Creating Layer ip_f
    I1230 18:39:01.155179 23363 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1230 18:39:01.155190 23363 net.cpp:411] ip_f -> ip_f
    I1230 18:39:01.155798 23363 net.cpp:150] Setting up ip_f
    I1230 18:39:01.155825 23363 net.cpp:157] Top shape: 100 100 (10000)
    I1230 18:39:01.155833 23363 net.cpp:165] Memory required for data: 59971208
    I1230 18:39:01.155846 23363 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1230 18:39:01.155858 23363 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1230 18:39:01.155894 23363 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1230 18:39:01.155905 23363 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1230 18:39:01.155916 23363 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1230 18:39:01.155953 23363 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1230 18:39:01.155963 23363 net.cpp:157] Top shape: 100 100 (10000)
    I1230 18:39:01.155972 23363 net.cpp:157] Top shape: 100 100 (10000)
    I1230 18:39:01.155978 23363 net.cpp:165] Memory required for data: 60051208
    I1230 18:39:01.155985 23363 layer_factory.hpp:77] Creating layer accuracy_f
    I1230 18:39:01.155994 23363 net.cpp:106] Creating Layer accuracy_f
    I1230 18:39:01.156015 23363 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1230 18:39:01.156024 23363 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1230 18:39:01.156034 23363 net.cpp:411] accuracy_f -> accuracy_f
    I1230 18:39:01.156047 23363 net.cpp:150] Setting up accuracy_f
    I1230 18:39:01.156070 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.156076 23363 net.cpp:165] Memory required for data: 60051212
    I1230 18:39:01.156083 23363 layer_factory.hpp:77] Creating layer loss_f
    I1230 18:39:01.156093 23363 net.cpp:106] Creating Layer loss_f
    I1230 18:39:01.156100 23363 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1230 18:39:01.156108 23363 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1230 18:39:01.156117 23363 net.cpp:411] loss_f -> loss_f
    I1230 18:39:01.156208 23363 layer_factory.hpp:77] Creating layer loss_f
    I1230 18:39:01.156352 23363 net.cpp:150] Setting up loss_f
    I1230 18:39:01.156365 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.156373 23363 net.cpp:160]     with loss weight 1
    I1230 18:39:01.156399 23363 net.cpp:165] Memory required for data: 60051216
    I1230 18:39:01.156405 23363 net.cpp:226] loss_f needs backward computation.
    I1230 18:39:01.156414 23363 net.cpp:228] accuracy_f does not need backward computation.
    I1230 18:39:01.156421 23363 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1230 18:39:01.156430 23363 net.cpp:226] ip_f needs backward computation.
    I1230 18:39:01.156436 23363 net.cpp:226] loss_c needs backward computation.
    I1230 18:39:01.156445 23363 net.cpp:228] accuracy_c does not need backward computation.
    I1230 18:39:01.156453 23363 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1230 18:39:01.156461 23363 net.cpp:226] ip_c needs backward computation.
    I1230 18:39:01.156468 23363 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1230 18:39:01.156492 23363 net.cpp:226] sig1 needs backward computation.
    I1230 18:39:01.156512 23363 net.cpp:226] ip1 needs backward computation.
    I1230 18:39:01.156520 23363 net.cpp:226] relu3 needs backward computation.
    I1230 18:39:01.156527 23363 net.cpp:226] pool3 needs backward computation.
    I1230 18:39:01.156546 23363 net.cpp:226] conv3 needs backward computation.
    I1230 18:39:01.156563 23363 net.cpp:226] relu2 needs backward computation.
    I1230 18:39:01.156571 23363 net.cpp:226] drop2 needs backward computation.
    I1230 18:39:01.156579 23363 net.cpp:226] pool2 needs backward computation.
    I1230 18:39:01.156597 23363 net.cpp:226] conv2 needs backward computation.
    I1230 18:39:01.156605 23363 net.cpp:226] relu1 needs backward computation.
    I1230 18:39:01.156610 23363 net.cpp:226] drop1 needs backward computation.
    I1230 18:39:01.156617 23363 net.cpp:226] pool1 needs backward computation.
    I1230 18:39:01.156625 23363 net.cpp:226] cccp2 needs backward computation.
    I1230 18:39:01.156641 23363 net.cpp:226] cccp1 needs backward computation.
    I1230 18:39:01.156647 23363 net.cpp:226] conv1 needs backward computation.
    I1230 18:39:01.156654 23363 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1230 18:39:01.156661 23363 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1230 18:39:01.156667 23363 net.cpp:228] data does not need backward computation.
    I1230 18:39:01.156673 23363 net.cpp:270] This network produces output accuracy_c
    I1230 18:39:01.156679 23363 net.cpp:270] This network produces output accuracy_f
    I1230 18:39:01.156685 23363 net.cpp:270] This network produces output loss_c
    I1230 18:39:01.156699 23363 net.cpp:270] This network produces output loss_f
    I1230 18:39:01.156724 23363 net.cpp:283] Network initialization done.
    I1230 18:39:01.157115 23363 solver.cpp:181] Creating test net (#0) specified by test_net file: cnn_test.prototxt
    I1230 18:39:01.157284 23363 net.cpp:49] Initializing net from parameters:
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label_coarse"
      top: "label_fine"
      hdf5_data_param {
        source: "cifar_100_caffe_hdf5/test.txt"
        batch_size: 120
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 4
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp2"
      type: "Convolution"
      bottom: "cccp1"
      top: "cccp2"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp2"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
        num_output: 42
        kernel_size: 4
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
        kernel_size: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool3"
      type: "Pooling"
      bottom: "conv3"
      top: "pool3"
      pooling_param {
        pool: AVE
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "pool3"
      top: "pool3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "sig1"
      type: "Sigmoid"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip_c"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_c"
      inner_product_param {
        num_output: 20
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_c"
      type: "Accuracy"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "accuracy_c"
    }
    layer {
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "loss_c"
    }
    layer {
      name: "ip_f"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_f"
      inner_product_param {
        num_output: 100
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_f"
      type: "Accuracy"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "accuracy_f"
    }
    layer {
      name: "loss_f"
      type: "SoftmaxWithLoss"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "loss_f"
    }
    I1230 18:39:01.158046 23363 layer_factory.hpp:77] Creating layer data
    I1230 18:39:01.158061 23363 net.cpp:106] Creating Layer data
    I1230 18:39:01.158069 23363 net.cpp:411] data -> data
    I1230 18:39:01.158080 23363 net.cpp:411] data -> label_coarse
    I1230 18:39:01.158090 23363 net.cpp:411] data -> label_fine
    I1230 18:39:01.158099 23363 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/test.txt
    I1230 18:39:01.158120 23363 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1230 18:39:01.559880 23363 net.cpp:150] Setting up data
    I1230 18:39:01.559934 23363 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1230 18:39:01.559944 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.559952 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.559985 23363 net.cpp:165] Memory required for data: 1475520
    I1230 18:39:01.559994 23363 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1230 18:39:01.560011 23363 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1230 18:39:01.560020 23363 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1230 18:39:01.560031 23363 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1230 18:39:01.560045 23363 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1230 18:39:01.560086 23363 net.cpp:150] Setting up label_coarse_data_1_split
    I1230 18:39:01.560096 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.560103 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.560109 23363 net.cpp:165] Memory required for data: 1476480
    I1230 18:39:01.560117 23363 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1230 18:39:01.560125 23363 net.cpp:106] Creating Layer label_fine_data_2_split
    I1230 18:39:01.560132 23363 net.cpp:454] label_fine_data_2_split <- label_fine
    I1230 18:39:01.560140 23363 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1230 18:39:01.560150 23363 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1230 18:39:01.560225 23363 net.cpp:150] Setting up label_fine_data_2_split
    I1230 18:39:01.560236 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.560245 23363 net.cpp:157] Top shape: 120 (120)
    I1230 18:39:01.560250 23363 net.cpp:165] Memory required for data: 1477440
    I1230 18:39:01.560257 23363 layer_factory.hpp:77] Creating layer conv1
    I1230 18:39:01.560272 23363 net.cpp:106] Creating Layer conv1
    I1230 18:39:01.560279 23363 net.cpp:454] conv1 <- data
    I1230 18:39:01.560289 23363 net.cpp:411] conv1 -> conv1
    I1230 18:39:01.560556 23363 net.cpp:150] Setting up conv1
    I1230 18:39:01.560570 23363 net.cpp:157] Top shape: 120 64 29 29 (6458880)
    I1230 18:39:01.560576 23363 net.cpp:165] Memory required for data: 27312960
    I1230 18:39:01.560592 23363 layer_factory.hpp:77] Creating layer cccp1
    I1230 18:39:01.560606 23363 net.cpp:106] Creating Layer cccp1
    I1230 18:39:01.560613 23363 net.cpp:454] cccp1 <- conv1
    I1230 18:39:01.560623 23363 net.cpp:411] cccp1 -> cccp1
    I1230 18:39:01.560852 23363 net.cpp:150] Setting up cccp1
    I1230 18:39:01.560864 23363 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1230 18:39:01.560871 23363 net.cpp:165] Memory required for data: 44267520
    I1230 18:39:01.560883 23363 layer_factory.hpp:77] Creating layer cccp2
    I1230 18:39:01.560892 23363 net.cpp:106] Creating Layer cccp2
    I1230 18:39:01.560899 23363 net.cpp:454] cccp2 <- cccp1
    I1230 18:39:01.560907 23363 net.cpp:411] cccp2 -> cccp2
    I1230 18:39:01.561321 23363 net.cpp:150] Setting up cccp2
    I1230 18:39:01.561352 23363 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1230 18:39:01.561370 23363 net.cpp:165] Memory required for data: 57185280
    I1230 18:39:01.561384 23363 layer_factory.hpp:77] Creating layer pool1
    I1230 18:39:01.561396 23363 net.cpp:106] Creating Layer pool1
    I1230 18:39:01.561415 23363 net.cpp:454] pool1 <- cccp2
    I1230 18:39:01.561425 23363 net.cpp:411] pool1 -> pool1
    I1230 18:39:01.561465 23363 net.cpp:150] Setting up pool1
    I1230 18:39:01.561473 23363 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 18:39:01.561480 23363 net.cpp:165] Memory required for data: 60195840
    I1230 18:39:01.561486 23363 layer_factory.hpp:77] Creating layer drop1
    I1230 18:39:01.561496 23363 net.cpp:106] Creating Layer drop1
    I1230 18:39:01.561502 23363 net.cpp:454] drop1 <- pool1
    I1230 18:39:01.561511 23363 net.cpp:397] drop1 -> pool1 (in-place)
    I1230 18:39:01.561533 23363 net.cpp:150] Setting up drop1
    I1230 18:39:01.561553 23363 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 18:39:01.561559 23363 net.cpp:165] Memory required for data: 63206400
    I1230 18:39:01.561566 23363 layer_factory.hpp:77] Creating layer relu1
    I1230 18:39:01.561578 23363 net.cpp:106] Creating Layer relu1
    I1230 18:39:01.561584 23363 net.cpp:454] relu1 <- pool1
    I1230 18:39:01.561594 23363 net.cpp:397] relu1 -> pool1 (in-place)
    I1230 18:39:01.561604 23363 net.cpp:150] Setting up relu1
    I1230 18:39:01.561645 23363 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 18:39:01.561650 23363 net.cpp:165] Memory required for data: 66216960
    I1230 18:39:01.561656 23363 layer_factory.hpp:77] Creating layer conv2
    I1230 18:39:01.561667 23363 net.cpp:106] Creating Layer conv2
    I1230 18:39:01.561673 23363 net.cpp:454] conv2 <- pool1
    I1230 18:39:01.561683 23363 net.cpp:411] conv2 -> conv2
    I1230 18:39:01.562122 23363 net.cpp:150] Setting up conv2
    I1230 18:39:01.562140 23363 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1230 18:39:01.562147 23363 net.cpp:165] Memory required for data: 68656320
    I1230 18:39:01.562158 23363 layer_factory.hpp:77] Creating layer pool2
    I1230 18:39:01.562170 23363 net.cpp:106] Creating Layer pool2
    I1230 18:39:01.562176 23363 net.cpp:454] pool2 <- conv2
    I1230 18:39:01.562186 23363 net.cpp:411] pool2 -> pool2
    I1230 18:39:01.562227 23363 net.cpp:150] Setting up pool2
    I1230 18:39:01.562237 23363 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 18:39:01.562244 23363 net.cpp:165] Memory required for data: 69160320
    I1230 18:39:01.562252 23363 layer_factory.hpp:77] Creating layer drop2
    I1230 18:39:01.562261 23363 net.cpp:106] Creating Layer drop2
    I1230 18:39:01.562268 23363 net.cpp:454] drop2 <- pool2
    I1230 18:39:01.562278 23363 net.cpp:397] drop2 -> pool2 (in-place)
    I1230 18:39:01.562300 23363 net.cpp:150] Setting up drop2
    I1230 18:39:01.562310 23363 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 18:39:01.562316 23363 net.cpp:165] Memory required for data: 69664320
    I1230 18:39:01.562324 23363 layer_factory.hpp:77] Creating layer relu2
    I1230 18:39:01.562333 23363 net.cpp:106] Creating Layer relu2
    I1230 18:39:01.562340 23363 net.cpp:454] relu2 <- pool2
    I1230 18:39:01.562348 23363 net.cpp:397] relu2 -> pool2 (in-place)
    I1230 18:39:01.562357 23363 net.cpp:150] Setting up relu2
    I1230 18:39:01.562377 23363 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 18:39:01.562383 23363 net.cpp:165] Memory required for data: 70168320
    I1230 18:39:01.562389 23363 layer_factory.hpp:77] Creating layer conv3
    I1230 18:39:01.562398 23363 net.cpp:106] Creating Layer conv3
    I1230 18:39:01.562404 23363 net.cpp:454] conv3 <- pool2
    I1230 18:39:01.562413 23363 net.cpp:411] conv3 -> conv3
    I1230 18:39:01.562722 23363 net.cpp:150] Setting up conv3
    I1230 18:39:01.562736 23363 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1230 18:39:01.562744 23363 net.cpp:165] Memory required for data: 70659840
    I1230 18:39:01.562757 23363 layer_factory.hpp:77] Creating layer pool3
    I1230 18:39:01.562768 23363 net.cpp:106] Creating Layer pool3
    I1230 18:39:01.562777 23363 net.cpp:454] pool3 <- conv3
    I1230 18:39:01.562786 23363 net.cpp:411] pool3 -> pool3
    I1230 18:39:01.562834 23363 net.cpp:150] Setting up pool3
    I1230 18:39:01.562933 23363 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 18:39:01.562942 23363 net.cpp:165] Memory required for data: 70782720
    I1230 18:39:01.562949 23363 layer_factory.hpp:77] Creating layer relu3
    I1230 18:39:01.562971 23363 net.cpp:106] Creating Layer relu3
    I1230 18:39:01.562978 23363 net.cpp:454] relu3 <- pool3
    I1230 18:39:01.562985 23363 net.cpp:397] relu3 -> pool3 (in-place)
    I1230 18:39:01.563005 23363 net.cpp:150] Setting up relu3
    I1230 18:39:01.563014 23363 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 18:39:01.563020 23363 net.cpp:165] Memory required for data: 70905600
    I1230 18:39:01.563026 23363 layer_factory.hpp:77] Creating layer ip1
    I1230 18:39:01.563040 23363 net.cpp:106] Creating Layer ip1
    I1230 18:39:01.563047 23363 net.cpp:454] ip1 <- pool3
    I1230 18:39:01.563056 23363 net.cpp:411] ip1 -> ip1
    I1230 18:39:01.564437 23363 net.cpp:150] Setting up ip1
    I1230 18:39:01.564479 23363 net.cpp:157] Top shape: 120 512 (61440)
    I1230 18:39:01.564488 23363 net.cpp:165] Memory required for data: 71151360
    I1230 18:39:01.564502 23363 layer_factory.hpp:77] Creating layer sig1
    I1230 18:39:01.564517 23363 net.cpp:106] Creating Layer sig1
    I1230 18:39:01.564539 23363 net.cpp:454] sig1 <- ip1
    I1230 18:39:01.564550 23363 net.cpp:397] sig1 -> ip1 (in-place)
    I1230 18:39:01.564561 23363 net.cpp:150] Setting up sig1
    I1230 18:39:01.564592 23363 net.cpp:157] Top shape: 120 512 (61440)
    I1230 18:39:01.564599 23363 net.cpp:165] Memory required for data: 71397120
    I1230 18:39:01.564606 23363 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1230 18:39:01.564616 23363 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1230 18:39:01.564623 23363 net.cpp:454] ip1_sig1_0_split <- ip1
    I1230 18:39:01.564632 23363 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1230 18:39:01.564646 23363 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1230 18:39:01.564702 23363 net.cpp:150] Setting up ip1_sig1_0_split
    I1230 18:39:01.564714 23363 net.cpp:157] Top shape: 120 512 (61440)
    I1230 18:39:01.564723 23363 net.cpp:157] Top shape: 120 512 (61440)
    I1230 18:39:01.564730 23363 net.cpp:165] Memory required for data: 71888640
    I1230 18:39:01.564750 23363 layer_factory.hpp:77] Creating layer ip_c
    I1230 18:39:01.564762 23363 net.cpp:106] Creating Layer ip_c
    I1230 18:39:01.564769 23363 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1230 18:39:01.564779 23363 net.cpp:411] ip_c -> ip_c
    I1230 18:39:01.564996 23363 net.cpp:150] Setting up ip_c
    I1230 18:39:01.565007 23363 net.cpp:157] Top shape: 120 20 (2400)
    I1230 18:39:01.565014 23363 net.cpp:165] Memory required for data: 71898240
    I1230 18:39:01.565023 23363 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1230 18:39:01.565033 23363 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1230 18:39:01.565040 23363 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1230 18:39:01.565049 23363 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1230 18:39:01.565059 23363 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1230 18:39:01.565111 23363 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1230 18:39:01.565121 23363 net.cpp:157] Top shape: 120 20 (2400)
    I1230 18:39:01.565130 23363 net.cpp:157] Top shape: 120 20 (2400)
    I1230 18:39:01.565137 23363 net.cpp:165] Memory required for data: 71917440
    I1230 18:39:01.565145 23363 layer_factory.hpp:77] Creating layer accuracy_c
    I1230 18:39:01.565156 23363 net.cpp:106] Creating Layer accuracy_c
    I1230 18:39:01.565165 23363 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1230 18:39:01.565187 23363 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1230 18:39:01.565197 23363 net.cpp:411] accuracy_c -> accuracy_c
    I1230 18:39:01.565208 23363 net.cpp:150] Setting up accuracy_c
    I1230 18:39:01.565217 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.565222 23363 net.cpp:165] Memory required for data: 71917444
    I1230 18:39:01.565229 23363 layer_factory.hpp:77] Creating layer loss_c
    I1230 18:39:01.565239 23363 net.cpp:106] Creating Layer loss_c
    I1230 18:39:01.565246 23363 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1230 18:39:01.565254 23363 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1230 18:39:01.565263 23363 net.cpp:411] loss_c -> loss_c
    I1230 18:39:01.565287 23363 layer_factory.hpp:77] Creating layer loss_c
    I1230 18:39:01.565446 23363 net.cpp:150] Setting up loss_c
    I1230 18:39:01.565464 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.565472 23363 net.cpp:160]     with loss weight 1
    I1230 18:39:01.565490 23363 net.cpp:165] Memory required for data: 71917448
    I1230 18:39:01.565498 23363 layer_factory.hpp:77] Creating layer ip_f
    I1230 18:39:01.565511 23363 net.cpp:106] Creating Layer ip_f
    I1230 18:39:01.565533 23363 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1230 18:39:01.565543 23363 net.cpp:411] ip_f -> ip_f
    I1230 18:39:01.566144 23363 net.cpp:150] Setting up ip_f
    I1230 18:39:01.566170 23363 net.cpp:157] Top shape: 120 100 (12000)
    I1230 18:39:01.566179 23363 net.cpp:165] Memory required for data: 71965448
    I1230 18:39:01.566190 23363 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1230 18:39:01.566202 23363 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1230 18:39:01.566223 23363 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1230 18:39:01.566232 23363 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1230 18:39:01.566242 23363 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1230 18:39:01.566278 23363 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1230 18:39:01.566308 23363 net.cpp:157] Top shape: 120 100 (12000)
    I1230 18:39:01.566316 23363 net.cpp:157] Top shape: 120 100 (12000)
    I1230 18:39:01.566323 23363 net.cpp:165] Memory required for data: 72061448
    I1230 18:39:01.566330 23363 layer_factory.hpp:77] Creating layer accuracy_f
    I1230 18:39:01.566340 23363 net.cpp:106] Creating Layer accuracy_f
    I1230 18:39:01.566361 23363 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1230 18:39:01.566370 23363 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1230 18:39:01.566380 23363 net.cpp:411] accuracy_f -> accuracy_f
    I1230 18:39:01.566393 23363 net.cpp:150] Setting up accuracy_f
    I1230 18:39:01.566416 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.566422 23363 net.cpp:165] Memory required for data: 72061452
    I1230 18:39:01.566429 23363 layer_factory.hpp:77] Creating layer loss_f
    I1230 18:39:01.566438 23363 net.cpp:106] Creating Layer loss_f
    I1230 18:39:01.566445 23363 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1230 18:39:01.566453 23363 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1230 18:39:01.566462 23363 net.cpp:411] loss_f -> loss_f
    I1230 18:39:01.566474 23363 layer_factory.hpp:77] Creating layer loss_f
    I1230 18:39:01.566593 23363 net.cpp:150] Setting up loss_f
    I1230 18:39:01.566607 23363 net.cpp:157] Top shape: (1)
    I1230 18:39:01.566614 23363 net.cpp:160]     with loss weight 1
    I1230 18:39:01.566627 23363 net.cpp:165] Memory required for data: 72061456
    I1230 18:39:01.566635 23363 net.cpp:226] loss_f needs backward computation.
    I1230 18:39:01.566644 23363 net.cpp:228] accuracy_f does not need backward computation.
    I1230 18:39:01.566666 23363 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1230 18:39:01.566674 23363 net.cpp:226] ip_f needs backward computation.
    I1230 18:39:01.566681 23363 net.cpp:226] loss_c needs backward computation.
    I1230 18:39:01.566689 23363 net.cpp:228] accuracy_c does not need backward computation.
    I1230 18:39:01.566697 23363 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1230 18:39:01.566705 23363 net.cpp:226] ip_c needs backward computation.
    I1230 18:39:01.566712 23363 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1230 18:39:01.566720 23363 net.cpp:226] sig1 needs backward computation.
    I1230 18:39:01.566726 23363 net.cpp:226] ip1 needs backward computation.
    I1230 18:39:01.566735 23363 net.cpp:226] relu3 needs backward computation.
    I1230 18:39:01.566742 23363 net.cpp:226] pool3 needs backward computation.
    I1230 18:39:01.566762 23363 net.cpp:226] conv3 needs backward computation.
    I1230 18:39:01.566771 23363 net.cpp:226] relu2 needs backward computation.
    I1230 18:39:01.566778 23363 net.cpp:226] drop2 needs backward computation.
    I1230 18:39:01.566787 23363 net.cpp:226] pool2 needs backward computation.
    I1230 18:39:01.566795 23363 net.cpp:226] conv2 needs backward computation.
    I1230 18:39:01.566802 23363 net.cpp:226] relu1 needs backward computation.
    I1230 18:39:01.566823 23363 net.cpp:226] drop1 needs backward computation.
    I1230 18:39:01.566830 23363 net.cpp:226] pool1 needs backward computation.
    I1230 18:39:01.566838 23363 net.cpp:226] cccp2 needs backward computation.
    I1230 18:39:01.566844 23363 net.cpp:226] cccp1 needs backward computation.
    I1230 18:39:01.566853 23363 net.cpp:226] conv1 needs backward computation.
    I1230 18:39:01.566860 23363 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1230 18:39:01.566869 23363 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1230 18:39:01.566876 23363 net.cpp:228] data does not need backward computation.
    I1230 18:39:01.566884 23363 net.cpp:270] This network produces output accuracy_c
    I1230 18:39:01.566890 23363 net.cpp:270] This network produces output accuracy_f
    I1230 18:39:01.566897 23363 net.cpp:270] This network produces output loss_c
    I1230 18:39:01.566905 23363 net.cpp:270] This network produces output loss_f
    I1230 18:39:01.566936 23363 net.cpp:283] Network initialization done.
    I1230 18:39:01.567080 23363 solver.cpp:60] Solver scaffolding done.
    I1230 18:39:01.567638 23363 caffe.cpp:212] Starting Optimization
    I1230 18:39:01.567713 23363 solver.cpp:288] Solving
    I1230 18:39:01.567733 23363 solver.cpp:289] Learning Rate Policy: inv
    I1230 18:39:01.569033 23363 solver.cpp:341] Iteration 0, Testing net (#0)
    I1230 18:39:08.036262 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.0505833
    I1230 18:39:08.036312 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.01
    I1230 18:39:08.036329 23363 solver.cpp:409]     Test net output #2: loss_c = 3.39392 (* 1 = 3.39392 loss)
    I1230 18:39:08.036350 23363 solver.cpp:409]     Test net output #3: loss_f = 4.79247 (* 1 = 4.79247 loss)
    I1230 18:39:08.170918 23363 solver.cpp:237] Iteration 0, loss = 8.06237
    I1230 18:39:08.170963 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.08
    I1230 18:39:08.170974 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.02
    I1230 18:39:08.170986 23363 solver.cpp:253]     Train net output #2: loss_c = 3.31471 (* 1 = 3.31471 loss)
    I1230 18:39:08.170999 23363 solver.cpp:253]     Train net output #3: loss_f = 4.74766 (* 1 = 4.74766 loss)
    I1230 18:39:08.171023 23363 sgd_solver.cpp:106] Iteration 0, lr = 0.0007
    I1230 18:39:23.971710 23363 solver.cpp:237] Iteration 100, loss = 7.60514
    I1230 18:39:23.971784 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.06
    I1230 18:39:23.971798 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 18:39:23.971812 23363 solver.cpp:253]     Train net output #2: loss_c = 2.99053 (* 1 = 2.99053 loss)
    I1230 18:39:23.971824 23363 solver.cpp:253]     Train net output #3: loss_f = 4.61461 (* 1 = 4.61461 loss)
    I1230 18:39:23.971837 23363 sgd_solver.cpp:106] Iteration 100, lr = 0.000694796
    I1230 18:39:39.656504 23363 solver.cpp:237] Iteration 200, loss = 7.63582
    I1230 18:39:39.656577 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.07
    I1230 18:39:39.656590 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 18:39:39.656602 23363 solver.cpp:253]     Train net output #2: loss_c = 3.01622 (* 1 = 3.01622 loss)
    I1230 18:39:39.656611 23363 solver.cpp:253]     Train net output #3: loss_f = 4.6196 (* 1 = 4.6196 loss)
    I1230 18:39:39.656621 23363 sgd_solver.cpp:106] Iteration 200, lr = 0.00068968
    I1230 18:39:55.218297 23363 solver.cpp:237] Iteration 300, loss = 7.66779
    I1230 18:39:55.218379 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.04
    I1230 18:39:55.218395 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.03
    I1230 18:39:55.218412 23363 solver.cpp:253]     Train net output #2: loss_c = 3.00958 (* 1 = 3.00958 loss)
    I1230 18:39:55.218426 23363 solver.cpp:253]     Train net output #3: loss_f = 4.65821 (* 1 = 4.65821 loss)
    I1230 18:39:55.218453 23363 sgd_solver.cpp:106] Iteration 300, lr = 0.000684652
    I1230 18:40:10.921149 23363 solver.cpp:237] Iteration 400, loss = 7.61258
    I1230 18:40:10.921269 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.07
    I1230 18:40:10.921284 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0
    I1230 18:40:10.921296 23363 solver.cpp:253]     Train net output #2: loss_c = 2.99638 (* 1 = 2.99638 loss)
    I1230 18:40:10.921306 23363 solver.cpp:253]     Train net output #3: loss_f = 4.6162 (* 1 = 4.6162 loss)
    I1230 18:40:10.921317 23363 sgd_solver.cpp:106] Iteration 400, lr = 0.000679709
    I1230 18:40:27.035276 23363 solver.cpp:237] Iteration 500, loss = 7.36931
    I1230 18:40:27.035326 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.11
    I1230 18:40:27.035337 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.04
    I1230 18:40:27.035348 23363 solver.cpp:253]     Train net output #2: loss_c = 2.89929 (* 1 = 2.89929 loss)
    I1230 18:40:27.035358 23363 solver.cpp:253]     Train net output #3: loss_f = 4.47002 (* 1 = 4.47002 loss)
    I1230 18:40:27.035369 23363 sgd_solver.cpp:106] Iteration 500, lr = 0.000674848
    I1230 18:40:42.890941 23363 solver.cpp:237] Iteration 600, loss = 7.08039
    I1230 18:40:42.891093 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.19
    I1230 18:40:42.891110 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.03
    I1230 18:40:42.891134 23363 solver.cpp:253]     Train net output #2: loss_c = 2.7697 (* 1 = 2.7697 loss)
    I1230 18:40:42.891144 23363 solver.cpp:253]     Train net output #3: loss_f = 4.31069 (* 1 = 4.31069 loss)
    I1230 18:40:42.891155 23363 sgd_solver.cpp:106] Iteration 600, lr = 0.000670068
    I1230 18:40:58.844640 23363 solver.cpp:237] Iteration 700, loss = 6.92457
    I1230 18:40:58.844689 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.2
    I1230 18:40:58.844702 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.06
    I1230 18:40:58.844717 23363 solver.cpp:253]     Train net output #2: loss_c = 2.66557 (* 1 = 2.66557 loss)
    I1230 18:40:58.844727 23363 solver.cpp:253]     Train net output #3: loss_f = 4.259 (* 1 = 4.259 loss)
    I1230 18:40:58.844740 23363 sgd_solver.cpp:106] Iteration 700, lr = 0.000665365
    I1230 18:41:14.893930 23363 solver.cpp:237] Iteration 800, loss = 7.0562
    I1230 18:41:14.894062 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.1
    I1230 18:41:14.894091 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.03
    I1230 18:41:14.894106 23363 solver.cpp:253]     Train net output #2: loss_c = 2.73943 (* 1 = 2.73943 loss)
    I1230 18:41:14.894119 23363 solver.cpp:253]     Train net output #3: loss_f = 4.31676 (* 1 = 4.31676 loss)
    I1230 18:41:14.894131 23363 sgd_solver.cpp:106] Iteration 800, lr = 0.000660739
    I1230 18:41:30.481817 23363 solver.cpp:237] Iteration 900, loss = 6.9631
    I1230 18:41:30.481914 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.1
    I1230 18:41:30.481940 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.03
    I1230 18:41:30.481963 23363 solver.cpp:253]     Train net output #2: loss_c = 2.77791 (* 1 = 2.77791 loss)
    I1230 18:41:30.481979 23363 solver.cpp:253]     Train net output #3: loss_f = 4.18519 (* 1 = 4.18519 loss)
    I1230 18:41:30.481999 23363 sgd_solver.cpp:106] Iteration 900, lr = 0.000656188
    I1230 18:41:46.085453 23363 solver.cpp:341] Iteration 1000, Testing net (#0)
    I1230 18:41:52.524164 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.153583
    I1230 18:41:52.524216 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.0464167
    I1230 18:41:52.524230 23363 solver.cpp:409]     Test net output #2: loss_c = 2.6987 (* 1 = 2.6987 loss)
    I1230 18:41:52.524240 23363 solver.cpp:409]     Test net output #3: loss_f = 4.20359 (* 1 = 4.20359 loss)
    I1230 18:41:52.610040 23363 solver.cpp:237] Iteration 1000, loss = 6.86557
    I1230 18:41:52.610088 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.1
    I1230 18:41:52.610100 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.06
    I1230 18:41:52.610111 23363 solver.cpp:253]     Train net output #2: loss_c = 2.67945 (* 1 = 2.67945 loss)
    I1230 18:41:52.610123 23363 solver.cpp:253]     Train net output #3: loss_f = 4.18613 (* 1 = 4.18613 loss)
    I1230 18:41:52.610136 23363 sgd_solver.cpp:106] Iteration 1000, lr = 0.000651709
    I1230 18:42:08.372619 23363 solver.cpp:237] Iteration 1100, loss = 6.62983
    I1230 18:42:08.372663 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.23
    I1230 18:42:08.372673 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.08
    I1230 18:42:08.372685 23363 solver.cpp:253]     Train net output #2: loss_c = 2.62181 (* 1 = 2.62181 loss)
    I1230 18:42:08.372695 23363 solver.cpp:253]     Train net output #3: loss_f = 4.00802 (* 1 = 4.00802 loss)
    I1230 18:42:08.372706 23363 sgd_solver.cpp:106] Iteration 1100, lr = 0.0006473
    I1230 18:42:24.416604 23363 solver.cpp:237] Iteration 1200, loss = 6.44881
    I1230 18:42:24.416744 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.26
    I1230 18:42:24.416760 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.07
    I1230 18:42:24.416774 23363 solver.cpp:253]     Train net output #2: loss_c = 2.47376 (* 1 = 2.47376 loss)
    I1230 18:42:24.416784 23363 solver.cpp:253]     Train net output #3: loss_f = 3.97505 (* 1 = 3.97505 loss)
    I1230 18:42:24.416795 23363 sgd_solver.cpp:106] Iteration 1200, lr = 0.000642961
    I1230 18:42:40.062433 23363 solver.cpp:237] Iteration 1300, loss = 6.69891
    I1230 18:42:40.062489 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.18
    I1230 18:42:40.062511 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.05
    I1230 18:42:40.062522 23363 solver.cpp:253]     Train net output #2: loss_c = 2.6036 (* 1 = 2.6036 loss)
    I1230 18:42:40.062532 23363 solver.cpp:253]     Train net output #3: loss_f = 4.0953 (* 1 = 4.0953 loss)
    I1230 18:42:40.062542 23363 sgd_solver.cpp:106] Iteration 1300, lr = 0.000638689
    I1230 18:42:55.556602 23363 solver.cpp:237] Iteration 1400, loss = 6.41039
    I1230 18:42:55.556773 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.15
    I1230 18:42:55.556793 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.07
    I1230 18:42:55.556807 23363 solver.cpp:253]     Train net output #2: loss_c = 2.54239 (* 1 = 2.54239 loss)
    I1230 18:42:55.556818 23363 solver.cpp:253]     Train net output #3: loss_f = 3.868 (* 1 = 3.868 loss)
    I1230 18:42:55.556829 23363 sgd_solver.cpp:106] Iteration 1400, lr = 0.000634482
    I1230 18:43:11.775095 23363 solver.cpp:237] Iteration 1500, loss = 6.38971
    I1230 18:43:11.775136 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.2
    I1230 18:43:11.775149 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.06
    I1230 18:43:11.775163 23363 solver.cpp:253]     Train net output #2: loss_c = 2.46619 (* 1 = 2.46619 loss)
    I1230 18:43:11.775176 23363 solver.cpp:253]     Train net output #3: loss_f = 3.92353 (* 1 = 3.92353 loss)
    I1230 18:43:11.775187 23363 sgd_solver.cpp:106] Iteration 1500, lr = 0.00063034
    I1230 18:43:27.299437 23363 solver.cpp:237] Iteration 1600, loss = 6.11462
    I1230 18:43:27.299558 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.24
    I1230 18:43:27.299581 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1230 18:43:27.299594 23363 solver.cpp:253]     Train net output #2: loss_c = 2.40577 (* 1 = 2.40577 loss)
    I1230 18:43:27.299605 23363 solver.cpp:253]     Train net output #3: loss_f = 3.70885 (* 1 = 3.70885 loss)
    I1230 18:43:27.299618 23363 sgd_solver.cpp:106] Iteration 1600, lr = 0.00062626
    I1230 18:43:42.878940 23363 solver.cpp:237] Iteration 1700, loss = 6.19874
    I1230 18:43:42.878988 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.24
    I1230 18:43:42.879000 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.11
    I1230 18:43:42.879014 23363 solver.cpp:253]     Train net output #2: loss_c = 2.38824 (* 1 = 2.38824 loss)
    I1230 18:43:42.879024 23363 solver.cpp:253]     Train net output #3: loss_f = 3.8105 (* 1 = 3.8105 loss)
    I1230 18:43:42.879036 23363 sgd_solver.cpp:106] Iteration 1700, lr = 0.000622241
    I1230 18:43:58.050297 23363 solver.cpp:237] Iteration 1800, loss = 6.33617
    I1230 18:43:58.050411 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.22
    I1230 18:43:58.050432 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.1
    I1230 18:43:58.050451 23363 solver.cpp:253]     Train net output #2: loss_c = 2.45854 (* 1 = 2.45854 loss)
    I1230 18:43:58.050467 23363 solver.cpp:253]     Train net output #3: loss_f = 3.87762 (* 1 = 3.87762 loss)
    I1230 18:43:58.050480 23363 sgd_solver.cpp:106] Iteration 1800, lr = 0.000618282
    I1230 18:44:13.490631 23363 solver.cpp:237] Iteration 1900, loss = 5.85429
    I1230 18:44:13.490675 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.26
    I1230 18:44:13.490687 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1230 18:44:13.490701 23363 solver.cpp:253]     Train net output #2: loss_c = 2.30168 (* 1 = 2.30168 loss)
    I1230 18:44:13.490713 23363 solver.cpp:253]     Train net output #3: loss_f = 3.55261 (* 1 = 3.55261 loss)
    I1230 18:44:13.490725 23363 sgd_solver.cpp:106] Iteration 1900, lr = 0.000614381
    I1230 18:44:28.784540 23363 solver.cpp:341] Iteration 2000, Testing net (#0)
    I1230 18:44:34.928217 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.264583
    I1230 18:44:34.928287 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.114667
    I1230 18:44:34.928313 23363 solver.cpp:409]     Test net output #2: loss_c = 2.36976 (* 1 = 2.36976 loss)
    I1230 18:44:34.928328 23363 solver.cpp:409]     Test net output #3: loss_f = 3.71768 (* 1 = 3.71768 loss)
    I1230 18:44:35.010829 23363 solver.cpp:237] Iteration 2000, loss = 5.96856
    I1230 18:44:35.010872 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:44:35.010884 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.18
    I1230 18:44:35.010896 23363 solver.cpp:253]     Train net output #2: loss_c = 2.32073 (* 1 = 2.32073 loss)
    I1230 18:44:35.010907 23363 solver.cpp:253]     Train net output #3: loss_f = 3.64782 (* 1 = 3.64782 loss)
    I1230 18:44:35.010920 23363 sgd_solver.cpp:106] Iteration 2000, lr = 0.000610537
    I1230 18:44:50.219336 23363 solver.cpp:237] Iteration 2100, loss = 6.02868
    I1230 18:44:50.219388 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.3
    I1230 18:44:50.219404 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.1
    I1230 18:44:50.219424 23363 solver.cpp:253]     Train net output #2: loss_c = 2.40659 (* 1 = 2.40659 loss)
    I1230 18:44:50.219441 23363 solver.cpp:253]     Train net output #3: loss_f = 3.62209 (* 1 = 3.62209 loss)
    I1230 18:44:50.219456 23363 sgd_solver.cpp:106] Iteration 2100, lr = 0.000606749
    I1230 18:45:06.297054 23363 solver.cpp:237] Iteration 2200, loss = 6.00307
    I1230 18:45:06.297189 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.33
    I1230 18:45:06.297211 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.15
    I1230 18:45:06.297229 23363 solver.cpp:253]     Train net output #2: loss_c = 2.27865 (* 1 = 2.27865 loss)
    I1230 18:45:06.297245 23363 solver.cpp:253]     Train net output #3: loss_f = 3.72442 (* 1 = 3.72442 loss)
    I1230 18:45:06.297258 23363 sgd_solver.cpp:106] Iteration 2200, lr = 0.000603015
    I1230 18:45:21.845185 23363 solver.cpp:237] Iteration 2300, loss = 6.27778
    I1230 18:45:21.845249 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.19
    I1230 18:45:21.845269 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.12
    I1230 18:45:21.845291 23363 solver.cpp:253]     Train net output #2: loss_c = 2.52783 (* 1 = 2.52783 loss)
    I1230 18:45:21.845309 23363 solver.cpp:253]     Train net output #3: loss_f = 3.74995 (* 1 = 3.74995 loss)
    I1230 18:45:21.845326 23363 sgd_solver.cpp:106] Iteration 2300, lr = 0.000599334
    I1230 18:45:37.958739 23363 solver.cpp:237] Iteration 2400, loss = 5.58488
    I1230 18:45:37.958858 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.31
    I1230 18:45:37.958879 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.17
    I1230 18:45:37.958899 23363 solver.cpp:253]     Train net output #2: loss_c = 2.22609 (* 1 = 2.22609 loss)
    I1230 18:45:37.958915 23363 solver.cpp:253]     Train net output #3: loss_f = 3.35879 (* 1 = 3.35879 loss)
    I1230 18:45:37.958930 23363 sgd_solver.cpp:106] Iteration 2400, lr = 0.000595706
    I1230 18:45:53.542995 23363 solver.cpp:237] Iteration 2500, loss = 6.04978
    I1230 18:45:53.543045 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.25
    I1230 18:45:53.543061 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1230 18:45:53.543078 23363 solver.cpp:253]     Train net output #2: loss_c = 2.39374 (* 1 = 2.39374 loss)
    I1230 18:45:53.543093 23363 solver.cpp:253]     Train net output #3: loss_f = 3.65604 (* 1 = 3.65604 loss)
    I1230 18:45:53.543108 23363 sgd_solver.cpp:106] Iteration 2500, lr = 0.000592128
    I1230 18:46:09.180642 23363 solver.cpp:237] Iteration 2600, loss = 5.80321
    I1230 18:46:09.180743 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.25
    I1230 18:46:09.180763 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.09
    I1230 18:46:09.180783 23363 solver.cpp:253]     Train net output #2: loss_c = 2.33882 (* 1 = 2.33882 loss)
    I1230 18:46:09.180799 23363 solver.cpp:253]     Train net output #3: loss_f = 3.46439 (* 1 = 3.46439 loss)
    I1230 18:46:09.180815 23363 sgd_solver.cpp:106] Iteration 2600, lr = 0.0005886
    I1230 18:46:25.089241 23363 solver.cpp:237] Iteration 2700, loss = 5.79144
    I1230 18:46:25.089284 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.31
    I1230 18:46:25.089298 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1230 18:46:25.089310 23363 solver.cpp:253]     Train net output #2: loss_c = 2.2022 (* 1 = 2.2022 loss)
    I1230 18:46:25.089323 23363 solver.cpp:253]     Train net output #3: loss_f = 3.58925 (* 1 = 3.58925 loss)
    I1230 18:46:25.089334 23363 sgd_solver.cpp:106] Iteration 2700, lr = 0.00058512
    I1230 18:46:40.711382 23363 solver.cpp:237] Iteration 2800, loss = 5.926
    I1230 18:46:40.711524 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:46:40.711544 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.16
    I1230 18:46:40.711561 23363 solver.cpp:253]     Train net output #2: loss_c = 2.34889 (* 1 = 2.34889 loss)
    I1230 18:46:40.711577 23363 solver.cpp:253]     Train net output #3: loss_f = 3.57711 (* 1 = 3.57711 loss)
    I1230 18:46:40.711591 23363 sgd_solver.cpp:106] Iteration 2800, lr = 0.000581689
    I1230 18:46:55.778092 23363 solver.cpp:237] Iteration 2900, loss = 5.29468
    I1230 18:46:55.778132 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:46:55.778143 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1230 18:46:55.778156 23363 solver.cpp:253]     Train net output #2: loss_c = 2.1083 (* 1 = 2.1083 loss)
    I1230 18:46:55.778167 23363 solver.cpp:253]     Train net output #3: loss_f = 3.18638 (* 1 = 3.18638 loss)
    I1230 18:46:55.778177 23363 sgd_solver.cpp:106] Iteration 2900, lr = 0.000578303
    I1230 18:47:10.891158 23363 solver.cpp:341] Iteration 3000, Testing net (#0)
    I1230 18:47:17.033658 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.30225
    I1230 18:47:17.033740 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.167333
    I1230 18:47:17.033767 23363 solver.cpp:409]     Test net output #2: loss_c = 2.27099 (* 1 = 2.27099 loss)
    I1230 18:47:17.033787 23363 solver.cpp:409]     Test net output #3: loss_f = 3.48574 (* 1 = 3.48574 loss)
    I1230 18:47:17.124716 23363 solver.cpp:237] Iteration 3000, loss = 5.83517
    I1230 18:47:17.124765 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.26
    I1230 18:47:17.124781 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1230 18:47:17.124799 23363 solver.cpp:253]     Train net output #2: loss_c = 2.29513 (* 1 = 2.29513 loss)
    I1230 18:47:17.124814 23363 solver.cpp:253]     Train net output #3: loss_f = 3.54004 (* 1 = 3.54004 loss)
    I1230 18:47:17.124830 23363 sgd_solver.cpp:106] Iteration 3000, lr = 0.000574964
    I1230 18:47:32.869230 23363 solver.cpp:237] Iteration 3100, loss = 5.7293
    I1230 18:47:32.869279 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:47:32.869294 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.15
    I1230 18:47:32.869312 23363 solver.cpp:253]     Train net output #2: loss_c = 2.28402 (* 1 = 2.28402 loss)
    I1230 18:47:32.869326 23363 solver.cpp:253]     Train net output #3: loss_f = 3.44528 (* 1 = 3.44528 loss)
    I1230 18:47:32.869341 23363 sgd_solver.cpp:106] Iteration 3100, lr = 0.000571669
    I1230 18:47:49.299301 23363 solver.cpp:237] Iteration 3200, loss = 5.66976
    I1230 18:47:49.299394 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:47:49.299412 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.15
    I1230 18:47:49.299428 23363 solver.cpp:253]     Train net output #2: loss_c = 2.16737 (* 1 = 2.16737 loss)
    I1230 18:47:49.299443 23363 solver.cpp:253]     Train net output #3: loss_f = 3.50239 (* 1 = 3.50239 loss)
    I1230 18:47:49.299458 23363 sgd_solver.cpp:106] Iteration 3200, lr = 0.000568418
    I1230 18:48:05.245184 23363 solver.cpp:237] Iteration 3300, loss = 5.57663
    I1230 18:48:05.245236 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:48:05.245254 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.19
    I1230 18:48:05.245272 23363 solver.cpp:253]     Train net output #2: loss_c = 2.22887 (* 1 = 2.22887 loss)
    I1230 18:48:05.245287 23363 solver.cpp:253]     Train net output #3: loss_f = 3.34776 (* 1 = 3.34776 loss)
    I1230 18:48:05.245302 23363 sgd_solver.cpp:106] Iteration 3300, lr = 0.000565209
    I1230 18:48:20.644140 23363 solver.cpp:237] Iteration 3400, loss = 5.18095
    I1230 18:48:20.644245 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:48:20.644259 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.26
    I1230 18:48:20.644271 23363 solver.cpp:253]     Train net output #2: loss_c = 2.13469 (* 1 = 2.13469 loss)
    I1230 18:48:20.644282 23363 solver.cpp:253]     Train net output #3: loss_f = 3.04626 (* 1 = 3.04626 loss)
    I1230 18:48:20.644294 23363 sgd_solver.cpp:106] Iteration 3400, lr = 0.000562043
    I1230 18:48:36.024065 23363 solver.cpp:237] Iteration 3500, loss = 5.69281
    I1230 18:48:36.024101 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.3
    I1230 18:48:36.024111 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.15
    I1230 18:48:36.024121 23363 solver.cpp:253]     Train net output #2: loss_c = 2.2453 (* 1 = 2.2453 loss)
    I1230 18:48:36.024129 23363 solver.cpp:253]     Train net output #3: loss_f = 3.4475 (* 1 = 3.4475 loss)
    I1230 18:48:36.024138 23363 sgd_solver.cpp:106] Iteration 3500, lr = 0.000558917
    I1230 18:48:51.994590 23363 solver.cpp:237] Iteration 3600, loss = 5.46274
    I1230 18:48:51.994702 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:48:51.994716 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:48:51.994727 23363 solver.cpp:253]     Train net output #2: loss_c = 2.18956 (* 1 = 2.18956 loss)
    I1230 18:48:51.994736 23363 solver.cpp:253]     Train net output #3: loss_f = 3.27318 (* 1 = 3.27318 loss)
    I1230 18:48:51.994746 23363 sgd_solver.cpp:106] Iteration 3600, lr = 0.000555832
    I1230 18:49:07.680043 23363 solver.cpp:237] Iteration 3700, loss = 5.53487
    I1230 18:49:07.680091 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:49:07.680104 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.11
    I1230 18:49:07.680116 23363 solver.cpp:253]     Train net output #2: loss_c = 2.10354 (* 1 = 2.10354 loss)
    I1230 18:49:07.680126 23363 solver.cpp:253]     Train net output #3: loss_f = 3.43132 (* 1 = 3.43132 loss)
    I1230 18:49:07.680136 23363 sgd_solver.cpp:106] Iteration 3700, lr = 0.000552787
    I1230 18:49:23.347484 23363 solver.cpp:237] Iteration 3800, loss = 5.44975
    I1230 18:49:23.347610 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:49:23.347633 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.22
    I1230 18:49:23.347662 23363 solver.cpp:253]     Train net output #2: loss_c = 2.19781 (* 1 = 2.19781 loss)
    I1230 18:49:23.347682 23363 solver.cpp:253]     Train net output #3: loss_f = 3.25194 (* 1 = 3.25194 loss)
    I1230 18:49:23.347702 23363 sgd_solver.cpp:106] Iteration 3800, lr = 0.00054978
    I1230 18:49:39.568938 23363 solver.cpp:237] Iteration 3900, loss = 4.81995
    I1230 18:49:39.568982 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:49:39.568994 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.22
    I1230 18:49:39.569008 23363 solver.cpp:253]     Train net output #2: loss_c = 1.95523 (* 1 = 1.95523 loss)
    I1230 18:49:39.569020 23363 solver.cpp:253]     Train net output #3: loss_f = 2.86473 (* 1 = 2.86473 loss)
    I1230 18:49:39.569031 23363 sgd_solver.cpp:106] Iteration 3900, lr = 0.000546811
    I1230 18:49:55.782342 23363 solver.cpp:341] Iteration 4000, Testing net (#0)
    I1230 18:50:02.152247 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.346583
    I1230 18:50:02.152348 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.210667
    I1230 18:50:02.152384 23363 solver.cpp:409]     Test net output #2: loss_c = 2.16425 (* 1 = 2.16425 loss)
    I1230 18:50:02.152411 23363 solver.cpp:409]     Test net output #3: loss_f = 3.25468 (* 1 = 3.25468 loss)
    I1230 18:50:02.239725 23363 solver.cpp:237] Iteration 4000, loss = 5.5727
    I1230 18:50:02.239814 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.33
    I1230 18:50:02.239841 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.16
    I1230 18:50:02.239856 23363 solver.cpp:253]     Train net output #2: loss_c = 2.21352 (* 1 = 2.21352 loss)
    I1230 18:50:02.239871 23363 solver.cpp:253]     Train net output #3: loss_f = 3.35919 (* 1 = 3.35919 loss)
    I1230 18:50:02.239883 23363 sgd_solver.cpp:106] Iteration 4000, lr = 0.000543879
    I1230 18:50:17.782287 23363 solver.cpp:237] Iteration 4100, loss = 5.28033
    I1230 18:50:17.782327 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1230 18:50:17.782341 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:50:17.782353 23363 solver.cpp:253]     Train net output #2: loss_c = 2.10251 (* 1 = 2.10251 loss)
    I1230 18:50:17.782363 23363 solver.cpp:253]     Train net output #3: loss_f = 3.17782 (* 1 = 3.17782 loss)
    I1230 18:50:17.782373 23363 sgd_solver.cpp:106] Iteration 4100, lr = 0.000540983
    I1230 18:50:33.052912 23363 solver.cpp:237] Iteration 4200, loss = 5.13327
    I1230 18:50:33.053067 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1230 18:50:33.053091 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1230 18:50:33.053107 23363 solver.cpp:253]     Train net output #2: loss_c = 1.94265 (* 1 = 1.94265 loss)
    I1230 18:50:33.053124 23363 solver.cpp:253]     Train net output #3: loss_f = 3.19062 (* 1 = 3.19062 loss)
    I1230 18:50:33.053140 23363 sgd_solver.cpp:106] Iteration 4200, lr = 0.000538123
    I1230 18:50:48.598567 23363 solver.cpp:237] Iteration 4300, loss = 5.12345
    I1230 18:50:48.598620 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:50:48.598639 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:50:48.598657 23363 solver.cpp:253]     Train net output #2: loss_c = 2.08686 (* 1 = 2.08686 loss)
    I1230 18:50:48.598675 23363 solver.cpp:253]     Train net output #3: loss_f = 3.03659 (* 1 = 3.03659 loss)
    I1230 18:50:48.598691 23363 sgd_solver.cpp:106] Iteration 4300, lr = 0.000535298
    I1230 18:51:04.323675 23363 solver.cpp:237] Iteration 4400, loss = 4.79842
    I1230 18:51:04.323794 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 18:51:04.323818 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:51:04.323839 23363 solver.cpp:253]     Train net output #2: loss_c = 1.95913 (* 1 = 1.95913 loss)
    I1230 18:51:04.323856 23363 solver.cpp:253]     Train net output #3: loss_f = 2.83929 (* 1 = 2.83929 loss)
    I1230 18:51:04.323873 23363 sgd_solver.cpp:106] Iteration 4400, lr = 0.000532508
    I1230 18:51:19.813587 23363 solver.cpp:237] Iteration 4500, loss = 5.22746
    I1230 18:51:19.813645 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:51:19.813663 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1230 18:51:19.813683 23363 solver.cpp:253]     Train net output #2: loss_c = 2.07404 (* 1 = 2.07404 loss)
    I1230 18:51:19.813700 23363 solver.cpp:253]     Train net output #3: loss_f = 3.15342 (* 1 = 3.15342 loss)
    I1230 18:51:19.813716 23363 sgd_solver.cpp:106] Iteration 4500, lr = 0.000529751
    I1230 18:51:35.374457 23363 solver.cpp:237] Iteration 4600, loss = 4.9046
    I1230 18:51:35.374570 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 18:51:35.374594 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 18:51:35.374614 23363 solver.cpp:253]     Train net output #2: loss_c = 1.97554 (* 1 = 1.97554 loss)
    I1230 18:51:35.374631 23363 solver.cpp:253]     Train net output #3: loss_f = 2.92906 (* 1 = 2.92906 loss)
    I1230 18:51:35.374647 23363 sgd_solver.cpp:106] Iteration 4600, lr = 0.000527028
    I1230 18:51:50.736906 23363 solver.cpp:237] Iteration 4700, loss = 5.16948
    I1230 18:51:50.736981 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 18:51:50.736999 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.26
    I1230 18:51:50.737015 23363 solver.cpp:253]     Train net output #2: loss_c = 2.00558 (* 1 = 2.00558 loss)
    I1230 18:51:50.737026 23363 solver.cpp:253]     Train net output #3: loss_f = 3.16391 (* 1 = 3.16391 loss)
    I1230 18:51:50.737040 23363 sgd_solver.cpp:106] Iteration 4700, lr = 0.000524336
    I1230 18:52:05.908151 23363 solver.cpp:237] Iteration 4800, loss = 5.11166
    I1230 18:52:05.908350 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.33
    I1230 18:52:05.908368 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:52:05.908380 23363 solver.cpp:253]     Train net output #2: loss_c = 2.05127 (* 1 = 2.05127 loss)
    I1230 18:52:05.908388 23363 solver.cpp:253]     Train net output #3: loss_f = 3.06039 (* 1 = 3.06039 loss)
    I1230 18:52:05.908398 23363 sgd_solver.cpp:106] Iteration 4800, lr = 0.000521677
    I1230 18:52:21.304862 23363 solver.cpp:237] Iteration 4900, loss = 4.55655
    I1230 18:52:21.304913 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 18:52:21.304924 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 18:52:21.304935 23363 solver.cpp:253]     Train net output #2: loss_c = 1.8788 (* 1 = 1.8788 loss)
    I1230 18:52:21.304945 23363 solver.cpp:253]     Train net output #3: loss_f = 2.67775 (* 1 = 2.67775 loss)
    I1230 18:52:21.304958 23363 sgd_solver.cpp:106] Iteration 4900, lr = 0.000519049
    I1230 18:52:36.529274 23363 solver.cpp:341] Iteration 5000, Testing net (#0)
    I1230 18:52:42.467710 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.380667
    I1230 18:52:42.467759 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.246083
    I1230 18:52:42.467778 23363 solver.cpp:409]     Test net output #2: loss_c = 2.02274 (* 1 = 2.02274 loss)
    I1230 18:52:42.467806 23363 solver.cpp:409]     Test net output #3: loss_f = 3.04998 (* 1 = 3.04998 loss)
    I1230 18:52:42.529412 23363 solver.cpp:237] Iteration 5000, loss = 5.09451
    I1230 18:52:42.529463 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 18:52:42.529474 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.23
    I1230 18:52:42.529487 23363 solver.cpp:253]     Train net output #2: loss_c = 1.99899 (* 1 = 1.99899 loss)
    I1230 18:52:42.529498 23363 solver.cpp:253]     Train net output #3: loss_f = 3.09552 (* 1 = 3.09552 loss)
    I1230 18:52:42.529510 23363 sgd_solver.cpp:106] Iteration 5000, lr = 0.000516452
    I1230 18:52:58.452447 23363 solver.cpp:237] Iteration 5100, loss = 5.17656
    I1230 18:52:58.452491 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1230 18:52:58.452502 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1230 18:52:58.452514 23363 solver.cpp:253]     Train net output #2: loss_c = 2.122 (* 1 = 2.122 loss)
    I1230 18:52:58.452524 23363 solver.cpp:253]     Train net output #3: loss_f = 3.05456 (* 1 = 3.05456 loss)
    I1230 18:52:58.452534 23363 sgd_solver.cpp:106] Iteration 5100, lr = 0.000513884
    I1230 18:53:13.995568 23363 solver.cpp:237] Iteration 5200, loss = 5.07382
    I1230 18:53:13.995751 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 18:53:13.995771 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.19
    I1230 18:53:13.995784 23363 solver.cpp:253]     Train net output #2: loss_c = 1.93506 (* 1 = 1.93506 loss)
    I1230 18:53:13.995795 23363 solver.cpp:253]     Train net output #3: loss_f = 3.13876 (* 1 = 3.13876 loss)
    I1230 18:53:13.995805 23363 sgd_solver.cpp:106] Iteration 5200, lr = 0.000511347
    I1230 18:53:29.544807 23363 solver.cpp:237] Iteration 5300, loss = 5.31331
    I1230 18:53:29.544852 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1230 18:53:29.544863 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.22
    I1230 18:53:29.544875 23363 solver.cpp:253]     Train net output #2: loss_c = 2.19337 (* 1 = 2.19337 loss)
    I1230 18:53:29.544888 23363 solver.cpp:253]     Train net output #3: loss_f = 3.11994 (* 1 = 3.11994 loss)
    I1230 18:53:29.544898 23363 sgd_solver.cpp:106] Iteration 5300, lr = 0.000508838
    I1230 18:53:44.841531 23363 solver.cpp:237] Iteration 5400, loss = 4.69766
    I1230 18:53:44.841645 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:53:44.841657 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:53:44.841667 23363 solver.cpp:253]     Train net output #2: loss_c = 1.93365 (* 1 = 1.93365 loss)
    I1230 18:53:44.841676 23363 solver.cpp:253]     Train net output #3: loss_f = 2.764 (* 1 = 2.764 loss)
    I1230 18:53:44.841686 23363 sgd_solver.cpp:106] Iteration 5400, lr = 0.000506358
    I1230 18:54:00.739778 23363 solver.cpp:237] Iteration 5500, loss = 5.00282
    I1230 18:54:00.739814 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.37
    I1230 18:54:00.739825 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.23
    I1230 18:54:00.739835 23363 solver.cpp:253]     Train net output #2: loss_c = 1.97746 (* 1 = 1.97746 loss)
    I1230 18:54:00.739845 23363 solver.cpp:253]     Train net output #3: loss_f = 3.02536 (* 1 = 3.02536 loss)
    I1230 18:54:00.739853 23363 sgd_solver.cpp:106] Iteration 5500, lr = 0.000503906
    I1230 18:54:14.036661 23363 solver.cpp:237] Iteration 5600, loss = 5.07631
    I1230 18:54:14.036711 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:54:14.036721 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1230 18:54:14.036732 23363 solver.cpp:253]     Train net output #2: loss_c = 2.05686 (* 1 = 2.05686 loss)
    I1230 18:54:14.036741 23363 solver.cpp:253]     Train net output #3: loss_f = 3.01945 (* 1 = 3.01945 loss)
    I1230 18:54:14.036751 23363 sgd_solver.cpp:106] Iteration 5600, lr = 0.000501481
    I1230 18:54:28.354560 23363 solver.cpp:237] Iteration 5700, loss = 4.998
    I1230 18:54:28.354745 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1230 18:54:28.354770 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 18:54:28.354789 23363 solver.cpp:253]     Train net output #2: loss_c = 1.93404 (* 1 = 1.93404 loss)
    I1230 18:54:28.354805 23363 solver.cpp:253]     Train net output #3: loss_f = 3.06396 (* 1 = 3.06396 loss)
    I1230 18:54:28.354821 23363 sgd_solver.cpp:106] Iteration 5700, lr = 0.000499084
    I1230 18:54:41.021113 23363 solver.cpp:237] Iteration 5800, loss = 4.97783
    I1230 18:54:41.021167 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:54:41.021178 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 18:54:41.021190 23363 solver.cpp:253]     Train net output #2: loss_c = 2.02706 (* 1 = 2.02706 loss)
    I1230 18:54:41.021201 23363 solver.cpp:253]     Train net output #3: loss_f = 2.95077 (* 1 = 2.95077 loss)
    I1230 18:54:41.021211 23363 sgd_solver.cpp:106] Iteration 5800, lr = 0.000496713
    I1230 18:54:53.585110 23363 solver.cpp:237] Iteration 5900, loss = 4.50412
    I1230 18:54:53.585150 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 18:54:53.585161 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 18:54:53.585172 23363 solver.cpp:253]     Train net output #2: loss_c = 1.79773 (* 1 = 1.79773 loss)
    I1230 18:54:53.585181 23363 solver.cpp:253]     Train net output #3: loss_f = 2.70639 (* 1 = 2.70639 loss)
    I1230 18:54:53.585191 23363 sgd_solver.cpp:106] Iteration 5900, lr = 0.000494368
    I1230 18:55:06.046293 23363 solver.cpp:341] Iteration 6000, Testing net (#0)
    I1230 18:55:10.733093 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.411
    I1230 18:55:10.733139 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.271333
    I1230 18:55:10.733152 23363 solver.cpp:409]     Test net output #2: loss_c = 1.9271 (* 1 = 1.9271 loss)
    I1230 18:55:10.733162 23363 solver.cpp:409]     Test net output #3: loss_f = 2.91889 (* 1 = 2.91889 loss)
    I1230 18:55:10.790782 23363 solver.cpp:237] Iteration 6000, loss = 4.84409
    I1230 18:55:10.790838 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 18:55:10.790848 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1230 18:55:10.790860 23363 solver.cpp:253]     Train net output #2: loss_c = 1.94181 (* 1 = 1.94181 loss)
    I1230 18:55:10.790873 23363 solver.cpp:253]     Train net output #3: loss_f = 2.90228 (* 1 = 2.90228 loss)
    I1230 18:55:10.790885 23363 sgd_solver.cpp:106] Iteration 6000, lr = 0.000492049
    I1230 18:55:23.403422 23363 solver.cpp:237] Iteration 6100, loss = 5.09579
    I1230 18:55:23.403465 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:55:23.403476 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.22
    I1230 18:55:23.403489 23363 solver.cpp:253]     Train net output #2: loss_c = 2.06523 (* 1 = 2.06523 loss)
    I1230 18:55:23.403501 23363 solver.cpp:253]     Train net output #3: loss_f = 3.03056 (* 1 = 3.03056 loss)
    I1230 18:55:23.403512 23363 sgd_solver.cpp:106] Iteration 6100, lr = 0.000489755
    I1230 18:55:36.000145 23363 solver.cpp:237] Iteration 6200, loss = 5.04198
    I1230 18:55:36.000187 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 18:55:36.000200 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:55:36.000211 23363 solver.cpp:253]     Train net output #2: loss_c = 1.95949 (* 1 = 1.95949 loss)
    I1230 18:55:36.000222 23363 solver.cpp:253]     Train net output #3: loss_f = 3.0825 (* 1 = 3.0825 loss)
    I1230 18:55:36.000233 23363 sgd_solver.cpp:106] Iteration 6200, lr = 0.000487486
    I1230 18:55:48.573055 23363 solver.cpp:237] Iteration 6300, loss = 4.93231
    I1230 18:55:48.573228 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.37
    I1230 18:55:48.573246 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:55:48.573258 23363 solver.cpp:253]     Train net output #2: loss_c = 2.00103 (* 1 = 2.00103 loss)
    I1230 18:55:48.573268 23363 solver.cpp:253]     Train net output #3: loss_f = 2.93128 (* 1 = 2.93128 loss)
    I1230 18:55:48.573281 23363 sgd_solver.cpp:106] Iteration 6300, lr = 0.000485241
    I1230 18:56:01.164453 23363 solver.cpp:237] Iteration 6400, loss = 4.66603
    I1230 18:56:01.164492 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1230 18:56:01.164505 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 18:56:01.164516 23363 solver.cpp:253]     Train net output #2: loss_c = 1.94498 (* 1 = 1.94498 loss)
    I1230 18:56:01.164526 23363 solver.cpp:253]     Train net output #3: loss_f = 2.72105 (* 1 = 2.72105 loss)
    I1230 18:56:01.164538 23363 sgd_solver.cpp:106] Iteration 6400, lr = 0.00048302
    I1230 18:56:13.751406 23363 solver.cpp:237] Iteration 6500, loss = 4.7476
    I1230 18:56:13.751447 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 18:56:13.751458 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1230 18:56:13.751471 23363 solver.cpp:253]     Train net output #2: loss_c = 1.82531 (* 1 = 1.82531 loss)
    I1230 18:56:13.751482 23363 solver.cpp:253]     Train net output #3: loss_f = 2.92228 (* 1 = 2.92228 loss)
    I1230 18:56:13.751492 23363 sgd_solver.cpp:106] Iteration 6500, lr = 0.000480823
    I1230 18:56:26.356456 23363 solver.cpp:237] Iteration 6600, loss = 4.97384
    I1230 18:56:26.356577 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 18:56:26.356595 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 18:56:26.356608 23363 solver.cpp:253]     Train net output #2: loss_c = 2.02764 (* 1 = 2.02764 loss)
    I1230 18:56:26.356619 23363 solver.cpp:253]     Train net output #3: loss_f = 2.94621 (* 1 = 2.94621 loss)
    I1230 18:56:26.356629 23363 sgd_solver.cpp:106] Iteration 6600, lr = 0.000478649
    I1230 18:56:38.941758 23363 solver.cpp:237] Iteration 6700, loss = 4.7181
    I1230 18:56:38.941800 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 18:56:38.941812 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:56:38.941824 23363 solver.cpp:253]     Train net output #2: loss_c = 1.82313 (* 1 = 1.82313 loss)
    I1230 18:56:38.941834 23363 solver.cpp:253]     Train net output #3: loss_f = 2.89497 (* 1 = 2.89497 loss)
    I1230 18:56:38.941845 23363 sgd_solver.cpp:106] Iteration 6700, lr = 0.000476498
    I1230 18:56:51.499776 23363 solver.cpp:237] Iteration 6800, loss = 4.92491
    I1230 18:56:51.499819 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.37
    I1230 18:56:51.499830 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:56:51.499842 23363 solver.cpp:253]     Train net output #2: loss_c = 2.00043 (* 1 = 2.00043 loss)
    I1230 18:56:51.499853 23363 solver.cpp:253]     Train net output #3: loss_f = 2.92448 (* 1 = 2.92448 loss)
    I1230 18:56:51.499863 23363 sgd_solver.cpp:106] Iteration 6800, lr = 0.000474369
    I1230 18:57:05.656189 23363 solver.cpp:237] Iteration 6900, loss = 4.26739
    I1230 18:57:05.656357 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 18:57:05.656373 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 18:57:05.656386 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73245 (* 1 = 1.73245 loss)
    I1230 18:57:05.656396 23363 solver.cpp:253]     Train net output #3: loss_f = 2.53494 (* 1 = 2.53494 loss)
    I1230 18:57:05.656409 23363 sgd_solver.cpp:106] Iteration 6900, lr = 0.000472262
    I1230 18:57:18.136935 23363 solver.cpp:341] Iteration 7000, Testing net (#0)
    I1230 18:57:22.859136 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.426333
    I1230 18:57:22.859174 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.28875
    I1230 18:57:22.859185 23363 solver.cpp:409]     Test net output #2: loss_c = 1.84118 (* 1 = 1.84118 loss)
    I1230 18:57:22.859194 23363 solver.cpp:409]     Test net output #3: loss_f = 2.82153 (* 1 = 2.82153 loss)
    I1230 18:57:22.926460 23363 solver.cpp:237] Iteration 7000, loss = 4.90354
    I1230 18:57:22.926492 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 18:57:22.926501 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:57:22.926511 23363 solver.cpp:253]     Train net output #2: loss_c = 1.96514 (* 1 = 1.96514 loss)
    I1230 18:57:22.926519 23363 solver.cpp:253]     Train net output #3: loss_f = 2.9384 (* 1 = 2.9384 loss)
    I1230 18:57:22.926539 23363 sgd_solver.cpp:106] Iteration 7000, lr = 0.000470177
    I1230 18:57:36.024359 23363 solver.cpp:237] Iteration 7100, loss = 4.8212
    I1230 18:57:36.024530 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 18:57:36.024544 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 18:57:36.024554 23363 solver.cpp:253]     Train net output #2: loss_c = 1.9633 (* 1 = 1.9633 loss)
    I1230 18:57:36.024562 23363 solver.cpp:253]     Train net output #3: loss_f = 2.8579 (* 1 = 2.8579 loss)
    I1230 18:57:36.024580 23363 sgd_solver.cpp:106] Iteration 7100, lr = 0.000468113
    I1230 18:57:49.014106 23363 solver.cpp:237] Iteration 7200, loss = 4.89473
    I1230 18:57:49.014153 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 18:57:49.014163 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.24
    I1230 18:57:49.014171 23363 solver.cpp:253]     Train net output #2: loss_c = 1.88641 (* 1 = 1.88641 loss)
    I1230 18:57:49.014180 23363 solver.cpp:253]     Train net output #3: loss_f = 3.00832 (* 1 = 3.00832 loss)
    I1230 18:57:49.014189 23363 sgd_solver.cpp:106] Iteration 7200, lr = 0.000466071
    I1230 18:58:02.248006 23363 solver.cpp:237] Iteration 7300, loss = 4.73245
    I1230 18:58:02.248060 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 18:58:02.248072 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:58:02.248083 23363 solver.cpp:253]     Train net output #2: loss_c = 1.90439 (* 1 = 1.90439 loss)
    I1230 18:58:02.248093 23363 solver.cpp:253]     Train net output #3: loss_f = 2.82807 (* 1 = 2.82807 loss)
    I1230 18:58:02.248103 23363 sgd_solver.cpp:106] Iteration 7300, lr = 0.000464049
    I1230 18:58:15.565800 23363 solver.cpp:237] Iteration 7400, loss = 4.47914
    I1230 18:58:15.565950 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 18:58:15.565968 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 18:58:15.565980 23363 solver.cpp:253]     Train net output #2: loss_c = 1.86001 (* 1 = 1.86001 loss)
    I1230 18:58:15.565991 23363 solver.cpp:253]     Train net output #3: loss_f = 2.61913 (* 1 = 2.61913 loss)
    I1230 18:58:15.566004 23363 sgd_solver.cpp:106] Iteration 7400, lr = 0.000462047
    I1230 18:58:28.867319 23363 solver.cpp:237] Iteration 7500, loss = 4.64934
    I1230 18:58:28.867362 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 18:58:28.867372 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:58:28.867383 23363 solver.cpp:253]     Train net output #2: loss_c = 1.82507 (* 1 = 1.82507 loss)
    I1230 18:58:28.867393 23363 solver.cpp:253]     Train net output #3: loss_f = 2.82427 (* 1 = 2.82427 loss)
    I1230 18:58:28.867403 23363 sgd_solver.cpp:106] Iteration 7500, lr = 0.000460065
    I1230 18:58:42.505550 23363 solver.cpp:237] Iteration 7600, loss = 5.08576
    I1230 18:58:42.505589 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1230 18:58:42.505600 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:58:42.505611 23363 solver.cpp:253]     Train net output #2: loss_c = 2.09359 (* 1 = 2.09359 loss)
    I1230 18:58:42.505620 23363 solver.cpp:253]     Train net output #3: loss_f = 2.99217 (* 1 = 2.99217 loss)
    I1230 18:58:42.505631 23363 sgd_solver.cpp:106] Iteration 7600, lr = 0.000458103
    I1230 18:58:55.968354 23363 solver.cpp:237] Iteration 7700, loss = 4.49872
    I1230 18:58:55.968534 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 18:58:55.968549 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 18:58:55.968559 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71172 (* 1 = 1.71172 loss)
    I1230 18:58:55.968567 23363 solver.cpp:253]     Train net output #3: loss_f = 2.78699 (* 1 = 2.78699 loss)
    I1230 18:58:55.968585 23363 sgd_solver.cpp:106] Iteration 7700, lr = 0.000456161
    I1230 18:59:09.064865 23363 solver.cpp:237] Iteration 7800, loss = 4.85473
    I1230 18:59:09.064905 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 18:59:09.064916 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 18:59:09.064926 23363 solver.cpp:253]     Train net output #2: loss_c = 2.01194 (* 1 = 2.01194 loss)
    I1230 18:59:09.064935 23363 solver.cpp:253]     Train net output #3: loss_f = 2.8428 (* 1 = 2.8428 loss)
    I1230 18:59:09.064945 23363 sgd_solver.cpp:106] Iteration 7800, lr = 0.000454238
    I1230 18:59:22.424520 23363 solver.cpp:237] Iteration 7900, loss = 4.11836
    I1230 18:59:22.424577 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 18:59:22.424595 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 18:59:22.424614 23363 solver.cpp:253]     Train net output #2: loss_c = 1.65672 (* 1 = 1.65672 loss)
    I1230 18:59:22.424630 23363 solver.cpp:253]     Train net output #3: loss_f = 2.46163 (* 1 = 2.46163 loss)
    I1230 18:59:22.424648 23363 sgd_solver.cpp:106] Iteration 7900, lr = 0.000452333
    I1230 18:59:36.012049 23363 solver.cpp:341] Iteration 8000, Testing net (#0)
    I1230 18:59:42.611726 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.43625
    I1230 18:59:42.611795 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.29625
    I1230 18:59:42.611809 23363 solver.cpp:409]     Test net output #2: loss_c = 1.80374 (* 1 = 1.80374 loss)
    I1230 18:59:42.611821 23363 solver.cpp:409]     Test net output #3: loss_f = 2.76596 (* 1 = 2.76596 loss)
    I1230 18:59:42.709667 23363 solver.cpp:237] Iteration 8000, loss = 4.5996
    I1230 18:59:42.709717 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 18:59:42.709728 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 18:59:42.709738 23363 solver.cpp:253]     Train net output #2: loss_c = 1.77818 (* 1 = 1.77818 loss)
    I1230 18:59:42.709748 23363 solver.cpp:253]     Train net output #3: loss_f = 2.82142 (* 1 = 2.82142 loss)
    I1230 18:59:42.709759 23363 sgd_solver.cpp:106] Iteration 8000, lr = 0.000450447
    I1230 18:59:56.631276 23363 solver.cpp:237] Iteration 8100, loss = 4.64826
    I1230 18:59:56.631322 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 18:59:56.631332 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 18:59:56.631342 23363 solver.cpp:253]     Train net output #2: loss_c = 1.84945 (* 1 = 1.84945 loss)
    I1230 18:59:56.631351 23363 solver.cpp:253]     Train net output #3: loss_f = 2.79881 (* 1 = 2.79881 loss)
    I1230 18:59:56.631361 23363 sgd_solver.cpp:106] Iteration 8100, lr = 0.000448579
    I1230 19:00:10.743047 23363 solver.cpp:237] Iteration 8200, loss = 4.78869
    I1230 19:00:10.743171 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 19:00:10.743185 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1230 19:00:10.743197 23363 solver.cpp:253]     Train net output #2: loss_c = 1.86079 (* 1 = 1.86079 loss)
    I1230 19:00:10.743207 23363 solver.cpp:253]     Train net output #3: loss_f = 2.9279 (* 1 = 2.9279 loss)
    I1230 19:00:10.743216 23363 sgd_solver.cpp:106] Iteration 8200, lr = 0.000446729
    I1230 19:00:23.844372 23363 solver.cpp:237] Iteration 8300, loss = 4.53778
    I1230 19:00:23.844429 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:00:23.844442 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:00:23.844456 23363 solver.cpp:253]     Train net output #2: loss_c = 1.83164 (* 1 = 1.83164 loss)
    I1230 19:00:23.844467 23363 solver.cpp:253]     Train net output #3: loss_f = 2.70613 (* 1 = 2.70613 loss)
    I1230 19:00:23.844478 23363 sgd_solver.cpp:106] Iteration 8300, lr = 0.000444897
    I1230 19:00:37.285648 23363 solver.cpp:237] Iteration 8400, loss = 4.05245
    I1230 19:00:37.285686 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:00:37.285696 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:00:37.285706 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63803 (* 1 = 1.63803 loss)
    I1230 19:00:37.285714 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41442 (* 1 = 2.41442 loss)
    I1230 19:00:37.285724 23363 sgd_solver.cpp:106] Iteration 8400, lr = 0.000443083
    I1230 19:00:49.935278 23363 solver.cpp:237] Iteration 8500, loss = 4.57974
    I1230 19:00:49.935451 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:00:49.935475 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 19:00:49.935494 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75536 (* 1 = 1.75536 loss)
    I1230 19:00:49.935511 23363 solver.cpp:253]     Train net output #3: loss_f = 2.82438 (* 1 = 2.82438 loss)
    I1230 19:00:49.935528 23363 sgd_solver.cpp:106] Iteration 8500, lr = 0.000441285
    I1230 19:01:02.501798 23363 solver.cpp:237] Iteration 8600, loss = 4.42117
    I1230 19:01:02.501845 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:01:02.501854 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:01:02.501864 23363 solver.cpp:253]     Train net output #2: loss_c = 1.81288 (* 1 = 1.81288 loss)
    I1230 19:01:02.501873 23363 solver.cpp:253]     Train net output #3: loss_f = 2.60829 (* 1 = 2.60829 loss)
    I1230 19:01:02.501881 23363 sgd_solver.cpp:106] Iteration 8600, lr = 0.000439505
    I1230 19:01:15.080929 23363 solver.cpp:237] Iteration 8700, loss = 4.46285
    I1230 19:01:15.080965 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:01:15.080976 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:01:15.080986 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71484 (* 1 = 1.71484 loss)
    I1230 19:01:15.080994 23363 solver.cpp:253]     Train net output #3: loss_f = 2.74801 (* 1 = 2.74801 loss)
    I1230 19:01:15.081003 23363 sgd_solver.cpp:106] Iteration 8700, lr = 0.000437741
    I1230 19:01:27.585119 23363 solver.cpp:237] Iteration 8800, loss = 4.51533
    I1230 19:01:27.585275 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1230 19:01:27.585288 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:01:27.585297 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85079 (* 1 = 1.85079 loss)
    I1230 19:01:27.585304 23363 solver.cpp:253]     Train net output #3: loss_f = 2.66454 (* 1 = 2.66454 loss)
    I1230 19:01:27.585314 23363 sgd_solver.cpp:106] Iteration 8800, lr = 0.000435993
    I1230 19:01:40.142513 23363 solver.cpp:237] Iteration 8900, loss = 4.14741
    I1230 19:01:40.142550 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:01:40.142560 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:01:40.142570 23363 solver.cpp:253]     Train net output #2: loss_c = 1.68913 (* 1 = 1.68913 loss)
    I1230 19:01:40.142578 23363 solver.cpp:253]     Train net output #3: loss_f = 2.45829 (* 1 = 2.45829 loss)
    I1230 19:01:40.142588 23363 sgd_solver.cpp:106] Iteration 8900, lr = 0.000434262
    I1230 19:01:52.566551 23363 solver.cpp:341] Iteration 9000, Testing net (#0)
    I1230 19:01:57.252904 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.443917
    I1230 19:01:57.252943 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.302583
    I1230 19:01:57.252954 23363 solver.cpp:409]     Test net output #2: loss_c = 1.79153 (* 1 = 1.79153 loss)
    I1230 19:01:57.252962 23363 solver.cpp:409]     Test net output #3: loss_f = 2.73411 (* 1 = 2.73411 loss)
    I1230 19:01:57.315235 23363 solver.cpp:237] Iteration 9000, loss = 4.59276
    I1230 19:01:57.315270 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:01:57.315279 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:01:57.315289 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85679 (* 1 = 1.85679 loss)
    I1230 19:01:57.315299 23363 solver.cpp:253]     Train net output #3: loss_f = 2.73597 (* 1 = 2.73597 loss)
    I1230 19:01:57.315309 23363 sgd_solver.cpp:106] Iteration 9000, lr = 0.000432547
    I1230 19:02:09.882243 23363 solver.cpp:237] Iteration 9100, loss = 4.66457
    I1230 19:02:09.882402 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:02:09.882414 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:02:09.882424 23363 solver.cpp:253]     Train net output #2: loss_c = 1.89589 (* 1 = 1.89589 loss)
    I1230 19:02:09.882432 23363 solver.cpp:253]     Train net output #3: loss_f = 2.76868 (* 1 = 2.76868 loss)
    I1230 19:02:09.882441 23363 sgd_solver.cpp:106] Iteration 9100, lr = 0.000430847
    I1230 19:02:22.387567 23363 solver.cpp:237] Iteration 9200, loss = 4.70863
    I1230 19:02:22.387603 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:02:22.387614 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 19:02:22.387624 23363 solver.cpp:253]     Train net output #2: loss_c = 1.84843 (* 1 = 1.84843 loss)
    I1230 19:02:22.387631 23363 solver.cpp:253]     Train net output #3: loss_f = 2.8602 (* 1 = 2.8602 loss)
    I1230 19:02:22.387641 23363 sgd_solver.cpp:106] Iteration 9200, lr = 0.000429163
    I1230 19:02:34.973098 23363 solver.cpp:237] Iteration 9300, loss = 4.55583
    I1230 19:02:34.973146 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:02:34.973156 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:02:34.973166 23363 solver.cpp:253]     Train net output #2: loss_c = 1.84471 (* 1 = 1.84471 loss)
    I1230 19:02:34.973176 23363 solver.cpp:253]     Train net output #3: loss_f = 2.71112 (* 1 = 2.71112 loss)
    I1230 19:02:34.973186 23363 sgd_solver.cpp:106] Iteration 9300, lr = 0.000427494
    I1230 19:02:48.251693 23363 solver.cpp:237] Iteration 9400, loss = 4.25985
    I1230 19:02:48.251806 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:02:48.251821 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:02:48.251832 23363 solver.cpp:253]     Train net output #2: loss_c = 1.76269 (* 1 = 1.76269 loss)
    I1230 19:02:48.251842 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49717 (* 1 = 2.49717 loss)
    I1230 19:02:48.251852 23363 sgd_solver.cpp:106] Iteration 9400, lr = 0.00042584
    I1230 19:03:01.513085 23363 solver.cpp:237] Iteration 9500, loss = 4.37748
    I1230 19:03:01.513123 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:03:01.513134 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:03:01.513144 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73265 (* 1 = 1.73265 loss)
    I1230 19:03:01.513152 23363 solver.cpp:253]     Train net output #3: loss_f = 2.64483 (* 1 = 2.64483 loss)
    I1230 19:03:01.513162 23363 sgd_solver.cpp:106] Iteration 9500, lr = 0.000424201
    I1230 19:03:14.127945 23363 solver.cpp:237] Iteration 9600, loss = 4.81147
    I1230 19:03:14.128005 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:03:14.128021 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 19:03:14.128041 23363 solver.cpp:253]     Train net output #2: loss_c = 1.93244 (* 1 = 1.93244 loss)
    I1230 19:03:14.128057 23363 solver.cpp:253]     Train net output #3: loss_f = 2.87903 (* 1 = 2.87903 loss)
    I1230 19:03:14.128072 23363 sgd_solver.cpp:106] Iteration 9600, lr = 0.000422577
    I1230 19:03:26.683648 23363 solver.cpp:237] Iteration 9700, loss = 4.49339
    I1230 19:03:26.683902 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:03:26.683917 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.26
    I1230 19:03:26.683928 23363 solver.cpp:253]     Train net output #2: loss_c = 1.69601 (* 1 = 1.69601 loss)
    I1230 19:03:26.683936 23363 solver.cpp:253]     Train net output #3: loss_f = 2.79738 (* 1 = 2.79738 loss)
    I1230 19:03:26.683945 23363 sgd_solver.cpp:106] Iteration 9700, lr = 0.000420967
    I1230 19:03:39.237118 23363 solver.cpp:237] Iteration 9800, loss = 4.52676
    I1230 19:03:39.237156 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 19:03:39.237165 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1230 19:03:39.237175 23363 solver.cpp:253]     Train net output #2: loss_c = 1.83433 (* 1 = 1.83433 loss)
    I1230 19:03:39.237184 23363 solver.cpp:253]     Train net output #3: loss_f = 2.69242 (* 1 = 2.69242 loss)
    I1230 19:03:39.237192 23363 sgd_solver.cpp:106] Iteration 9800, lr = 0.000419372
    I1230 19:03:51.821562 23363 solver.cpp:237] Iteration 9900, loss = 3.99797
    I1230 19:03:51.821607 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:03:51.821617 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:03:51.821629 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6046 (* 1 = 1.6046 loss)
    I1230 19:03:51.821636 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39337 (* 1 = 2.39337 loss)
    I1230 19:03:51.821646 23363 sgd_solver.cpp:106] Iteration 9900, lr = 0.00041779
    I1230 19:04:04.310377 23363 solver.cpp:341] Iteration 10000, Testing net (#0)
    I1230 19:04:09.033440 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.454583
    I1230 19:04:09.033488 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.311
    I1230 19:04:09.033500 23363 solver.cpp:409]     Test net output #2: loss_c = 1.75141 (* 1 = 1.75141 loss)
    I1230 19:04:09.033510 23363 solver.cpp:409]     Test net output #3: loss_f = 2.68382 (* 1 = 2.68382 loss)
    I1230 19:04:09.090920 23363 solver.cpp:237] Iteration 10000, loss = 4.34446
    I1230 19:04:09.090971 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:04:09.091003 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:04:09.091017 23363 solver.cpp:253]     Train net output #2: loss_c = 1.65145 (* 1 = 1.65145 loss)
    I1230 19:04:09.091028 23363 solver.cpp:253]     Train net output #3: loss_f = 2.69301 (* 1 = 2.69301 loss)
    I1230 19:04:09.091042 23363 sgd_solver.cpp:106] Iteration 10000, lr = 0.000416222
    I1230 19:04:21.730368 23363 solver.cpp:237] Iteration 10100, loss = 4.73584
    I1230 19:04:21.730412 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:04:21.730424 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:04:21.730437 23363 solver.cpp:253]     Train net output #2: loss_c = 1.90236 (* 1 = 1.90236 loss)
    I1230 19:04:21.730448 23363 solver.cpp:253]     Train net output #3: loss_f = 2.83348 (* 1 = 2.83348 loss)
    I1230 19:04:21.730459 23363 sgd_solver.cpp:106] Iteration 10100, lr = 0.000414668
    I1230 19:04:34.244562 23363 solver.cpp:237] Iteration 10200, loss = 4.74118
    I1230 19:04:34.244608 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:04:34.244619 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:04:34.244629 23363 solver.cpp:253]     Train net output #2: loss_c = 1.83179 (* 1 = 1.83179 loss)
    I1230 19:04:34.244638 23363 solver.cpp:253]     Train net output #3: loss_f = 2.90939 (* 1 = 2.90939 loss)
    I1230 19:04:34.244647 23363 sgd_solver.cpp:106] Iteration 10200, lr = 0.000413128
    I1230 19:04:46.805227 23363 solver.cpp:237] Iteration 10300, loss = 4.49609
    I1230 19:04:46.805402 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 19:04:46.805424 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:04:46.805435 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85121 (* 1 = 1.85121 loss)
    I1230 19:04:46.805444 23363 solver.cpp:253]     Train net output #3: loss_f = 2.64488 (* 1 = 2.64488 loss)
    I1230 19:04:46.805452 23363 sgd_solver.cpp:106] Iteration 10300, lr = 0.000411601
    I1230 19:04:59.328805 23363 solver.cpp:237] Iteration 10400, loss = 3.92966
    I1230 19:04:59.328843 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:04:59.328855 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:04:59.328865 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59201 (* 1 = 1.59201 loss)
    I1230 19:04:59.328872 23363 solver.cpp:253]     Train net output #3: loss_f = 2.33765 (* 1 = 2.33765 loss)
    I1230 19:04:59.328882 23363 sgd_solver.cpp:106] Iteration 10400, lr = 0.000410086
    I1230 19:05:11.905987 23363 solver.cpp:237] Iteration 10500, loss = 4.58812
    I1230 19:05:11.906025 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:05:11.906035 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 19:05:11.906044 23363 solver.cpp:253]     Train net output #2: loss_c = 1.77228 (* 1 = 1.77228 loss)
    I1230 19:05:11.906054 23363 solver.cpp:253]     Train net output #3: loss_f = 2.81584 (* 1 = 2.81584 loss)
    I1230 19:05:11.906062 23363 sgd_solver.cpp:106] Iteration 10500, lr = 0.000408585
    I1230 19:05:24.520833 23363 solver.cpp:237] Iteration 10600, loss = 4.60772
    I1230 19:05:24.520939 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:05:24.520954 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 19:05:24.520964 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85869 (* 1 = 1.85869 loss)
    I1230 19:05:24.520972 23363 solver.cpp:253]     Train net output #3: loss_f = 2.74904 (* 1 = 2.74904 loss)
    I1230 19:05:24.520979 23363 sgd_solver.cpp:106] Iteration 10600, lr = 0.000407097
    I1230 19:05:37.118717 23363 solver.cpp:237] Iteration 10700, loss = 4.67384
    I1230 19:05:37.118762 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:05:37.118777 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:05:37.118791 23363 solver.cpp:253]     Train net output #2: loss_c = 1.78239 (* 1 = 1.78239 loss)
    I1230 19:05:37.118803 23363 solver.cpp:253]     Train net output #3: loss_f = 2.89145 (* 1 = 2.89145 loss)
    I1230 19:05:37.118814 23363 sgd_solver.cpp:106] Iteration 10700, lr = 0.000405621
    I1230 19:05:49.718430 23363 solver.cpp:237] Iteration 10800, loss = 4.61201
    I1230 19:05:49.718475 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 19:05:49.718484 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 19:05:49.718494 23363 solver.cpp:253]     Train net output #2: loss_c = 1.86271 (* 1 = 1.86271 loss)
    I1230 19:05:49.718503 23363 solver.cpp:253]     Train net output #3: loss_f = 2.7493 (* 1 = 2.7493 loss)
    I1230 19:05:49.718513 23363 sgd_solver.cpp:106] Iteration 10800, lr = 0.000404157
    I1230 19:06:02.282021 23363 solver.cpp:237] Iteration 10900, loss = 4.00983
    I1230 19:06:02.282182 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:06:02.282201 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:06:02.282212 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6172 (* 1 = 1.6172 loss)
    I1230 19:06:02.282222 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39263 (* 1 = 2.39263 loss)
    I1230 19:06:02.282244 23363 sgd_solver.cpp:106] Iteration 10900, lr = 0.000402706
    I1230 19:06:14.796644 23363 solver.cpp:341] Iteration 11000, Testing net (#0)
    I1230 19:06:19.500902 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.45175
    I1230 19:06:19.500946 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.320417
    I1230 19:06:19.500960 23363 solver.cpp:409]     Test net output #2: loss_c = 1.75761 (* 1 = 1.75761 loss)
    I1230 19:06:19.500973 23363 solver.cpp:409]     Test net output #3: loss_f = 2.66831 (* 1 = 2.66831 loss)
    I1230 19:06:19.558490 23363 solver.cpp:237] Iteration 11000, loss = 4.61613
    I1230 19:06:19.558534 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:06:19.558547 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:06:19.558562 23363 solver.cpp:253]     Train net output #2: loss_c = 1.84319 (* 1 = 1.84319 loss)
    I1230 19:06:19.558574 23363 solver.cpp:253]     Train net output #3: loss_f = 2.77293 (* 1 = 2.77293 loss)
    I1230 19:06:19.558586 23363 sgd_solver.cpp:106] Iteration 11000, lr = 0.000401267
    I1230 19:06:32.135931 23363 solver.cpp:237] Iteration 11100, loss = 4.55343
    I1230 19:06:32.135967 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:06:32.135977 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:06:32.135987 23363 solver.cpp:253]     Train net output #2: loss_c = 1.86866 (* 1 = 1.86866 loss)
    I1230 19:06:32.135998 23363 solver.cpp:253]     Train net output #3: loss_f = 2.68477 (* 1 = 2.68477 loss)
    I1230 19:06:32.136006 23363 sgd_solver.cpp:106] Iteration 11100, lr = 0.00039984
    I1230 19:06:44.724581 23363 solver.cpp:237] Iteration 11200, loss = 4.52236
    I1230 19:06:44.724730 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:06:44.724745 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 19:06:44.724756 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73511 (* 1 = 1.73511 loss)
    I1230 19:06:44.724767 23363 solver.cpp:253]     Train net output #3: loss_f = 2.78725 (* 1 = 2.78725 loss)
    I1230 19:06:44.724777 23363 sgd_solver.cpp:106] Iteration 11200, lr = 0.000398425
    I1230 19:06:57.312041 23363 solver.cpp:237] Iteration 11300, loss = 4.53106
    I1230 19:06:57.312083 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 19:06:57.312096 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:06:57.312109 23363 solver.cpp:253]     Train net output #2: loss_c = 1.87846 (* 1 = 1.87846 loss)
    I1230 19:06:57.312120 23363 solver.cpp:253]     Train net output #3: loss_f = 2.6526 (* 1 = 2.6526 loss)
    I1230 19:06:57.312131 23363 sgd_solver.cpp:106] Iteration 11300, lr = 0.000397021
    I1230 19:07:09.921035 23363 solver.cpp:237] Iteration 11400, loss = 3.79257
    I1230 19:07:09.921082 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:07:09.921092 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:07:09.921103 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52693 (* 1 = 1.52693 loss)
    I1230 19:07:09.921113 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26564 (* 1 = 2.26564 loss)
    I1230 19:07:09.921133 23363 sgd_solver.cpp:106] Iteration 11400, lr = 0.000395629
    I1230 19:07:22.499800 23363 solver.cpp:237] Iteration 11500, loss = 4.40711
    I1230 19:07:22.499941 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:07:22.499964 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:07:22.499984 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71765 (* 1 = 1.71765 loss)
    I1230 19:07:22.500001 23363 solver.cpp:253]     Train net output #3: loss_f = 2.68946 (* 1 = 2.68946 loss)
    I1230 19:07:22.500017 23363 sgd_solver.cpp:106] Iteration 11500, lr = 0.000394248
    I1230 19:07:35.129896 23363 solver.cpp:237] Iteration 11600, loss = 4.40865
    I1230 19:07:35.129933 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:07:35.129945 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:07:35.129956 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73793 (* 1 = 1.73793 loss)
    I1230 19:07:35.129966 23363 solver.cpp:253]     Train net output #3: loss_f = 2.67072 (* 1 = 2.67072 loss)
    I1230 19:07:35.129976 23363 sgd_solver.cpp:106] Iteration 11600, lr = 0.000392878
    I1230 19:07:47.767457 23363 solver.cpp:237] Iteration 11700, loss = 4.40352
    I1230 19:07:47.767493 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:07:47.767503 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:07:47.767513 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70663 (* 1 = 1.70663 loss)
    I1230 19:07:47.767524 23363 solver.cpp:253]     Train net output #3: loss_f = 2.69689 (* 1 = 2.69689 loss)
    I1230 19:07:47.767532 23363 sgd_solver.cpp:106] Iteration 11700, lr = 0.000391519
    I1230 19:08:06.511163 23363 solver.cpp:237] Iteration 11800, loss = 4.54006
    I1230 19:08:06.511373 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:08:06.511392 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 19:08:06.511404 23363 solver.cpp:253]     Train net output #2: loss_c = 1.8309 (* 1 = 1.8309 loss)
    I1230 19:08:06.511415 23363 solver.cpp:253]     Train net output #3: loss_f = 2.70915 (* 1 = 2.70915 loss)
    I1230 19:08:06.511425 23363 sgd_solver.cpp:106] Iteration 11800, lr = 0.000390172
    I1230 19:08:20.079463 23363 solver.cpp:237] Iteration 11900, loss = 4.006
    I1230 19:08:20.079516 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:08:20.079535 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:08:20.079552 23363 solver.cpp:253]     Train net output #2: loss_c = 1.65735 (* 1 = 1.65735 loss)
    I1230 19:08:20.079568 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34866 (* 1 = 2.34866 loss)
    I1230 19:08:20.079586 23363 sgd_solver.cpp:106] Iteration 11900, lr = 0.000388835
    I1230 19:08:33.508924 23363 solver.cpp:341] Iteration 12000, Testing net (#0)
    I1230 19:08:38.649328 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.48175
    I1230 19:08:38.649437 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.34025
    I1230 19:08:38.649458 23363 solver.cpp:409]     Test net output #2: loss_c = 1.65793 (* 1 = 1.65793 loss)
    I1230 19:08:38.649471 23363 solver.cpp:409]     Test net output #3: loss_f = 2.55638 (* 1 = 2.55638 loss)
    I1230 19:08:38.725561 23363 solver.cpp:237] Iteration 12000, loss = 4.07187
    I1230 19:08:38.725606 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:08:38.725620 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:08:38.725636 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55092 (* 1 = 1.55092 loss)
    I1230 19:08:38.725647 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52095 (* 1 = 2.52095 loss)
    I1230 19:08:38.725661 23363 sgd_solver.cpp:106] Iteration 12000, lr = 0.000387508
    I1230 19:08:52.262689 23363 solver.cpp:237] Iteration 12100, loss = 4.6739
    I1230 19:08:52.262727 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:08:52.262737 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:08:52.262751 23363 solver.cpp:253]     Train net output #2: loss_c = 1.88641 (* 1 = 1.88641 loss)
    I1230 19:08:52.262763 23363 solver.cpp:253]     Train net output #3: loss_f = 2.7875 (* 1 = 2.7875 loss)
    I1230 19:08:52.262773 23363 sgd_solver.cpp:106] Iteration 12100, lr = 0.000386192
    I1230 19:09:05.854622 23363 solver.cpp:237] Iteration 12200, loss = 4.30223
    I1230 19:09:05.854670 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:09:05.854683 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:09:05.854697 23363 solver.cpp:253]     Train net output #2: loss_c = 1.65956 (* 1 = 1.65956 loss)
    I1230 19:09:05.854709 23363 solver.cpp:253]     Train net output #3: loss_f = 2.64267 (* 1 = 2.64267 loss)
    I1230 19:09:05.854722 23363 sgd_solver.cpp:106] Iteration 12200, lr = 0.000384887
    I1230 19:09:19.400482 23363 solver.cpp:237] Iteration 12300, loss = 4.43381
    I1230 19:09:19.400620 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1230 19:09:19.400638 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 19:09:19.400651 23363 solver.cpp:253]     Train net output #2: loss_c = 1.84291 (* 1 = 1.84291 loss)
    I1230 19:09:19.400662 23363 solver.cpp:253]     Train net output #3: loss_f = 2.59091 (* 1 = 2.59091 loss)
    I1230 19:09:19.400673 23363 sgd_solver.cpp:106] Iteration 12300, lr = 0.000383592
    I1230 19:09:32.889272 23363 solver.cpp:237] Iteration 12400, loss = 3.93479
    I1230 19:09:32.889310 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:09:32.889322 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 19:09:32.889336 23363 solver.cpp:253]     Train net output #2: loss_c = 1.60107 (* 1 = 1.60107 loss)
    I1230 19:09:32.889348 23363 solver.cpp:253]     Train net output #3: loss_f = 2.33372 (* 1 = 2.33372 loss)
    I1230 19:09:32.889360 23363 sgd_solver.cpp:106] Iteration 12400, lr = 0.000382307
    I1230 19:09:46.396675 23363 solver.cpp:237] Iteration 12500, loss = 4.2071
    I1230 19:09:46.396720 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:09:46.396731 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:09:46.396742 23363 solver.cpp:253]     Train net output #2: loss_c = 1.66036 (* 1 = 1.66036 loss)
    I1230 19:09:46.396751 23363 solver.cpp:253]     Train net output #3: loss_f = 2.54674 (* 1 = 2.54674 loss)
    I1230 19:09:46.396761 23363 sgd_solver.cpp:106] Iteration 12500, lr = 0.000381032
    I1230 19:09:59.889528 23363 solver.cpp:237] Iteration 12600, loss = 4.42813
    I1230 19:09:59.889698 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:09:59.889721 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:09:59.889734 23363 solver.cpp:253]     Train net output #2: loss_c = 1.79854 (* 1 = 1.79854 loss)
    I1230 19:09:59.889742 23363 solver.cpp:253]     Train net output #3: loss_f = 2.62959 (* 1 = 2.62959 loss)
    I1230 19:09:59.889751 23363 sgd_solver.cpp:106] Iteration 12600, lr = 0.000379767
    I1230 19:10:13.432409 23363 solver.cpp:237] Iteration 12700, loss = 4.23503
    I1230 19:10:13.432467 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:10:13.432490 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:10:13.432517 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59977 (* 1 = 1.59977 loss)
    I1230 19:10:13.432538 23363 solver.cpp:253]     Train net output #3: loss_f = 2.63526 (* 1 = 2.63526 loss)
    I1230 19:10:13.432559 23363 sgd_solver.cpp:106] Iteration 12700, lr = 0.000378511
    I1230 19:10:27.044522 23363 solver.cpp:237] Iteration 12800, loss = 4.53326
    I1230 19:10:27.044567 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.37
    I1230 19:10:27.044577 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 19:10:27.044589 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85959 (* 1 = 1.85959 loss)
    I1230 19:10:27.044597 23363 solver.cpp:253]     Train net output #3: loss_f = 2.67367 (* 1 = 2.67367 loss)
    I1230 19:10:27.044607 23363 sgd_solver.cpp:106] Iteration 12800, lr = 0.000377265
    I1230 19:10:40.473956 23363 solver.cpp:237] Iteration 12900, loss = 4.09448
    I1230 19:10:40.474081 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:10:40.474092 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:10:40.474102 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70464 (* 1 = 1.70464 loss)
    I1230 19:10:40.474112 23363 solver.cpp:253]     Train net output #3: loss_f = 2.38983 (* 1 = 2.38983 loss)
    I1230 19:10:40.474120 23363 sgd_solver.cpp:106] Iteration 12900, lr = 0.000376029
    I1230 19:10:53.933639 23363 solver.cpp:341] Iteration 13000, Testing net (#0)
    I1230 19:10:59.444403 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.472417
    I1230 19:10:59.444453 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.335333
    I1230 19:10:59.444466 23363 solver.cpp:409]     Test net output #2: loss_c = 1.70772 (* 1 = 1.70772 loss)
    I1230 19:10:59.444478 23363 solver.cpp:409]     Test net output #3: loss_f = 2.6056 (* 1 = 2.6056 loss)
    I1230 19:10:59.503428 23363 solver.cpp:237] Iteration 13000, loss = 4.3508
    I1230 19:10:59.503473 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:10:59.503486 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:10:59.503501 23363 solver.cpp:253]     Train net output #2: loss_c = 1.72542 (* 1 = 1.72542 loss)
    I1230 19:10:59.503514 23363 solver.cpp:253]     Train net output #3: loss_f = 2.62538 (* 1 = 2.62538 loss)
    I1230 19:10:59.503526 23363 sgd_solver.cpp:106] Iteration 13000, lr = 0.000374802
    I1230 19:11:12.645072 23363 solver.cpp:237] Iteration 13100, loss = 4.48682
    I1230 19:11:12.645222 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:11:12.645238 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:11:12.645251 23363 solver.cpp:253]     Train net output #2: loss_c = 1.82449 (* 1 = 1.82449 loss)
    I1230 19:11:12.645261 23363 solver.cpp:253]     Train net output #3: loss_f = 2.66233 (* 1 = 2.66233 loss)
    I1230 19:11:12.645270 23363 sgd_solver.cpp:106] Iteration 13100, lr = 0.000373585
    I1230 19:11:27.233325 23363 solver.cpp:237] Iteration 13200, loss = 4.38658
    I1230 19:11:27.233363 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:11:27.233374 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1230 19:11:27.233386 23363 solver.cpp:253]     Train net output #2: loss_c = 1.68172 (* 1 = 1.68172 loss)
    I1230 19:11:27.233396 23363 solver.cpp:253]     Train net output #3: loss_f = 2.70485 (* 1 = 2.70485 loss)
    I1230 19:11:27.233404 23363 sgd_solver.cpp:106] Iteration 13200, lr = 0.000372376
    I1230 19:11:40.046030 23363 solver.cpp:237] Iteration 13300, loss = 4.29107
    I1230 19:11:40.046072 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:11:40.046083 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:11:40.046097 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75232 (* 1 = 1.75232 loss)
    I1230 19:11:40.046108 23363 solver.cpp:253]     Train net output #3: loss_f = 2.53875 (* 1 = 2.53875 loss)
    I1230 19:11:40.046118 23363 sgd_solver.cpp:106] Iteration 13300, lr = 0.000371177
    I1230 19:11:53.096134 23363 solver.cpp:237] Iteration 13400, loss = 3.96742
    I1230 19:11:53.096278 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:11:53.096303 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:11:53.096318 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62013 (* 1 = 1.62013 loss)
    I1230 19:11:53.096330 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34728 (* 1 = 2.34728 loss)
    I1230 19:11:53.096343 23363 sgd_solver.cpp:106] Iteration 13400, lr = 0.000369987
    I1230 19:12:06.309654 23363 solver.cpp:237] Iteration 13500, loss = 4.45282
    I1230 19:12:06.309711 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 19:12:06.309731 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 19:12:06.309751 23363 solver.cpp:253]     Train net output #2: loss_c = 1.78013 (* 1 = 1.78013 loss)
    I1230 19:12:06.309769 23363 solver.cpp:253]     Train net output #3: loss_f = 2.67269 (* 1 = 2.67269 loss)
    I1230 19:12:06.309787 23363 sgd_solver.cpp:106] Iteration 13500, lr = 0.000368805
    I1230 19:12:19.252997 23363 solver.cpp:237] Iteration 13600, loss = 4.58833
    I1230 19:12:19.253041 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:12:19.253051 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:12:19.253062 23363 solver.cpp:253]     Train net output #2: loss_c = 1.85787 (* 1 = 1.85787 loss)
    I1230 19:12:19.253072 23363 solver.cpp:253]     Train net output #3: loss_f = 2.73046 (* 1 = 2.73046 loss)
    I1230 19:12:19.253090 23363 sgd_solver.cpp:106] Iteration 13600, lr = 0.000367633
    I1230 19:12:34.248030 23363 solver.cpp:237] Iteration 13700, loss = 4.16996
    I1230 19:12:34.248152 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:12:34.248165 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:12:34.248178 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5492 (* 1 = 1.5492 loss)
    I1230 19:12:34.248186 23363 solver.cpp:253]     Train net output #3: loss_f = 2.62075 (* 1 = 2.62075 loss)
    I1230 19:12:34.248196 23363 sgd_solver.cpp:106] Iteration 13700, lr = 0.000366469
    I1230 19:12:47.242650 23363 solver.cpp:237] Iteration 13800, loss = 4.23273
    I1230 19:12:47.242689 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 19:12:47.242702 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:12:47.242715 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75559 (* 1 = 1.75559 loss)
    I1230 19:12:47.242727 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47714 (* 1 = 2.47714 loss)
    I1230 19:12:47.242738 23363 sgd_solver.cpp:106] Iteration 13800, lr = 0.000365313
    I1230 19:13:00.417023 23363 solver.cpp:237] Iteration 13900, loss = 3.79056
    I1230 19:13:00.417068 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:13:00.417078 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:13:00.417088 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56084 (* 1 = 1.56084 loss)
    I1230 19:13:00.417096 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22972 (* 1 = 2.22972 loss)
    I1230 19:13:00.417106 23363 sgd_solver.cpp:106] Iteration 13900, lr = 0.000364166
    I1230 19:13:13.571627 23363 solver.cpp:341] Iteration 14000, Testing net (#0)
    I1230 19:13:18.303647 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.4805
    I1230 19:13:18.303691 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.355
    I1230 19:13:18.303704 23363 solver.cpp:409]     Test net output #2: loss_c = 1.64064 (* 1 = 1.64064 loss)
    I1230 19:13:18.303715 23363 solver.cpp:409]     Test net output #3: loss_f = 2.51601 (* 1 = 2.51601 loss)
    I1230 19:13:18.376399 23363 solver.cpp:237] Iteration 14000, loss = 4.0602
    I1230 19:13:18.376438 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:13:18.376451 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:13:18.376463 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59644 (* 1 = 1.59644 loss)
    I1230 19:13:18.376474 23363 solver.cpp:253]     Train net output #3: loss_f = 2.46375 (* 1 = 2.46375 loss)
    I1230 19:13:18.376487 23363 sgd_solver.cpp:106] Iteration 14000, lr = 0.000363028
    I1230 19:13:31.398483 23363 solver.cpp:237] Iteration 14100, loss = 4.2877
    I1230 19:13:31.398551 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:13:31.398567 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:13:31.398582 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70858 (* 1 = 1.70858 loss)
    I1230 19:13:31.398596 23363 solver.cpp:253]     Train net output #3: loss_f = 2.57912 (* 1 = 2.57912 loss)
    I1230 19:13:31.398608 23363 sgd_solver.cpp:106] Iteration 14100, lr = 0.000361897
    I1230 19:13:45.934365 23363 solver.cpp:237] Iteration 14200, loss = 4.19514
    I1230 19:13:45.934505 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:13:45.934530 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:13:45.934542 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5786 (* 1 = 1.5786 loss)
    I1230 19:13:45.934552 23363 solver.cpp:253]     Train net output #3: loss_f = 2.61654 (* 1 = 2.61654 loss)
    I1230 19:13:45.934562 23363 sgd_solver.cpp:106] Iteration 14200, lr = 0.000360775
    I1230 19:13:59.232992 23363 solver.cpp:237] Iteration 14300, loss = 4.26482
    I1230 19:13:59.233034 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:13:59.233044 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:13:59.233054 23363 solver.cpp:253]     Train net output #2: loss_c = 1.67841 (* 1 = 1.67841 loss)
    I1230 19:13:59.233064 23363 solver.cpp:253]     Train net output #3: loss_f = 2.58641 (* 1 = 2.58641 loss)
    I1230 19:13:59.233073 23363 sgd_solver.cpp:106] Iteration 14300, lr = 0.000359661
    I1230 19:14:12.134986 23363 solver.cpp:237] Iteration 14400, loss = 3.83564
    I1230 19:14:12.135032 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:14:12.135041 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:14:12.135052 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56911 (* 1 = 1.56911 loss)
    I1230 19:14:12.135061 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26654 (* 1 = 2.26654 loss)
    I1230 19:14:12.135069 23363 sgd_solver.cpp:106] Iteration 14400, lr = 0.000358555
    I1230 19:14:24.845885 23363 solver.cpp:237] Iteration 14500, loss = 4.32728
    I1230 19:14:24.847743 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:14:24.847770 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1230 19:14:24.847782 23363 solver.cpp:253]     Train net output #2: loss_c = 1.69617 (* 1 = 1.69617 loss)
    I1230 19:14:24.847793 23363 solver.cpp:253]     Train net output #3: loss_f = 2.6311 (* 1 = 2.6311 loss)
    I1230 19:14:24.847805 23363 sgd_solver.cpp:106] Iteration 14500, lr = 0.000357457
    I1230 19:14:37.511915 23363 solver.cpp:237] Iteration 14600, loss = 4.11967
    I1230 19:14:37.511960 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:14:37.511970 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:14:37.511981 23363 solver.cpp:253]     Train net output #2: loss_c = 1.66275 (* 1 = 1.66275 loss)
    I1230 19:14:37.511989 23363 solver.cpp:253]     Train net output #3: loss_f = 2.45692 (* 1 = 2.45692 loss)
    I1230 19:14:37.511997 23363 sgd_solver.cpp:106] Iteration 14600, lr = 0.000356366
    I1230 19:14:50.241729 23363 solver.cpp:237] Iteration 14700, loss = 4.52858
    I1230 19:14:50.241798 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:14:50.241819 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1230 19:14:50.241840 23363 solver.cpp:253]     Train net output #2: loss_c = 1.7054 (* 1 = 1.7054 loss)
    I1230 19:14:50.241849 23363 solver.cpp:253]     Train net output #3: loss_f = 2.82318 (* 1 = 2.82318 loss)
    I1230 19:14:50.241859 23363 sgd_solver.cpp:106] Iteration 14700, lr = 0.000355284
    I1230 19:15:03.379186 23363 solver.cpp:237] Iteration 14800, loss = 4.34178
    I1230 19:15:03.379324 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:15:03.379336 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:15:03.379348 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75163 (* 1 = 1.75163 loss)
    I1230 19:15:03.379355 23363 solver.cpp:253]     Train net output #3: loss_f = 2.59016 (* 1 = 2.59016 loss)
    I1230 19:15:03.379364 23363 sgd_solver.cpp:106] Iteration 14800, lr = 0.000354209
    I1230 19:15:16.931903 23363 solver.cpp:237] Iteration 14900, loss = 3.67464
    I1230 19:15:16.931959 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:15:16.931977 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:15:16.931994 23363 solver.cpp:253]     Train net output #2: loss_c = 1.49627 (* 1 = 1.49627 loss)
    I1230 19:15:16.932010 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17837 (* 1 = 2.17837 loss)
    I1230 19:15:16.932027 23363 sgd_solver.cpp:106] Iteration 14900, lr = 0.000353141
    I1230 19:15:29.568473 23363 solver.cpp:341] Iteration 15000, Testing net (#0)
    I1230 19:15:34.298753 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.4755
    I1230 19:15:34.298857 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.345083
    I1230 19:15:34.298877 23363 solver.cpp:409]     Test net output #2: loss_c = 1.67917 (* 1 = 1.67917 loss)
    I1230 19:15:34.298888 23363 solver.cpp:409]     Test net output #3: loss_f = 2.56131 (* 1 = 2.56131 loss)
    I1230 19:15:34.361232 23363 solver.cpp:237] Iteration 15000, loss = 4.16628
    I1230 19:15:34.361274 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:15:34.361287 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:15:34.361300 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63898 (* 1 = 1.63898 loss)
    I1230 19:15:34.361313 23363 solver.cpp:253]     Train net output #3: loss_f = 2.5273 (* 1 = 2.5273 loss)
    I1230 19:15:34.361325 23363 sgd_solver.cpp:106] Iteration 15000, lr = 0.000352081
    I1230 19:15:47.302328 23363 solver.cpp:237] Iteration 15100, loss = 3.97343
    I1230 19:15:47.302373 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:15:47.302382 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:15:47.302393 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58144 (* 1 = 1.58144 loss)
    I1230 19:15:47.302402 23363 solver.cpp:253]     Train net output #3: loss_f = 2.392 (* 1 = 2.392 loss)
    I1230 19:15:47.302410 23363 sgd_solver.cpp:106] Iteration 15100, lr = 0.000351029
    I1230 19:16:00.835028 23363 solver.cpp:237] Iteration 15200, loss = 3.94227
    I1230 19:16:00.835073 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:16:00.835083 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:16:00.835093 23363 solver.cpp:253]     Train net output #2: loss_c = 1.514 (* 1 = 1.514 loss)
    I1230 19:16:00.835101 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42827 (* 1 = 2.42827 loss)
    I1230 19:16:00.835110 23363 sgd_solver.cpp:106] Iteration 15200, lr = 0.000349984
    I1230 19:16:14.637555 23363 solver.cpp:237] Iteration 15300, loss = 4.28594
    I1230 19:16:14.637672 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1230 19:16:14.637687 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:16:14.637701 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73759 (* 1 = 1.73759 loss)
    I1230 19:16:14.637711 23363 solver.cpp:253]     Train net output #3: loss_f = 2.54835 (* 1 = 2.54835 loss)
    I1230 19:16:14.637722 23363 sgd_solver.cpp:106] Iteration 15300, lr = 0.000348946
    I1230 19:16:28.648464 23363 solver.cpp:237] Iteration 15400, loss = 3.70013
    I1230 19:16:28.648550 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:16:28.648561 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 19:16:28.648574 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5242 (* 1 = 1.5242 loss)
    I1230 19:16:28.648584 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17594 (* 1 = 2.17594 loss)
    I1230 19:16:28.648596 23363 sgd_solver.cpp:106] Iteration 15400, lr = 0.000347915
    I1230 19:16:43.742707 23363 solver.cpp:237] Iteration 15500, loss = 4.15734
    I1230 19:16:43.742764 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:16:43.742774 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:16:43.742785 23363 solver.cpp:253]     Train net output #2: loss_c = 1.66555 (* 1 = 1.66555 loss)
    I1230 19:16:43.742795 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49179 (* 1 = 2.49179 loss)
    I1230 19:16:43.742815 23363 sgd_solver.cpp:106] Iteration 15500, lr = 0.000346891
    I1230 19:16:56.798537 23363 solver.cpp:237] Iteration 15600, loss = 4.26592
    I1230 19:16:56.798655 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:16:56.798668 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:16:56.798679 23363 solver.cpp:253]     Train net output #2: loss_c = 1.7611 (* 1 = 1.7611 loss)
    I1230 19:16:56.798689 23363 solver.cpp:253]     Train net output #3: loss_f = 2.50481 (* 1 = 2.50481 loss)
    I1230 19:16:56.798699 23363 sgd_solver.cpp:106] Iteration 15600, lr = 0.000345874
    I1230 19:17:10.333382 23363 solver.cpp:237] Iteration 15700, loss = 4.2378
    I1230 19:17:10.333417 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:17:10.333427 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:17:10.333437 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59324 (* 1 = 1.59324 loss)
    I1230 19:17:10.333446 23363 solver.cpp:253]     Train net output #3: loss_f = 2.64455 (* 1 = 2.64455 loss)
    I1230 19:17:10.333454 23363 sgd_solver.cpp:106] Iteration 15700, lr = 0.000344864
    I1230 19:17:25.393311 23363 solver.cpp:237] Iteration 15800, loss = 4.22227
    I1230 19:17:25.393357 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:17:25.393368 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:17:25.393378 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6977 (* 1 = 1.6977 loss)
    I1230 19:17:25.393388 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52456 (* 1 = 2.52456 loss)
    I1230 19:17:25.393396 23363 sgd_solver.cpp:106] Iteration 15800, lr = 0.000343861
    I1230 19:17:41.872617 23363 solver.cpp:237] Iteration 15900, loss = 3.64854
    I1230 19:17:41.872901 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:17:41.872917 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 19:17:41.872938 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52054 (* 1 = 1.52054 loss)
    I1230 19:17:41.872947 23363 solver.cpp:253]     Train net output #3: loss_f = 2.128 (* 1 = 2.128 loss)
    I1230 19:17:41.872957 23363 sgd_solver.cpp:106] Iteration 15900, lr = 0.000342865
    I1230 19:17:56.245306 23363 solver.cpp:341] Iteration 16000, Testing net (#0)
    I1230 19:18:00.763228 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.490833
    I1230 19:18:00.763284 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.35325
    I1230 19:18:00.763305 23363 solver.cpp:409]     Test net output #2: loss_c = 1.63084 (* 1 = 1.63084 loss)
    I1230 19:18:00.763322 23363 solver.cpp:409]     Test net output #3: loss_f = 2.50169 (* 1 = 2.50169 loss)
    I1230 19:18:00.821239 23363 solver.cpp:237] Iteration 16000, loss = 4.13499
    I1230 19:18:00.821290 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:18:00.821305 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:18:00.821323 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63898 (* 1 = 1.63898 loss)
    I1230 19:18:00.821341 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49602 (* 1 = 2.49602 loss)
    I1230 19:18:00.821357 23363 sgd_solver.cpp:106] Iteration 16000, lr = 0.000341876
    I1230 19:18:13.820116 23363 solver.cpp:237] Iteration 16100, loss = 4.29058
    I1230 19:18:13.820241 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:18:13.820252 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:18:13.820263 23363 solver.cpp:253]     Train net output #2: loss_c = 1.72853 (* 1 = 1.72853 loss)
    I1230 19:18:13.820272 23363 solver.cpp:253]     Train net output #3: loss_f = 2.56205 (* 1 = 2.56205 loss)
    I1230 19:18:13.820281 23363 sgd_solver.cpp:106] Iteration 16100, lr = 0.000340893
    I1230 19:18:27.480310 23363 solver.cpp:237] Iteration 16200, loss = 4.02646
    I1230 19:18:27.480365 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:18:27.480383 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:18:27.480401 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4964 (* 1 = 1.4964 loss)
    I1230 19:18:27.480417 23363 solver.cpp:253]     Train net output #3: loss_f = 2.53006 (* 1 = 2.53006 loss)
    I1230 19:18:27.480432 23363 sgd_solver.cpp:106] Iteration 16200, lr = 0.000339916
    I1230 19:18:41.640228 23363 solver.cpp:237] Iteration 16300, loss = 4.23979
    I1230 19:18:41.640270 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:18:41.640280 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:18:41.640290 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70439 (* 1 = 1.70439 loss)
    I1230 19:18:41.640300 23363 solver.cpp:253]     Train net output #3: loss_f = 2.53541 (* 1 = 2.53541 loss)
    I1230 19:18:41.640308 23363 sgd_solver.cpp:106] Iteration 16300, lr = 0.000338947
    I1230 19:18:54.917009 23363 solver.cpp:237] Iteration 16400, loss = 3.40376
    I1230 19:18:54.917143 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 19:18:54.917156 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 19:18:54.917167 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37795 (* 1 = 1.37795 loss)
    I1230 19:18:54.917176 23363 solver.cpp:253]     Train net output #3: loss_f = 2.02581 (* 1 = 2.02581 loss)
    I1230 19:18:54.917186 23363 sgd_solver.cpp:106] Iteration 16400, lr = 0.000337983
    I1230 19:19:07.600865 23363 solver.cpp:237] Iteration 16500, loss = 3.69409
    I1230 19:19:07.600916 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:19:07.600929 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:19:07.600939 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41056 (* 1 = 1.41056 loss)
    I1230 19:19:07.600950 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28353 (* 1 = 2.28353 loss)
    I1230 19:19:07.600960 23363 sgd_solver.cpp:106] Iteration 16500, lr = 0.000337026
    I1230 19:19:20.812244 23363 solver.cpp:237] Iteration 16600, loss = 4.19467
    I1230 19:19:20.812284 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:19:20.812295 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:19:20.812307 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6662 (* 1 = 1.6662 loss)
    I1230 19:19:20.812317 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52848 (* 1 = 2.52848 loss)
    I1230 19:19:20.812327 23363 sgd_solver.cpp:106] Iteration 16600, lr = 0.000336075
    I1230 19:19:34.479856 23363 solver.cpp:237] Iteration 16700, loss = 4.08871
    I1230 19:19:34.480069 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:19:34.480095 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:19:34.480114 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50382 (* 1 = 1.50382 loss)
    I1230 19:19:34.480129 23363 solver.cpp:253]     Train net output #3: loss_f = 2.58489 (* 1 = 2.58489 loss)
    I1230 19:19:34.480144 23363 sgd_solver.cpp:106] Iteration 16700, lr = 0.000335131
    I1230 19:19:49.171059 23363 solver.cpp:237] Iteration 16800, loss = 4.46296
    I1230 19:19:49.171103 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:19:49.171111 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:19:49.171121 23363 solver.cpp:253]     Train net output #2: loss_c = 1.86642 (* 1 = 1.86642 loss)
    I1230 19:19:49.171129 23363 solver.cpp:253]     Train net output #3: loss_f = 2.59654 (* 1 = 2.59654 loss)
    I1230 19:19:49.171139 23363 sgd_solver.cpp:106] Iteration 16800, lr = 0.000334193
    I1230 19:20:02.480319 23363 solver.cpp:237] Iteration 16900, loss = 3.67436
    I1230 19:20:02.480360 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:20:02.480371 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:20:02.480383 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45618 (* 1 = 1.45618 loss)
    I1230 19:20:02.480394 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21818 (* 1 = 2.21818 loss)
    I1230 19:20:02.480404 23363 sgd_solver.cpp:106] Iteration 16900, lr = 0.00033326
    I1230 19:20:14.816980 23363 solver.cpp:341] Iteration 17000, Testing net (#0)
    I1230 19:20:19.568814 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.479583
    I1230 19:20:19.568862 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.353
    I1230 19:20:19.568879 23363 solver.cpp:409]     Test net output #2: loss_c = 1.64308 (* 1 = 1.64308 loss)
    I1230 19:20:19.568893 23363 solver.cpp:409]     Test net output #3: loss_f = 2.50645 (* 1 = 2.50645 loss)
    I1230 19:20:19.636034 23363 solver.cpp:237] Iteration 17000, loss = 4.31187
    I1230 19:20:19.636075 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:20:19.636087 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:20:19.636099 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71576 (* 1 = 1.71576 loss)
    I1230 19:20:19.636111 23363 solver.cpp:253]     Train net output #3: loss_f = 2.59611 (* 1 = 2.59611 loss)
    I1230 19:20:19.636122 23363 sgd_solver.cpp:106] Iteration 17000, lr = 0.000332334
    I1230 19:20:31.820389 23363 solver.cpp:237] Iteration 17100, loss = 4.27317
    I1230 19:20:31.820428 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:20:31.820441 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:20:31.820452 23363 solver.cpp:253]     Train net output #2: loss_c = 1.69529 (* 1 = 1.69529 loss)
    I1230 19:20:31.820463 23363 solver.cpp:253]     Train net output #3: loss_f = 2.57788 (* 1 = 2.57788 loss)
    I1230 19:20:31.820473 23363 sgd_solver.cpp:106] Iteration 17100, lr = 0.000331414
    I1230 19:20:43.985848 23363 solver.cpp:237] Iteration 17200, loss = 4.07721
    I1230 19:20:43.985888 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:20:43.985899 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1230 19:20:43.985911 23363 solver.cpp:253]     Train net output #2: loss_c = 1.53128 (* 1 = 1.53128 loss)
    I1230 19:20:43.985923 23363 solver.cpp:253]     Train net output #3: loss_f = 2.54593 (* 1 = 2.54593 loss)
    I1230 19:20:43.985932 23363 sgd_solver.cpp:106] Iteration 17200, lr = 0.0003305
    I1230 19:20:56.688915 23363 solver.cpp:237] Iteration 17300, loss = 4.25499
    I1230 19:20:56.689043 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:20:56.689059 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:20:56.689071 23363 solver.cpp:253]     Train net output #2: loss_c = 1.76422 (* 1 = 1.76422 loss)
    I1230 19:20:56.689081 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49078 (* 1 = 2.49078 loss)
    I1230 19:20:56.689093 23363 sgd_solver.cpp:106] Iteration 17300, lr = 0.000329592
    I1230 19:21:09.148308 23363 solver.cpp:237] Iteration 17400, loss = 3.58559
    I1230 19:21:09.148352 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:21:09.148365 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:21:09.148378 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45213 (* 1 = 1.45213 loss)
    I1230 19:21:09.148389 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13346 (* 1 = 2.13346 loss)
    I1230 19:21:09.148401 23363 sgd_solver.cpp:106] Iteration 17400, lr = 0.000328689
    I1230 19:21:21.749274 23363 solver.cpp:237] Iteration 17500, loss = 4.11713
    I1230 19:21:21.749323 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:21:21.749333 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:21:21.749344 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59312 (* 1 = 1.59312 loss)
    I1230 19:21:21.749354 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52401 (* 1 = 2.52401 loss)
    I1230 19:21:21.749364 23363 sgd_solver.cpp:106] Iteration 17500, lr = 0.000327792
    I1230 19:21:34.174198 23363 solver.cpp:237] Iteration 17600, loss = 4.29078
    I1230 19:21:34.174336 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:21:34.174357 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:21:34.174368 23363 solver.cpp:253]     Train net output #2: loss_c = 1.768 (* 1 = 1.768 loss)
    I1230 19:21:34.174376 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52278 (* 1 = 2.52278 loss)
    I1230 19:21:34.174386 23363 sgd_solver.cpp:106] Iteration 17600, lr = 0.000326901
    I1230 19:21:46.685664 23363 solver.cpp:237] Iteration 17700, loss = 3.97436
    I1230 19:21:46.685719 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:21:46.685736 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:21:46.685755 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5155 (* 1 = 1.5155 loss)
    I1230 19:21:46.685770 23363 solver.cpp:253]     Train net output #3: loss_f = 2.45886 (* 1 = 2.45886 loss)
    I1230 19:21:46.685786 23363 sgd_solver.cpp:106] Iteration 17700, lr = 0.000326015
    I1230 19:21:58.661763 23363 solver.cpp:237] Iteration 17800, loss = 4.46727
    I1230 19:21:58.661813 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:21:58.661828 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:21:58.661847 23363 solver.cpp:253]     Train net output #2: loss_c = 1.83447 (* 1 = 1.83447 loss)
    I1230 19:21:58.661864 23363 solver.cpp:253]     Train net output #3: loss_f = 2.6328 (* 1 = 2.6328 loss)
    I1230 19:21:58.661880 23363 sgd_solver.cpp:106] Iteration 17800, lr = 0.000325136
    I1230 19:22:10.957496 23363 solver.cpp:237] Iteration 17900, loss = 3.61284
    I1230 19:22:10.957605 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:22:10.957623 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:22:10.957636 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46929 (* 1 = 1.46929 loss)
    I1230 19:22:10.957648 23363 solver.cpp:253]     Train net output #3: loss_f = 2.14355 (* 1 = 2.14355 loss)
    I1230 19:22:10.957660 23363 sgd_solver.cpp:106] Iteration 17900, lr = 0.000324261
    I1230 19:22:22.972522 23363 solver.cpp:341] Iteration 18000, Testing net (#0)
    I1230 19:22:27.510604 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.50575
    I1230 19:22:27.510644 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.37475
    I1230 19:22:27.510658 23363 solver.cpp:409]     Test net output #2: loss_c = 1.58016 (* 1 = 1.58016 loss)
    I1230 19:22:27.510671 23363 solver.cpp:409]     Test net output #3: loss_f = 2.43023 (* 1 = 2.43023 loss)
    I1230 19:22:27.568338 23363 solver.cpp:237] Iteration 18000, loss = 3.94321
    I1230 19:22:27.568382 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:22:27.568394 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:22:27.568408 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55458 (* 1 = 1.55458 loss)
    I1230 19:22:27.568419 23363 solver.cpp:253]     Train net output #3: loss_f = 2.38863 (* 1 = 2.38863 loss)
    I1230 19:22:27.568431 23363 sgd_solver.cpp:106] Iteration 18000, lr = 0.000323392
    I1230 19:22:39.693269 23363 solver.cpp:237] Iteration 18100, loss = 4.09981
    I1230 19:22:39.693310 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:22:39.693321 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:22:39.693333 23363 solver.cpp:253]     Train net output #2: loss_c = 1.61974 (* 1 = 1.61974 loss)
    I1230 19:22:39.693344 23363 solver.cpp:253]     Train net output #3: loss_f = 2.48007 (* 1 = 2.48007 loss)
    I1230 19:22:39.693356 23363 sgd_solver.cpp:106] Iteration 18100, lr = 0.000322529
    I1230 19:22:54.059288 23363 solver.cpp:237] Iteration 18200, loss = 3.99131
    I1230 19:22:54.059413 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:22:54.059429 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:22:54.059442 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48794 (* 1 = 1.48794 loss)
    I1230 19:22:54.059453 23363 solver.cpp:253]     Train net output #3: loss_f = 2.50337 (* 1 = 2.50337 loss)
    I1230 19:22:54.059465 23363 sgd_solver.cpp:106] Iteration 18200, lr = 0.00032167
    I1230 19:23:06.651248 23363 solver.cpp:237] Iteration 18300, loss = 4.03637
    I1230 19:23:06.651283 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:23:06.651293 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:23:06.651303 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63809 (* 1 = 1.63809 loss)
    I1230 19:23:06.651311 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39828 (* 1 = 2.39828 loss)
    I1230 19:23:06.651319 23363 sgd_solver.cpp:106] Iteration 18300, lr = 0.000320818
    I1230 19:23:19.974767 23363 solver.cpp:237] Iteration 18400, loss = 3.61528
    I1230 19:23:19.974819 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 19:23:19.974835 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:23:19.974854 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44791 (* 1 = 1.44791 loss)
    I1230 19:23:19.974870 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16737 (* 1 = 2.16737 loss)
    I1230 19:23:19.974886 23363 sgd_solver.cpp:106] Iteration 18400, lr = 0.00031997
    I1230 19:23:34.594851 23363 solver.cpp:237] Iteration 18500, loss = 3.70408
    I1230 19:23:34.594962 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:23:34.594976 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 19:23:34.594988 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46375 (* 1 = 1.46375 loss)
    I1230 19:23:34.595000 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24032 (* 1 = 2.24032 loss)
    I1230 19:23:34.595011 23363 sgd_solver.cpp:106] Iteration 18500, lr = 0.000319128
    I1230 19:23:50.361151 23363 solver.cpp:237] Iteration 18600, loss = 4.12376
    I1230 19:23:50.361186 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:23:50.361196 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:23:50.361207 23363 solver.cpp:253]     Train net output #2: loss_c = 1.68126 (* 1 = 1.68126 loss)
    I1230 19:23:50.361214 23363 solver.cpp:253]     Train net output #3: loss_f = 2.44251 (* 1 = 2.44251 loss)
    I1230 19:23:50.361224 23363 sgd_solver.cpp:106] Iteration 18600, lr = 0.00031829
    I1230 19:24:05.266484 23363 solver.cpp:237] Iteration 18700, loss = 4.03301
    I1230 19:24:05.266679 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:24:05.266708 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:24:05.266723 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51055 (* 1 = 1.51055 loss)
    I1230 19:24:05.266736 23363 solver.cpp:253]     Train net output #3: loss_f = 2.52246 (* 1 = 2.52246 loss)
    I1230 19:24:05.266748 23363 sgd_solver.cpp:106] Iteration 18700, lr = 0.000317458
    I1230 19:24:19.126358 23363 solver.cpp:237] Iteration 18800, loss = 4.24299
    I1230 19:24:19.126395 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:24:19.126406 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:24:19.126417 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73167 (* 1 = 1.73167 loss)
    I1230 19:24:19.126426 23363 solver.cpp:253]     Train net output #3: loss_f = 2.51132 (* 1 = 2.51132 loss)
    I1230 19:24:19.126436 23363 sgd_solver.cpp:106] Iteration 18800, lr = 0.000316631
    I1230 19:24:33.293084 23363 solver.cpp:237] Iteration 18900, loss = 3.69367
    I1230 19:24:33.293138 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:24:33.293149 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:24:33.293162 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50231 (* 1 = 1.50231 loss)
    I1230 19:24:33.293174 23363 solver.cpp:253]     Train net output #3: loss_f = 2.19136 (* 1 = 2.19136 loss)
    I1230 19:24:33.293186 23363 sgd_solver.cpp:106] Iteration 18900, lr = 0.000315809
    I1230 19:24:46.315984 23363 solver.cpp:341] Iteration 19000, Testing net (#0)
    I1230 19:24:52.930533 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.496417
    I1230 19:24:52.930584 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.365667
    I1230 19:24:52.930598 23363 solver.cpp:409]     Test net output #2: loss_c = 1.60426 (* 1 = 1.60426 loss)
    I1230 19:24:52.930608 23363 solver.cpp:409]     Test net output #3: loss_f = 2.45373 (* 1 = 2.45373 loss)
    I1230 19:24:53.019183 23363 solver.cpp:237] Iteration 19000, loss = 3.86471
    I1230 19:24:53.019232 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:24:53.019242 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:24:53.019253 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50128 (* 1 = 1.50128 loss)
    I1230 19:24:53.019261 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36343 (* 1 = 2.36343 loss)
    I1230 19:24:53.019273 23363 sgd_solver.cpp:106] Iteration 19000, lr = 0.000314992
    I1230 19:25:10.910694 23363 solver.cpp:237] Iteration 19100, loss = 4.02006
    I1230 19:25:10.910735 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:25:10.910747 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:25:10.910758 23363 solver.cpp:253]     Train net output #2: loss_c = 1.61767 (* 1 = 1.61767 loss)
    I1230 19:25:10.910768 23363 solver.cpp:253]     Train net output #3: loss_f = 2.40239 (* 1 = 2.40239 loss)
    I1230 19:25:10.910778 23363 sgd_solver.cpp:106] Iteration 19100, lr = 0.00031418
    I1230 19:25:24.244194 23363 solver.cpp:237] Iteration 19200, loss = 4.07478
    I1230 19:25:24.244333 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:25:24.244344 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:25:24.244354 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56792 (* 1 = 1.56792 loss)
    I1230 19:25:24.244364 23363 solver.cpp:253]     Train net output #3: loss_f = 2.50685 (* 1 = 2.50685 loss)
    I1230 19:25:24.244372 23363 sgd_solver.cpp:106] Iteration 19200, lr = 0.000313372
    I1230 19:25:38.529777 23363 solver.cpp:237] Iteration 19300, loss = 4.04195
    I1230 19:25:38.529839 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:25:38.529855 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:25:38.529872 23363 solver.cpp:253]     Train net output #2: loss_c = 1.65601 (* 1 = 1.65601 loss)
    I1230 19:25:38.529887 23363 solver.cpp:253]     Train net output #3: loss_f = 2.38595 (* 1 = 2.38595 loss)
    I1230 19:25:38.529902 23363 sgd_solver.cpp:106] Iteration 19300, lr = 0.00031257
    I1230 19:25:51.768836 23363 solver.cpp:237] Iteration 19400, loss = 3.64185
    I1230 19:25:51.768890 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:25:51.768908 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:25:51.768926 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46984 (* 1 = 1.46984 loss)
    I1230 19:25:51.768944 23363 solver.cpp:253]     Train net output #3: loss_f = 2.172 (* 1 = 2.172 loss)
    I1230 19:25:51.768961 23363 sgd_solver.cpp:106] Iteration 19400, lr = 0.000311772
    I1230 19:26:04.970157 23363 solver.cpp:237] Iteration 19500, loss = 3.94945
    I1230 19:26:04.970372 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:26:04.970401 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:26:04.970420 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5515 (* 1 = 1.5515 loss)
    I1230 19:26:04.970438 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39795 (* 1 = 2.39795 loss)
    I1230 19:26:04.970456 23363 sgd_solver.cpp:106] Iteration 19500, lr = 0.000310979
    I1230 19:26:18.449712 23363 solver.cpp:237] Iteration 19600, loss = 4.2439
    I1230 19:26:18.449765 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:26:18.449779 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:26:18.449805 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75387 (* 1 = 1.75387 loss)
    I1230 19:26:18.449817 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49004 (* 1 = 2.49004 loss)
    I1230 19:26:18.449828 23363 sgd_solver.cpp:106] Iteration 19600, lr = 0.000310191
    I1230 19:26:31.129751 23363 solver.cpp:237] Iteration 19700, loss = 3.94274
    I1230 19:26:31.129784 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:26:31.129793 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:26:31.129803 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46698 (* 1 = 1.46698 loss)
    I1230 19:26:31.129812 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47575 (* 1 = 2.47575 loss)
    I1230 19:26:31.129822 23363 sgd_solver.cpp:106] Iteration 19700, lr = 0.000309407
    I1230 19:26:45.762805 23363 solver.cpp:237] Iteration 19800, loss = 4.06251
    I1230 19:26:45.762915 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:26:45.762929 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:26:45.762943 23363 solver.cpp:253]     Train net output #2: loss_c = 1.66649 (* 1 = 1.66649 loss)
    I1230 19:26:45.762953 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39602 (* 1 = 2.39602 loss)
    I1230 19:26:45.762964 23363 sgd_solver.cpp:106] Iteration 19800, lr = 0.000308628
    I1230 19:26:58.129775 23363 solver.cpp:237] Iteration 19900, loss = 3.58216
    I1230 19:26:58.129832 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:26:58.129849 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:26:58.129868 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4524 (* 1 = 1.4524 loss)
    I1230 19:26:58.129883 23363 solver.cpp:253]     Train net output #3: loss_f = 2.12975 (* 1 = 2.12975 loss)
    I1230 19:26:58.129899 23363 sgd_solver.cpp:106] Iteration 19900, lr = 0.000307854
    I1230 19:27:13.813454 23363 solver.cpp:341] Iteration 20000, Testing net (#0)
    I1230 19:27:19.096516 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.502833
    I1230 19:27:19.097569 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.370167
    I1230 19:27:19.097592 23363 solver.cpp:409]     Test net output #2: loss_c = 1.58201 (* 1 = 1.58201 loss)
    I1230 19:27:19.097602 23363 solver.cpp:409]     Test net output #3: loss_f = 2.42893 (* 1 = 2.42893 loss)
    I1230 19:27:19.181303 23363 solver.cpp:237] Iteration 20000, loss = 3.91208
    I1230 19:27:19.181360 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:27:19.181372 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:27:19.181385 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48864 (* 1 = 1.48864 loss)
    I1230 19:27:19.181396 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42343 (* 1 = 2.42343 loss)
    I1230 19:27:19.181408 23363 sgd_solver.cpp:106] Iteration 20000, lr = 0.000307084
    I1230 19:27:35.860146 23363 solver.cpp:237] Iteration 20100, loss = 4.24318
    I1230 19:27:35.860188 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:27:35.860199 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:27:35.860211 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70367 (* 1 = 1.70367 loss)
    I1230 19:27:35.860221 23363 solver.cpp:253]     Train net output #3: loss_f = 2.53951 (* 1 = 2.53951 loss)
    I1230 19:27:35.860231 23363 sgd_solver.cpp:106] Iteration 20100, lr = 0.000306318
    I1230 19:27:49.497618 23363 solver.cpp:237] Iteration 20200, loss = 4.0819
    I1230 19:27:49.497776 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:27:49.497800 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:27:49.497819 23363 solver.cpp:253]     Train net output #2: loss_c = 1.53781 (* 1 = 1.53781 loss)
    I1230 19:27:49.497834 23363 solver.cpp:253]     Train net output #3: loss_f = 2.54409 (* 1 = 2.54409 loss)
    I1230 19:27:49.497849 23363 sgd_solver.cpp:106] Iteration 20200, lr = 0.000305557
    I1230 19:28:03.906977 23363 solver.cpp:237] Iteration 20300, loss = 4.07396
    I1230 19:28:03.907037 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:28:03.907057 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:28:03.907076 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70026 (* 1 = 1.70026 loss)
    I1230 19:28:03.907093 23363 solver.cpp:253]     Train net output #3: loss_f = 2.3737 (* 1 = 2.3737 loss)
    I1230 19:28:03.907109 23363 sgd_solver.cpp:106] Iteration 20300, lr = 0.000304801
    I1230 19:28:19.957597 23363 solver.cpp:237] Iteration 20400, loss = 3.55382
    I1230 19:28:19.957736 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:28:19.957757 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:28:19.957775 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4612 (* 1 = 1.4612 loss)
    I1230 19:28:19.957792 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09262 (* 1 = 2.09262 loss)
    I1230 19:28:19.957808 23363 sgd_solver.cpp:106] Iteration 20400, lr = 0.000304048
    I1230 19:28:36.490494 23363 solver.cpp:237] Iteration 20500, loss = 3.77177
    I1230 19:28:36.490557 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:28:36.490579 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:28:36.490604 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48195 (* 1 = 1.48195 loss)
    I1230 19:28:36.490628 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28983 (* 1 = 2.28983 loss)
    I1230 19:28:36.490648 23363 sgd_solver.cpp:106] Iteration 20500, lr = 0.000303301
    I1230 19:28:52.941220 23363 solver.cpp:237] Iteration 20600, loss = 4.1979
    I1230 19:28:52.941349 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:28:52.941372 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:28:52.941385 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75465 (* 1 = 1.75465 loss)
    I1230 19:28:52.941395 23363 solver.cpp:253]     Train net output #3: loss_f = 2.44325 (* 1 = 2.44325 loss)
    I1230 19:28:52.941404 23363 sgd_solver.cpp:106] Iteration 20600, lr = 0.000302557
    I1230 19:29:08.923715 23363 solver.cpp:237] Iteration 20700, loss = 4.07261
    I1230 19:29:08.923774 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:29:08.923785 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:29:08.923796 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55871 (* 1 = 1.55871 loss)
    I1230 19:29:08.923806 23363 solver.cpp:253]     Train net output #3: loss_f = 2.5139 (* 1 = 2.5139 loss)
    I1230 19:29:08.923816 23363 sgd_solver.cpp:106] Iteration 20700, lr = 0.000301817
    I1230 19:29:24.015619 23363 solver.cpp:237] Iteration 20800, loss = 4.19835
    I1230 19:29:24.015789 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:29:24.015813 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1230 19:29:24.015825 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70344 (* 1 = 1.70344 loss)
    I1230 19:29:24.015835 23363 solver.cpp:253]     Train net output #3: loss_f = 2.49491 (* 1 = 2.49491 loss)
    I1230 19:29:24.015846 23363 sgd_solver.cpp:106] Iteration 20800, lr = 0.000301082
    I1230 19:29:39.134259 23363 solver.cpp:237] Iteration 20900, loss = 3.6055
    I1230 19:29:39.134305 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 19:29:39.134315 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 19:29:39.134325 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44372 (* 1 = 1.44372 loss)
    I1230 19:29:39.134333 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16178 (* 1 = 2.16178 loss)
    I1230 19:29:39.134342 23363 sgd_solver.cpp:106] Iteration 20900, lr = 0.000300351
    I1230 19:29:55.695543 23363 solver.cpp:341] Iteration 21000, Testing net (#0)
    I1230 19:30:01.446091 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.51525
    I1230 19:30:01.446131 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.381917
    I1230 19:30:01.446142 23363 solver.cpp:409]     Test net output #2: loss_c = 1.55695 (* 1 = 1.55695 loss)
    I1230 19:30:01.446151 23363 solver.cpp:409]     Test net output #3: loss_f = 2.39289 (* 1 = 2.39289 loss)
    I1230 19:30:01.520004 23363 solver.cpp:237] Iteration 21000, loss = 3.9493
    I1230 19:30:01.520045 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:30:01.520056 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:30:01.520066 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5001 (* 1 = 1.5001 loss)
    I1230 19:30:01.520076 23363 solver.cpp:253]     Train net output #3: loss_f = 2.4492 (* 1 = 2.4492 loss)
    I1230 19:30:01.520088 23363 sgd_solver.cpp:106] Iteration 21000, lr = 0.000299624
    I1230 19:30:17.763258 23363 solver.cpp:237] Iteration 21100, loss = 4.21143
    I1230 19:30:17.763304 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:30:17.763316 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:30:17.763329 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73941 (* 1 = 1.73941 loss)
    I1230 19:30:17.763340 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47202 (* 1 = 2.47202 loss)
    I1230 19:30:17.763351 23363 sgd_solver.cpp:106] Iteration 21100, lr = 0.000298901
    I1230 19:30:34.242583 23363 solver.cpp:237] Iteration 21200, loss = 3.86969
    I1230 19:30:34.242727 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:30:34.242749 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:30:34.242769 23363 solver.cpp:253]     Train net output #2: loss_c = 1.47954 (* 1 = 1.47954 loss)
    I1230 19:30:34.242784 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39014 (* 1 = 2.39014 loss)
    I1230 19:30:34.242799 23363 sgd_solver.cpp:106] Iteration 21200, lr = 0.000298182
    I1230 19:30:49.981202 23363 solver.cpp:237] Iteration 21300, loss = 4.01231
    I1230 19:30:49.981243 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:30:49.981256 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:30:49.981266 23363 solver.cpp:253]     Train net output #2: loss_c = 1.66425 (* 1 = 1.66425 loss)
    I1230 19:30:49.981276 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34805 (* 1 = 2.34805 loss)
    I1230 19:30:49.981286 23363 sgd_solver.cpp:106] Iteration 21300, lr = 0.000297468
    I1230 19:31:06.466269 23363 solver.cpp:237] Iteration 21400, loss = 3.78295
    I1230 19:31:06.466408 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:31:06.466421 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:31:06.466431 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5394 (* 1 = 1.5394 loss)
    I1230 19:31:06.466440 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24354 (* 1 = 2.24354 loss)
    I1230 19:31:06.466451 23363 sgd_solver.cpp:106] Iteration 21400, lr = 0.000296757
    I1230 19:31:21.994837 23363 solver.cpp:237] Iteration 21500, loss = 3.8646
    I1230 19:31:21.994885 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:31:21.994894 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:31:21.994904 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51689 (* 1 = 1.51689 loss)
    I1230 19:31:21.994913 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34771 (* 1 = 2.34771 loss)
    I1230 19:31:21.994922 23363 sgd_solver.cpp:106] Iteration 21500, lr = 0.00029605
    I1230 19:31:37.901584 23363 solver.cpp:237] Iteration 21600, loss = 4.1238
    I1230 19:31:37.901739 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:31:37.901763 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:31:37.901783 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64739 (* 1 = 1.64739 loss)
    I1230 19:31:37.901798 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47641 (* 1 = 2.47641 loss)
    I1230 19:31:37.901814 23363 sgd_solver.cpp:106] Iteration 21600, lr = 0.000295347
    I1230 19:31:53.507407 23363 solver.cpp:237] Iteration 21700, loss = 3.70419
    I1230 19:31:53.507457 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:31:53.507472 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:31:53.507489 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4028 (* 1 = 1.4028 loss)
    I1230 19:31:53.507504 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30139 (* 1 = 2.30139 loss)
    I1230 19:31:53.507519 23363 sgd_solver.cpp:106] Iteration 21700, lr = 0.000294648
    I1230 19:32:08.754401 23363 solver.cpp:237] Iteration 21800, loss = 4.19329
    I1230 19:32:08.754547 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:32:08.754567 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:32:08.754581 23363 solver.cpp:253]     Train net output #2: loss_c = 1.7395 (* 1 = 1.7395 loss)
    I1230 19:32:08.754593 23363 solver.cpp:253]     Train net output #3: loss_f = 2.45379 (* 1 = 2.45379 loss)
    I1230 19:32:08.754604 23363 sgd_solver.cpp:106] Iteration 21800, lr = 0.000293953
    I1230 19:32:23.917417 23363 solver.cpp:237] Iteration 21900, loss = 3.5
    I1230 19:32:23.917459 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:32:23.917471 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:32:23.917484 23363 solver.cpp:253]     Train net output #2: loss_c = 1.38925 (* 1 = 1.38925 loss)
    I1230 19:32:23.917495 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11075 (* 1 = 2.11075 loss)
    I1230 19:32:23.917506 23363 sgd_solver.cpp:106] Iteration 21900, lr = 0.000293261
    I1230 19:32:39.064638 23363 solver.cpp:341] Iteration 22000, Testing net (#0)
    I1230 19:32:44.796670 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.51525
    I1230 19:32:44.796712 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.377833
    I1230 19:32:44.796726 23363 solver.cpp:409]     Test net output #2: loss_c = 1.55947 (* 1 = 1.55947 loss)
    I1230 19:32:44.796737 23363 solver.cpp:409]     Test net output #3: loss_f = 2.38903 (* 1 = 2.38903 loss)
    I1230 19:32:44.867911 23363 solver.cpp:237] Iteration 22000, loss = 3.83262
    I1230 19:32:44.867946 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:32:44.867959 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:32:44.867970 23363 solver.cpp:253]     Train net output #2: loss_c = 1.53022 (* 1 = 1.53022 loss)
    I1230 19:32:44.867981 23363 solver.cpp:253]     Train net output #3: loss_f = 2.3024 (* 1 = 2.3024 loss)
    I1230 19:32:44.867992 23363 sgd_solver.cpp:106] Iteration 22000, lr = 0.000292574
    I1230 19:33:00.122577 23363 solver.cpp:237] Iteration 22100, loss = 4.05741
    I1230 19:33:00.122627 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:33:00.122637 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:33:00.122648 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64441 (* 1 = 1.64441 loss)
    I1230 19:33:00.122668 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41299 (* 1 = 2.41299 loss)
    I1230 19:33:00.122679 23363 sgd_solver.cpp:106] Iteration 22100, lr = 0.00029189
    I1230 19:33:15.386448 23363 solver.cpp:237] Iteration 22200, loss = 3.86516
    I1230 19:33:15.386625 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:33:15.386637 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:33:15.386648 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45941 (* 1 = 1.45941 loss)
    I1230 19:33:15.386656 23363 solver.cpp:253]     Train net output #3: loss_f = 2.40575 (* 1 = 2.40575 loss)
    I1230 19:33:15.386667 23363 sgd_solver.cpp:106] Iteration 22200, lr = 0.00029121
    I1230 19:33:30.530498 23363 solver.cpp:237] Iteration 22300, loss = 4.11699
    I1230 19:33:30.530553 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:33:30.530565 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:33:30.530576 23363 solver.cpp:253]     Train net output #2: loss_c = 1.68897 (* 1 = 1.68897 loss)
    I1230 19:33:30.530586 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42802 (* 1 = 2.42802 loss)
    I1230 19:33:30.530597 23363 sgd_solver.cpp:106] Iteration 22300, lr = 0.000290533
    I1230 19:33:46.161586 23363 solver.cpp:237] Iteration 22400, loss = 3.37422
    I1230 19:33:46.161715 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:33:46.161728 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 19:33:46.161738 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31883 (* 1 = 1.31883 loss)
    I1230 19:33:46.161747 23363 solver.cpp:253]     Train net output #3: loss_f = 2.0554 (* 1 = 2.0554 loss)
    I1230 19:33:46.161757 23363 sgd_solver.cpp:106] Iteration 22400, lr = 0.000289861
    I1230 19:34:01.305734 23363 solver.cpp:237] Iteration 22500, loss = 3.70706
    I1230 19:34:01.305771 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:34:01.305781 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:34:01.305791 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48319 (* 1 = 1.48319 loss)
    I1230 19:34:01.305800 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22387 (* 1 = 2.22387 loss)
    I1230 19:34:01.305809 23363 sgd_solver.cpp:106] Iteration 22500, lr = 0.000289191
    I1230 19:34:16.446955 23363 solver.cpp:237] Iteration 22600, loss = 4.08209
    I1230 19:34:16.447113 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:34:16.447126 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:34:16.447136 23363 solver.cpp:253]     Train net output #2: loss_c = 1.67774 (* 1 = 1.67774 loss)
    I1230 19:34:16.447146 23363 solver.cpp:253]     Train net output #3: loss_f = 2.40434 (* 1 = 2.40434 loss)
    I1230 19:34:16.447155 23363 sgd_solver.cpp:106] Iteration 22600, lr = 0.000288526
    I1230 19:34:31.540184 23363 solver.cpp:237] Iteration 22700, loss = 3.82968
    I1230 19:34:31.540228 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:34:31.540237 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:34:31.540247 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37308 (* 1 = 1.37308 loss)
    I1230 19:34:31.540256 23363 solver.cpp:253]     Train net output #3: loss_f = 2.4566 (* 1 = 2.4566 loss)
    I1230 19:34:31.540266 23363 sgd_solver.cpp:106] Iteration 22700, lr = 0.000287864
    I1230 19:34:46.705771 23363 solver.cpp:237] Iteration 22800, loss = 3.91183
    I1230 19:34:46.705909 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:34:46.705922 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:34:46.705932 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59071 (* 1 = 1.59071 loss)
    I1230 19:34:46.705940 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32112 (* 1 = 2.32112 loss)
    I1230 19:34:46.705950 23363 sgd_solver.cpp:106] Iteration 22800, lr = 0.000287205
    I1230 19:35:01.789789 23363 solver.cpp:237] Iteration 22900, loss = 3.58832
    I1230 19:35:01.789825 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:35:01.789834 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:35:01.789844 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45613 (* 1 = 1.45613 loss)
    I1230 19:35:01.789854 23363 solver.cpp:253]     Train net output #3: loss_f = 2.1322 (* 1 = 2.1322 loss)
    I1230 19:35:01.789863 23363 sgd_solver.cpp:106] Iteration 22900, lr = 0.00028655
    I1230 19:35:16.793772 23363 solver.cpp:341] Iteration 23000, Testing net (#0)
    I1230 19:35:22.458587 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.520833
    I1230 19:35:22.458627 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.387167
    I1230 19:35:22.458638 23363 solver.cpp:409]     Test net output #2: loss_c = 1.52291 (* 1 = 1.52291 loss)
    I1230 19:35:22.458647 23363 solver.cpp:409]     Test net output #3: loss_f = 2.35768 (* 1 = 2.35768 loss)
    I1230 19:35:22.531626 23363 solver.cpp:237] Iteration 23000, loss = 3.80439
    I1230 19:35:22.531680 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:35:22.531693 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:35:22.531704 23363 solver.cpp:253]     Train net output #2: loss_c = 1.47626 (* 1 = 1.47626 loss)
    I1230 19:35:22.531725 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32813 (* 1 = 2.32813 loss)
    I1230 19:35:22.531735 23363 sgd_solver.cpp:106] Iteration 23000, lr = 0.000285899
    I1230 19:35:37.693192 23363 solver.cpp:237] Iteration 23100, loss = 4.03968
    I1230 19:35:37.693234 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:35:37.693244 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:35:37.693256 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62734 (* 1 = 1.62734 loss)
    I1230 19:35:37.693265 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41235 (* 1 = 2.41235 loss)
    I1230 19:35:37.693275 23363 sgd_solver.cpp:106] Iteration 23100, lr = 0.000285251
    I1230 19:35:52.905505 23363 solver.cpp:237] Iteration 23200, loss = 3.75462
    I1230 19:35:52.905625 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:35:52.905647 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:35:52.905658 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37559 (* 1 = 1.37559 loss)
    I1230 19:35:52.905666 23363 solver.cpp:253]     Train net output #3: loss_f = 2.37903 (* 1 = 2.37903 loss)
    I1230 19:35:52.905675 23363 sgd_solver.cpp:106] Iteration 23200, lr = 0.000284606
    I1230 19:36:08.091179 23363 solver.cpp:237] Iteration 23300, loss = 4.12356
    I1230 19:36:08.091222 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 19:36:08.091231 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:36:08.091243 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71015 (* 1 = 1.71015 loss)
    I1230 19:36:08.091253 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41342 (* 1 = 2.41342 loss)
    I1230 19:36:08.091262 23363 sgd_solver.cpp:106] Iteration 23300, lr = 0.000283965
    I1230 19:36:23.229184 23363 solver.cpp:237] Iteration 23400, loss = 3.49617
    I1230 19:36:23.229315 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:36:23.229334 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:36:23.229344 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41882 (* 1 = 1.41882 loss)
    I1230 19:36:23.229352 23363 solver.cpp:253]     Train net output #3: loss_f = 2.07735 (* 1 = 2.07735 loss)
    I1230 19:36:23.229362 23363 sgd_solver.cpp:106] Iteration 23400, lr = 0.000283327
    I1230 19:36:38.387212 23363 solver.cpp:237] Iteration 23500, loss = 3.82855
    I1230 19:36:38.387259 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:36:38.387269 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:36:38.387277 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48532 (* 1 = 1.48532 loss)
    I1230 19:36:38.387286 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34323 (* 1 = 2.34323 loss)
    I1230 19:36:38.387295 23363 sgd_solver.cpp:106] Iteration 23500, lr = 0.000282693
    I1230 19:36:53.561627 23363 solver.cpp:237] Iteration 23600, loss = 4.32259
    I1230 19:36:53.561779 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 19:36:53.561800 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:36:53.561818 23363 solver.cpp:253]     Train net output #2: loss_c = 1.7683 (* 1 = 1.7683 loss)
    I1230 19:36:53.561835 23363 solver.cpp:253]     Train net output #3: loss_f = 2.5543 (* 1 = 2.5543 loss)
    I1230 19:36:53.561849 23363 sgd_solver.cpp:106] Iteration 23600, lr = 0.000282061
    I1230 19:37:08.780535 23363 solver.cpp:237] Iteration 23700, loss = 4.07445
    I1230 19:37:08.780591 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:37:08.780601 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 19:37:08.780612 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55721 (* 1 = 1.55721 loss)
    I1230 19:37:08.780622 23363 solver.cpp:253]     Train net output #3: loss_f = 2.51724 (* 1 = 2.51724 loss)
    I1230 19:37:08.780632 23363 sgd_solver.cpp:106] Iteration 23700, lr = 0.000281433
    I1230 19:37:23.911046 23363 solver.cpp:237] Iteration 23800, loss = 3.76805
    I1230 19:37:23.911172 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:37:23.911193 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:37:23.911212 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52978 (* 1 = 1.52978 loss)
    I1230 19:37:23.911228 23363 solver.cpp:253]     Train net output #3: loss_f = 2.23828 (* 1 = 2.23828 loss)
    I1230 19:37:23.911245 23363 sgd_solver.cpp:106] Iteration 23800, lr = 0.000280809
    I1230 19:37:39.115154 23363 solver.cpp:237] Iteration 23900, loss = 3.40031
    I1230 19:37:39.115190 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:37:39.115200 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:37:39.115211 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34372 (* 1 = 1.34372 loss)
    I1230 19:37:39.115218 23363 solver.cpp:253]     Train net output #3: loss_f = 2.0566 (* 1 = 2.0566 loss)
    I1230 19:37:39.115229 23363 sgd_solver.cpp:106] Iteration 23900, lr = 0.000280187
    I1230 19:37:54.129039 23363 solver.cpp:341] Iteration 24000, Testing net (#0)
    I1230 19:37:59.853363 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.511583
    I1230 19:37:59.853427 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.376583
    I1230 19:37:59.853443 23363 solver.cpp:409]     Test net output #2: loss_c = 1.5474 (* 1 = 1.5474 loss)
    I1230 19:37:59.853456 23363 solver.cpp:409]     Test net output #3: loss_f = 2.38045 (* 1 = 2.38045 loss)
    I1230 19:37:59.930573 23363 solver.cpp:237] Iteration 24000, loss = 4.11341
    I1230 19:37:59.930619 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:37:59.930629 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:37:59.930640 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64161 (* 1 = 1.64161 loss)
    I1230 19:37:59.930650 23363 solver.cpp:253]     Train net output #3: loss_f = 2.4718 (* 1 = 2.4718 loss)
    I1230 19:37:59.930661 23363 sgd_solver.cpp:106] Iteration 24000, lr = 0.000279569
    I1230 19:38:15.182718 23363 solver.cpp:237] Iteration 24100, loss = 4.01326
    I1230 19:38:15.182766 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:38:15.182778 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:38:15.182790 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64781 (* 1 = 1.64781 loss)
    I1230 19:38:15.182801 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36545 (* 1 = 2.36545 loss)
    I1230 19:38:15.182811 23363 sgd_solver.cpp:106] Iteration 24100, lr = 0.000278954
    I1230 19:38:30.348116 23363 solver.cpp:237] Iteration 24200, loss = 3.94213
    I1230 19:38:30.348323 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:38:30.348337 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:38:30.348347 23363 solver.cpp:253]     Train net output #2: loss_c = 1.47714 (* 1 = 1.47714 loss)
    I1230 19:38:30.348356 23363 solver.cpp:253]     Train net output #3: loss_f = 2.46499 (* 1 = 2.46499 loss)
    I1230 19:38:30.348366 23363 sgd_solver.cpp:106] Iteration 24200, lr = 0.000278342
    I1230 19:38:50.215028 23363 solver.cpp:237] Iteration 24300, loss = 3.90114
    I1230 19:38:50.215070 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:38:50.215080 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:38:50.215090 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59868 (* 1 = 1.59868 loss)
    I1230 19:38:50.215101 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30246 (* 1 = 2.30246 loss)
    I1230 19:38:50.215109 23363 sgd_solver.cpp:106] Iteration 24300, lr = 0.000277733
    I1230 19:39:11.939754 23363 solver.cpp:237] Iteration 24400, loss = 3.47092
    I1230 19:39:11.939885 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:39:11.939908 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:39:11.939919 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40698 (* 1 = 1.40698 loss)
    I1230 19:39:11.939929 23363 solver.cpp:253]     Train net output #3: loss_f = 2.06393 (* 1 = 2.06393 loss)
    I1230 19:39:11.939939 23363 sgd_solver.cpp:106] Iteration 24400, lr = 0.000277127
    I1230 19:39:33.779594 23363 solver.cpp:237] Iteration 24500, loss = 3.6275
    I1230 19:39:33.779642 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:39:33.779654 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:39:33.779664 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41306 (* 1 = 1.41306 loss)
    I1230 19:39:33.779674 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21444 (* 1 = 2.21444 loss)
    I1230 19:39:33.779682 23363 sgd_solver.cpp:106] Iteration 24500, lr = 0.000276525
    I1230 19:39:55.590658 23363 solver.cpp:237] Iteration 24600, loss = 3.9454
    I1230 19:39:55.590786 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 19:39:55.590798 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:39:55.590808 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58888 (* 1 = 1.58888 loss)
    I1230 19:39:55.590817 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35652 (* 1 = 2.35652 loss)
    I1230 19:39:55.590827 23363 sgd_solver.cpp:106] Iteration 24600, lr = 0.000275925
    I1230 19:40:17.366322 23363 solver.cpp:237] Iteration 24700, loss = 3.82444
    I1230 19:40:17.366371 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:40:17.366384 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:40:17.366395 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39573 (* 1 = 1.39573 loss)
    I1230 19:40:17.366406 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42871 (* 1 = 2.42871 loss)
    I1230 19:40:17.366416 23363 sgd_solver.cpp:106] Iteration 24700, lr = 0.000275328
    I1230 19:40:39.000674 23363 solver.cpp:237] Iteration 24800, loss = 4.03653
    I1230 19:40:39.000833 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:40:39.000849 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:40:39.000859 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6856 (* 1 = 1.6856 loss)
    I1230 19:40:39.000867 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35093 (* 1 = 2.35093 loss)
    I1230 19:40:39.000877 23363 sgd_solver.cpp:106] Iteration 24800, lr = 0.000274735
    I1230 19:41:00.800228 23363 solver.cpp:237] Iteration 24900, loss = 3.38589
    I1230 19:41:00.800273 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:41:00.800283 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 19:41:00.800293 23363 solver.cpp:253]     Train net output #2: loss_c = 1.33413 (* 1 = 1.33413 loss)
    I1230 19:41:00.800304 23363 solver.cpp:253]     Train net output #3: loss_f = 2.05176 (* 1 = 2.05176 loss)
    I1230 19:41:00.800314 23363 sgd_solver.cpp:106] Iteration 24900, lr = 0.000274144
    I1230 19:41:22.296912 23363 solver.cpp:341] Iteration 25000, Testing net (#0)
    I1230 19:41:30.363411 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.520166
    I1230 19:41:30.363466 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.391083
    I1230 19:41:30.363487 23363 solver.cpp:409]     Test net output #2: loss_c = 1.5212 (* 1 = 1.5212 loss)
    I1230 19:41:30.363503 23363 solver.cpp:409]     Test net output #3: loss_f = 2.33894 (* 1 = 2.33894 loss)
    I1230 19:41:30.479708 23363 solver.cpp:237] Iteration 25000, loss = 3.7703
    I1230 19:41:30.479759 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:41:30.479776 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 19:41:30.479794 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51753 (* 1 = 1.51753 loss)
    I1230 19:41:30.479810 23363 solver.cpp:253]     Train net output #3: loss_f = 2.25278 (* 1 = 2.25278 loss)
    I1230 19:41:30.479827 23363 sgd_solver.cpp:106] Iteration 25000, lr = 0.000273556
    I1230 19:41:52.232960 23363 solver.cpp:237] Iteration 25100, loss = 4.04741
    I1230 19:41:52.233009 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:41:52.233019 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:41:52.233042 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62233 (* 1 = 1.62233 loss)
    I1230 19:41:52.233053 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42508 (* 1 = 2.42508 loss)
    I1230 19:41:52.233074 23363 sgd_solver.cpp:106] Iteration 25100, lr = 0.000272972
    I1230 19:42:13.754323 23363 solver.cpp:237] Iteration 25200, loss = 3.8952
    I1230 19:42:13.754554 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 19:42:13.754570 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:42:13.754593 23363 solver.cpp:253]     Train net output #2: loss_c = 1.43479 (* 1 = 1.43479 loss)
    I1230 19:42:13.754603 23363 solver.cpp:253]     Train net output #3: loss_f = 2.46042 (* 1 = 2.46042 loss)
    I1230 19:42:13.754612 23363 sgd_solver.cpp:106] Iteration 25200, lr = 0.00027239
    I1230 19:42:35.043452 23363 solver.cpp:237] Iteration 25300, loss = 4.11946
    I1230 19:42:35.043495 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.39
    I1230 19:42:35.043505 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:42:35.043514 23363 solver.cpp:253]     Train net output #2: loss_c = 1.72639 (* 1 = 1.72639 loss)
    I1230 19:42:35.043524 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39307 (* 1 = 2.39307 loss)
    I1230 19:42:35.043531 23363 sgd_solver.cpp:106] Iteration 25300, lr = 0.000271811
    I1230 19:42:56.906453 23363 solver.cpp:237] Iteration 25400, loss = 3.42483
    I1230 19:42:56.906569 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 19:42:56.906582 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 19:42:56.906594 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35628 (* 1 = 1.35628 loss)
    I1230 19:42:56.906602 23363 solver.cpp:253]     Train net output #3: loss_f = 2.06855 (* 1 = 2.06855 loss)
    I1230 19:42:56.906613 23363 sgd_solver.cpp:106] Iteration 25400, lr = 0.000271235
    I1230 19:43:18.606766 23363 solver.cpp:237] Iteration 25500, loss = 3.73515
    I1230 19:43:18.606822 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 19:43:18.606837 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:43:18.606850 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4391 (* 1 = 1.4391 loss)
    I1230 19:43:18.606863 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29606 (* 1 = 2.29606 loss)
    I1230 19:43:18.606876 23363 sgd_solver.cpp:106] Iteration 25500, lr = 0.000270662
    I1230 19:43:40.270603 23363 solver.cpp:237] Iteration 25600, loss = 3.84498
    I1230 19:43:40.270721 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:43:40.270736 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 19:43:40.270750 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56979 (* 1 = 1.56979 loss)
    I1230 19:43:40.270759 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27519 (* 1 = 2.27519 loss)
    I1230 19:43:40.270771 23363 sgd_solver.cpp:106] Iteration 25600, lr = 0.000270091
    I1230 19:44:02.363164 23363 solver.cpp:237] Iteration 25700, loss = 3.75392
    I1230 19:44:02.363209 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:44:02.363221 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:44:02.363234 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37482 (* 1 = 1.37482 loss)
    I1230 19:44:02.363245 23363 solver.cpp:253]     Train net output #3: loss_f = 2.3791 (* 1 = 2.3791 loss)
    I1230 19:44:02.363257 23363 sgd_solver.cpp:106] Iteration 25700, lr = 0.000269524
    I1230 19:44:24.253684 23363 solver.cpp:237] Iteration 25800, loss = 4.11236
    I1230 19:44:24.253826 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:44:24.253837 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:44:24.253849 23363 solver.cpp:253]     Train net output #2: loss_c = 1.75427 (* 1 = 1.75427 loss)
    I1230 19:44:24.253856 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35809 (* 1 = 2.35809 loss)
    I1230 19:44:24.253865 23363 sgd_solver.cpp:106] Iteration 25800, lr = 0.000268959
    I1230 19:44:45.993731 23363 solver.cpp:237] Iteration 25900, loss = 3.39117
    I1230 19:44:45.993768 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:44:45.993777 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 19:44:45.993788 23363 solver.cpp:253]     Train net output #2: loss_c = 1.33879 (* 1 = 1.33879 loss)
    I1230 19:44:45.993796 23363 solver.cpp:253]     Train net output #3: loss_f = 2.05237 (* 1 = 2.05237 loss)
    I1230 19:44:45.993805 23363 sgd_solver.cpp:106] Iteration 25900, lr = 0.000268397
    I1230 19:45:07.616653 23363 solver.cpp:341] Iteration 26000, Testing net (#0)
    I1230 19:45:15.779309 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.525167
    I1230 19:45:15.779356 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.390833
    I1230 19:45:15.779367 23363 solver.cpp:409]     Test net output #2: loss_c = 1.50488 (* 1 = 1.50488 loss)
    I1230 19:45:15.779377 23363 solver.cpp:409]     Test net output #3: loss_f = 2.32782 (* 1 = 2.32782 loss)
    I1230 19:45:15.866679 23363 solver.cpp:237] Iteration 26000, loss = 3.40521
    I1230 19:45:15.866726 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:45:15.866739 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:45:15.866750 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31516 (* 1 = 1.31516 loss)
    I1230 19:45:15.866761 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09005 (* 1 = 2.09005 loss)
    I1230 19:45:15.866773 23363 sgd_solver.cpp:106] Iteration 26000, lr = 0.000267837
    I1230 19:45:37.577028 23363 solver.cpp:237] Iteration 26100, loss = 4.01797
    I1230 19:45:37.577086 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:45:37.577106 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:45:37.577118 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6148 (* 1 = 1.6148 loss)
    I1230 19:45:37.577137 23363 solver.cpp:253]     Train net output #3: loss_f = 2.40317 (* 1 = 2.40317 loss)
    I1230 19:45:37.577147 23363 sgd_solver.cpp:106] Iteration 26100, lr = 0.000267281
    I1230 19:45:59.270617 23363 solver.cpp:237] Iteration 26200, loss = 3.93607
    I1230 19:45:59.270720 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:45:59.270745 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:45:59.270756 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4735 (* 1 = 1.4735 loss)
    I1230 19:45:59.270766 23363 solver.cpp:253]     Train net output #3: loss_f = 2.46257 (* 1 = 2.46257 loss)
    I1230 19:45:59.270776 23363 sgd_solver.cpp:106] Iteration 26200, lr = 0.000266727
    I1230 19:46:21.138085 23363 solver.cpp:237] Iteration 26300, loss = 4.20948
    I1230 19:46:21.138123 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:46:21.138134 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:46:21.138142 23363 solver.cpp:253]     Train net output #2: loss_c = 1.73339 (* 1 = 1.73339 loss)
    I1230 19:46:21.138151 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47609 (* 1 = 2.47609 loss)
    I1230 19:46:21.138160 23363 sgd_solver.cpp:106] Iteration 26300, lr = 0.000266175
    I1230 19:46:42.916208 23363 solver.cpp:237] Iteration 26400, loss = 3.23612
    I1230 19:46:42.916404 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:46:42.916419 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 19:46:42.916429 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29193 (* 1 = 1.29193 loss)
    I1230 19:46:42.916437 23363 solver.cpp:253]     Train net output #3: loss_f = 1.9442 (* 1 = 1.9442 loss)
    I1230 19:46:42.916447 23363 sgd_solver.cpp:106] Iteration 26400, lr = 0.000265627
    I1230 19:47:04.733984 23363 solver.cpp:237] Iteration 26500, loss = 4.00687
    I1230 19:47:04.734033 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:47:04.734045 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:47:04.734055 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55867 (* 1 = 1.55867 loss)
    I1230 19:47:04.734066 23363 solver.cpp:253]     Train net output #3: loss_f = 2.4482 (* 1 = 2.4482 loss)
    I1230 19:47:04.734076 23363 sgd_solver.cpp:106] Iteration 26500, lr = 0.000265081
    I1230 19:47:26.749248 23363 solver.cpp:237] Iteration 26600, loss = 4.01414
    I1230 19:47:26.749379 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:47:26.749392 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:47:26.749402 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62188 (* 1 = 1.62188 loss)
    I1230 19:47:26.749410 23363 solver.cpp:253]     Train net output #3: loss_f = 2.39226 (* 1 = 2.39226 loss)
    I1230 19:47:26.749420 23363 sgd_solver.cpp:106] Iteration 26600, lr = 0.000264537
    I1230 19:47:48.291405 23363 solver.cpp:237] Iteration 26700, loss = 3.96342
    I1230 19:47:48.291465 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:47:48.291483 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 19:47:48.291503 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48914 (* 1 = 1.48914 loss)
    I1230 19:47:48.291520 23363 solver.cpp:253]     Train net output #3: loss_f = 2.47428 (* 1 = 2.47428 loss)
    I1230 19:47:48.291535 23363 sgd_solver.cpp:106] Iteration 26700, lr = 0.000263997
    I1230 19:48:10.270347 23363 solver.cpp:237] Iteration 26800, loss = 4.07318
    I1230 19:48:10.270488 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:48:10.270500 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:48:10.270511 23363 solver.cpp:253]     Train net output #2: loss_c = 1.70611 (* 1 = 1.70611 loss)
    I1230 19:48:10.270519 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36707 (* 1 = 2.36707 loss)
    I1230 19:48:10.270529 23363 sgd_solver.cpp:106] Iteration 26800, lr = 0.000263458
    I1230 19:48:31.803679 23363 solver.cpp:237] Iteration 26900, loss = 3.37699
    I1230 19:48:31.803724 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:48:31.803735 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 19:48:31.803748 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34208 (* 1 = 1.34208 loss)
    I1230 19:48:31.803758 23363 solver.cpp:253]     Train net output #3: loss_f = 2.03491 (* 1 = 2.03491 loss)
    I1230 19:48:31.803769 23363 sgd_solver.cpp:106] Iteration 26900, lr = 0.000262923
    I1230 19:48:53.479444 23363 solver.cpp:341] Iteration 27000, Testing net (#0)
    I1230 19:49:01.583441 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.518917
    I1230 19:49:01.583498 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.389083
    I1230 19:49:01.583514 23363 solver.cpp:409]     Test net output #2: loss_c = 1.52072 (* 1 = 1.52072 loss)
    I1230 19:49:01.583529 23363 solver.cpp:409]     Test net output #3: loss_f = 2.33474 (* 1 = 2.33474 loss)
    I1230 19:49:01.746937 23363 solver.cpp:237] Iteration 27000, loss = 3.6861
    I1230 19:49:01.747011 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:49:01.747026 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:49:01.747054 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40564 (* 1 = 1.40564 loss)
    I1230 19:49:01.747068 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28045 (* 1 = 2.28045 loss)
    I1230 19:49:01.747082 23363 sgd_solver.cpp:106] Iteration 27000, lr = 0.00026239
    I1230 19:49:23.395489 23363 solver.cpp:237] Iteration 27100, loss = 4.05943
    I1230 19:49:23.395526 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 19:49:23.395535 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:49:23.395556 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64137 (* 1 = 1.64137 loss)
    I1230 19:49:23.395566 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41806 (* 1 = 2.41806 loss)
    I1230 19:49:23.395576 23363 sgd_solver.cpp:106] Iteration 27100, lr = 0.000261859
    I1230 19:49:45.074384 23363 solver.cpp:237] Iteration 27200, loss = 3.7238
    I1230 19:49:45.074569 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:49:45.074584 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:49:45.074594 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37125 (* 1 = 1.37125 loss)
    I1230 19:49:45.074601 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35255 (* 1 = 2.35255 loss)
    I1230 19:49:45.074610 23363 sgd_solver.cpp:106] Iteration 27200, lr = 0.000261331
    I1230 19:50:06.828634 23363 solver.cpp:237] Iteration 27300, loss = 4.0504
    I1230 19:50:06.828670 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 19:50:06.828680 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:50:06.828701 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64153 (* 1 = 1.64153 loss)
    I1230 19:50:06.828711 23363 solver.cpp:253]     Train net output #3: loss_f = 2.40887 (* 1 = 2.40887 loss)
    I1230 19:50:06.828722 23363 sgd_solver.cpp:106] Iteration 27300, lr = 0.000260805
    I1230 19:50:28.421161 23363 solver.cpp:237] Iteration 27400, loss = 3.5178
    I1230 19:50:28.421264 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 19:50:28.421277 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 19:50:28.421286 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39248 (* 1 = 1.39248 loss)
    I1230 19:50:28.421295 23363 solver.cpp:253]     Train net output #3: loss_f = 2.12532 (* 1 = 2.12532 loss)
    I1230 19:50:28.421304 23363 sgd_solver.cpp:106] Iteration 27400, lr = 0.000260282
    I1230 19:50:49.910936 23363 solver.cpp:237] Iteration 27500, loss = 3.8438
    I1230 19:50:49.910974 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 19:50:49.910984 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:50:49.910994 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51645 (* 1 = 1.51645 loss)
    I1230 19:50:49.911002 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32736 (* 1 = 2.32736 loss)
    I1230 19:50:49.911011 23363 sgd_solver.cpp:106] Iteration 27500, lr = 0.000259761
    I1230 19:51:11.707681 23363 solver.cpp:237] Iteration 27600, loss = 3.91747
    I1230 19:51:11.707849 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:51:11.707864 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:51:11.707875 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58118 (* 1 = 1.58118 loss)
    I1230 19:51:11.707882 23363 solver.cpp:253]     Train net output #3: loss_f = 2.3363 (* 1 = 2.3363 loss)
    I1230 19:51:11.707892 23363 sgd_solver.cpp:106] Iteration 27600, lr = 0.000259243
    I1230 19:51:33.295879 23363 solver.cpp:237] Iteration 27700, loss = 4.08273
    I1230 19:51:33.295917 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 19:51:33.295928 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:51:33.295938 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56523 (* 1 = 1.56523 loss)
    I1230 19:51:33.295945 23363 solver.cpp:253]     Train net output #3: loss_f = 2.51751 (* 1 = 2.51751 loss)
    I1230 19:51:33.295954 23363 sgd_solver.cpp:106] Iteration 27700, lr = 0.000258727
    I1230 19:51:54.990795 23363 solver.cpp:237] Iteration 27800, loss = 4.08421
    I1230 19:51:54.991011 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:51:54.991039 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:51:54.991058 23363 solver.cpp:253]     Train net output #2: loss_c = 1.74328 (* 1 = 1.74328 loss)
    I1230 19:51:54.991075 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34093 (* 1 = 2.34093 loss)
    I1230 19:51:54.991091 23363 sgd_solver.cpp:106] Iteration 27800, lr = 0.000258214
    I1230 19:52:16.676307 23363 solver.cpp:237] Iteration 27900, loss = 3.26814
    I1230 19:52:16.676367 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 19:52:16.676385 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.52
    I1230 19:52:16.676404 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34885 (* 1 = 1.34885 loss)
    I1230 19:52:16.676420 23363 solver.cpp:253]     Train net output #3: loss_f = 1.9193 (* 1 = 1.9193 loss)
    I1230 19:52:16.676437 23363 sgd_solver.cpp:106] Iteration 27900, lr = 0.000257702
    I1230 19:52:38.502650 23363 solver.cpp:341] Iteration 28000, Testing net (#0)
    I1230 19:52:46.920908 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.51725
    I1230 19:52:46.920946 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.388667
    I1230 19:52:46.920958 23363 solver.cpp:409]     Test net output #2: loss_c = 1.53249 (* 1 = 1.53249 loss)
    I1230 19:52:46.920969 23363 solver.cpp:409]     Test net output #3: loss_f = 2.35134 (* 1 = 2.35134 loss)
    I1230 19:52:47.025931 23363 solver.cpp:237] Iteration 28000, loss = 3.72982
    I1230 19:52:47.025977 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 19:52:47.025987 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:52:47.025998 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42124 (* 1 = 1.42124 loss)
    I1230 19:52:47.026008 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30857 (* 1 = 2.30857 loss)
    I1230 19:52:47.026018 23363 sgd_solver.cpp:106] Iteration 28000, lr = 0.000257194
    I1230 19:53:08.777300 23363 solver.cpp:237] Iteration 28100, loss = 4.07766
    I1230 19:53:08.777442 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:53:08.777458 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:53:08.777469 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64123 (* 1 = 1.64123 loss)
    I1230 19:53:08.777478 23363 solver.cpp:253]     Train net output #3: loss_f = 2.43643 (* 1 = 2.43643 loss)
    I1230 19:53:08.777487 23363 sgd_solver.cpp:106] Iteration 28100, lr = 0.000256687
    I1230 19:53:30.546931 23363 solver.cpp:237] Iteration 28200, loss = 3.84096
    I1230 19:53:30.546977 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:53:30.546986 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:53:30.546995 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42033 (* 1 = 1.42033 loss)
    I1230 19:53:30.547004 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42063 (* 1 = 2.42063 loss)
    I1230 19:53:30.547013 23363 sgd_solver.cpp:106] Iteration 28200, lr = 0.000256183
    I1230 19:53:52.463095 23363 solver.cpp:237] Iteration 28300, loss = 3.98736
    I1230 19:53:52.463235 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:53:52.463253 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:53:52.463264 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63578 (* 1 = 1.63578 loss)
    I1230 19:53:52.463275 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35158 (* 1 = 2.35158 loss)
    I1230 19:53:52.463287 23363 sgd_solver.cpp:106] Iteration 28300, lr = 0.000255681
    I1230 19:54:14.254964 23363 solver.cpp:237] Iteration 28400, loss = 3.13213
    I1230 19:54:14.255009 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 19:54:14.255023 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.49
    I1230 19:54:14.255036 23363 solver.cpp:253]     Train net output #2: loss_c = 1.23098 (* 1 = 1.23098 loss)
    I1230 19:54:14.255049 23363 solver.cpp:253]     Train net output #3: loss_f = 1.90115 (* 1 = 1.90115 loss)
    I1230 19:54:14.255061 23363 sgd_solver.cpp:106] Iteration 28400, lr = 0.000255182
    I1230 19:54:36.318631 23363 solver.cpp:237] Iteration 28500, loss = 3.44733
    I1230 19:54:36.318744 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:54:36.318763 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:54:36.318780 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31451 (* 1 = 1.31451 loss)
    I1230 19:54:36.318797 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13282 (* 1 = 2.13282 loss)
    I1230 19:54:36.318812 23363 sgd_solver.cpp:106] Iteration 28500, lr = 0.000254684
    I1230 19:54:58.042755 23363 solver.cpp:237] Iteration 28600, loss = 3.77652
    I1230 19:54:58.042804 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 19:54:58.042815 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:54:58.042827 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55667 (* 1 = 1.55667 loss)
    I1230 19:54:58.042839 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21985 (* 1 = 2.21985 loss)
    I1230 19:54:58.042848 23363 sgd_solver.cpp:106] Iteration 28600, lr = 0.000254189
    I1230 19:55:19.844661 23363 solver.cpp:237] Iteration 28700, loss = 3.78049
    I1230 19:55:19.847846 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 19:55:19.847903 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1230 19:55:19.847928 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41344 (* 1 = 1.41344 loss)
    I1230 19:55:19.847945 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36705 (* 1 = 2.36705 loss)
    I1230 19:55:19.847960 23363 sgd_solver.cpp:106] Iteration 28700, lr = 0.000253697
    I1230 19:55:41.726081 23363 solver.cpp:237] Iteration 28800, loss = 3.99727
    I1230 19:55:41.726128 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1230 19:55:41.726140 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 19:55:41.726150 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71687 (* 1 = 1.71687 loss)
    I1230 19:55:41.726160 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2804 (* 1 = 2.2804 loss)
    I1230 19:55:41.726168 23363 sgd_solver.cpp:106] Iteration 28800, lr = 0.000253206
    I1230 19:56:03.578795 23363 solver.cpp:237] Iteration 28900, loss = 3.44657
    I1230 19:56:03.578923 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 19:56:03.578936 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 19:56:03.578948 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37268 (* 1 = 1.37268 loss)
    I1230 19:56:03.578956 23363 solver.cpp:253]     Train net output #3: loss_f = 2.07389 (* 1 = 2.07389 loss)
    I1230 19:56:03.578966 23363 sgd_solver.cpp:106] Iteration 28900, lr = 0.000252718
    I1230 19:56:24.911571 23363 solver.cpp:341] Iteration 29000, Testing net (#0)
    I1230 19:56:33.216095 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.520917
    I1230 19:56:33.216138 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.386083
    I1230 19:56:33.216151 23363 solver.cpp:409]     Test net output #2: loss_c = 1.51125 (* 1 = 1.51125 loss)
    I1230 19:56:33.216162 23363 solver.cpp:409]     Test net output #3: loss_f = 2.33078 (* 1 = 2.33078 loss)
    I1230 19:56:33.312477 23363 solver.cpp:237] Iteration 29000, loss = 3.69841
    I1230 19:56:33.312525 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:56:33.312536 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 19:56:33.312547 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48414 (* 1 = 1.48414 loss)
    I1230 19:56:33.312569 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21427 (* 1 = 2.21427 loss)
    I1230 19:56:33.312579 23363 sgd_solver.cpp:106] Iteration 29000, lr = 0.000252232
    I1230 19:56:54.935711 23363 solver.cpp:237] Iteration 29100, loss = 3.98827
    I1230 19:56:54.935859 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 19:56:54.935875 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 19:56:54.935888 23363 solver.cpp:253]     Train net output #2: loss_c = 1.64504 (* 1 = 1.64504 loss)
    I1230 19:56:54.935897 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34323 (* 1 = 2.34323 loss)
    I1230 19:56:54.935907 23363 sgd_solver.cpp:106] Iteration 29100, lr = 0.000251748
    I1230 19:57:16.534010 23363 solver.cpp:237] Iteration 29200, loss = 3.50537
    I1230 19:57:16.534045 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:57:16.534055 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 19:57:16.534065 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26083 (* 1 = 1.26083 loss)
    I1230 19:57:16.534073 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24455 (* 1 = 2.24455 loss)
    I1230 19:57:16.534082 23363 sgd_solver.cpp:106] Iteration 29200, lr = 0.000251266
    I1230 19:57:38.301373 23363 solver.cpp:237] Iteration 29300, loss = 4.14393
    I1230 19:57:38.301551 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 19:57:38.301576 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 19:57:38.301595 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71692 (* 1 = 1.71692 loss)
    I1230 19:57:38.301611 23363 solver.cpp:253]     Train net output #3: loss_f = 2.42701 (* 1 = 2.42701 loss)
    I1230 19:57:38.301626 23363 sgd_solver.cpp:106] Iteration 29300, lr = 0.000250786
    I1230 19:58:00.146411 23363 solver.cpp:237] Iteration 29400, loss = 3.30409
    I1230 19:58:00.146448 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 19:58:00.146457 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 19:58:00.146467 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31962 (* 1 = 1.31962 loss)
    I1230 19:58:00.146476 23363 solver.cpp:253]     Train net output #3: loss_f = 1.98447 (* 1 = 1.98447 loss)
    I1230 19:58:00.146486 23363 sgd_solver.cpp:106] Iteration 29400, lr = 0.000250309
    I1230 19:58:21.942414 23363 solver.cpp:237] Iteration 29500, loss = 3.76858
    I1230 19:58:21.942543 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 19:58:21.942554 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:58:21.942564 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44415 (* 1 = 1.44415 loss)
    I1230 19:58:21.942574 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32442 (* 1 = 2.32442 loss)
    I1230 19:58:21.942582 23363 sgd_solver.cpp:106] Iteration 29500, lr = 0.000249833
    I1230 19:58:43.588636 23363 solver.cpp:237] Iteration 29600, loss = 3.98458
    I1230 19:58:43.588680 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 19:58:43.588690 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 19:58:43.588701 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62037 (* 1 = 1.62037 loss)
    I1230 19:58:43.588708 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36421 (* 1 = 2.36421 loss)
    I1230 19:58:43.588716 23363 sgd_solver.cpp:106] Iteration 29600, lr = 0.00024936
    I1230 19:59:05.363867 23363 solver.cpp:237] Iteration 29700, loss = 3.60034
    I1230 19:59:05.364028 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 19:59:05.364040 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 19:59:05.364049 23363 solver.cpp:253]     Train net output #2: loss_c = 1.33314 (* 1 = 1.33314 loss)
    I1230 19:59:05.364058 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2672 (* 1 = 2.2672 loss)
    I1230 19:59:05.364066 23363 sgd_solver.cpp:106] Iteration 29700, lr = 0.000248889
    I1230 19:59:27.258380 23363 solver.cpp:237] Iteration 29800, loss = 3.95189
    I1230 19:59:27.258416 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 19:59:27.258425 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 19:59:27.258435 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63541 (* 1 = 1.63541 loss)
    I1230 19:59:27.258445 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31649 (* 1 = 2.31649 loss)
    I1230 19:59:27.258453 23363 sgd_solver.cpp:106] Iteration 29800, lr = 0.00024842
    I1230 19:59:49.190554 23363 solver.cpp:237] Iteration 29900, loss = 3.21745
    I1230 19:59:49.190721 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 19:59:49.190735 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.52
    I1230 19:59:49.190747 23363 solver.cpp:253]     Train net output #2: loss_c = 1.2827 (* 1 = 1.2827 loss)
    I1230 19:59:49.190755 23363 solver.cpp:253]     Train net output #3: loss_f = 1.93475 (* 1 = 1.93475 loss)
    I1230 19:59:49.190765 23363 sgd_solver.cpp:106] Iteration 29900, lr = 0.000247952
    I1230 20:00:10.771775 23363 solver.cpp:341] Iteration 30000, Testing net (#0)
    I1230 20:00:18.959815 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.525167
    I1230 20:00:18.959867 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.394667
    I1230 20:00:18.959879 23363 solver.cpp:409]     Test net output #2: loss_c = 1.50338 (* 1 = 1.50338 loss)
    I1230 20:00:18.959890 23363 solver.cpp:409]     Test net output #3: loss_f = 2.30911 (* 1 = 2.30911 loss)
    I1230 20:00:19.107834 23363 solver.cpp:237] Iteration 30000, loss = 3.69913
    I1230 20:00:19.107880 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:00:19.107892 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:00:19.107903 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45898 (* 1 = 1.45898 loss)
    I1230 20:00:19.107914 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24014 (* 1 = 2.24014 loss)
    I1230 20:00:19.107925 23363 sgd_solver.cpp:106] Iteration 30000, lr = 0.000247487
    I1230 20:00:41.289892 23363 solver.cpp:237] Iteration 30100, loss = 3.87669
    I1230 20:00:41.290029 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:00:41.290055 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:00:41.290069 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56801 (* 1 = 1.56801 loss)
    I1230 20:00:41.290081 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30868 (* 1 = 2.30868 loss)
    I1230 20:00:41.290091 23363 sgd_solver.cpp:106] Iteration 30100, lr = 0.000247024
    I1230 20:01:03.268498 23363 solver.cpp:237] Iteration 30200, loss = 3.6853
    I1230 20:01:03.268558 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:01:03.268569 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:01:03.268582 23363 solver.cpp:253]     Train net output #2: loss_c = 1.38441 (* 1 = 1.38441 loss)
    I1230 20:01:03.268592 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30089 (* 1 = 2.30089 loss)
    I1230 20:01:03.268602 23363 sgd_solver.cpp:106] Iteration 30200, lr = 0.000246563
    I1230 20:01:20.450855 23363 solver.cpp:237] Iteration 30300, loss = 4.00146
    I1230 20:01:20.450999 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 20:01:20.451012 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:01:20.451022 23363 solver.cpp:253]     Train net output #2: loss_c = 1.67811 (* 1 = 1.67811 loss)
    I1230 20:01:20.451031 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32334 (* 1 = 2.32334 loss)
    I1230 20:01:20.451040 23363 sgd_solver.cpp:106] Iteration 30300, lr = 0.000246104
    I1230 20:01:37.450713 23363 solver.cpp:237] Iteration 30400, loss = 3.35864
    I1230 20:01:37.450768 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:01:37.450781 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:01:37.450793 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32091 (* 1 = 1.32091 loss)
    I1230 20:01:37.450804 23363 solver.cpp:253]     Train net output #3: loss_f = 2.03773 (* 1 = 2.03773 loss)
    I1230 20:01:37.450816 23363 sgd_solver.cpp:106] Iteration 30400, lr = 0.000245647
    I1230 20:01:52.682862 23363 solver.cpp:237] Iteration 30500, loss = 3.5518
    I1230 20:01:52.683015 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:01:52.683028 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:01:52.683038 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39325 (* 1 = 1.39325 loss)
    I1230 20:01:52.683048 23363 solver.cpp:253]     Train net output #3: loss_f = 2.15855 (* 1 = 2.15855 loss)
    I1230 20:01:52.683058 23363 sgd_solver.cpp:106] Iteration 30500, lr = 0.000245192
    I1230 20:02:08.926539 23363 solver.cpp:237] Iteration 30600, loss = 3.77784
    I1230 20:02:08.926587 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:02:08.926597 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:02:08.926609 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55942 (* 1 = 1.55942 loss)
    I1230 20:02:08.926617 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21842 (* 1 = 2.21842 loss)
    I1230 20:02:08.926626 23363 sgd_solver.cpp:106] Iteration 30600, lr = 0.000244739
    I1230 20:02:25.330240 23363 solver.cpp:237] Iteration 30700, loss = 3.73784
    I1230 20:02:25.330392 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:02:25.330405 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:02:25.330415 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42198 (* 1 = 1.42198 loss)
    I1230 20:02:25.330423 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31585 (* 1 = 2.31585 loss)
    I1230 20:02:25.330432 23363 sgd_solver.cpp:106] Iteration 30700, lr = 0.000244288
    I1230 20:02:42.388056 23363 solver.cpp:237] Iteration 30800, loss = 4.0255
    I1230 20:02:42.388094 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1230 20:02:42.388104 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:02:42.388114 23363 solver.cpp:253]     Train net output #2: loss_c = 1.71893 (* 1 = 1.71893 loss)
    I1230 20:02:42.388123 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30657 (* 1 = 2.30657 loss)
    I1230 20:02:42.388133 23363 sgd_solver.cpp:106] Iteration 30800, lr = 0.000243839
    I1230 20:02:58.547322 23363 solver.cpp:237] Iteration 30900, loss = 3.09992
    I1230 20:02:58.547426 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:02:58.547441 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:02:58.547454 23363 solver.cpp:253]     Train net output #2: loss_c = 1.2444 (* 1 = 1.2444 loss)
    I1230 20:02:58.547466 23363 solver.cpp:253]     Train net output #3: loss_f = 1.85552 (* 1 = 1.85552 loss)
    I1230 20:02:58.547477 23363 sgd_solver.cpp:106] Iteration 30900, lr = 0.000243392
    I1230 20:03:13.546319 23363 solver.cpp:341] Iteration 31000, Testing net (#0)
    I1230 20:03:19.525828 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.5275
    I1230 20:03:19.525868 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.39875
    I1230 20:03:19.525881 23363 solver.cpp:409]     Test net output #2: loss_c = 1.50742 (* 1 = 1.50742 loss)
    I1230 20:03:19.525890 23363 solver.cpp:409]     Test net output #3: loss_f = 2.30899 (* 1 = 2.30899 loss)
    I1230 20:03:19.595341 23363 solver.cpp:237] Iteration 31000, loss = 3.44233
    I1230 20:03:19.595391 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:03:19.595404 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:03:19.595419 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32316 (* 1 = 1.32316 loss)
    I1230 20:03:19.595432 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11917 (* 1 = 2.11917 loss)
    I1230 20:03:19.595449 23363 sgd_solver.cpp:106] Iteration 31000, lr = 0.000242946
    I1230 20:03:35.659981 23363 solver.cpp:237] Iteration 31100, loss = 3.76368
    I1230 20:03:35.660138 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:03:35.660156 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:03:35.660168 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48836 (* 1 = 1.48836 loss)
    I1230 20:03:35.660177 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27532 (* 1 = 2.27532 loss)
    I1230 20:03:35.660197 23363 sgd_solver.cpp:106] Iteration 31100, lr = 0.000242503
    I1230 20:03:51.045476 23363 solver.cpp:237] Iteration 31200, loss = 3.81436
    I1230 20:03:51.045536 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:03:51.045554 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:03:51.045573 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44727 (* 1 = 1.44727 loss)
    I1230 20:03:51.045589 23363 solver.cpp:253]     Train net output #3: loss_f = 2.36709 (* 1 = 2.36709 loss)
    I1230 20:03:51.045606 23363 sgd_solver.cpp:106] Iteration 31200, lr = 0.000242061
    I1230 20:04:06.285527 23363 solver.cpp:237] Iteration 31300, loss = 4.04967
    I1230 20:04:06.285635 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 20:04:06.285652 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:04:06.285665 23363 solver.cpp:253]     Train net output #2: loss_c = 1.67366 (* 1 = 1.67366 loss)
    I1230 20:04:06.285677 23363 solver.cpp:253]     Train net output #3: loss_f = 2.37601 (* 1 = 2.37601 loss)
    I1230 20:04:06.285688 23363 sgd_solver.cpp:106] Iteration 31300, lr = 0.000241621
    I1230 20:04:23.488517 23363 solver.cpp:237] Iteration 31400, loss = 3.34328
    I1230 20:04:23.488564 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:04:23.488575 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:04:23.488587 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31238 (* 1 = 1.31238 loss)
    I1230 20:04:23.488600 23363 solver.cpp:253]     Train net output #3: loss_f = 2.0309 (* 1 = 2.0309 loss)
    I1230 20:04:23.488611 23363 sgd_solver.cpp:106] Iteration 31400, lr = 0.000241184
    I1230 20:04:39.424054 23363 solver.cpp:237] Iteration 31500, loss = 3.5171
    I1230 20:04:39.424306 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:04:39.424322 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.52
    I1230 20:04:39.424334 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40426 (* 1 = 1.40426 loss)
    I1230 20:04:39.424352 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11284 (* 1 = 2.11284 loss)
    I1230 20:04:39.424362 23363 sgd_solver.cpp:106] Iteration 31500, lr = 0.000240748
    I1230 20:04:55.499933 23363 solver.cpp:237] Iteration 31600, loss = 3.84228
    I1230 20:04:55.499972 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:04:55.499982 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:04:55.499992 23363 solver.cpp:253]     Train net output #2: loss_c = 1.54894 (* 1 = 1.54894 loss)
    I1230 20:04:55.500001 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29334 (* 1 = 2.29334 loss)
    I1230 20:04:55.500010 23363 sgd_solver.cpp:106] Iteration 31600, lr = 0.000240313
    I1230 20:05:11.575397 23363 solver.cpp:237] Iteration 31700, loss = 3.68625
    I1230 20:05:11.575577 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:05:11.575592 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:05:11.575603 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39568 (* 1 = 1.39568 loss)
    I1230 20:05:11.575610 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29056 (* 1 = 2.29056 loss)
    I1230 20:05:11.575620 23363 sgd_solver.cpp:106] Iteration 31700, lr = 0.000239881
    I1230 20:05:27.412926 23363 solver.cpp:237] Iteration 31800, loss = 3.85935
    I1230 20:05:27.412974 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 20:05:27.412983 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:05:27.412993 23363 solver.cpp:253]     Train net output #2: loss_c = 1.59696 (* 1 = 1.59696 loss)
    I1230 20:05:27.413002 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26239 (* 1 = 2.26239 loss)
    I1230 20:05:27.413012 23363 sgd_solver.cpp:106] Iteration 31800, lr = 0.000239451
    I1230 20:05:43.236651 23363 solver.cpp:237] Iteration 31900, loss = 3.37709
    I1230 20:05:43.236814 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:05:43.236845 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:05:43.236865 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37395 (* 1 = 1.37395 loss)
    I1230 20:05:43.236882 23363 solver.cpp:253]     Train net output #3: loss_f = 2.00314 (* 1 = 2.00314 loss)
    I1230 20:05:43.236898 23363 sgd_solver.cpp:106] Iteration 31900, lr = 0.000239022
    I1230 20:05:58.522091 23363 solver.cpp:341] Iteration 32000, Testing net (#0)
    I1230 20:06:04.164829 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.53625
    I1230 20:06:04.164876 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.405833
    I1230 20:06:04.164887 23363 solver.cpp:409]     Test net output #2: loss_c = 1.48236 (* 1 = 1.48236 loss)
    I1230 20:06:04.164896 23363 solver.cpp:409]     Test net output #3: loss_f = 2.27497 (* 1 = 2.27497 loss)
    I1230 20:06:04.239259 23363 solver.cpp:237] Iteration 32000, loss = 3.46398
    I1230 20:06:04.239305 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1230 20:06:04.239317 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:06:04.239330 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32366 (* 1 = 1.32366 loss)
    I1230 20:06:04.239341 23363 solver.cpp:253]     Train net output #3: loss_f = 2.14032 (* 1 = 2.14032 loss)
    I1230 20:06:04.239353 23363 sgd_solver.cpp:106] Iteration 32000, lr = 0.000238595
    I1230 20:06:19.492243 23363 solver.cpp:237] Iteration 32100, loss = 3.8278
    I1230 20:06:19.492420 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:06:19.492444 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:06:19.492462 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5536 (* 1 = 1.5536 loss)
    I1230 20:06:19.492480 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27419 (* 1 = 2.27419 loss)
    I1230 20:06:19.492496 23363 sgd_solver.cpp:106] Iteration 32100, lr = 0.00023817
    I1230 20:06:34.689616 23363 solver.cpp:237] Iteration 32200, loss = 3.58148
    I1230 20:06:34.689663 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:06:34.689673 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 20:06:34.689683 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31699 (* 1 = 1.31699 loss)
    I1230 20:06:34.689692 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26448 (* 1 = 2.26448 loss)
    I1230 20:06:34.689702 23363 sgd_solver.cpp:106] Iteration 32200, lr = 0.000237746
    I1230 20:06:50.351866 23363 solver.cpp:237] Iteration 32300, loss = 3.87816
    I1230 20:06:50.352044 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:06:50.352056 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:06:50.352066 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58765 (* 1 = 1.58765 loss)
    I1230 20:06:50.352074 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29052 (* 1 = 2.29052 loss)
    I1230 20:06:50.352084 23363 sgd_solver.cpp:106] Iteration 32300, lr = 0.000237325
    I1230 20:07:05.489727 23363 solver.cpp:237] Iteration 32400, loss = 3.43937
    I1230 20:07:05.489769 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:07:05.489780 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:07:05.489791 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41531 (* 1 = 1.41531 loss)
    I1230 20:07:05.489801 23363 solver.cpp:253]     Train net output #3: loss_f = 2.02406 (* 1 = 2.02406 loss)
    I1230 20:07:05.489812 23363 sgd_solver.cpp:106] Iteration 32400, lr = 0.000236905
    I1230 20:07:20.563082 23363 solver.cpp:237] Iteration 32500, loss = 3.6241
    I1230 20:07:20.563231 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:07:20.563253 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:07:20.563263 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40691 (* 1 = 1.40691 loss)
    I1230 20:07:20.563271 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21719 (* 1 = 2.21719 loss)
    I1230 20:07:20.563280 23363 sgd_solver.cpp:106] Iteration 32500, lr = 0.000236486
    I1230 20:07:36.990814 23363 solver.cpp:237] Iteration 32600, loss = 3.9648
    I1230 20:07:36.990854 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:07:36.990862 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 20:07:36.990874 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62206 (* 1 = 1.62206 loss)
    I1230 20:07:36.990882 23363 solver.cpp:253]     Train net output #3: loss_f = 2.34274 (* 1 = 2.34274 loss)
    I1230 20:07:36.990891 23363 sgd_solver.cpp:106] Iteration 32600, lr = 0.00023607
    I1230 20:07:53.375205 23363 solver.cpp:237] Iteration 32700, loss = 3.66851
    I1230 20:07:53.375355 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:07:53.375380 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:07:53.375398 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35045 (* 1 = 1.35045 loss)
    I1230 20:07:53.375413 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31806 (* 1 = 2.31806 loss)
    I1230 20:07:53.375429 23363 sgd_solver.cpp:106] Iteration 32700, lr = 0.000235655
    I1230 20:08:09.417667 23363 solver.cpp:237] Iteration 32800, loss = 3.81871
    I1230 20:08:09.417714 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:08:09.417724 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:08:09.417733 23363 solver.cpp:253]     Train net output #2: loss_c = 1.57067 (* 1 = 1.57067 loss)
    I1230 20:08:09.417742 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24804 (* 1 = 2.24804 loss)
    I1230 20:08:09.417752 23363 sgd_solver.cpp:106] Iteration 32800, lr = 0.000235242
    I1230 20:08:25.924906 23363 solver.cpp:237] Iteration 32900, loss = 3.23372
    I1230 20:08:25.925034 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:08:25.925047 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:08:25.925057 23363 solver.cpp:253]     Train net output #2: loss_c = 1.27077 (* 1 = 1.27077 loss)
    I1230 20:08:25.925065 23363 solver.cpp:253]     Train net output #3: loss_f = 1.96295 (* 1 = 1.96295 loss)
    I1230 20:08:25.925076 23363 sgd_solver.cpp:106] Iteration 32900, lr = 0.000234831
    I1230 20:08:42.847544 23363 solver.cpp:341] Iteration 33000, Testing net (#0)
    I1230 20:08:48.551956 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.527417
    I1230 20:08:48.552002 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.39825
    I1230 20:08:48.552016 23363 solver.cpp:409]     Test net output #2: loss_c = 1.51245 (* 1 = 1.51245 loss)
    I1230 20:08:48.552027 23363 solver.cpp:409]     Test net output #3: loss_f = 2.30477 (* 1 = 2.30477 loss)
    I1230 20:08:48.619045 23363 solver.cpp:237] Iteration 33000, loss = 3.58613
    I1230 20:08:48.619088 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:08:48.619101 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:08:48.619112 23363 solver.cpp:253]     Train net output #2: loss_c = 1.33177 (* 1 = 1.33177 loss)
    I1230 20:08:48.619123 23363 solver.cpp:253]     Train net output #3: loss_f = 2.25436 (* 1 = 2.25436 loss)
    I1230 20:08:48.619135 23363 sgd_solver.cpp:106] Iteration 33000, lr = 0.000234421
    I1230 20:09:03.744212 23363 solver.cpp:237] Iteration 33100, loss = 3.80873
    I1230 20:09:03.744312 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:09:03.744325 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:09:03.744338 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52471 (* 1 = 1.52471 loss)
    I1230 20:09:03.744349 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28402 (* 1 = 2.28402 loss)
    I1230 20:09:03.744360 23363 sgd_solver.cpp:106] Iteration 33100, lr = 0.000234013
    I1230 20:09:19.459239 23363 solver.cpp:237] Iteration 33200, loss = 3.82785
    I1230 20:09:19.459278 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:09:19.459290 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:09:19.459300 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41067 (* 1 = 1.41067 loss)
    I1230 20:09:19.459307 23363 solver.cpp:253]     Train net output #3: loss_f = 2.41718 (* 1 = 2.41718 loss)
    I1230 20:09:19.459327 23363 sgd_solver.cpp:106] Iteration 33200, lr = 0.000233607
    I1230 20:09:34.840347 23363 solver.cpp:237] Iteration 33300, loss = 3.93569
    I1230 20:09:34.840523 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1230 20:09:34.840538 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:09:34.840548 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62403 (* 1 = 1.62403 loss)
    I1230 20:09:34.840558 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31165 (* 1 = 2.31165 loss)
    I1230 20:09:34.840567 23363 sgd_solver.cpp:106] Iteration 33300, lr = 0.000233202
    I1230 20:09:50.007210 23363 solver.cpp:237] Iteration 33400, loss = 3.48643
    I1230 20:09:50.007256 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:09:50.007267 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:09:50.007278 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42896 (* 1 = 1.42896 loss)
    I1230 20:09:50.007288 23363 solver.cpp:253]     Train net output #3: loss_f = 2.05747 (* 1 = 2.05747 loss)
    I1230 20:09:50.007298 23363 sgd_solver.cpp:106] Iteration 33400, lr = 0.000232799
    I1230 20:10:05.189265 23363 solver.cpp:237] Iteration 33500, loss = 3.52916
    I1230 20:10:05.189394 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:10:05.189414 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:10:05.189434 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35671 (* 1 = 1.35671 loss)
    I1230 20:10:05.189450 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17245 (* 1 = 2.17245 loss)
    I1230 20:10:05.189466 23363 sgd_solver.cpp:106] Iteration 33500, lr = 0.000232397
    I1230 20:10:20.429580 23363 solver.cpp:237] Iteration 33600, loss = 3.93152
    I1230 20:10:20.429622 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:10:20.429635 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:10:20.429647 23363 solver.cpp:253]     Train net output #2: loss_c = 1.54806 (* 1 = 1.54806 loss)
    I1230 20:10:20.429657 23363 solver.cpp:253]     Train net output #3: loss_f = 2.38346 (* 1 = 2.38346 loss)
    I1230 20:10:20.429668 23363 sgd_solver.cpp:106] Iteration 33600, lr = 0.000231997
    I1230 20:10:35.649418 23363 solver.cpp:237] Iteration 33700, loss = 3.64999
    I1230 20:10:35.649513 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:10:35.649528 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:10:35.649540 23363 solver.cpp:253]     Train net output #2: loss_c = 1.3493 (* 1 = 1.3493 loss)
    I1230 20:10:35.649551 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30069 (* 1 = 2.30069 loss)
    I1230 20:10:35.649562 23363 sgd_solver.cpp:106] Iteration 33700, lr = 0.000231599
    I1230 20:10:51.028919 23363 solver.cpp:237] Iteration 33800, loss = 3.76874
    I1230 20:10:51.028967 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:10:51.028980 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:10:51.028991 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51543 (* 1 = 1.51543 loss)
    I1230 20:10:51.029002 23363 solver.cpp:253]     Train net output #3: loss_f = 2.25331 (* 1 = 2.25331 loss)
    I1230 20:10:51.029014 23363 sgd_solver.cpp:106] Iteration 33800, lr = 0.000231202
    I1230 20:11:07.452165 23363 solver.cpp:237] Iteration 33900, loss = 3.21095
    I1230 20:11:07.452316 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1230 20:11:07.452327 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.49
    I1230 20:11:07.452337 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29561 (* 1 = 1.29561 loss)
    I1230 20:11:07.452347 23363 solver.cpp:253]     Train net output #3: loss_f = 1.91533 (* 1 = 1.91533 loss)
    I1230 20:11:07.452355 23363 sgd_solver.cpp:106] Iteration 33900, lr = 0.000230807
    I1230 20:11:23.844168 23363 solver.cpp:341] Iteration 34000, Testing net (#0)
    I1230 20:11:30.416842 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.530583
    I1230 20:11:30.416883 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.405333
    I1230 20:11:30.416894 23363 solver.cpp:409]     Test net output #2: loss_c = 1.48617 (* 1 = 1.48617 loss)
    I1230 20:11:30.416903 23363 solver.cpp:409]     Test net output #3: loss_f = 2.27884 (* 1 = 2.27884 loss)
    I1230 20:11:30.487951 23363 solver.cpp:237] Iteration 34000, loss = 3.68425
    I1230 20:11:30.487999 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:11:30.488009 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:11:30.488019 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40228 (* 1 = 1.40228 loss)
    I1230 20:11:30.488029 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28197 (* 1 = 2.28197 loss)
    I1230 20:11:30.488040 23363 sgd_solver.cpp:106] Iteration 34000, lr = 0.000230414
    I1230 20:11:46.780211 23363 solver.cpp:237] Iteration 34100, loss = 3.78065
    I1230 20:11:46.780396 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:11:46.780412 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:11:46.780424 23363 solver.cpp:253]     Train net output #2: loss_c = 1.51999 (* 1 = 1.51999 loss)
    I1230 20:11:46.780434 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26066 (* 1 = 2.26066 loss)
    I1230 20:11:46.780444 23363 sgd_solver.cpp:106] Iteration 34100, lr = 0.000230022
    I1230 20:12:00.288475 23363 solver.cpp:237] Iteration 34200, loss = 3.51075
    I1230 20:12:00.288524 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:12:00.288538 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:12:00.288552 23363 solver.cpp:253]     Train net output #2: loss_c = 1.30953 (* 1 = 1.30953 loss)
    I1230 20:12:00.288564 23363 solver.cpp:253]     Train net output #3: loss_f = 2.20122 (* 1 = 2.20122 loss)
    I1230 20:12:00.288575 23363 sgd_solver.cpp:106] Iteration 34200, lr = 0.000229631
    I1230 20:12:13.743981 23363 solver.cpp:237] Iteration 34300, loss = 3.91607
    I1230 20:12:13.744038 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:12:13.744056 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:12:13.744073 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58898 (* 1 = 1.58898 loss)
    I1230 20:12:13.744088 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32709 (* 1 = 2.32709 loss)
    I1230 20:12:13.744103 23363 sgd_solver.cpp:106] Iteration 34300, lr = 0.000229243
    I1230 20:12:27.405197 23363 solver.cpp:237] Iteration 34400, loss = 3.48526
    I1230 20:12:27.405313 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:12:27.405333 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:12:27.405352 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42904 (* 1 = 1.42904 loss)
    I1230 20:12:27.405369 23363 solver.cpp:253]     Train net output #3: loss_f = 2.05622 (* 1 = 2.05622 loss)
    I1230 20:12:27.405385 23363 sgd_solver.cpp:106] Iteration 34400, lr = 0.000228855
    I1230 20:12:41.002938 23363 solver.cpp:237] Iteration 34500, loss = 3.41619
    I1230 20:12:41.002991 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:12:41.003002 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:12:41.003015 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32677 (* 1 = 1.32677 loss)
    I1230 20:12:41.003023 23363 solver.cpp:253]     Train net output #3: loss_f = 2.08941 (* 1 = 2.08941 loss)
    I1230 20:12:41.003033 23363 sgd_solver.cpp:106] Iteration 34500, lr = 0.000228469
    I1230 20:12:54.452261 23363 solver.cpp:237] Iteration 34600, loss = 3.76973
    I1230 20:12:54.452323 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:12:54.452337 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:12:54.452350 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52301 (* 1 = 1.52301 loss)
    I1230 20:12:54.452361 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24672 (* 1 = 2.24672 loss)
    I1230 20:12:54.452374 23363 sgd_solver.cpp:106] Iteration 34600, lr = 0.000228085
    I1230 20:13:07.930820 23363 solver.cpp:237] Iteration 34700, loss = 3.76035
    I1230 20:13:07.930975 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:13:07.930990 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 20:13:07.931004 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4036 (* 1 = 1.4036 loss)
    I1230 20:13:07.931013 23363 solver.cpp:253]     Train net output #3: loss_f = 2.35676 (* 1 = 2.35676 loss)
    I1230 20:13:07.931025 23363 sgd_solver.cpp:106] Iteration 34700, lr = 0.000227702
    I1230 20:13:21.420743 23363 solver.cpp:237] Iteration 34800, loss = 3.80763
    I1230 20:13:21.420804 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:13:21.420820 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:13:21.420838 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56052 (* 1 = 1.56052 loss)
    I1230 20:13:21.420853 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2471 (* 1 = 2.2471 loss)
    I1230 20:13:21.420869 23363 sgd_solver.cpp:106] Iteration 34800, lr = 0.000227321
    I1230 20:13:35.090361 23363 solver.cpp:237] Iteration 34900, loss = 3.28116
    I1230 20:13:35.090415 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:13:35.090425 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:13:35.090436 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29739 (* 1 = 1.29739 loss)
    I1230 20:13:35.090445 23363 solver.cpp:253]     Train net output #3: loss_f = 1.98377 (* 1 = 1.98377 loss)
    I1230 20:13:35.090456 23363 sgd_solver.cpp:106] Iteration 34900, lr = 0.000226941
    I1230 20:13:48.421114 23363 solver.cpp:341] Iteration 35000, Testing net (#0)
    I1230 20:13:53.490164 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.536833
    I1230 20:13:53.490228 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.407417
    I1230 20:13:53.490242 23363 solver.cpp:409]     Test net output #2: loss_c = 1.4726 (* 1 = 1.4726 loss)
    I1230 20:13:53.490252 23363 solver.cpp:409]     Test net output #3: loss_f = 2.26531 (* 1 = 2.26531 loss)
    I1230 20:13:53.553433 23363 solver.cpp:237] Iteration 35000, loss = 3.52541
    I1230 20:13:53.553486 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:13:53.553498 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:13:53.553510 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34678 (* 1 = 1.34678 loss)
    I1230 20:13:53.553521 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17863 (* 1 = 2.17863 loss)
    I1230 20:13:53.553534 23363 sgd_solver.cpp:106] Iteration 35000, lr = 0.000226563
    I1230 20:14:07.117341 23363 solver.cpp:237] Iteration 35100, loss = 3.94145
    I1230 20:14:07.117390 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 20:14:07.117403 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:14:07.117419 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62003 (* 1 = 1.62003 loss)
    I1230 20:14:07.117429 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32142 (* 1 = 2.32142 loss)
    I1230 20:14:07.117441 23363 sgd_solver.cpp:106] Iteration 35100, lr = 0.000226186
    I1230 20:14:20.836774 23363 solver.cpp:237] Iteration 35200, loss = 3.61601
    I1230 20:14:20.836918 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:14:20.836935 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:14:20.836948 23363 solver.cpp:253]     Train net output #2: loss_c = 1.38208 (* 1 = 1.38208 loss)
    I1230 20:14:20.836958 23363 solver.cpp:253]     Train net output #3: loss_f = 2.23393 (* 1 = 2.23393 loss)
    I1230 20:14:20.836971 23363 sgd_solver.cpp:106] Iteration 35200, lr = 0.000225811
    I1230 20:14:34.294862 23363 solver.cpp:237] Iteration 35300, loss = 3.89668
    I1230 20:14:34.294915 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 20:14:34.294931 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:14:34.294948 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58255 (* 1 = 1.58255 loss)
    I1230 20:14:34.294963 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31414 (* 1 = 2.31414 loss)
    I1230 20:14:34.294978 23363 sgd_solver.cpp:106] Iteration 35300, lr = 0.000225437
    I1230 20:14:47.974968 23363 solver.cpp:237] Iteration 35400, loss = 3.19983
    I1230 20:14:47.975019 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:14:47.975031 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:14:47.975047 23363 solver.cpp:253]     Train net output #2: loss_c = 1.27934 (* 1 = 1.27934 loss)
    I1230 20:14:47.975059 23363 solver.cpp:253]     Train net output #3: loss_f = 1.92049 (* 1 = 1.92049 loss)
    I1230 20:14:47.975074 23363 sgd_solver.cpp:106] Iteration 35400, lr = 0.000225064
    I1230 20:15:01.706593 23363 solver.cpp:237] Iteration 35500, loss = 3.44276
    I1230 20:15:01.706750 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:15:01.706765 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:15:01.706778 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36126 (* 1 = 1.36126 loss)
    I1230 20:15:01.706789 23363 solver.cpp:253]     Train net output #3: loss_f = 2.0815 (* 1 = 2.0815 loss)
    I1230 20:15:01.706799 23363 sgd_solver.cpp:106] Iteration 35500, lr = 0.000224693
    I1230 20:15:15.183248 23363 solver.cpp:237] Iteration 35600, loss = 3.85794
    I1230 20:15:15.183298 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:15:15.183310 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:15:15.183322 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56601 (* 1 = 1.56601 loss)
    I1230 20:15:15.183333 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29193 (* 1 = 2.29193 loss)
    I1230 20:15:15.183346 23363 sgd_solver.cpp:106] Iteration 35600, lr = 0.000224323
    I1230 20:15:28.632702 23363 solver.cpp:237] Iteration 35700, loss = 3.5477
    I1230 20:15:28.632742 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:15:28.632753 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:15:28.632764 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31849 (* 1 = 1.31849 loss)
    I1230 20:15:28.632774 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22921 (* 1 = 2.22921 loss)
    I1230 20:15:28.632784 23363 sgd_solver.cpp:106] Iteration 35700, lr = 0.000223955
    I1230 20:15:41.949000 23363 solver.cpp:237] Iteration 35800, loss = 3.82675
    I1230 20:15:41.949185 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 20:15:41.949199 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:15:41.949209 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56844 (* 1 = 1.56844 loss)
    I1230 20:15:41.949218 23363 solver.cpp:253]     Train net output #3: loss_f = 2.25831 (* 1 = 2.25831 loss)
    I1230 20:15:41.949226 23363 sgd_solver.cpp:106] Iteration 35800, lr = 0.000223588
    I1230 20:15:55.306402 23363 solver.cpp:237] Iteration 35900, loss = 3.27467
    I1230 20:15:55.306439 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:15:55.306449 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:15:55.306459 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31685 (* 1 = 1.31685 loss)
    I1230 20:15:55.306468 23363 solver.cpp:253]     Train net output #3: loss_f = 1.95781 (* 1 = 1.95781 loss)
    I1230 20:15:55.306478 23363 sgd_solver.cpp:106] Iteration 35900, lr = 0.000223223
    I1230 20:16:08.503829 23363 solver.cpp:341] Iteration 36000, Testing net (#0)
    I1230 20:16:13.535919 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.539917
    I1230 20:16:13.536133 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.40875
    I1230 20:16:13.536161 23363 solver.cpp:409]     Test net output #2: loss_c = 1.46761 (* 1 = 1.46761 loss)
    I1230 20:16:13.536178 23363 solver.cpp:409]     Test net output #3: loss_f = 2.25866 (* 1 = 2.25866 loss)
    I1230 20:16:13.600993 23363 solver.cpp:237] Iteration 36000, loss = 3.48276
    I1230 20:16:13.601048 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:16:13.601065 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:16:13.601083 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37603 (* 1 = 1.37603 loss)
    I1230 20:16:13.601099 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10673 (* 1 = 2.10673 loss)
    I1230 20:16:13.601115 23363 sgd_solver.cpp:106] Iteration 36000, lr = 0.000222859
    I1230 20:16:27.369112 23363 solver.cpp:237] Iteration 36100, loss = 3.69479
    I1230 20:16:27.369163 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:16:27.369174 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:16:27.369187 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48764 (* 1 = 1.48764 loss)
    I1230 20:16:27.369197 23363 solver.cpp:253]     Train net output #3: loss_f = 2.20715 (* 1 = 2.20715 loss)
    I1230 20:16:27.369207 23363 sgd_solver.cpp:106] Iteration 36100, lr = 0.000222496
    I1230 20:16:40.783522 23363 solver.cpp:237] Iteration 36200, loss = 3.75645
    I1230 20:16:40.783566 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:16:40.783578 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.34
    I1230 20:16:40.783591 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48098 (* 1 = 1.48098 loss)
    I1230 20:16:40.783601 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27546 (* 1 = 2.27546 loss)
    I1230 20:16:40.783612 23363 sgd_solver.cpp:106] Iteration 36200, lr = 0.000222135
    I1230 20:16:54.287550 23363 solver.cpp:237] Iteration 36300, loss = 3.57961
    I1230 20:16:54.287714 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:16:54.287739 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:16:54.287757 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44557 (* 1 = 1.44557 loss)
    I1230 20:16:54.287772 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13404 (* 1 = 2.13404 loss)
    I1230 20:16:54.287789 23363 sgd_solver.cpp:106] Iteration 36300, lr = 0.000221775
    I1230 20:17:07.878845 23363 solver.cpp:237] Iteration 36400, loss = 3.36489
    I1230 20:17:07.878909 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1230 20:17:07.878928 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:17:07.878947 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36243 (* 1 = 1.36243 loss)
    I1230 20:17:07.878962 23363 solver.cpp:253]     Train net output #3: loss_f = 2.00246 (* 1 = 2.00246 loss)
    I1230 20:17:07.878980 23363 sgd_solver.cpp:106] Iteration 36400, lr = 0.000221416
    I1230 20:17:21.485616 23363 solver.cpp:237] Iteration 36500, loss = 3.30969
    I1230 20:17:21.485671 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1230 20:17:21.485682 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:17:21.485695 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29286 (* 1 = 1.29286 loss)
    I1230 20:17:21.485707 23363 solver.cpp:253]     Train net output #3: loss_f = 2.01683 (* 1 = 2.01683 loss)
    I1230 20:17:21.485718 23363 sgd_solver.cpp:106] Iteration 36500, lr = 0.000221059
    I1230 20:17:34.999934 23363 solver.cpp:237] Iteration 36600, loss = 3.7703
    I1230 20:17:35.000047 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:17:35.000059 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:17:35.000071 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52903 (* 1 = 1.52903 loss)
    I1230 20:17:35.000078 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24127 (* 1 = 2.24127 loss)
    I1230 20:17:35.000087 23363 sgd_solver.cpp:106] Iteration 36600, lr = 0.000220703
    I1230 20:17:48.359637 23363 solver.cpp:237] Iteration 36700, loss = 3.64784
    I1230 20:17:48.359689 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:17:48.359714 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.35
    I1230 20:17:48.359736 23363 solver.cpp:253]     Train net output #2: loss_c = 1.3585 (* 1 = 1.3585 loss)
    I1230 20:17:48.359746 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28934 (* 1 = 2.28934 loss)
    I1230 20:17:48.359766 23363 sgd_solver.cpp:106] Iteration 36700, lr = 0.000220349
    I1230 20:18:01.673107 23363 solver.cpp:237] Iteration 36800, loss = 3.81647
    I1230 20:18:01.673167 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 20:18:01.673182 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:18:01.673195 23363 solver.cpp:253]     Train net output #2: loss_c = 1.55058 (* 1 = 1.55058 loss)
    I1230 20:18:01.673207 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26589 (* 1 = 2.26589 loss)
    I1230 20:18:01.673220 23363 sgd_solver.cpp:106] Iteration 36800, lr = 0.000219995
    I1230 20:18:15.180713 23363 solver.cpp:237] Iteration 36900, loss = 3.26614
    I1230 20:18:15.180835 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:18:15.180851 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:18:15.180869 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29083 (* 1 = 1.29083 loss)
    I1230 20:18:15.180886 23363 solver.cpp:253]     Train net output #3: loss_f = 1.97531 (* 1 = 1.97531 loss)
    I1230 20:18:15.180901 23363 sgd_solver.cpp:106] Iteration 36900, lr = 0.000219644
    I1230 20:18:28.648686 23363 solver.cpp:341] Iteration 37000, Testing net (#0)
    I1230 20:18:33.879055 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.5395
    I1230 20:18:33.879113 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.409333
    I1230 20:18:33.879128 23363 solver.cpp:409]     Test net output #2: loss_c = 1.46588 (* 1 = 1.46588 loss)
    I1230 20:18:33.879142 23363 solver.cpp:409]     Test net output #3: loss_f = 2.24489 (* 1 = 2.24489 loss)
    I1230 20:18:33.950856 23363 solver.cpp:237] Iteration 37000, loss = 3.36421
    I1230 20:18:33.950903 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:18:33.950917 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:18:33.950930 23363 solver.cpp:253]     Train net output #2: loss_c = 1.28535 (* 1 = 1.28535 loss)
    I1230 20:18:33.950943 23363 solver.cpp:253]     Train net output #3: loss_f = 2.07886 (* 1 = 2.07886 loss)
    I1230 20:18:33.950956 23363 sgd_solver.cpp:106] Iteration 37000, lr = 0.000219293
    I1230 20:18:47.123529 23363 solver.cpp:237] Iteration 37100, loss = 3.8463
    I1230 20:18:47.123662 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:18:47.123677 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:18:47.123688 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50785 (* 1 = 1.50785 loss)
    I1230 20:18:47.123698 23363 solver.cpp:253]     Train net output #3: loss_f = 2.33846 (* 1 = 2.33846 loss)
    I1230 20:18:47.123708 23363 sgd_solver.cpp:106] Iteration 37100, lr = 0.000218944
    I1230 20:19:00.693385 23363 solver.cpp:237] Iteration 37200, loss = 3.56865
    I1230 20:19:00.693445 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:19:00.693464 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:19:00.693483 23363 solver.cpp:253]     Train net output #2: loss_c = 1.3355 (* 1 = 1.3355 loss)
    I1230 20:19:00.693500 23363 solver.cpp:253]     Train net output #3: loss_f = 2.23315 (* 1 = 2.23315 loss)
    I1230 20:19:00.693516 23363 sgd_solver.cpp:106] Iteration 37200, lr = 0.000218596
    I1230 20:19:14.043874 23363 solver.cpp:237] Iteration 37300, loss = 3.8727
    I1230 20:19:14.043941 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.47
    I1230 20:19:14.043962 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:19:14.043982 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63891 (* 1 = 1.63891 loss)
    I1230 20:19:14.044003 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2338 (* 1 = 2.2338 loss)
    I1230 20:19:14.044021 23363 sgd_solver.cpp:106] Iteration 37300, lr = 0.000218249
    I1230 20:19:27.439968 23363 solver.cpp:237] Iteration 37400, loss = 3.39777
    I1230 20:19:27.440130 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:19:27.440155 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.49
    I1230 20:19:27.440167 23363 solver.cpp:253]     Train net output #2: loss_c = 1.37337 (* 1 = 1.37337 loss)
    I1230 20:19:27.440177 23363 solver.cpp:253]     Train net output #3: loss_f = 2.02441 (* 1 = 2.02441 loss)
    I1230 20:19:27.440187 23363 sgd_solver.cpp:106] Iteration 37400, lr = 0.000217904
    I1230 20:19:40.921977 23363 solver.cpp:237] Iteration 37500, loss = 3.5903
    I1230 20:19:40.922036 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:19:40.922051 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:19:40.922065 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40214 (* 1 = 1.40214 loss)
    I1230 20:19:40.922078 23363 solver.cpp:253]     Train net output #3: loss_f = 2.18816 (* 1 = 2.18816 loss)
    I1230 20:19:40.922091 23363 sgd_solver.cpp:106] Iteration 37500, lr = 0.000217559
    I1230 20:19:54.366472 23363 solver.cpp:237] Iteration 37600, loss = 3.82965
    I1230 20:19:54.366515 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 20:19:54.366526 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:19:54.366539 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52976 (* 1 = 1.52976 loss)
    I1230 20:19:54.366549 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29989 (* 1 = 2.29989 loss)
    I1230 20:19:54.366560 23363 sgd_solver.cpp:106] Iteration 37600, lr = 0.000217216
    I1230 20:20:07.799590 23363 solver.cpp:237] Iteration 37700, loss = 3.45802
    I1230 20:20:07.799705 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:20:07.799718 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:20:07.799731 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26958 (* 1 = 1.26958 loss)
    I1230 20:20:07.799739 23363 solver.cpp:253]     Train net output #3: loss_f = 2.18844 (* 1 = 2.18844 loss)
    I1230 20:20:07.799749 23363 sgd_solver.cpp:106] Iteration 37700, lr = 0.000216875
    I1230 20:20:21.371446 23363 solver.cpp:237] Iteration 37800, loss = 3.58678
    I1230 20:20:21.371505 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:20:21.371520 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 20:20:21.371538 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42904 (* 1 = 1.42904 loss)
    I1230 20:20:21.371554 23363 solver.cpp:253]     Train net output #3: loss_f = 2.15774 (* 1 = 2.15774 loss)
    I1230 20:20:21.371568 23363 sgd_solver.cpp:106] Iteration 37800, lr = 0.000216535
    I1230 20:20:34.774282 23363 solver.cpp:237] Iteration 37900, loss = 3.18729
    I1230 20:20:34.774338 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1230 20:20:34.774354 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.52
    I1230 20:20:34.774372 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26049 (* 1 = 1.26049 loss)
    I1230 20:20:34.774389 23363 solver.cpp:253]     Train net output #3: loss_f = 1.9268 (* 1 = 1.9268 loss)
    I1230 20:20:34.774405 23363 sgd_solver.cpp:106] Iteration 37900, lr = 0.000216195
    I1230 20:20:48.018666 23363 solver.cpp:341] Iteration 38000, Testing net (#0)
    I1230 20:20:52.992128 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.5405
    I1230 20:20:52.992190 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.41175
    I1230 20:20:52.992216 23363 solver.cpp:409]     Test net output #2: loss_c = 1.45898 (* 1 = 1.45898 loss)
    I1230 20:20:52.992239 23363 solver.cpp:409]     Test net output #3: loss_f = 2.23989 (* 1 = 2.23989 loss)
    I1230 20:20:53.056301 23363 solver.cpp:237] Iteration 38000, loss = 3.41848
    I1230 20:20:53.056344 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1230 20:20:53.056354 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:20:53.056366 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34036 (* 1 = 1.34036 loss)
    I1230 20:20:53.056377 23363 solver.cpp:253]     Train net output #3: loss_f = 2.07812 (* 1 = 2.07812 loss)
    I1230 20:20:53.056388 23363 sgd_solver.cpp:106] Iteration 38000, lr = 0.000215857
    I1230 20:21:06.583917 23363 solver.cpp:237] Iteration 38100, loss = 3.87833
    I1230 20:21:06.583978 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:21:06.583992 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:21:06.584008 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5765 (* 1 = 1.5765 loss)
    I1230 20:21:06.584022 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30183 (* 1 = 2.30183 loss)
    I1230 20:21:06.584033 23363 sgd_solver.cpp:106] Iteration 38100, lr = 0.000215521
    I1230 20:21:20.084401 23363 solver.cpp:237] Iteration 38200, loss = 3.67109
    I1230 20:21:20.084548 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:21:20.084566 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1230 20:21:20.084579 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36161 (* 1 = 1.36161 loss)
    I1230 20:21:20.084590 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30948 (* 1 = 2.30948 loss)
    I1230 20:21:20.084601 23363 sgd_solver.cpp:106] Iteration 38200, lr = 0.000215185
    I1230 20:21:33.615435 23363 solver.cpp:237] Iteration 38300, loss = 3.88286
    I1230 20:21:33.615502 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 20:21:33.615521 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:21:33.615540 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6052 (* 1 = 1.6052 loss)
    I1230 20:21:33.615556 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27766 (* 1 = 2.27766 loss)
    I1230 20:21:33.615572 23363 sgd_solver.cpp:106] Iteration 38300, lr = 0.000214851
    I1230 20:21:47.277034 23363 solver.cpp:237] Iteration 38400, loss = 3.35484
    I1230 20:21:47.277092 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:21:47.277111 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:21:47.277129 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36709 (* 1 = 1.36709 loss)
    I1230 20:21:47.277146 23363 solver.cpp:253]     Train net output #3: loss_f = 1.98775 (* 1 = 1.98775 loss)
    I1230 20:21:47.277163 23363 sgd_solver.cpp:106] Iteration 38400, lr = 0.000214518
    I1230 20:22:00.721961 23363 solver.cpp:237] Iteration 38500, loss = 3.55449
    I1230 20:22:00.722128 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:22:00.722146 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:22:00.722158 23363 solver.cpp:253]     Train net output #2: loss_c = 1.38965 (* 1 = 1.38965 loss)
    I1230 20:22:00.722168 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16484 (* 1 = 2.16484 loss)
    I1230 20:22:00.722179 23363 sgd_solver.cpp:106] Iteration 38500, lr = 0.000214186
    I1230 20:22:14.148950 23363 solver.cpp:237] Iteration 38600, loss = 3.74395
    I1230 20:22:14.149024 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:22:14.149036 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:22:14.149049 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52514 (* 1 = 1.52514 loss)
    I1230 20:22:14.149060 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21881 (* 1 = 2.21881 loss)
    I1230 20:22:14.149072 23363 sgd_solver.cpp:106] Iteration 38600, lr = 0.000213856
    I1230 20:22:27.606051 23363 solver.cpp:237] Iteration 38700, loss = 3.59427
    I1230 20:22:27.606094 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:22:27.606106 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:22:27.606118 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36824 (* 1 = 1.36824 loss)
    I1230 20:22:27.606127 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22602 (* 1 = 2.22602 loss)
    I1230 20:22:27.606137 23363 sgd_solver.cpp:106] Iteration 38700, lr = 0.000213526
    I1230 20:22:41.058537 23363 solver.cpp:237] Iteration 38800, loss = 3.95821
    I1230 20:22:41.058709 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:22:41.058743 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:22:41.058758 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6522 (* 1 = 1.6522 loss)
    I1230 20:22:41.058768 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30601 (* 1 = 2.30601 loss)
    I1230 20:22:41.058781 23363 sgd_solver.cpp:106] Iteration 38800, lr = 0.000213198
    I1230 20:22:54.209595 23363 solver.cpp:237] Iteration 38900, loss = 3.20289
    I1230 20:22:54.209653 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:22:54.209671 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:22:54.209688 23363 solver.cpp:253]     Train net output #2: loss_c = 1.27696 (* 1 = 1.27696 loss)
    I1230 20:22:54.209703 23363 solver.cpp:253]     Train net output #3: loss_f = 1.92593 (* 1 = 1.92593 loss)
    I1230 20:22:54.209719 23363 sgd_solver.cpp:106] Iteration 38900, lr = 0.000212871
    I1230 20:23:07.877442 23363 solver.cpp:341] Iteration 39000, Testing net (#0)
    I1230 20:23:13.169530 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.538333
    I1230 20:23:13.169644 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.407833
    I1230 20:23:13.169668 23363 solver.cpp:409]     Test net output #2: loss_c = 1.46045 (* 1 = 1.46045 loss)
    I1230 20:23:13.169685 23363 solver.cpp:409]     Test net output #3: loss_f = 2.24659 (* 1 = 2.24659 loss)
    I1230 20:23:13.238528 23363 solver.cpp:237] Iteration 39000, loss = 3.48626
    I1230 20:23:13.238581 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:23:13.238597 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:23:13.238616 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35091 (* 1 = 1.35091 loss)
    I1230 20:23:13.238632 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13535 (* 1 = 2.13535 loss)
    I1230 20:23:13.238649 23363 sgd_solver.cpp:106] Iteration 39000, lr = 0.000212545
    I1230 20:23:26.747669 23363 solver.cpp:237] Iteration 39100, loss = 3.91741
    I1230 20:23:26.747730 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:23:26.747748 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:23:26.747769 23363 solver.cpp:253]     Train net output #2: loss_c = 1.60813 (* 1 = 1.60813 loss)
    I1230 20:23:26.747786 23363 solver.cpp:253]     Train net output #3: loss_f = 2.30928 (* 1 = 2.30928 loss)
    I1230 20:23:26.747802 23363 sgd_solver.cpp:106] Iteration 39100, lr = 0.00021222
    I1230 20:23:40.218045 23363 solver.cpp:237] Iteration 39200, loss = 3.53173
    I1230 20:23:40.218104 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:23:40.218116 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:23:40.218128 23363 solver.cpp:253]     Train net output #2: loss_c = 1.30414 (* 1 = 1.30414 loss)
    I1230 20:23:40.218138 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2276 (* 1 = 2.2276 loss)
    I1230 20:23:40.218149 23363 sgd_solver.cpp:106] Iteration 39200, lr = 0.000211897
    I1230 20:23:53.683063 23363 solver.cpp:237] Iteration 39300, loss = 3.56457
    I1230 20:23:53.683260 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:23:53.683277 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:23:53.683291 23363 solver.cpp:253]     Train net output #2: loss_c = 1.43041 (* 1 = 1.43041 loss)
    I1230 20:23:53.683302 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13417 (* 1 = 2.13417 loss)
    I1230 20:23:53.683315 23363 sgd_solver.cpp:106] Iteration 39300, lr = 0.000211574
    I1230 20:24:07.240514 23363 solver.cpp:237] Iteration 39400, loss = 3.16811
    I1230 20:24:07.240559 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1230 20:24:07.240571 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:24:07.240584 23363 solver.cpp:253]     Train net output #2: loss_c = 1.21318 (* 1 = 1.21318 loss)
    I1230 20:24:07.240594 23363 solver.cpp:253]     Train net output #3: loss_f = 1.95493 (* 1 = 1.95493 loss)
    I1230 20:24:07.240607 23363 sgd_solver.cpp:106] Iteration 39400, lr = 0.000211253
    I1230 20:24:20.685367 23363 solver.cpp:237] Iteration 39500, loss = 3.46416
    I1230 20:24:20.685426 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:24:20.685444 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:24:20.685461 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34738 (* 1 = 1.34738 loss)
    I1230 20:24:20.685478 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11678 (* 1 = 2.11678 loss)
    I1230 20:24:20.685495 23363 sgd_solver.cpp:106] Iteration 39500, lr = 0.000210933
    I1230 20:24:34.313621 23363 solver.cpp:237] Iteration 39600, loss = 3.78631
    I1230 20:24:34.313803 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 20:24:34.313820 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:24:34.313832 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50916 (* 1 = 1.50916 loss)
    I1230 20:24:34.313841 23363 solver.cpp:253]     Train net output #3: loss_f = 2.27715 (* 1 = 2.27715 loss)
    I1230 20:24:34.313853 23363 sgd_solver.cpp:106] Iteration 39600, lr = 0.000210614
    I1230 20:24:47.681617 23363 solver.cpp:237] Iteration 39700, loss = 3.64498
    I1230 20:24:47.681679 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:24:47.681692 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:24:47.681705 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34595 (* 1 = 1.34595 loss)
    I1230 20:24:47.681715 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29903 (* 1 = 2.29903 loss)
    I1230 20:24:47.681727 23363 sgd_solver.cpp:106] Iteration 39700, lr = 0.000210296
    I1230 20:25:01.067533 23363 solver.cpp:237] Iteration 39800, loss = 3.71923
    I1230 20:25:01.067589 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 20:25:01.067601 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:25:01.067613 23363 solver.cpp:253]     Train net output #2: loss_c = 1.50343 (* 1 = 1.50343 loss)
    I1230 20:25:01.067623 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2158 (* 1 = 2.2158 loss)
    I1230 20:25:01.067636 23363 sgd_solver.cpp:106] Iteration 39800, lr = 0.000209979
    I1230 20:25:14.562098 23363 solver.cpp:237] Iteration 39900, loss = 3.23253
    I1230 20:25:14.562232 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.69
    I1230 20:25:14.562248 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:25:14.562261 23363 solver.cpp:253]     Train net output #2: loss_c = 1.25368 (* 1 = 1.25368 loss)
    I1230 20:25:14.562271 23363 solver.cpp:253]     Train net output #3: loss_f = 1.97885 (* 1 = 1.97885 loss)
    I1230 20:25:14.562283 23363 sgd_solver.cpp:106] Iteration 39900, lr = 0.000209663
    I1230 20:25:27.650719 23363 solver.cpp:341] Iteration 40000, Testing net (#0)
    I1230 20:25:32.679924 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.532833
    I1230 20:25:32.679972 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.401583
    I1230 20:25:32.679990 23363 solver.cpp:409]     Test net output #2: loss_c = 1.4854 (* 1 = 1.4854 loss)
    I1230 20:25:32.680002 23363 solver.cpp:409]     Test net output #3: loss_f = 2.27422 (* 1 = 2.27422 loss)
    I1230 20:25:32.742674 23363 solver.cpp:237] Iteration 40000, loss = 3.5021
    I1230 20:25:32.742730 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:25:32.742743 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:25:32.742758 23363 solver.cpp:253]     Train net output #2: loss_c = 1.3986 (* 1 = 1.3986 loss)
    I1230 20:25:32.742769 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10351 (* 1 = 2.10351 loss)
    I1230 20:25:32.742781 23363 sgd_solver.cpp:106] Iteration 40000, lr = 0.000209349
    I1230 20:25:46.176789 23363 solver.cpp:237] Iteration 40100, loss = 3.48481
    I1230 20:25:46.176921 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:25:46.176936 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:25:46.176950 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36065 (* 1 = 1.36065 loss)
    I1230 20:25:46.176961 23363 solver.cpp:253]     Train net output #3: loss_f = 2.12416 (* 1 = 2.12416 loss)
    I1230 20:25:46.176971 23363 sgd_solver.cpp:106] Iteration 40100, lr = 0.000209035
    I1230 20:25:59.508059 23363 solver.cpp:237] Iteration 40200, loss = 3.38404
    I1230 20:25:59.508117 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:25:59.508136 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:25:59.508153 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26568 (* 1 = 1.26568 loss)
    I1230 20:25:59.508170 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11836 (* 1 = 2.11836 loss)
    I1230 20:25:59.508185 23363 sgd_solver.cpp:106] Iteration 40200, lr = 0.000208723
    I1230 20:26:13.038036 23363 solver.cpp:237] Iteration 40300, loss = 3.60521
    I1230 20:26:13.038091 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:26:13.038102 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:26:13.038113 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48124 (* 1 = 1.48124 loss)
    I1230 20:26:13.038122 23363 solver.cpp:253]     Train net output #3: loss_f = 2.12397 (* 1 = 2.12397 loss)
    I1230 20:26:13.038132 23363 sgd_solver.cpp:106] Iteration 40300, lr = 0.000208412
    I1230 20:26:26.476691 23363 solver.cpp:237] Iteration 40400, loss = 3.34486
    I1230 20:26:26.476794 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:26:26.476807 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:26:26.476819 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32109 (* 1 = 1.32109 loss)
    I1230 20:26:26.476830 23363 solver.cpp:253]     Train net output #3: loss_f = 2.02377 (* 1 = 2.02377 loss)
    I1230 20:26:26.476841 23363 sgd_solver.cpp:106] Iteration 40400, lr = 0.000208101
    I1230 20:26:40.130203 23363 solver.cpp:237] Iteration 40500, loss = 3.57776
    I1230 20:26:40.130249 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:26:40.130260 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:26:40.130273 23363 solver.cpp:253]     Train net output #2: loss_c = 1.3996 (* 1 = 1.3996 loss)
    I1230 20:26:40.130285 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17816 (* 1 = 2.17816 loss)
    I1230 20:26:40.130295 23363 sgd_solver.cpp:106] Iteration 40500, lr = 0.000207792
    I1230 20:26:53.834105 23363 solver.cpp:237] Iteration 40600, loss = 3.76095
    I1230 20:26:53.834177 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:26:53.834190 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:26:53.834205 23363 solver.cpp:253]     Train net output #2: loss_c = 1.47097 (* 1 = 1.47097 loss)
    I1230 20:26:53.834216 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28998 (* 1 = 2.28998 loss)
    I1230 20:26:53.834228 23363 sgd_solver.cpp:106] Iteration 40600, lr = 0.000207484
    I1230 20:27:07.837545 23363 solver.cpp:237] Iteration 40700, loss = 3.65937
    I1230 20:27:07.837636 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:27:07.837651 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:27:07.837663 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42757 (* 1 = 1.42757 loss)
    I1230 20:27:07.837673 23363 solver.cpp:253]     Train net output #3: loss_f = 2.23181 (* 1 = 2.23181 loss)
    I1230 20:27:07.837684 23363 sgd_solver.cpp:106] Iteration 40700, lr = 0.000207177
    I1230 20:27:25.395051 23363 solver.cpp:237] Iteration 40800, loss = 3.75623
    I1230 20:27:25.395107 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:27:25.395117 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:27:25.395128 23363 solver.cpp:253]     Train net output #2: loss_c = 1.54003 (* 1 = 1.54003 loss)
    I1230 20:27:25.395138 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21621 (* 1 = 2.21621 loss)
    I1230 20:27:25.395149 23363 sgd_solver.cpp:106] Iteration 40800, lr = 0.000206871
    I1230 20:27:40.911139 23363 solver.cpp:237] Iteration 40900, loss = 3.21583
    I1230 20:27:40.911291 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:27:40.911306 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:27:40.911319 23363 solver.cpp:253]     Train net output #2: loss_c = 1.30701 (* 1 = 1.30701 loss)
    I1230 20:27:40.911329 23363 solver.cpp:253]     Train net output #3: loss_f = 1.90882 (* 1 = 1.90882 loss)
    I1230 20:27:40.911340 23363 sgd_solver.cpp:106] Iteration 40900, lr = 0.000206566
    I1230 20:27:57.327298 23363 solver.cpp:341] Iteration 41000, Testing net (#0)
    I1230 20:28:03.457258 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.545417
    I1230 20:28:03.457320 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.412917
    I1230 20:28:03.457341 23363 solver.cpp:409]     Test net output #2: loss_c = 1.45889 (* 1 = 1.45889 loss)
    I1230 20:28:03.457358 23363 solver.cpp:409]     Test net output #3: loss_f = 2.24016 (* 1 = 2.24016 loss)
    I1230 20:28:03.537927 23363 solver.cpp:237] Iteration 41000, loss = 3.65932
    I1230 20:28:03.537981 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:28:03.537998 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:28:03.538015 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40162 (* 1 = 1.40162 loss)
    I1230 20:28:03.538033 23363 solver.cpp:253]     Train net output #3: loss_f = 2.2577 (* 1 = 2.2577 loss)
    I1230 20:28:03.538049 23363 sgd_solver.cpp:106] Iteration 41000, lr = 0.000206263
    I1230 20:28:19.269079 23363 solver.cpp:237] Iteration 41100, loss = 3.64879
    I1230 20:28:19.269227 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:28:19.269242 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.49
    I1230 20:28:19.269253 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44533 (* 1 = 1.44533 loss)
    I1230 20:28:19.269263 23363 solver.cpp:253]     Train net output #3: loss_f = 2.20346 (* 1 = 2.20346 loss)
    I1230 20:28:19.269273 23363 sgd_solver.cpp:106] Iteration 41100, lr = 0.00020596
    I1230 20:28:34.713493 23363 solver.cpp:237] Iteration 41200, loss = 3.48581
    I1230 20:28:34.713534 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:28:34.713544 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:28:34.713554 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31342 (* 1 = 1.31342 loss)
    I1230 20:28:34.713563 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17239 (* 1 = 2.17239 loss)
    I1230 20:28:34.713572 23363 sgd_solver.cpp:106] Iteration 41200, lr = 0.000205658
    I1230 20:28:52.482867 23363 solver.cpp:237] Iteration 41300, loss = 3.61344
    I1230 20:28:52.482955 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:28:52.482970 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:28:52.482982 23363 solver.cpp:253]     Train net output #2: loss_c = 1.49423 (* 1 = 1.49423 loss)
    I1230 20:28:52.482995 23363 solver.cpp:253]     Train net output #3: loss_f = 2.1192 (* 1 = 2.1192 loss)
    I1230 20:28:52.483008 23363 sgd_solver.cpp:106] Iteration 41300, lr = 0.000205357
    I1230 20:29:07.841410 23363 solver.cpp:237] Iteration 41400, loss = 3.15105
    I1230 20:29:07.841470 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:29:07.841488 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:29:07.841506 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26209 (* 1 = 1.26209 loss)
    I1230 20:29:07.841523 23363 solver.cpp:253]     Train net output #3: loss_f = 1.88897 (* 1 = 1.88897 loss)
    I1230 20:29:07.841539 23363 sgd_solver.cpp:106] Iteration 41400, lr = 0.000205058
    I1230 20:29:24.590963 23363 solver.cpp:237] Iteration 41500, loss = 3.37006
    I1230 20:29:24.591172 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:29:24.591187 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:29:24.591197 23363 solver.cpp:253]     Train net output #2: loss_c = 1.27392 (* 1 = 1.27392 loss)
    I1230 20:29:24.591207 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09613 (* 1 = 2.09613 loss)
    I1230 20:29:24.591217 23363 sgd_solver.cpp:106] Iteration 41500, lr = 0.000204759
    I1230 20:29:40.104645 23363 solver.cpp:237] Iteration 41600, loss = 3.64767
    I1230 20:29:40.104694 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:29:40.104706 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:29:40.104717 23363 solver.cpp:253]     Train net output #2: loss_c = 1.48207 (* 1 = 1.48207 loss)
    I1230 20:29:40.104727 23363 solver.cpp:253]     Train net output #3: loss_f = 2.1656 (* 1 = 2.1656 loss)
    I1230 20:29:40.104737 23363 sgd_solver.cpp:106] Iteration 41600, lr = 0.000204461
    I1230 20:29:55.599869 23363 solver.cpp:237] Iteration 41700, loss = 3.60649
    I1230 20:29:55.600009 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:29:55.600033 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1230 20:29:55.600052 23363 solver.cpp:253]     Train net output #2: loss_c = 1.40391 (* 1 = 1.40391 loss)
    I1230 20:29:55.600069 23363 solver.cpp:253]     Train net output #3: loss_f = 2.20258 (* 1 = 2.20258 loss)
    I1230 20:29:55.600083 23363 sgd_solver.cpp:106] Iteration 41700, lr = 0.000204164
    I1230 20:30:11.663408 23363 solver.cpp:237] Iteration 41800, loss = 3.74336
    I1230 20:30:11.663458 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:30:11.663470 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:30:11.663480 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56187 (* 1 = 1.56187 loss)
    I1230 20:30:11.663489 23363 solver.cpp:253]     Train net output #3: loss_f = 2.18149 (* 1 = 2.18149 loss)
    I1230 20:30:11.663499 23363 sgd_solver.cpp:106] Iteration 41800, lr = 0.000203869
    I1230 20:30:27.541476 23363 solver.cpp:237] Iteration 41900, loss = 3.02195
    I1230 20:30:27.541635 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:30:27.541648 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:30:27.541659 23363 solver.cpp:253]     Train net output #2: loss_c = 1.18888 (* 1 = 1.18888 loss)
    I1230 20:30:27.541668 23363 solver.cpp:253]     Train net output #3: loss_f = 1.83307 (* 1 = 1.83307 loss)
    I1230 20:30:27.541678 23363 sgd_solver.cpp:106] Iteration 41900, lr = 0.000203574
    I1230 20:30:43.922672 23363 solver.cpp:341] Iteration 42000, Testing net (#0)
    I1230 20:30:50.298585 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.545
    I1230 20:30:50.298641 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.41775
    I1230 20:30:50.298655 23363 solver.cpp:409]     Test net output #2: loss_c = 1.437 (* 1 = 1.437 loss)
    I1230 20:30:50.298665 23363 solver.cpp:409]     Test net output #3: loss_f = 2.21431 (* 1 = 2.21431 loss)
    I1230 20:30:50.382761 23363 solver.cpp:237] Iteration 42000, loss = 3.13218
    I1230 20:30:50.382807 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.67
    I1230 20:30:50.382817 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:30:50.382828 23363 solver.cpp:253]     Train net output #2: loss_c = 1.19121 (* 1 = 1.19121 loss)
    I1230 20:30:50.382838 23363 solver.cpp:253]     Train net output #3: loss_f = 1.94097 (* 1 = 1.94097 loss)
    I1230 20:30:50.382848 23363 sgd_solver.cpp:106] Iteration 42000, lr = 0.00020328
    I1230 20:31:06.626233 23363 solver.cpp:237] Iteration 42100, loss = 3.6567
    I1230 20:31:06.626327 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:31:06.626343 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:31:06.626356 23363 solver.cpp:253]     Train net output #2: loss_c = 1.49203 (* 1 = 1.49203 loss)
    I1230 20:31:06.626368 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16467 (* 1 = 2.16467 loss)
    I1230 20:31:06.626379 23363 sgd_solver.cpp:106] Iteration 42100, lr = 0.000202988
    I1230 20:31:22.814038 23363 solver.cpp:237] Iteration 42200, loss = 3.60925
    I1230 20:31:22.814098 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:31:22.814110 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.36
    I1230 20:31:22.814121 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32463 (* 1 = 1.32463 loss)
    I1230 20:31:22.814131 23363 solver.cpp:253]     Train net output #3: loss_f = 2.28461 (* 1 = 2.28461 loss)
    I1230 20:31:22.814141 23363 sgd_solver.cpp:106] Iteration 42200, lr = 0.000202696
    I1230 20:31:39.087155 23363 solver.cpp:237] Iteration 42300, loss = 3.90285
    I1230 20:31:39.087308 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:31:39.087322 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:31:39.087332 23363 solver.cpp:253]     Train net output #2: loss_c = 1.61098 (* 1 = 1.61098 loss)
    I1230 20:31:39.087340 23363 solver.cpp:253]     Train net output #3: loss_f = 2.29187 (* 1 = 2.29187 loss)
    I1230 20:31:39.087349 23363 sgd_solver.cpp:106] Iteration 42300, lr = 0.000202405
    I1230 20:31:55.282207 23363 solver.cpp:237] Iteration 42400, loss = 3.14183
    I1230 20:31:55.282258 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:31:55.282269 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:31:55.282279 23363 solver.cpp:253]     Train net output #2: loss_c = 1.24449 (* 1 = 1.24449 loss)
    I1230 20:31:55.282289 23363 solver.cpp:253]     Train net output #3: loss_f = 1.89734 (* 1 = 1.89734 loss)
    I1230 20:31:55.282299 23363 sgd_solver.cpp:106] Iteration 42400, lr = 0.000202115
    I1230 20:32:11.453356 23363 solver.cpp:237] Iteration 42500, loss = 3.20577
    I1230 20:32:11.453490 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:32:11.453502 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1230 20:32:11.453513 23363 solver.cpp:253]     Train net output #2: loss_c = 1.23181 (* 1 = 1.23181 loss)
    I1230 20:32:11.453522 23363 solver.cpp:253]     Train net output #3: loss_f = 1.97396 (* 1 = 1.97396 loss)
    I1230 20:32:11.453531 23363 sgd_solver.cpp:106] Iteration 42500, lr = 0.000201827
    I1230 20:32:25.737704 23363 solver.cpp:237] Iteration 42600, loss = 3.84942
    I1230 20:32:25.737757 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:32:25.737769 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:32:25.737781 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58027 (* 1 = 1.58027 loss)
    I1230 20:32:25.737790 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26915 (* 1 = 2.26915 loss)
    I1230 20:32:25.737800 23363 sgd_solver.cpp:106] Iteration 42600, lr = 0.000201539
    I1230 20:32:39.235071 23363 solver.cpp:237] Iteration 42700, loss = 3.45523
    I1230 20:32:39.235123 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:32:39.235134 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:32:39.235146 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29996 (* 1 = 1.29996 loss)
    I1230 20:32:39.235154 23363 solver.cpp:253]     Train net output #3: loss_f = 2.15527 (* 1 = 2.15527 loss)
    I1230 20:32:39.235164 23363 sgd_solver.cpp:106] Iteration 42700, lr = 0.000201252
    I1230 20:32:52.801653 23363 solver.cpp:237] Iteration 42800, loss = 3.5
    I1230 20:32:52.801767 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:32:52.801782 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:32:52.801795 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46164 (* 1 = 1.46164 loss)
    I1230 20:32:52.801805 23363 solver.cpp:253]     Train net output #3: loss_f = 2.03836 (* 1 = 2.03836 loss)
    I1230 20:32:52.801815 23363 sgd_solver.cpp:106] Iteration 42800, lr = 0.000200966
    I1230 20:33:06.313663 23363 solver.cpp:237] Iteration 42900, loss = 3.37339
    I1230 20:33:06.313706 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:33:06.313717 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.55
    I1230 20:33:06.313729 23363 solver.cpp:253]     Train net output #2: loss_c = 1.36313 (* 1 = 1.36313 loss)
    I1230 20:33:06.313738 23363 solver.cpp:253]     Train net output #3: loss_f = 2.01025 (* 1 = 2.01025 loss)
    I1230 20:33:06.313750 23363 sgd_solver.cpp:106] Iteration 42900, lr = 0.000200681
    I1230 20:33:19.441606 23363 solver.cpp:341] Iteration 43000, Testing net (#0)
    I1230 20:33:24.434763 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.538583
    I1230 20:33:24.434911 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.411167
    I1230 20:33:24.434932 23363 solver.cpp:409]     Test net output #2: loss_c = 1.46583 (* 1 = 1.46583 loss)
    I1230 20:33:24.434945 23363 solver.cpp:409]     Test net output #3: loss_f = 2.24329 (* 1 = 2.24329 loss)
    I1230 20:33:24.498119 23363 solver.cpp:237] Iteration 43000, loss = 3.49767
    I1230 20:33:24.498172 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:33:24.498183 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:33:24.498193 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34509 (* 1 = 1.34509 loss)
    I1230 20:33:24.498203 23363 solver.cpp:253]     Train net output #3: loss_f = 2.15258 (* 1 = 2.15258 loss)
    I1230 20:33:24.498214 23363 sgd_solver.cpp:106] Iteration 43000, lr = 0.000200397
    I1230 20:33:38.001880 23363 solver.cpp:237] Iteration 43100, loss = 3.6642
    I1230 20:33:38.001925 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.52
    I1230 20:33:38.001937 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:33:38.001950 23363 solver.cpp:253]     Train net output #2: loss_c = 1.46716 (* 1 = 1.46716 loss)
    I1230 20:33:38.001960 23363 solver.cpp:253]     Train net output #3: loss_f = 2.19704 (* 1 = 2.19704 loss)
    I1230 20:33:38.001971 23363 sgd_solver.cpp:106] Iteration 43100, lr = 0.000200114
    I1230 20:33:51.445405 23363 solver.cpp:237] Iteration 43200, loss = 3.67292
    I1230 20:33:51.445451 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:33:51.445463 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:33:51.445475 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35881 (* 1 = 1.35881 loss)
    I1230 20:33:51.445484 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31412 (* 1 = 2.31412 loss)
    I1230 20:33:51.445507 23363 sgd_solver.cpp:106] Iteration 43200, lr = 0.000199832
    I1230 20:34:04.815470 23363 solver.cpp:237] Iteration 43300, loss = 3.59932
    I1230 20:34:04.815609 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.46
    I1230 20:34:04.815631 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:34:04.815656 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4912 (* 1 = 1.4912 loss)
    I1230 20:34:04.815675 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10812 (* 1 = 2.10812 loss)
    I1230 20:34:04.815690 23363 sgd_solver.cpp:106] Iteration 43300, lr = 0.00019955
    I1230 20:34:18.396595 23363 solver.cpp:237] Iteration 43400, loss = 3.1302
    I1230 20:34:18.396649 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:34:18.396661 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:34:18.396672 23363 solver.cpp:253]     Train net output #2: loss_c = 1.26034 (* 1 = 1.26034 loss)
    I1230 20:34:18.396682 23363 solver.cpp:253]     Train net output #3: loss_f = 1.86986 (* 1 = 1.86986 loss)
    I1230 20:34:18.396692 23363 sgd_solver.cpp:106] Iteration 43400, lr = 0.00019927
    I1230 20:34:31.803408 23363 solver.cpp:237] Iteration 43500, loss = 3.40583
    I1230 20:34:31.803478 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:34:31.803490 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:34:31.803503 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31398 (* 1 = 1.31398 loss)
    I1230 20:34:31.803514 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09184 (* 1 = 2.09184 loss)
    I1230 20:34:31.803525 23363 sgd_solver.cpp:106] Iteration 43500, lr = 0.000198991
    I1230 20:34:45.335930 23363 solver.cpp:237] Iteration 43600, loss = 3.62944
    I1230 20:34:45.336082 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:34:45.336104 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:34:45.336119 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45411 (* 1 = 1.45411 loss)
    I1230 20:34:45.336129 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17532 (* 1 = 2.17532 loss)
    I1230 20:34:45.336141 23363 sgd_solver.cpp:106] Iteration 43600, lr = 0.000198712
    I1230 20:34:58.515131 23363 solver.cpp:237] Iteration 43700, loss = 3.38802
    I1230 20:34:58.515177 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:34:58.515188 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:34:58.515199 23363 solver.cpp:253]     Train net output #2: loss_c = 1.25638 (* 1 = 1.25638 loss)
    I1230 20:34:58.515209 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13164 (* 1 = 2.13164 loss)
    I1230 20:34:58.515219 23363 sgd_solver.cpp:106] Iteration 43700, lr = 0.000198435
    I1230 20:35:12.059348 23363 solver.cpp:237] Iteration 43800, loss = 3.45612
    I1230 20:35:12.059391 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:35:12.059402 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:35:12.059414 23363 solver.cpp:253]     Train net output #2: loss_c = 1.43786 (* 1 = 1.43786 loss)
    I1230 20:35:12.059424 23363 solver.cpp:253]     Train net output #3: loss_f = 2.01827 (* 1 = 2.01827 loss)
    I1230 20:35:12.059434 23363 sgd_solver.cpp:106] Iteration 43800, lr = 0.000198158
    I1230 20:35:25.507558 23363 solver.cpp:237] Iteration 43900, loss = 3.15314
    I1230 20:35:25.507699 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:35:25.507721 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:35:25.507740 23363 solver.cpp:253]     Train net output #2: loss_c = 1.24595 (* 1 = 1.24595 loss)
    I1230 20:35:25.507756 23363 solver.cpp:253]     Train net output #3: loss_f = 1.90719 (* 1 = 1.90719 loss)
    I1230 20:35:25.507773 23363 sgd_solver.cpp:106] Iteration 43900, lr = 0.000197882
    I1230 20:35:38.978971 23363 solver.cpp:341] Iteration 44000, Testing net (#0)
    I1230 20:35:43.987157 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.53975
    I1230 20:35:43.987210 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.415417
    I1230 20:35:43.987228 23363 solver.cpp:409]     Test net output #2: loss_c = 1.44979 (* 1 = 1.44979 loss)
    I1230 20:35:43.987241 23363 solver.cpp:409]     Test net output #3: loss_f = 2.2241 (* 1 = 2.2241 loss)
    I1230 20:35:44.050017 23363 solver.cpp:237] Iteration 44000, loss = 3.43174
    I1230 20:35:44.050060 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:35:44.050071 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:35:44.050084 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32495 (* 1 = 1.32495 loss)
    I1230 20:35:44.050096 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10679 (* 1 = 2.10679 loss)
    I1230 20:35:44.050108 23363 sgd_solver.cpp:106] Iteration 44000, lr = 0.000197607
    I1230 20:35:57.503392 23363 solver.cpp:237] Iteration 44100, loss = 3.51076
    I1230 20:35:57.503521 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:35:57.503536 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:35:57.503550 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41712 (* 1 = 1.41712 loss)
    I1230 20:35:57.503561 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09363 (* 1 = 2.09363 loss)
    I1230 20:35:57.503571 23363 sgd_solver.cpp:106] Iteration 44100, lr = 0.000197333
    I1230 20:36:11.046625 23363 solver.cpp:237] Iteration 44200, loss = 3.61505
    I1230 20:36:11.046682 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:36:11.046700 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:36:11.046720 23363 solver.cpp:253]     Train net output #2: loss_c = 1.34742 (* 1 = 1.34742 loss)
    I1230 20:36:11.046735 23363 solver.cpp:253]     Train net output #3: loss_f = 2.26763 (* 1 = 2.26763 loss)
    I1230 20:36:11.046751 23363 sgd_solver.cpp:106] Iteration 44200, lr = 0.00019706
    I1230 20:36:24.762395 23363 solver.cpp:237] Iteration 44300, loss = 3.95084
    I1230 20:36:24.762456 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.45
    I1230 20:36:24.762467 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:36:24.762478 23363 solver.cpp:253]     Train net output #2: loss_c = 1.63451 (* 1 = 1.63451 loss)
    I1230 20:36:24.762488 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31633 (* 1 = 2.31633 loss)
    I1230 20:36:24.762500 23363 sgd_solver.cpp:106] Iteration 44300, lr = 0.000196788
    I1230 20:36:38.232200 23363 solver.cpp:237] Iteration 44400, loss = 3.02882
    I1230 20:36:38.232343 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:36:38.232362 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1230 20:36:38.232377 23363 solver.cpp:253]     Train net output #2: loss_c = 1.17625 (* 1 = 1.17625 loss)
    I1230 20:36:38.232388 23363 solver.cpp:253]     Train net output #3: loss_f = 1.85257 (* 1 = 1.85257 loss)
    I1230 20:36:38.232399 23363 sgd_solver.cpp:106] Iteration 44400, lr = 0.000196516
    I1230 20:36:51.890167 23363 solver.cpp:237] Iteration 44500, loss = 3.26861
    I1230 20:36:51.890216 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1230 20:36:51.890229 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:36:51.890243 23363 solver.cpp:253]     Train net output #2: loss_c = 1.28605 (* 1 = 1.28605 loss)
    I1230 20:36:51.890255 23363 solver.cpp:253]     Train net output #3: loss_f = 1.98256 (* 1 = 1.98256 loss)
    I1230 20:36:51.890267 23363 sgd_solver.cpp:106] Iteration 44500, lr = 0.000196246
    I1230 20:37:05.544114 23363 solver.cpp:237] Iteration 44600, loss = 3.6125
    I1230 20:37:05.544165 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1230 20:37:05.544179 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:37:05.544193 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44872 (* 1 = 1.44872 loss)
    I1230 20:37:05.544205 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16378 (* 1 = 2.16378 loss)
    I1230 20:37:05.544219 23363 sgd_solver.cpp:106] Iteration 44600, lr = 0.000195976
    I1230 20:37:19.278445 23363 solver.cpp:237] Iteration 44700, loss = 3.38361
    I1230 20:37:19.278584 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.68
    I1230 20:37:19.278602 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:37:19.278615 23363 solver.cpp:253]     Train net output #2: loss_c = 1.1915 (* 1 = 1.1915 loss)
    I1230 20:37:19.278626 23363 solver.cpp:253]     Train net output #3: loss_f = 2.19211 (* 1 = 2.19211 loss)
    I1230 20:37:19.278637 23363 sgd_solver.cpp:106] Iteration 44700, lr = 0.000195708
    I1230 20:37:32.703364 23363 solver.cpp:237] Iteration 44800, loss = 3.80769
    I1230 20:37:32.703413 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:37:32.703426 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:37:32.703439 23363 solver.cpp:253]     Train net output #2: loss_c = 1.58185 (* 1 = 1.58185 loss)
    I1230 20:37:32.703450 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22584 (* 1 = 2.22584 loss)
    I1230 20:37:32.703462 23363 sgd_solver.cpp:106] Iteration 44800, lr = 0.00019544
    I1230 20:37:46.248600 23363 solver.cpp:237] Iteration 44900, loss = 3.05534
    I1230 20:37:46.248658 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:37:46.248675 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1230 20:37:46.248693 23363 solver.cpp:253]     Train net output #2: loss_c = 1.19268 (* 1 = 1.19268 loss)
    I1230 20:37:46.248709 23363 solver.cpp:253]     Train net output #3: loss_f = 1.86266 (* 1 = 1.86266 loss)
    I1230 20:37:46.248725 23363 sgd_solver.cpp:106] Iteration 44900, lr = 0.000195173
    I1230 20:37:59.443449 23363 solver.cpp:341] Iteration 45000, Testing net (#0)
    I1230 20:38:04.883569 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.546833
    I1230 20:38:04.883617 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.421333
    I1230 20:38:04.883632 23363 solver.cpp:409]     Test net output #2: loss_c = 1.43217 (* 1 = 1.43217 loss)
    I1230 20:38:04.883646 23363 solver.cpp:409]     Test net output #3: loss_f = 2.20675 (* 1 = 2.20675 loss)
    I1230 20:38:04.948232 23363 solver.cpp:237] Iteration 45000, loss = 3.46306
    I1230 20:38:04.948277 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1230 20:38:04.948290 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:38:04.948303 23363 solver.cpp:253]     Train net output #2: loss_c = 1.298 (* 1 = 1.298 loss)
    I1230 20:38:04.948317 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16506 (* 1 = 2.16506 loss)
    I1230 20:38:04.948329 23363 sgd_solver.cpp:106] Iteration 45000, lr = 0.000194906
    I1230 20:38:18.597746 23363 solver.cpp:237] Iteration 45100, loss = 3.58725
    I1230 20:38:18.597796 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:38:18.597810 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:38:18.597823 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44756 (* 1 = 1.44756 loss)
    I1230 20:38:18.597834 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13969 (* 1 = 2.13969 loss)
    I1230 20:38:18.597847 23363 sgd_solver.cpp:106] Iteration 45100, lr = 0.000194641
    I1230 20:38:32.276546 23363 solver.cpp:237] Iteration 45200, loss = 3.36598
    I1230 20:38:32.276703 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:38:32.276729 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:38:32.276742 23363 solver.cpp:253]     Train net output #2: loss_c = 1.23289 (* 1 = 1.23289 loss)
    I1230 20:38:32.276752 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13309 (* 1 = 2.13309 loss)
    I1230 20:38:32.276763 23363 sgd_solver.cpp:106] Iteration 45200, lr = 0.000194376
    I1230 20:38:45.782313 23363 solver.cpp:237] Iteration 45300, loss = 3.63082
    I1230 20:38:45.782363 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:38:45.782376 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:38:45.782389 23363 solver.cpp:253]     Train net output #2: loss_c = 1.52305 (* 1 = 1.52305 loss)
    I1230 20:38:45.782402 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10777 (* 1 = 2.10777 loss)
    I1230 20:38:45.782413 23363 sgd_solver.cpp:106] Iteration 45300, lr = 0.000194113
    I1230 20:38:59.235690 23363 solver.cpp:237] Iteration 45400, loss = 3.04879
    I1230 20:38:59.235734 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1230 20:38:59.235746 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1230 20:38:59.235759 23363 solver.cpp:253]     Train net output #2: loss_c = 1.19269 (* 1 = 1.19269 loss)
    I1230 20:38:59.235769 23363 solver.cpp:253]     Train net output #3: loss_f = 1.8561 (* 1 = 1.8561 loss)
    I1230 20:38:59.235780 23363 sgd_solver.cpp:106] Iteration 45400, lr = 0.00019385
    I1230 20:39:12.720767 23363 solver.cpp:237] Iteration 45500, loss = 3.46416
    I1230 20:39:12.720871 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:39:12.720886 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:39:12.720901 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31459 (* 1 = 1.31459 loss)
    I1230 20:39:12.720913 23363 solver.cpp:253]     Train net output #3: loss_f = 2.14957 (* 1 = 2.14957 loss)
    I1230 20:39:12.720924 23363 sgd_solver.cpp:106] Iteration 45500, lr = 0.000193588
    I1230 20:39:26.233249 23363 solver.cpp:237] Iteration 45600, loss = 3.44959
    I1230 20:39:26.233306 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:39:26.233324 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:39:26.233342 23363 solver.cpp:253]     Train net output #2: loss_c = 1.35845 (* 1 = 1.35845 loss)
    I1230 20:39:26.233358 23363 solver.cpp:253]     Train net output #3: loss_f = 2.09114 (* 1 = 2.09114 loss)
    I1230 20:39:26.233374 23363 sgd_solver.cpp:106] Iteration 45600, lr = 0.000193327
    I1230 20:39:39.857537 23363 solver.cpp:237] Iteration 45700, loss = 3.44392
    I1230 20:39:39.857599 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:39:39.857619 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:39:39.857636 23363 solver.cpp:253]     Train net output #2: loss_c = 1.24606 (* 1 = 1.24606 loss)
    I1230 20:39:39.857651 23363 solver.cpp:253]     Train net output #3: loss_f = 2.19786 (* 1 = 2.19786 loss)
    I1230 20:39:39.857668 23363 sgd_solver.cpp:106] Iteration 45700, lr = 0.000193066
    I1230 20:39:54.076776 23363 solver.cpp:237] Iteration 45800, loss = 3.69521
    I1230 20:39:54.076927 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:39:54.076941 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:39:54.076951 23363 solver.cpp:253]     Train net output #2: loss_c = 1.57054 (* 1 = 1.57054 loss)
    I1230 20:39:54.076958 23363 solver.cpp:253]     Train net output #3: loss_f = 2.12467 (* 1 = 2.12467 loss)
    I1230 20:39:54.076968 23363 sgd_solver.cpp:106] Iteration 45800, lr = 0.000192807
    I1230 20:40:10.382778 23363 solver.cpp:237] Iteration 45900, loss = 3.0297
    I1230 20:40:10.382819 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:40:10.382828 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1230 20:40:10.382838 23363 solver.cpp:253]     Train net output #2: loss_c = 1.23772 (* 1 = 1.23772 loss)
    I1230 20:40:10.382848 23363 solver.cpp:253]     Train net output #3: loss_f = 1.79198 (* 1 = 1.79198 loss)
    I1230 20:40:10.382858 23363 sgd_solver.cpp:106] Iteration 45900, lr = 0.000192548
    I1230 20:40:26.689352 23363 solver.cpp:341] Iteration 46000, Testing net (#0)
    I1230 20:40:33.125424 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.54825
    I1230 20:40:33.125481 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.42175
    I1230 20:40:33.125502 23363 solver.cpp:409]     Test net output #2: loss_c = 1.44268 (* 1 = 1.44268 loss)
    I1230 20:40:33.125519 23363 solver.cpp:409]     Test net output #3: loss_f = 2.20965 (* 1 = 2.20965 loss)
    I1230 20:40:33.200208 23363 solver.cpp:237] Iteration 46000, loss = 3.46909
    I1230 20:40:33.200270 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:40:33.200289 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:40:33.200305 23363 solver.cpp:253]     Train net output #2: loss_c = 1.30387 (* 1 = 1.30387 loss)
    I1230 20:40:33.200321 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16522 (* 1 = 2.16522 loss)
    I1230 20:40:33.200338 23363 sgd_solver.cpp:106] Iteration 46000, lr = 0.00019229
    I1230 20:40:48.834395 23363 solver.cpp:237] Iteration 46100, loss = 3.55172
    I1230 20:40:48.834434 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:40:48.834444 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:40:48.834455 23363 solver.cpp:253]     Train net output #2: loss_c = 1.4655 (* 1 = 1.4655 loss)
    I1230 20:40:48.834465 23363 solver.cpp:253]     Train net output #3: loss_f = 2.08622 (* 1 = 2.08622 loss)
    I1230 20:40:48.834475 23363 sgd_solver.cpp:106] Iteration 46100, lr = 0.000192033
    I1230 20:41:04.216502 23363 solver.cpp:237] Iteration 46200, loss = 3.25614
    I1230 20:41:04.216657 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:41:04.216681 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:41:04.216691 23363 solver.cpp:253]     Train net output #2: loss_c = 1.19546 (* 1 = 1.19546 loss)
    I1230 20:41:04.216698 23363 solver.cpp:253]     Train net output #3: loss_f = 2.06068 (* 1 = 2.06068 loss)
    I1230 20:41:04.216708 23363 sgd_solver.cpp:106] Iteration 46200, lr = 0.000191777
    I1230 20:41:19.377074 23363 solver.cpp:237] Iteration 46300, loss = 3.73954
    I1230 20:41:19.377115 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:41:19.377127 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:41:19.377137 23363 solver.cpp:253]     Train net output #2: loss_c = 1.56156 (* 1 = 1.56156 loss)
    I1230 20:41:19.377147 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17798 (* 1 = 2.17798 loss)
    I1230 20:41:19.377157 23363 sgd_solver.cpp:106] Iteration 46300, lr = 0.000191521
    I1230 20:41:34.531729 23363 solver.cpp:237] Iteration 46400, loss = 3.16376
    I1230 20:41:34.531884 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:41:34.531898 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:41:34.531908 23363 solver.cpp:253]     Train net output #2: loss_c = 1.297 (* 1 = 1.297 loss)
    I1230 20:41:34.531916 23363 solver.cpp:253]     Train net output #3: loss_f = 1.86675 (* 1 = 1.86675 loss)
    I1230 20:41:34.531925 23363 sgd_solver.cpp:106] Iteration 46400, lr = 0.000191266
    I1230 20:41:49.770284 23363 solver.cpp:237] Iteration 46500, loss = 3.16429
    I1230 20:41:49.770340 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.62
    I1230 20:41:49.770356 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:41:49.770373 23363 solver.cpp:253]     Train net output #2: loss_c = 1.20115 (* 1 = 1.20115 loss)
    I1230 20:41:49.770388 23363 solver.cpp:253]     Train net output #3: loss_f = 1.96314 (* 1 = 1.96314 loss)
    I1230 20:41:49.770403 23363 sgd_solver.cpp:106] Iteration 46500, lr = 0.000191012
    I1230 20:42:05.082808 23363 solver.cpp:237] Iteration 46600, loss = 3.50781
    I1230 20:42:05.082989 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:42:05.083004 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:42:05.083015 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39466 (* 1 = 1.39466 loss)
    I1230 20:42:05.083024 23363 solver.cpp:253]     Train net output #3: loss_f = 2.11315 (* 1 = 2.11315 loss)
    I1230 20:42:05.083034 23363 sgd_solver.cpp:106] Iteration 46600, lr = 0.000190759
    I1230 20:42:21.135967 23363 solver.cpp:237] Iteration 46700, loss = 3.52847
    I1230 20:42:21.136004 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:42:21.136016 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:42:21.136028 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29934 (* 1 = 1.29934 loss)
    I1230 20:42:21.136036 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22913 (* 1 = 2.22913 loss)
    I1230 20:42:21.136046 23363 sgd_solver.cpp:106] Iteration 46700, lr = 0.000190507
    I1230 20:42:37.204324 23363 solver.cpp:237] Iteration 46800, loss = 3.7435
    I1230 20:42:37.204444 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:42:37.204463 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:42:37.204476 23363 solver.cpp:253]     Train net output #2: loss_c = 1.53417 (* 1 = 1.53417 loss)
    I1230 20:42:37.204488 23363 solver.cpp:253]     Train net output #3: loss_f = 2.20933 (* 1 = 2.20933 loss)
    I1230 20:42:37.204499 23363 sgd_solver.cpp:106] Iteration 46800, lr = 0.000190255
    I1230 20:42:52.878273 23363 solver.cpp:237] Iteration 46900, loss = 2.93703
    I1230 20:42:52.878314 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1230 20:42:52.878326 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1230 20:42:52.878337 23363 solver.cpp:253]     Train net output #2: loss_c = 1.12168 (* 1 = 1.12168 loss)
    I1230 20:42:52.878346 23363 solver.cpp:253]     Train net output #3: loss_f = 1.81535 (* 1 = 1.81535 loss)
    I1230 20:42:52.878356 23363 sgd_solver.cpp:106] Iteration 46900, lr = 0.000190004
    I1230 20:43:08.496866 23363 solver.cpp:341] Iteration 47000, Testing net (#0)
    I1230 20:43:14.413559 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.548333
    I1230 20:43:14.413599 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.426333
    I1230 20:43:14.413610 23363 solver.cpp:409]     Test net output #2: loss_c = 1.42713 (* 1 = 1.42713 loss)
    I1230 20:43:14.413620 23363 solver.cpp:409]     Test net output #3: loss_f = 2.18678 (* 1 = 2.18678 loss)
    I1230 20:43:14.483762 23363 solver.cpp:237] Iteration 47000, loss = 3.43707
    I1230 20:43:14.483808 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:43:14.483819 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:43:14.483830 23363 solver.cpp:253]     Train net output #2: loss_c = 1.29127 (* 1 = 1.29127 loss)
    I1230 20:43:14.483840 23363 solver.cpp:253]     Train net output #3: loss_f = 2.1458 (* 1 = 2.1458 loss)
    I1230 20:43:14.483851 23363 sgd_solver.cpp:106] Iteration 47000, lr = 0.000189754
    I1230 20:43:30.220726 23363 solver.cpp:237] Iteration 47100, loss = 3.51955
    I1230 20:43:30.220767 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:43:30.220777 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:43:30.220789 23363 solver.cpp:253]     Train net output #2: loss_c = 1.41601 (* 1 = 1.41601 loss)
    I1230 20:43:30.220799 23363 solver.cpp:253]     Train net output #3: loss_f = 2.10354 (* 1 = 2.10354 loss)
    I1230 20:43:30.220808 23363 sgd_solver.cpp:106] Iteration 47100, lr = 0.000189505
    I1230 20:43:45.732604 23363 solver.cpp:237] Iteration 47200, loss = 3.4395
    I1230 20:43:45.732777 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:43:45.732795 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.4
    I1230 20:43:45.732810 23363 solver.cpp:253]     Train net output #2: loss_c = 1.25494 (* 1 = 1.25494 loss)
    I1230 20:43:45.732820 23363 solver.cpp:253]     Train net output #3: loss_f = 2.18456 (* 1 = 2.18456 loss)
    I1230 20:43:45.732831 23363 sgd_solver.cpp:106] Iteration 47200, lr = 0.000189257
    I1230 20:44:01.431488 23363 solver.cpp:237] Iteration 47300, loss = 3.85741
    I1230 20:44:01.431535 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:44:01.431548 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:44:01.431561 23363 solver.cpp:253]     Train net output #2: loss_c = 1.60891 (* 1 = 1.60891 loss)
    I1230 20:44:01.431572 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24849 (* 1 = 2.24849 loss)
    I1230 20:44:01.431583 23363 sgd_solver.cpp:106] Iteration 47300, lr = 0.000189009
    I1230 20:44:19.050417 23363 solver.cpp:237] Iteration 47400, loss = 3.12181
    I1230 20:44:19.050516 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:44:19.050539 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:44:19.050562 23363 solver.cpp:253]     Train net output #2: loss_c = 1.22041 (* 1 = 1.22041 loss)
    I1230 20:44:19.050575 23363 solver.cpp:253]     Train net output #3: loss_f = 1.9014 (* 1 = 1.9014 loss)
    I1230 20:44:19.050590 23363 sgd_solver.cpp:106] Iteration 47400, lr = 0.000188762
    I1230 20:44:35.469887 23363 solver.cpp:237] Iteration 47500, loss = 3.36959
    I1230 20:44:35.469949 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:44:35.469967 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:44:35.469986 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31354 (* 1 = 1.31354 loss)
    I1230 20:44:35.470003 23363 solver.cpp:253]     Train net output #3: loss_f = 2.05605 (* 1 = 2.05605 loss)
    I1230 20:44:35.470019 23363 sgd_solver.cpp:106] Iteration 47500, lr = 0.000188516
    I1230 20:44:51.298913 23363 solver.cpp:237] Iteration 47600, loss = 3.56828
    I1230 20:44:51.299037 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:44:51.299052 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:44:51.299064 23363 solver.cpp:253]     Train net output #2: loss_c = 1.39919 (* 1 = 1.39919 loss)
    I1230 20:44:51.299075 23363 solver.cpp:253]     Train net output #3: loss_f = 2.16909 (* 1 = 2.16909 loss)
    I1230 20:44:51.299088 23363 sgd_solver.cpp:106] Iteration 47600, lr = 0.00018827
    I1230 20:45:06.645056 23363 solver.cpp:237] Iteration 47700, loss = 3.27158
    I1230 20:45:06.645107 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:45:06.645119 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.39
    I1230 20:45:06.645143 23363 solver.cpp:253]     Train net output #2: loss_c = 1.19291 (* 1 = 1.19291 loss)
    I1230 20:45:06.645165 23363 solver.cpp:253]     Train net output #3: loss_f = 2.07868 (* 1 = 2.07868 loss)
    I1230 20:45:06.645176 23363 sgd_solver.cpp:106] Iteration 47700, lr = 0.000188025
    I1230 20:45:21.960418 23363 solver.cpp:237] Iteration 47800, loss = 3.86935
    I1230 20:45:21.960628 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1230 20:45:21.960643 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:45:21.960652 23363 solver.cpp:253]     Train net output #2: loss_c = 1.62712 (* 1 = 1.62712 loss)
    I1230 20:45:21.960661 23363 solver.cpp:253]     Train net output #3: loss_f = 2.24223 (* 1 = 2.24223 loss)
    I1230 20:45:21.960670 23363 sgd_solver.cpp:106] Iteration 47800, lr = 0.000187781
    I1230 20:45:38.818790 23363 solver.cpp:237] Iteration 47900, loss = 3.26624
    I1230 20:45:38.818846 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.55
    I1230 20:45:38.818858 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:45:38.818871 23363 solver.cpp:253]     Train net output #2: loss_c = 1.30638 (* 1 = 1.30638 loss)
    I1230 20:45:38.818881 23363 solver.cpp:253]     Train net output #3: loss_f = 1.95986 (* 1 = 1.95986 loss)
    I1230 20:45:38.818904 23363 sgd_solver.cpp:106] Iteration 47900, lr = 0.000187538
    I1230 20:45:54.310696 23363 solver.cpp:341] Iteration 48000, Testing net (#0)
    I1230 20:46:00.433106 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.552917
    I1230 20:46:00.433156 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.426667
    I1230 20:46:00.433169 23363 solver.cpp:409]     Test net output #2: loss_c = 1.4222 (* 1 = 1.4222 loss)
    I1230 20:46:00.433179 23363 solver.cpp:409]     Test net output #3: loss_f = 2.18278 (* 1 = 2.18278 loss)
    I1230 20:46:00.507673 23363 solver.cpp:237] Iteration 48000, loss = 3.56396
    I1230 20:46:00.507720 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.59
    I1230 20:46:00.507730 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:46:00.507741 23363 solver.cpp:253]     Train net output #2: loss_c = 1.38804 (* 1 = 1.38804 loss)
    I1230 20:46:00.507751 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17592 (* 1 = 2.17592 loss)
    I1230 20:46:00.507761 23363 sgd_solver.cpp:106] Iteration 48000, lr = 0.000187295
    I1230 20:46:16.165117 23363 solver.cpp:237] Iteration 48100, loss = 3.5617
    I1230 20:46:16.165168 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.56
    I1230 20:46:16.165177 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.42
    I1230 20:46:16.165189 23363 solver.cpp:253]     Train net output #2: loss_c = 1.42041 (* 1 = 1.42041 loss)
    I1230 20:46:16.165199 23363 solver.cpp:253]     Train net output #3: loss_f = 2.14129 (* 1 = 2.14129 loss)
    I1230 20:46:16.165210 23363 sgd_solver.cpp:106] Iteration 48100, lr = 0.000187054
    I1230 20:46:31.622238 23363 solver.cpp:237] Iteration 48200, loss = 3.50702
    I1230 20:46:31.622347 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:46:31.622380 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:46:31.622393 23363 solver.cpp:253]     Train net output #2: loss_c = 1.32157 (* 1 = 1.32157 loss)
    I1230 20:46:31.622405 23363 solver.cpp:253]     Train net output #3: loss_f = 2.18546 (* 1 = 2.18546 loss)
    I1230 20:46:31.622416 23363 sgd_solver.cpp:106] Iteration 48200, lr = 0.000186812
    I1230 20:46:47.258054 23363 solver.cpp:237] Iteration 48300, loss = 3.58409
    I1230 20:46:47.258091 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:46:47.258100 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:46:47.258111 23363 solver.cpp:253]     Train net output #2: loss_c = 1.44579 (* 1 = 1.44579 loss)
    I1230 20:46:47.258119 23363 solver.cpp:253]     Train net output #3: loss_f = 2.13829 (* 1 = 2.13829 loss)
    I1230 20:46:47.258128 23363 sgd_solver.cpp:106] Iteration 48300, lr = 0.000186572
    I1230 20:47:02.655074 23363 solver.cpp:237] Iteration 48400, loss = 2.96106
    I1230 20:47:02.655232 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1230 20:47:02.655247 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.52
    I1230 20:47:02.655259 23363 solver.cpp:253]     Train net output #2: loss_c = 1.13815 (* 1 = 1.13815 loss)
    I1230 20:47:02.655268 23363 solver.cpp:253]     Train net output #3: loss_f = 1.82291 (* 1 = 1.82291 loss)
    I1230 20:47:02.655279 23363 sgd_solver.cpp:106] Iteration 48400, lr = 0.000186332
    I1230 20:47:18.809640 23363 solver.cpp:237] Iteration 48500, loss = 3.31256
    I1230 20:47:18.809692 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:47:18.809705 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:47:18.809716 23363 solver.cpp:253]     Train net output #2: loss_c = 1.2839 (* 1 = 1.2839 loss)
    I1230 20:47:18.809727 23363 solver.cpp:253]     Train net output #3: loss_f = 2.02866 (* 1 = 2.02866 loss)
    I1230 20:47:18.809738 23363 sgd_solver.cpp:106] Iteration 48500, lr = 0.000186093
    I1230 20:47:35.549257 23363 solver.cpp:237] Iteration 48600, loss = 3.78616
    I1230 20:47:35.549372 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.5
    I1230 20:47:35.549387 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.43
    I1230 20:47:35.549401 23363 solver.cpp:253]     Train net output #2: loss_c = 1.5752 (* 1 = 1.5752 loss)
    I1230 20:47:35.549412 23363 solver.cpp:253]     Train net output #3: loss_f = 2.21096 (* 1 = 2.21096 loss)
    I1230 20:47:35.549423 23363 sgd_solver.cpp:106] Iteration 48600, lr = 0.000185855
    I1230 20:47:51.332171 23363 solver.cpp:237] Iteration 48700, loss = 3.21268
    I1230 20:47:51.332229 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1230 20:47:51.332242 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.44
    I1230 20:47:51.332255 23363 solver.cpp:253]     Train net output #2: loss_c = 1.18035 (* 1 = 1.18035 loss)
    I1230 20:47:51.332265 23363 solver.cpp:253]     Train net output #3: loss_f = 2.03233 (* 1 = 2.03233 loss)
    I1230 20:47:51.332276 23363 sgd_solver.cpp:106] Iteration 48700, lr = 0.000185618
    I1230 20:48:09.090114 23363 solver.cpp:237] Iteration 48800, loss = 3.76991
    I1230 20:48:09.090216 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:48:09.090232 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:48:09.090245 23363 solver.cpp:253]     Train net output #2: loss_c = 1.57329 (* 1 = 1.57329 loss)
    I1230 20:48:09.090255 23363 solver.cpp:253]     Train net output #3: loss_f = 2.19662 (* 1 = 2.19662 loss)
    I1230 20:48:09.090267 23363 sgd_solver.cpp:106] Iteration 48800, lr = 0.000185381
    I1230 20:48:26.389976 23363 solver.cpp:237] Iteration 48900, loss = 3.11128
    I1230 20:48:26.390028 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:48:26.390040 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.5
    I1230 20:48:26.390053 23363 solver.cpp:253]     Train net output #2: loss_c = 1.25108 (* 1 = 1.25108 loss)
    I1230 20:48:26.390063 23363 solver.cpp:253]     Train net output #3: loss_f = 1.8602 (* 1 = 1.8602 loss)
    I1230 20:48:26.390074 23363 sgd_solver.cpp:106] Iteration 48900, lr = 0.000185145
    I1230 20:48:41.856417 23363 solver.cpp:341] Iteration 49000, Testing net (#0)
    I1230 20:48:47.865113 23363 solver.cpp:409]     Test net output #0: accuracy_c = 0.549333
    I1230 20:48:47.865159 23363 solver.cpp:409]     Test net output #1: accuracy_f = 0.427333
    I1230 20:48:47.865173 23363 solver.cpp:409]     Test net output #2: loss_c = 1.42702 (* 1 = 1.42702 loss)
    I1230 20:48:47.865185 23363 solver.cpp:409]     Test net output #3: loss_f = 2.19429 (* 1 = 2.19429 loss)
    I1230 20:48:47.937688 23363 solver.cpp:237] Iteration 49000, loss = 3.3167
    I1230 20:48:47.937760 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.57
    I1230 20:48:47.937772 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.46
    I1230 20:48:47.937783 23363 solver.cpp:253]     Train net output #2: loss_c = 1.27839 (* 1 = 1.27839 loss)
    I1230 20:48:47.937793 23363 solver.cpp:253]     Train net output #3: loss_f = 2.03831 (* 1 = 2.03831 loss)
    I1230 20:48:47.937804 23363 sgd_solver.cpp:106] Iteration 49000, lr = 0.000184909
    I1230 20:49:03.821797 23363 solver.cpp:237] Iteration 49100, loss = 3.68313
    I1230 20:49:03.821847 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.54
    I1230 20:49:03.821861 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1230 20:49:03.821873 23363 solver.cpp:253]     Train net output #2: loss_c = 1.45776 (* 1 = 1.45776 loss)
    I1230 20:49:03.821885 23363 solver.cpp:253]     Train net output #3: loss_f = 2.22537 (* 1 = 2.22537 loss)
    I1230 20:49:03.821897 23363 sgd_solver.cpp:106] Iteration 49100, lr = 0.000184675
    I1230 20:49:19.900197 23363 solver.cpp:237] Iteration 49200, loss = 3.20203
    I1230 20:49:19.900344 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:49:19.900364 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:49:19.900378 23363 solver.cpp:253]     Train net output #2: loss_c = 1.18838 (* 1 = 1.18838 loss)
    I1230 20:49:19.900391 23363 solver.cpp:253]     Train net output #3: loss_f = 2.01365 (* 1 = 2.01365 loss)
    I1230 20:49:19.900404 23363 sgd_solver.cpp:106] Iteration 49200, lr = 0.000184441
    I1230 20:49:35.258577 23363 solver.cpp:237] Iteration 49300, loss = 3.94265
    I1230 20:49:35.258625 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.48
    I1230 20:49:35.258635 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.38
    I1230 20:49:35.258646 23363 solver.cpp:253]     Train net output #2: loss_c = 1.6266 (* 1 = 1.6266 loss)
    I1230 20:49:35.258656 23363 solver.cpp:253]     Train net output #3: loss_f = 2.31605 (* 1 = 2.31605 loss)
    I1230 20:49:35.258664 23363 sgd_solver.cpp:106] Iteration 49300, lr = 0.000184207
    I1230 20:49:50.641644 23363 solver.cpp:237] Iteration 49400, loss = 2.9684
    I1230 20:49:50.641777 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.68
    I1230 20:49:50.641801 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.54
    I1230 20:49:50.641821 23363 solver.cpp:253]     Train net output #2: loss_c = 1.16508 (* 1 = 1.16508 loss)
    I1230 20:49:50.641839 23363 solver.cpp:253]     Train net output #3: loss_f = 1.80332 (* 1 = 1.80332 loss)
    I1230 20:49:50.641857 23363 sgd_solver.cpp:106] Iteration 49400, lr = 0.000183975
    I1230 20:50:06.229737 23363 solver.cpp:237] Iteration 49500, loss = 3.39031
    I1230 20:50:06.229775 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1230 20:50:06.229785 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:50:06.229796 23363 solver.cpp:253]     Train net output #2: loss_c = 1.31631 (* 1 = 1.31631 loss)
    I1230 20:50:06.229807 23363 solver.cpp:253]     Train net output #3: loss_f = 2.074 (* 1 = 2.074 loss)
    I1230 20:50:06.229816 23363 sgd_solver.cpp:106] Iteration 49500, lr = 0.000183743
    I1230 20:50:22.041056 23363 solver.cpp:237] Iteration 49600, loss = 3.87282
    I1230 20:50:22.041193 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.51
    I1230 20:50:22.041206 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:50:22.041218 23363 solver.cpp:253]     Train net output #2: loss_c = 1.54589 (* 1 = 1.54589 loss)
    I1230 20:50:22.041226 23363 solver.cpp:253]     Train net output #3: loss_f = 2.32693 (* 1 = 2.32693 loss)
    I1230 20:50:22.041236 23363 sgd_solver.cpp:106] Iteration 49600, lr = 0.000183512
    I1230 20:50:37.754669 23363 solver.cpp:237] Iteration 49700, loss = 3.36482
    I1230 20:50:37.754725 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1230 20:50:37.754744 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.48
    I1230 20:50:37.754761 23363 solver.cpp:253]     Train net output #2: loss_c = 1.21442 (* 1 = 1.21442 loss)
    I1230 20:50:37.754777 23363 solver.cpp:253]     Train net output #3: loss_f = 2.15041 (* 1 = 2.15041 loss)
    I1230 20:50:37.754793 23363 sgd_solver.cpp:106] Iteration 49700, lr = 0.000183281
    I1230 20:50:53.133919 23363 solver.cpp:237] Iteration 49800, loss = 3.71086
    I1230 20:50:53.134110 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.53
    I1230 20:50:53.134124 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.41
    I1230 20:50:53.134135 23363 solver.cpp:253]     Train net output #2: loss_c = 1.53939 (* 1 = 1.53939 loss)
    I1230 20:50:53.134145 23363 solver.cpp:253]     Train net output #3: loss_f = 2.17147 (* 1 = 2.17147 loss)
    I1230 20:50:53.134155 23363 sgd_solver.cpp:106] Iteration 49800, lr = 0.000183051
    I1230 20:51:08.682876 23363 solver.cpp:237] Iteration 49900, loss = 3.08998
    I1230 20:51:08.682926 23363 solver.cpp:253]     Train net output #0: accuracy_c = 0.6
    I1230 20:51:08.682939 23363 solver.cpp:253]     Train net output #1: accuracy_f = 0.45
    I1230 20:51:08.682950 23363 solver.cpp:253]     Train net output #2: loss_c = 1.25242 (* 1 = 1.25242 loss)
    I1230 20:51:08.682962 23363 solver.cpp:253]     Train net output #3: loss_f = 1.83756 (* 1 = 1.83756 loss)
    I1230 20:51:08.682972 23363 sgd_solver.cpp:106] Iteration 49900, lr = 0.000182822
    I1230 20:51:23.961479 23363 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_50000.caffemodel
    I1230 20:51:24.030045 23363 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_50000.solverstate
    I1230 20:51:24.031729 23363 solver.cpp:341] Iteration 50000, Testing net (#0)
    ^CI1230 20:51:29.566437 23363 solver.cpp:391] Test interrupted.
    I1230 20:51:29.566483 23363 solver.cpp:309] Optimization stopped early.
    I1230 20:51:29.566496 23363 caffe.cpp:215] Optimization Done.

    CPU times: user 27.3 s, sys: 3.44 s, total: 30.7 s
    Wall time: 2h 12min 31s


Caffe brewed.
## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_50000.caffemodel -iterations 83
```

    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1230 20:51:46.782888 27818 caffe.cpp:234] Use CPU.
    I1230 20:51:46.993034 27818 net.cpp:49] Initializing net from parameters:
    state {
      phase: TEST
    }
    layer {
      name: "data"
      type: "HDF5Data"
      top: "data"
      top: "label_coarse"
      top: "label_fine"
      hdf5_data_param {
        source: "cifar_100_caffe_hdf5/test.txt"
        batch_size: 120
      }
    }
    layer {
      name: "conv1"
      type: "Convolution"
      bottom: "data"
      top: "conv1"
      convolution_param {
        num_output: 64
        kernel_size: 4
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp1"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "cccp2"
      type: "Convolution"
      bottom: "cccp1"
      top: "cccp2"
      convolution_param {
        num_output: 32
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool1"
      type: "Pooling"
      bottom: "cccp2"
      top: "pool1"
      pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop1"
      type: "Dropout"
      bottom: "pool1"
      top: "pool1"
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
        num_output: 42
        kernel_size: 4
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
        kernel_size: 3
        stride: 2
      }
    }
    layer {
      name: "drop2"
      type: "Dropout"
      bottom: "pool2"
      top: "pool2"
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
        kernel_size: 2
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "pool3"
      type: "Pooling"
      bottom: "conv3"
      top: "pool3"
      pooling_param {
        pool: AVE
        kernel_size: 2
        stride: 2
      }
    }
    layer {
      name: "relu3"
      type: "ReLU"
      bottom: "pool3"
      top: "pool3"
    }
    layer {
      name: "ip1"
      type: "InnerProduct"
      bottom: "pool3"
      top: "ip1"
      inner_product_param {
        num_output: 512
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "sig1"
      type: "Sigmoid"
      bottom: "ip1"
      top: "ip1"
    }
    layer {
      name: "ip_c"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_c"
      inner_product_param {
        num_output: 20
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_c"
      type: "Accuracy"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "accuracy_c"
    }
    layer {
      name: "loss_c"
      type: "SoftmaxWithLoss"
      bottom: "ip_c"
      bottom: "label_coarse"
      top: "loss_c"
    }
    layer {
      name: "ip_f"
      type: "InnerProduct"
      bottom: "ip1"
      top: "ip_f"
      inner_product_param {
        num_output: 100
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "accuracy_f"
      type: "Accuracy"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "accuracy_f"
    }
    layer {
      name: "loss_f"
      type: "SoftmaxWithLoss"
      bottom: "ip_f"
      bottom: "label_fine"
      top: "loss_f"
    }
    I1230 20:51:46.994405 27818 layer_factory.hpp:77] Creating layer data
    I1230 20:51:46.994446 27818 net.cpp:106] Creating Layer data
    I1230 20:51:46.994464 27818 net.cpp:411] data -> data
    I1230 20:51:46.994500 27818 net.cpp:411] data -> label_coarse
    I1230 20:51:46.994524 27818 net.cpp:411] data -> label_fine
    I1230 20:51:46.994545 27818 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/test.txt
    I1230 20:51:46.994616 27818 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1230 20:51:46.996502 27818 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1230 20:51:47.330950 27818 net.cpp:150] Setting up data
    I1230 20:51:47.330987 27818 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1230 20:51:47.330996 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331001 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331007 27818 net.cpp:165] Memory required for data: 1475520
    I1230 20:51:47.331017 27818 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1230 20:51:47.331037 27818 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1230 20:51:47.331065 27818 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1230 20:51:47.331078 27818 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1230 20:51:47.331089 27818 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1230 20:51:47.331099 27818 net.cpp:150] Setting up label_coarse_data_1_split
    I1230 20:51:47.331106 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331112 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331117 27818 net.cpp:165] Memory required for data: 1476480
    I1230 20:51:47.331123 27818 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1230 20:51:47.331130 27818 net.cpp:106] Creating Layer label_fine_data_2_split
    I1230 20:51:47.331136 27818 net.cpp:454] label_fine_data_2_split <- label_fine
    I1230 20:51:47.331142 27818 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1230 20:51:47.331151 27818 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1230 20:51:47.331157 27818 net.cpp:150] Setting up label_fine_data_2_split
    I1230 20:51:47.331163 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331169 27818 net.cpp:157] Top shape: 120 (120)
    I1230 20:51:47.331174 27818 net.cpp:165] Memory required for data: 1477440
    I1230 20:51:47.331179 27818 layer_factory.hpp:77] Creating layer conv1
    I1230 20:51:47.331189 27818 net.cpp:106] Creating Layer conv1
    I1230 20:51:47.331194 27818 net.cpp:454] conv1 <- data
    I1230 20:51:47.331202 27818 net.cpp:411] conv1 -> conv1
    I1230 20:51:47.331619 27818 net.cpp:150] Setting up conv1
    I1230 20:51:47.331631 27818 net.cpp:157] Top shape: 120 64 29 29 (6458880)
    I1230 20:51:47.331636 27818 net.cpp:165] Memory required for data: 27312960
    I1230 20:51:47.331648 27818 layer_factory.hpp:77] Creating layer cccp1
    I1230 20:51:47.331665 27818 net.cpp:106] Creating Layer cccp1
    I1230 20:51:47.331670 27818 net.cpp:454] cccp1 <- conv1
    I1230 20:51:47.331677 27818 net.cpp:411] cccp1 -> cccp1
    I1230 20:51:47.331714 27818 net.cpp:150] Setting up cccp1
    I1230 20:51:47.331722 27818 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1230 20:51:47.331727 27818 net.cpp:165] Memory required for data: 44267520
    I1230 20:51:47.331735 27818 layer_factory.hpp:77] Creating layer cccp2
    I1230 20:51:47.331743 27818 net.cpp:106] Creating Layer cccp2
    I1230 20:51:47.331748 27818 net.cpp:454] cccp2 <- cccp1
    I1230 20:51:47.331755 27818 net.cpp:411] cccp2 -> cccp2
    I1230 20:51:47.331779 27818 net.cpp:150] Setting up cccp2
    I1230 20:51:47.331786 27818 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1230 20:51:47.331791 27818 net.cpp:165] Memory required for data: 57185280
    I1230 20:51:47.331799 27818 layer_factory.hpp:77] Creating layer pool1
    I1230 20:51:47.331809 27818 net.cpp:106] Creating Layer pool1
    I1230 20:51:47.331815 27818 net.cpp:454] pool1 <- cccp2
    I1230 20:51:47.331820 27818 net.cpp:411] pool1 -> pool1
    I1230 20:51:47.331840 27818 net.cpp:150] Setting up pool1
    I1230 20:51:47.331846 27818 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 20:51:47.331851 27818 net.cpp:165] Memory required for data: 60195840
    I1230 20:51:47.331856 27818 layer_factory.hpp:77] Creating layer drop1
    I1230 20:51:47.331866 27818 net.cpp:106] Creating Layer drop1
    I1230 20:51:47.331872 27818 net.cpp:454] drop1 <- pool1
    I1230 20:51:47.331878 27818 net.cpp:397] drop1 -> pool1 (in-place)
    I1230 20:51:47.331890 27818 net.cpp:150] Setting up drop1
    I1230 20:51:47.331897 27818 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 20:51:47.331902 27818 net.cpp:165] Memory required for data: 63206400
    I1230 20:51:47.331907 27818 layer_factory.hpp:77] Creating layer relu1
    I1230 20:51:47.331914 27818 net.cpp:106] Creating Layer relu1
    I1230 20:51:47.331919 27818 net.cpp:454] relu1 <- pool1
    I1230 20:51:47.331925 27818 net.cpp:397] relu1 -> pool1 (in-place)
    I1230 20:51:47.331933 27818 net.cpp:150] Setting up relu1
    I1230 20:51:47.331938 27818 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 20:51:47.331943 27818 net.cpp:165] Memory required for data: 66216960
    I1230 20:51:47.331948 27818 layer_factory.hpp:77] Creating layer conv2
    I1230 20:51:47.331964 27818 net.cpp:106] Creating Layer conv2
    I1230 20:51:47.331970 27818 net.cpp:454] conv2 <- pool1
    I1230 20:51:47.331977 27818 net.cpp:411] conv2 -> conv2
    I1230 20:51:47.332125 27818 net.cpp:150] Setting up conv2
    I1230 20:51:47.332134 27818 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1230 20:51:47.332139 27818 net.cpp:165] Memory required for data: 68656320
    I1230 20:51:47.332145 27818 layer_factory.hpp:77] Creating layer pool2
    I1230 20:51:47.332154 27818 net.cpp:106] Creating Layer pool2
    I1230 20:51:47.332159 27818 net.cpp:454] pool2 <- conv2
    I1230 20:51:47.332165 27818 net.cpp:411] pool2 -> pool2
    I1230 20:51:47.332175 27818 net.cpp:150] Setting up pool2
    I1230 20:51:47.332180 27818 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 20:51:47.332185 27818 net.cpp:165] Memory required for data: 69160320
    I1230 20:51:47.332190 27818 layer_factory.hpp:77] Creating layer drop2
    I1230 20:51:47.332196 27818 net.cpp:106] Creating Layer drop2
    I1230 20:51:47.332201 27818 net.cpp:454] drop2 <- pool2
    I1230 20:51:47.332208 27818 net.cpp:397] drop2 -> pool2 (in-place)
    I1230 20:51:47.332216 27818 net.cpp:150] Setting up drop2
    I1230 20:51:47.332221 27818 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 20:51:47.332226 27818 net.cpp:165] Memory required for data: 69664320
    I1230 20:51:47.332231 27818 layer_factory.hpp:77] Creating layer relu2
    I1230 20:51:47.332237 27818 net.cpp:106] Creating Layer relu2
    I1230 20:51:47.332243 27818 net.cpp:454] relu2 <- pool2
    I1230 20:51:47.332248 27818 net.cpp:397] relu2 -> pool2 (in-place)
    I1230 20:51:47.332255 27818 net.cpp:150] Setting up relu2
    I1230 20:51:47.332262 27818 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 20:51:47.332267 27818 net.cpp:165] Memory required for data: 70168320
    I1230 20:51:47.332272 27818 layer_factory.hpp:77] Creating layer conv3
    I1230 20:51:47.332278 27818 net.cpp:106] Creating Layer conv3
    I1230 20:51:47.332283 27818 net.cpp:454] conv3 <- pool2
    I1230 20:51:47.332289 27818 net.cpp:411] conv3 -> conv3
    I1230 20:51:47.332398 27818 net.cpp:150] Setting up conv3
    I1230 20:51:47.332406 27818 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1230 20:51:47.332411 27818 net.cpp:165] Memory required for data: 70659840
    I1230 20:51:47.332417 27818 layer_factory.hpp:77] Creating layer pool3
    I1230 20:51:47.332424 27818 net.cpp:106] Creating Layer pool3
    I1230 20:51:47.332429 27818 net.cpp:454] pool3 <- conv3
    I1230 20:51:47.332445 27818 net.cpp:411] pool3 -> pool3
    I1230 20:51:47.332453 27818 net.cpp:150] Setting up pool3
    I1230 20:51:47.332459 27818 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 20:51:47.332464 27818 net.cpp:165] Memory required for data: 70782720
    I1230 20:51:47.332469 27818 layer_factory.hpp:77] Creating layer relu3
    I1230 20:51:47.332475 27818 net.cpp:106] Creating Layer relu3
    I1230 20:51:47.332480 27818 net.cpp:454] relu3 <- pool3
    I1230 20:51:47.332486 27818 net.cpp:397] relu3 -> pool3 (in-place)
    I1230 20:51:47.332492 27818 net.cpp:150] Setting up relu3
    I1230 20:51:47.332499 27818 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 20:51:47.332504 27818 net.cpp:165] Memory required for data: 70905600
    I1230 20:51:47.332509 27818 layer_factory.hpp:77] Creating layer ip1
    I1230 20:51:47.332540 27818 net.cpp:106] Creating Layer ip1
    I1230 20:51:47.332545 27818 net.cpp:454] ip1 <- pool3
    I1230 20:51:47.332552 27818 net.cpp:411] ip1 -> ip1
    I1230 20:51:47.333366 27818 net.cpp:150] Setting up ip1
    I1230 20:51:47.333390 27818 net.cpp:157] Top shape: 120 512 (61440)
    I1230 20:51:47.333396 27818 net.cpp:165] Memory required for data: 71151360
    I1230 20:51:47.333405 27818 layer_factory.hpp:77] Creating layer sig1
    I1230 20:51:47.333413 27818 net.cpp:106] Creating Layer sig1
    I1230 20:51:47.333418 27818 net.cpp:454] sig1 <- ip1
    I1230 20:51:47.333436 27818 net.cpp:397] sig1 -> ip1 (in-place)
    I1230 20:51:47.333442 27818 net.cpp:150] Setting up sig1
    I1230 20:51:47.333447 27818 net.cpp:157] Top shape: 120 512 (61440)
    I1230 20:51:47.333452 27818 net.cpp:165] Memory required for data: 71397120
    I1230 20:51:47.333457 27818 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1230 20:51:47.333472 27818 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1230 20:51:47.333477 27818 net.cpp:454] ip1_sig1_0_split <- ip1
    I1230 20:51:47.333482 27818 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1230 20:51:47.333490 27818 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1230 20:51:47.333498 27818 net.cpp:150] Setting up ip1_sig1_0_split
    I1230 20:51:47.333504 27818 net.cpp:157] Top shape: 120 512 (61440)
    I1230 20:51:47.333510 27818 net.cpp:157] Top shape: 120 512 (61440)
    I1230 20:51:47.333514 27818 net.cpp:165] Memory required for data: 71888640
    I1230 20:51:47.333519 27818 layer_factory.hpp:77] Creating layer ip_c
    I1230 20:51:47.333525 27818 net.cpp:106] Creating Layer ip_c
    I1230 20:51:47.333530 27818 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1230 20:51:47.333536 27818 net.cpp:411] ip_c -> ip_c
    I1230 20:51:47.333628 27818 net.cpp:150] Setting up ip_c
    I1230 20:51:47.333634 27818 net.cpp:157] Top shape: 120 20 (2400)
    I1230 20:51:47.333639 27818 net.cpp:165] Memory required for data: 71898240
    I1230 20:51:47.333645 27818 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1230 20:51:47.333652 27818 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1230 20:51:47.333657 27818 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1230 20:51:47.333662 27818 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1230 20:51:47.333669 27818 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1230 20:51:47.333678 27818 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1230 20:51:47.333683 27818 net.cpp:157] Top shape: 120 20 (2400)
    I1230 20:51:47.333688 27818 net.cpp:157] Top shape: 120 20 (2400)
    I1230 20:51:47.333693 27818 net.cpp:165] Memory required for data: 71917440
    I1230 20:51:47.333698 27818 layer_factory.hpp:77] Creating layer accuracy_c
    I1230 20:51:47.333706 27818 net.cpp:106] Creating Layer accuracy_c
    I1230 20:51:47.333711 27818 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1230 20:51:47.333717 27818 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1230 20:51:47.333724 27818 net.cpp:411] accuracy_c -> accuracy_c
    I1230 20:51:47.333730 27818 net.cpp:150] Setting up accuracy_c
    I1230 20:51:47.333736 27818 net.cpp:157] Top shape: (1)
    I1230 20:51:47.333740 27818 net.cpp:165] Memory required for data: 71917444
    I1230 20:51:47.333745 27818 layer_factory.hpp:77] Creating layer loss_c
    I1230 20:51:47.333762 27818 net.cpp:106] Creating Layer loss_c
    I1230 20:51:47.333767 27818 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1230 20:51:47.333773 27818 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1230 20:51:47.333780 27818 net.cpp:411] loss_c -> loss_c
    I1230 20:51:47.333791 27818 layer_factory.hpp:77] Creating layer loss_c
    I1230 20:51:47.333814 27818 net.cpp:150] Setting up loss_c
    I1230 20:51:47.333822 27818 net.cpp:157] Top shape: (1)
    I1230 20:51:47.333825 27818 net.cpp:160]     with loss weight 1
    I1230 20:51:47.333844 27818 net.cpp:165] Memory required for data: 71917448
    I1230 20:51:47.333849 27818 layer_factory.hpp:77] Creating layer ip_f
    I1230 20:51:47.333855 27818 net.cpp:106] Creating Layer ip_f
    I1230 20:51:47.333860 27818 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1230 20:51:47.333866 27818 net.cpp:411] ip_f -> ip_f
    I1230 20:51:47.334215 27818 net.cpp:150] Setting up ip_f
    I1230 20:51:47.334234 27818 net.cpp:157] Top shape: 120 100 (12000)
    I1230 20:51:47.334239 27818 net.cpp:165] Memory required for data: 71965448
    I1230 20:51:47.334246 27818 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1230 20:51:47.334254 27818 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1230 20:51:47.334269 27818 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1230 20:51:47.334275 27818 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1230 20:51:47.334281 27818 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1230 20:51:47.334288 27818 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1230 20:51:47.334295 27818 net.cpp:157] Top shape: 120 100 (12000)
    I1230 20:51:47.334300 27818 net.cpp:157] Top shape: 120 100 (12000)
    I1230 20:51:47.334305 27818 net.cpp:165] Memory required for data: 72061448
    I1230 20:51:47.334309 27818 layer_factory.hpp:77] Creating layer accuracy_f
    I1230 20:51:47.334321 27818 net.cpp:106] Creating Layer accuracy_f
    I1230 20:51:47.334326 27818 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1230 20:51:47.334332 27818 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1230 20:51:47.334338 27818 net.cpp:411] accuracy_f -> accuracy_f
    I1230 20:51:47.334345 27818 net.cpp:150] Setting up accuracy_f
    I1230 20:51:47.334350 27818 net.cpp:157] Top shape: (1)
    I1230 20:51:47.334355 27818 net.cpp:165] Memory required for data: 72061452
    I1230 20:51:47.334360 27818 layer_factory.hpp:77] Creating layer loss_f
    I1230 20:51:47.334365 27818 net.cpp:106] Creating Layer loss_f
    I1230 20:51:47.334370 27818 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1230 20:51:47.334377 27818 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1230 20:51:47.334381 27818 net.cpp:411] loss_f -> loss_f
    I1230 20:51:47.334388 27818 layer_factory.hpp:77] Creating layer loss_f
    I1230 20:51:47.334419 27818 net.cpp:150] Setting up loss_f
    I1230 20:51:47.334425 27818 net.cpp:157] Top shape: (1)
    I1230 20:51:47.334430 27818 net.cpp:160]     with loss weight 1
    I1230 20:51:47.334447 27818 net.cpp:165] Memory required for data: 72061456
    I1230 20:51:47.334452 27818 net.cpp:226] loss_f needs backward computation.
    I1230 20:51:47.334457 27818 net.cpp:228] accuracy_f does not need backward computation.
    I1230 20:51:47.334463 27818 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1230 20:51:47.334468 27818 net.cpp:226] ip_f needs backward computation.
    I1230 20:51:47.334473 27818 net.cpp:226] loss_c needs backward computation.
    I1230 20:51:47.334478 27818 net.cpp:228] accuracy_c does not need backward computation.
    I1230 20:51:47.334484 27818 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1230 20:51:47.334488 27818 net.cpp:226] ip_c needs backward computation.
    I1230 20:51:47.334493 27818 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1230 20:51:47.334498 27818 net.cpp:226] sig1 needs backward computation.
    I1230 20:51:47.334503 27818 net.cpp:226] ip1 needs backward computation.
    I1230 20:51:47.334508 27818 net.cpp:226] relu3 needs backward computation.
    I1230 20:51:47.334512 27818 net.cpp:226] pool3 needs backward computation.
    I1230 20:51:47.334517 27818 net.cpp:226] conv3 needs backward computation.
    I1230 20:51:47.334522 27818 net.cpp:226] relu2 needs backward computation.
    I1230 20:51:47.334527 27818 net.cpp:226] drop2 needs backward computation.
    I1230 20:51:47.334532 27818 net.cpp:226] pool2 needs backward computation.
    I1230 20:51:47.334537 27818 net.cpp:226] conv2 needs backward computation.
    I1230 20:51:47.334542 27818 net.cpp:226] relu1 needs backward computation.
    I1230 20:51:47.334545 27818 net.cpp:226] drop1 needs backward computation.
    I1230 20:51:47.334550 27818 net.cpp:226] pool1 needs backward computation.
    I1230 20:51:47.334555 27818 net.cpp:226] cccp2 needs backward computation.
    I1230 20:51:47.334560 27818 net.cpp:226] cccp1 needs backward computation.
    I1230 20:51:47.334565 27818 net.cpp:226] conv1 needs backward computation.
    I1230 20:51:47.334570 27818 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1230 20:51:47.334576 27818 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1230 20:51:47.334592 27818 net.cpp:228] data does not need backward computation.
    I1230 20:51:47.334597 27818 net.cpp:270] This network produces output accuracy_c
    I1230 20:51:47.334604 27818 net.cpp:270] This network produces output accuracy_f
    I1230 20:51:47.334609 27818 net.cpp:270] This network produces output loss_c
    I1230 20:51:47.334614 27818 net.cpp:270] This network produces output loss_f
    I1230 20:51:47.334641 27818 net.cpp:283] Network initialization done.
    I1230 20:51:47.335988 27818 caffe.cpp:240] Running for 83 iterations.
    I1230 20:51:47.634114 27818 caffe.cpp:264] Batch 0, accuracy_c = 0.55
    I1230 20:51:47.634163 27818 caffe.cpp:264] Batch 0, accuracy_f = 0.5
    I1230 20:51:47.634171 27818 caffe.cpp:264] Batch 0, loss_c = 1.35177
    I1230 20:51:47.634176 27818 caffe.cpp:264] Batch 0, loss_f = 2.14364
    I1230 20:51:47.921151 27818 caffe.cpp:264] Batch 1, accuracy_c = 0.591667
    I1230 20:51:47.921187 27818 caffe.cpp:264] Batch 1, accuracy_f = 0.425
    I1230 20:51:47.921195 27818 caffe.cpp:264] Batch 1, loss_c = 1.33962
    I1230 20:51:47.921201 27818 caffe.cpp:264] Batch 1, loss_f = 2.07294
    I1230 20:51:48.202461 27818 caffe.cpp:264] Batch 2, accuracy_c = 0.541667
    I1230 20:51:48.202505 27818 caffe.cpp:264] Batch 2, accuracy_f = 0.441667
    I1230 20:51:48.202512 27818 caffe.cpp:264] Batch 2, loss_c = 1.51729
    I1230 20:51:48.202518 27818 caffe.cpp:264] Batch 2, loss_f = 2.07398
    I1230 20:51:48.503707 27818 caffe.cpp:264] Batch 3, accuracy_c = 0.508333
    I1230 20:51:48.503751 27818 caffe.cpp:264] Batch 3, accuracy_f = 0.375
    I1230 20:51:48.503759 27818 caffe.cpp:264] Batch 3, loss_c = 1.48949
    I1230 20:51:48.503767 27818 caffe.cpp:264] Batch 3, loss_f = 2.35849
    I1230 20:51:48.799191 27818 caffe.cpp:264] Batch 4, accuracy_c = 0.558333
    I1230 20:51:48.799228 27818 caffe.cpp:264] Batch 4, accuracy_f = 0.441667
    I1230 20:51:48.799235 27818 caffe.cpp:264] Batch 4, loss_c = 1.39641
    I1230 20:51:48.799242 27818 caffe.cpp:264] Batch 4, loss_f = 2.02641
    I1230 20:51:49.092330 27818 caffe.cpp:264] Batch 5, accuracy_c = 0.558333
    I1230 20:51:49.092391 27818 caffe.cpp:264] Batch 5, accuracy_f = 0.375
    I1230 20:51:49.092401 27818 caffe.cpp:264] Batch 5, loss_c = 1.27608
    I1230 20:51:49.092409 27818 caffe.cpp:264] Batch 5, loss_f = 2.1975
    I1230 20:51:49.382412 27818 caffe.cpp:264] Batch 6, accuracy_c = 0.5
    I1230 20:51:49.382463 27818 caffe.cpp:264] Batch 6, accuracy_f = 0.433333
    I1230 20:51:49.382472 27818 caffe.cpp:264] Batch 6, loss_c = 1.72905
    I1230 20:51:49.382478 27818 caffe.cpp:264] Batch 6, loss_f = 2.42162
    I1230 20:51:49.702271 27818 caffe.cpp:264] Batch 7, accuracy_c = 0.533333
    I1230 20:51:49.702330 27818 caffe.cpp:264] Batch 7, accuracy_f = 0.383333
    I1230 20:51:49.702342 27818 caffe.cpp:264] Batch 7, loss_c = 1.43004
    I1230 20:51:49.702350 27818 caffe.cpp:264] Batch 7, loss_f = 2.32242
    I1230 20:51:50.038444 27818 caffe.cpp:264] Batch 8, accuracy_c = 0.575
    I1230 20:51:50.038537 27818 caffe.cpp:264] Batch 8, accuracy_f = 0.441667
    I1230 20:51:50.038553 27818 caffe.cpp:264] Batch 8, loss_c = 1.42234
    I1230 20:51:50.038563 27818 caffe.cpp:264] Batch 8, loss_f = 2.06416
    I1230 20:51:50.343201 27818 caffe.cpp:264] Batch 9, accuracy_c = 0.6
    I1230 20:51:50.343238 27818 caffe.cpp:264] Batch 9, accuracy_f = 0.458333
    I1230 20:51:50.343246 27818 caffe.cpp:264] Batch 9, loss_c = 1.51612
    I1230 20:51:50.343253 27818 caffe.cpp:264] Batch 9, loss_f = 2.28193
    I1230 20:51:50.653162 27818 caffe.cpp:264] Batch 10, accuracy_c = 0.558333
    I1230 20:51:50.653200 27818 caffe.cpp:264] Batch 10, accuracy_f = 0.375
    I1230 20:51:50.653210 27818 caffe.cpp:264] Batch 10, loss_c = 1.47147
    I1230 20:51:50.653218 27818 caffe.cpp:264] Batch 10, loss_f = 2.19219
    I1230 20:51:50.987845 27818 caffe.cpp:264] Batch 11, accuracy_c = 0.45
    I1230 20:51:50.987898 27818 caffe.cpp:264] Batch 11, accuracy_f = 0.316667
    I1230 20:51:50.987911 27818 caffe.cpp:264] Batch 11, loss_c = 1.6478
    I1230 20:51:50.987921 27818 caffe.cpp:264] Batch 11, loss_f = 2.56317
    I1230 20:51:51.304235 27818 caffe.cpp:264] Batch 12, accuracy_c = 0.625
    I1230 20:51:51.304282 27818 caffe.cpp:264] Batch 12, accuracy_f = 0.375
    I1230 20:51:51.304292 27818 caffe.cpp:264] Batch 12, loss_c = 1.24851
    I1230 20:51:51.304299 27818 caffe.cpp:264] Batch 12, loss_f = 2.16012
    I1230 20:51:51.616380 27818 caffe.cpp:264] Batch 13, accuracy_c = 0.508333
    I1230 20:51:51.616449 27818 caffe.cpp:264] Batch 13, accuracy_f = 0.358333
    I1230 20:51:51.616461 27818 caffe.cpp:264] Batch 13, loss_c = 1.5979
    I1230 20:51:51.616471 27818 caffe.cpp:264] Batch 13, loss_f = 2.28916
    I1230 20:51:51.946919 27818 caffe.cpp:264] Batch 14, accuracy_c = 0.516667
    I1230 20:51:51.946960 27818 caffe.cpp:264] Batch 14, accuracy_f = 0.35
    I1230 20:51:51.946970 27818 caffe.cpp:264] Batch 14, loss_c = 1.55324
    I1230 20:51:51.946977 27818 caffe.cpp:264] Batch 14, loss_f = 2.36315
    I1230 20:51:52.223165 27818 caffe.cpp:264] Batch 15, accuracy_c = 0.516667
    I1230 20:51:52.223209 27818 caffe.cpp:264] Batch 15, accuracy_f = 0.4
    I1230 20:51:52.223239 27818 caffe.cpp:264] Batch 15, loss_c = 1.36414
    I1230 20:51:52.223248 27818 caffe.cpp:264] Batch 15, loss_f = 2.11476
    I1230 20:51:52.497256 27818 caffe.cpp:264] Batch 16, accuracy_c = 0.5
    I1230 20:51:52.497315 27818 caffe.cpp:264] Batch 16, accuracy_f = 0.433333
    I1230 20:51:52.497325 27818 caffe.cpp:264] Batch 16, loss_c = 1.47081
    I1230 20:51:52.497334 27818 caffe.cpp:264] Batch 16, loss_f = 2.20141
    I1230 20:51:52.819108 27818 caffe.cpp:264] Batch 17, accuracy_c = 0.475
    I1230 20:51:52.819154 27818 caffe.cpp:264] Batch 17, accuracy_f = 0.391667
    I1230 20:51:52.819169 27818 caffe.cpp:264] Batch 17, loss_c = 1.52299
    I1230 20:51:52.819177 27818 caffe.cpp:264] Batch 17, loss_f = 2.33815
    I1230 20:51:53.115423 27818 caffe.cpp:264] Batch 18, accuracy_c = 0.575
    I1230 20:51:53.115483 27818 caffe.cpp:264] Batch 18, accuracy_f = 0.4
    I1230 20:51:53.115494 27818 caffe.cpp:264] Batch 18, loss_c = 1.27159
    I1230 20:51:53.115504 27818 caffe.cpp:264] Batch 18, loss_f = 2.13942
    I1230 20:51:53.407189 27818 caffe.cpp:264] Batch 19, accuracy_c = 0.55
    I1230 20:51:53.407232 27818 caffe.cpp:264] Batch 19, accuracy_f = 0.416667
    I1230 20:51:53.407240 27818 caffe.cpp:264] Batch 19, loss_c = 1.40948
    I1230 20:51:53.407248 27818 caffe.cpp:264] Batch 19, loss_f = 2.23321
    I1230 20:51:53.729380 27818 caffe.cpp:264] Batch 20, accuracy_c = 0.575
    I1230 20:51:53.729419 27818 caffe.cpp:264] Batch 20, accuracy_f = 0.416667
    I1230 20:51:53.729428 27818 caffe.cpp:264] Batch 20, loss_c = 1.42868
    I1230 20:51:53.729434 27818 caffe.cpp:264] Batch 20, loss_f = 2.21944
    I1230 20:51:54.016561 27818 caffe.cpp:264] Batch 21, accuracy_c = 0.541667
    I1230 20:51:54.016638 27818 caffe.cpp:264] Batch 21, accuracy_f = 0.358333
    I1230 20:51:54.016655 27818 caffe.cpp:264] Batch 21, loss_c = 1.55045
    I1230 20:51:54.016669 27818 caffe.cpp:264] Batch 21, loss_f = 2.32876
    I1230 20:51:54.297724 27818 caffe.cpp:264] Batch 22, accuracy_c = 0.558333
    I1230 20:51:54.297767 27818 caffe.cpp:264] Batch 22, accuracy_f = 0.491667
    I1230 20:51:54.297777 27818 caffe.cpp:264] Batch 22, loss_c = 1.39355
    I1230 20:51:54.297785 27818 caffe.cpp:264] Batch 22, loss_f = 1.96797
    I1230 20:51:54.592418 27818 caffe.cpp:264] Batch 23, accuracy_c = 0.55
    I1230 20:51:54.592478 27818 caffe.cpp:264] Batch 23, accuracy_f = 0.366667
    I1230 20:51:54.592489 27818 caffe.cpp:264] Batch 23, loss_c = 1.57801
    I1230 20:51:54.592497 27818 caffe.cpp:264] Batch 23, loss_f = 2.42852
    I1230 20:51:54.870738 27818 caffe.cpp:264] Batch 24, accuracy_c = 0.533333
    I1230 20:51:54.870791 27818 caffe.cpp:264] Batch 24, accuracy_f = 0.475
    I1230 20:51:54.870807 27818 caffe.cpp:264] Batch 24, loss_c = 1.4252
    I1230 20:51:54.870829 27818 caffe.cpp:264] Batch 24, loss_f = 2.12537
    I1230 20:51:55.138977 27818 caffe.cpp:264] Batch 25, accuracy_c = 0.65
    I1230 20:51:55.139020 27818 caffe.cpp:264] Batch 25, accuracy_f = 0.508333
    I1230 20:51:55.139027 27818 caffe.cpp:264] Batch 25, loss_c = 1.21504
    I1230 20:51:55.139034 27818 caffe.cpp:264] Batch 25, loss_f = 1.9041
    I1230 20:51:55.419037 27818 caffe.cpp:264] Batch 26, accuracy_c = 0.541667
    I1230 20:51:55.419088 27818 caffe.cpp:264] Batch 26, accuracy_f = 0.466667
    I1230 20:51:55.419096 27818 caffe.cpp:264] Batch 26, loss_c = 1.52711
    I1230 20:51:55.419102 27818 caffe.cpp:264] Batch 26, loss_f = 2.19451
    I1230 20:51:55.700847 27818 caffe.cpp:264] Batch 27, accuracy_c = 0.516667
    I1230 20:51:55.700891 27818 caffe.cpp:264] Batch 27, accuracy_f = 0.325
    I1230 20:51:55.700899 27818 caffe.cpp:264] Batch 27, loss_c = 1.6577
    I1230 20:51:55.700906 27818 caffe.cpp:264] Batch 27, loss_f = 2.5921
    I1230 20:51:55.983932 27818 caffe.cpp:264] Batch 28, accuracy_c = 0.625
    I1230 20:51:55.983988 27818 caffe.cpp:264] Batch 28, accuracy_f = 0.425
    I1230 20:51:55.983999 27818 caffe.cpp:264] Batch 28, loss_c = 1.43796
    I1230 20:51:55.984007 27818 caffe.cpp:264] Batch 28, loss_f = 2.25764
    I1230 20:51:56.276005 27818 caffe.cpp:264] Batch 29, accuracy_c = 0.516667
    I1230 20:51:56.276046 27818 caffe.cpp:264] Batch 29, accuracy_f = 0.4
    I1230 20:51:56.276053 27818 caffe.cpp:264] Batch 29, loss_c = 1.68441
    I1230 20:51:56.276072 27818 caffe.cpp:264] Batch 29, loss_f = 2.46314
    I1230 20:51:56.554996 27818 caffe.cpp:264] Batch 30, accuracy_c = 0.558333
    I1230 20:51:56.555029 27818 caffe.cpp:264] Batch 30, accuracy_f = 0.383333
    I1230 20:51:56.555035 27818 caffe.cpp:264] Batch 30, loss_c = 1.48341
    I1230 20:51:56.555042 27818 caffe.cpp:264] Batch 30, loss_f = 2.37941
    I1230 20:51:56.831357 27818 caffe.cpp:264] Batch 31, accuracy_c = 0.575
    I1230 20:51:56.831404 27818 caffe.cpp:264] Batch 31, accuracy_f = 0.316667
    I1230 20:51:56.831413 27818 caffe.cpp:264] Batch 31, loss_c = 1.41192
    I1230 20:51:56.831419 27818 caffe.cpp:264] Batch 31, loss_f = 2.2271
    I1230 20:51:57.116518 27818 caffe.cpp:264] Batch 32, accuracy_c = 0.466667
    I1230 20:51:57.116564 27818 caffe.cpp:264] Batch 32, accuracy_f = 0.433333
    I1230 20:51:57.116577 27818 caffe.cpp:264] Batch 32, loss_c = 1.72848
    I1230 20:51:57.116593 27818 caffe.cpp:264] Batch 32, loss_f = 2.35772
    I1230 20:51:57.445528 27818 caffe.cpp:264] Batch 33, accuracy_c = 0.466667
    I1230 20:51:57.445577 27818 caffe.cpp:264] Batch 33, accuracy_f = 0.375
    I1230 20:51:57.445590 27818 caffe.cpp:264] Batch 33, loss_c = 1.65193
    I1230 20:51:57.445600 27818 caffe.cpp:264] Batch 33, loss_f = 2.47395
    I1230 20:51:57.754428 27818 caffe.cpp:264] Batch 34, accuracy_c = 0.458333
    I1230 20:51:57.754478 27818 caffe.cpp:264] Batch 34, accuracy_f = 0.416667
    I1230 20:51:57.754505 27818 caffe.cpp:264] Batch 34, loss_c = 1.59148
    I1230 20:51:57.754515 27818 caffe.cpp:264] Batch 34, loss_f = 2.24714
    I1230 20:51:58.065567 27818 caffe.cpp:264] Batch 35, accuracy_c = 0.616667
    I1230 20:51:58.065613 27818 caffe.cpp:264] Batch 35, accuracy_f = 0.45
    I1230 20:51:58.065621 27818 caffe.cpp:264] Batch 35, loss_c = 1.34552
    I1230 20:51:58.065629 27818 caffe.cpp:264] Batch 35, loss_f = 2.07402
    I1230 20:51:58.345988 27818 caffe.cpp:264] Batch 36, accuracy_c = 0.575
    I1230 20:51:58.346029 27818 caffe.cpp:264] Batch 36, accuracy_f = 0.433333
    I1230 20:51:58.346037 27818 caffe.cpp:264] Batch 36, loss_c = 1.30405
    I1230 20:51:58.346045 27818 caffe.cpp:264] Batch 36, loss_f = 2.08939
    I1230 20:51:58.620493 27818 caffe.cpp:264] Batch 37, accuracy_c = 0.658333
    I1230 20:51:58.620539 27818 caffe.cpp:264] Batch 37, accuracy_f = 0.45
    I1230 20:51:58.620548 27818 caffe.cpp:264] Batch 37, loss_c = 1.24077
    I1230 20:51:58.620554 27818 caffe.cpp:264] Batch 37, loss_f = 1.97374
    I1230 20:51:58.903950 27818 caffe.cpp:264] Batch 38, accuracy_c = 0.583333
    I1230 20:51:58.903992 27818 caffe.cpp:264] Batch 38, accuracy_f = 0.475
    I1230 20:51:58.904000 27818 caffe.cpp:264] Batch 38, loss_c = 1.25474
    I1230 20:51:58.904006 27818 caffe.cpp:264] Batch 38, loss_f = 1.95928
    I1230 20:51:59.176537 27818 caffe.cpp:264] Batch 39, accuracy_c = 0.608333
    I1230 20:51:59.176585 27818 caffe.cpp:264] Batch 39, accuracy_f = 0.458333
    I1230 20:51:59.176597 27818 caffe.cpp:264] Batch 39, loss_c = 1.41309
    I1230 20:51:59.176606 27818 caffe.cpp:264] Batch 39, loss_f = 2.18064
    I1230 20:51:59.454358 27818 caffe.cpp:264] Batch 40, accuracy_c = 0.6
    I1230 20:51:59.454402 27818 caffe.cpp:264] Batch 40, accuracy_f = 0.433333
    I1230 20:51:59.454411 27818 caffe.cpp:264] Batch 40, loss_c = 1.36862
    I1230 20:51:59.454418 27818 caffe.cpp:264] Batch 40, loss_f = 2.19536
    I1230 20:51:59.729863 27818 caffe.cpp:264] Batch 41, accuracy_c = 0.591667
    I1230 20:51:59.729912 27818 caffe.cpp:264] Batch 41, accuracy_f = 0.425
    I1230 20:51:59.729920 27818 caffe.cpp:264] Batch 41, loss_c = 1.23418
    I1230 20:51:59.729928 27818 caffe.cpp:264] Batch 41, loss_f = 2.08628
    I1230 20:52:00.018235 27818 caffe.cpp:264] Batch 42, accuracy_c = 0.508333
    I1230 20:52:00.018286 27818 caffe.cpp:264] Batch 42, accuracy_f = 0.425
    I1230 20:52:00.018301 27818 caffe.cpp:264] Batch 42, loss_c = 1.47723
    I1230 20:52:00.018308 27818 caffe.cpp:264] Batch 42, loss_f = 2.0922
    I1230 20:52:00.303361 27818 caffe.cpp:264] Batch 43, accuracy_c = 0.566667
    I1230 20:52:00.303406 27818 caffe.cpp:264] Batch 43, accuracy_f = 0.416667
    I1230 20:52:00.303413 27818 caffe.cpp:264] Batch 43, loss_c = 1.3443
    I1230 20:52:00.303421 27818 caffe.cpp:264] Batch 43, loss_f = 2.1077
    I1230 20:52:00.582871 27818 caffe.cpp:264] Batch 44, accuracy_c = 0.541667
    I1230 20:52:00.582908 27818 caffe.cpp:264] Batch 44, accuracy_f = 0.458333
    I1230 20:52:00.582917 27818 caffe.cpp:264] Batch 44, loss_c = 1.47063
    I1230 20:52:00.582924 27818 caffe.cpp:264] Batch 44, loss_f = 2.17266
    I1230 20:52:00.876426 27818 caffe.cpp:264] Batch 45, accuracy_c = 0.616667
    I1230 20:52:00.876462 27818 caffe.cpp:264] Batch 45, accuracy_f = 0.516667
    I1230 20:52:00.876471 27818 caffe.cpp:264] Batch 45, loss_c = 1.35112
    I1230 20:52:00.876477 27818 caffe.cpp:264] Batch 45, loss_f = 1.98334
    I1230 20:52:01.148978 27818 caffe.cpp:264] Batch 46, accuracy_c = 0.55
    I1230 20:52:01.149024 27818 caffe.cpp:264] Batch 46, accuracy_f = 0.433333
    I1230 20:52:01.149034 27818 caffe.cpp:264] Batch 46, loss_c = 1.33742
    I1230 20:52:01.149040 27818 caffe.cpp:264] Batch 46, loss_f = 2.06631
    I1230 20:52:01.423593 27818 caffe.cpp:264] Batch 47, accuracy_c = 0.6
    I1230 20:52:01.423657 27818 caffe.cpp:264] Batch 47, accuracy_f = 0.416667
    I1230 20:52:01.423679 27818 caffe.cpp:264] Batch 47, loss_c = 1.26474
    I1230 20:52:01.423697 27818 caffe.cpp:264] Batch 47, loss_f = 2.10544
    I1230 20:52:01.707409 27818 caffe.cpp:264] Batch 48, accuracy_c = 0.5
    I1230 20:52:01.707447 27818 caffe.cpp:264] Batch 48, accuracy_f = 0.358333
    I1230 20:52:01.707454 27818 caffe.cpp:264] Batch 48, loss_c = 1.6389
    I1230 20:52:01.707460 27818 caffe.cpp:264] Batch 48, loss_f = 2.27175
    I1230 20:52:01.991158 27818 caffe.cpp:264] Batch 49, accuracy_c = 0.616667
    I1230 20:52:01.991204 27818 caffe.cpp:264] Batch 49, accuracy_f = 0.508333
    I1230 20:52:01.991212 27818 caffe.cpp:264] Batch 49, loss_c = 1.16806
    I1230 20:52:01.991219 27818 caffe.cpp:264] Batch 49, loss_f = 1.78866
    I1230 20:52:02.294574 27818 caffe.cpp:264] Batch 50, accuracy_c = 0.583333
    I1230 20:52:02.294627 27818 caffe.cpp:264] Batch 50, accuracy_f = 0.491667
    I1230 20:52:02.294638 27818 caffe.cpp:264] Batch 50, loss_c = 1.25036
    I1230 20:52:02.294647 27818 caffe.cpp:264] Batch 50, loss_f = 1.90412
    I1230 20:52:02.579349 27818 caffe.cpp:264] Batch 51, accuracy_c = 0.616667
    I1230 20:52:02.579391 27818 caffe.cpp:264] Batch 51, accuracy_f = 0.433333
    I1230 20:52:02.579399 27818 caffe.cpp:264] Batch 51, loss_c = 1.227
    I1230 20:52:02.579406 27818 caffe.cpp:264] Batch 51, loss_f = 2.10521
    I1230 20:52:02.872795 27818 caffe.cpp:264] Batch 52, accuracy_c = 0.575
    I1230 20:52:02.872831 27818 caffe.cpp:264] Batch 52, accuracy_f = 0.416667
    I1230 20:52:02.872839 27818 caffe.cpp:264] Batch 52, loss_c = 1.24546
    I1230 20:52:02.872846 27818 caffe.cpp:264] Batch 52, loss_f = 2.00973
    I1230 20:52:03.158696 27818 caffe.cpp:264] Batch 53, accuracy_c = 0.5
    I1230 20:52:03.158751 27818 caffe.cpp:264] Batch 53, accuracy_f = 0.433333
    I1230 20:52:03.158761 27818 caffe.cpp:264] Batch 53, loss_c = 1.53845
    I1230 20:52:03.158766 27818 caffe.cpp:264] Batch 53, loss_f = 2.15144
    I1230 20:52:03.446853 27818 caffe.cpp:264] Batch 54, accuracy_c = 0.583333
    I1230 20:52:03.446895 27818 caffe.cpp:264] Batch 54, accuracy_f = 0.45
    I1230 20:52:03.446903 27818 caffe.cpp:264] Batch 54, loss_c = 1.41592
    I1230 20:52:03.446909 27818 caffe.cpp:264] Batch 54, loss_f = 2.22875
    I1230 20:52:03.720019 27818 caffe.cpp:264] Batch 55, accuracy_c = 0.541667
    I1230 20:52:03.720077 27818 caffe.cpp:264] Batch 55, accuracy_f = 0.416667
    I1230 20:52:03.720087 27818 caffe.cpp:264] Batch 55, loss_c = 1.46741
    I1230 20:52:03.720094 27818 caffe.cpp:264] Batch 55, loss_f = 2.15518
    I1230 20:52:03.999639 27818 caffe.cpp:264] Batch 56, accuracy_c = 0.558333
    I1230 20:52:03.999691 27818 caffe.cpp:264] Batch 56, accuracy_f = 0.458333
    I1230 20:52:03.999703 27818 caffe.cpp:264] Batch 56, loss_c = 1.27675
    I1230 20:52:03.999711 27818 caffe.cpp:264] Batch 56, loss_f = 2.0617
    I1230 20:52:04.277698 27818 caffe.cpp:264] Batch 57, accuracy_c = 0.566667
    I1230 20:52:04.277741 27818 caffe.cpp:264] Batch 57, accuracy_f = 0.366667
    I1230 20:52:04.277750 27818 caffe.cpp:264] Batch 57, loss_c = 1.45045
    I1230 20:52:04.277758 27818 caffe.cpp:264] Batch 57, loss_f = 2.24805
    I1230 20:52:04.565021 27818 caffe.cpp:264] Batch 58, accuracy_c = 0.516667
    I1230 20:52:04.565078 27818 caffe.cpp:264] Batch 58, accuracy_f = 0.375
    I1230 20:52:04.565086 27818 caffe.cpp:264] Batch 58, loss_c = 1.55607
    I1230 20:52:04.565093 27818 caffe.cpp:264] Batch 58, loss_f = 2.39568
    I1230 20:52:04.858546 27818 caffe.cpp:264] Batch 59, accuracy_c = 0.575
    I1230 20:52:04.858583 27818 caffe.cpp:264] Batch 59, accuracy_f = 0.458333
    I1230 20:52:04.858592 27818 caffe.cpp:264] Batch 59, loss_c = 1.51123
    I1230 20:52:04.858599 27818 caffe.cpp:264] Batch 59, loss_f = 2.26248
    I1230 20:52:05.143522 27818 caffe.cpp:264] Batch 60, accuracy_c = 0.558333
    I1230 20:52:05.143560 27818 caffe.cpp:264] Batch 60, accuracy_f = 0.416667
    I1230 20:52:05.143568 27818 caffe.cpp:264] Batch 60, loss_c = 1.36479
    I1230 20:52:05.143575 27818 caffe.cpp:264] Batch 60, loss_f = 2.2494
    I1230 20:52:05.434281 27818 caffe.cpp:264] Batch 61, accuracy_c = 0.458333
    I1230 20:52:05.434319 27818 caffe.cpp:264] Batch 61, accuracy_f = 0.35
    I1230 20:52:05.434327 27818 caffe.cpp:264] Batch 61, loss_c = 1.66668
    I1230 20:52:05.434334 27818 caffe.cpp:264] Batch 61, loss_f = 2.36949
    I1230 20:52:05.718719 27818 caffe.cpp:264] Batch 62, accuracy_c = 0.633333
    I1230 20:52:05.718771 27818 caffe.cpp:264] Batch 62, accuracy_f = 0.425
    I1230 20:52:05.718781 27818 caffe.cpp:264] Batch 62, loss_c = 1.27415
    I1230 20:52:05.718789 27818 caffe.cpp:264] Batch 62, loss_f = 2.1578
    I1230 20:52:06.004834 27818 caffe.cpp:264] Batch 63, accuracy_c = 0.558333
    I1230 20:52:06.004879 27818 caffe.cpp:264] Batch 63, accuracy_f = 0.458333
    I1230 20:52:06.004889 27818 caffe.cpp:264] Batch 63, loss_c = 1.36506
    I1230 20:52:06.004897 27818 caffe.cpp:264] Batch 63, loss_f = 2.15929
    I1230 20:52:06.296030 27818 caffe.cpp:264] Batch 64, accuracy_c = 0.641667
    I1230 20:52:06.296092 27818 caffe.cpp:264] Batch 64, accuracy_f = 0.525
    I1230 20:52:06.296114 27818 caffe.cpp:264] Batch 64, loss_c = 1.13444
    I1230 20:52:06.296123 27818 caffe.cpp:264] Batch 64, loss_f = 1.84139
    I1230 20:52:06.677666 27818 caffe.cpp:264] Batch 65, accuracy_c = 0.466667
    I1230 20:52:06.677708 27818 caffe.cpp:264] Batch 65, accuracy_f = 0.375
    I1230 20:52:06.677716 27818 caffe.cpp:264] Batch 65, loss_c = 1.69584
    I1230 20:52:06.677721 27818 caffe.cpp:264] Batch 65, loss_f = 2.36638
    I1230 20:52:06.972074 27818 caffe.cpp:264] Batch 66, accuracy_c = 0.516667
    I1230 20:52:06.972112 27818 caffe.cpp:264] Batch 66, accuracy_f = 0.425
    I1230 20:52:06.972120 27818 caffe.cpp:264] Batch 66, loss_c = 1.48074
    I1230 20:52:06.972126 27818 caffe.cpp:264] Batch 66, loss_f = 2.17279
    I1230 20:52:07.259160 27818 caffe.cpp:264] Batch 67, accuracy_c = 0.525
    I1230 20:52:07.259207 27818 caffe.cpp:264] Batch 67, accuracy_f = 0.416667
    I1230 20:52:07.259217 27818 caffe.cpp:264] Batch 67, loss_c = 1.45827
    I1230 20:52:07.259224 27818 caffe.cpp:264] Batch 67, loss_f = 2.09786
    I1230 20:52:07.539926 27818 caffe.cpp:264] Batch 68, accuracy_c = 0.591667
    I1230 20:52:07.539959 27818 caffe.cpp:264] Batch 68, accuracy_f = 0.458333
    I1230 20:52:07.539968 27818 caffe.cpp:264] Batch 68, loss_c = 1.36812
    I1230 20:52:07.539974 27818 caffe.cpp:264] Batch 68, loss_f = 2.18938
    I1230 20:52:07.826297 27818 caffe.cpp:264] Batch 69, accuracy_c = 0.575
    I1230 20:52:07.826331 27818 caffe.cpp:264] Batch 69, accuracy_f = 0.441667
    I1230 20:52:07.826339 27818 caffe.cpp:264] Batch 69, loss_c = 1.32241
    I1230 20:52:07.826344 27818 caffe.cpp:264] Batch 69, loss_f = 2.08982
    I1230 20:52:08.124995 27818 caffe.cpp:264] Batch 70, accuracy_c = 0.558333
    I1230 20:52:08.125036 27818 caffe.cpp:264] Batch 70, accuracy_f = 0.391667
    I1230 20:52:08.125046 27818 caffe.cpp:264] Batch 70, loss_c = 1.47831
    I1230 20:52:08.125052 27818 caffe.cpp:264] Batch 70, loss_f = 2.19515
    I1230 20:52:08.423852 27818 caffe.cpp:264] Batch 71, accuracy_c = 0.508333
    I1230 20:52:08.423903 27818 caffe.cpp:264] Batch 71, accuracy_f = 0.375
    I1230 20:52:08.423924 27818 caffe.cpp:264] Batch 71, loss_c = 1.58004
    I1230 20:52:08.423936 27818 caffe.cpp:264] Batch 71, loss_f = 2.3199
    I1230 20:52:08.763730 27818 caffe.cpp:264] Batch 72, accuracy_c = 0.441667
    I1230 20:52:08.763772 27818 caffe.cpp:264] Batch 72, accuracy_f = 0.325
    I1230 20:52:08.763808 27818 caffe.cpp:264] Batch 72, loss_c = 1.70517
    I1230 20:52:08.763818 27818 caffe.cpp:264] Batch 72, loss_f = 2.55888
    I1230 20:52:09.040221 27818 caffe.cpp:264] Batch 73, accuracy_c = 0.466667
    I1230 20:52:09.040269 27818 caffe.cpp:264] Batch 73, accuracy_f = 0.283333
    I1230 20:52:09.040290 27818 caffe.cpp:264] Batch 73, loss_c = 1.80665
    I1230 20:52:09.040299 27818 caffe.cpp:264] Batch 73, loss_f = 2.7043
    I1230 20:52:09.323402 27818 caffe.cpp:264] Batch 74, accuracy_c = 0.516667
    I1230 20:52:09.323451 27818 caffe.cpp:264] Batch 74, accuracy_f = 0.4
    I1230 20:52:09.323459 27818 caffe.cpp:264] Batch 74, loss_c = 1.40707
    I1230 20:52:09.323467 27818 caffe.cpp:264] Batch 74, loss_f = 2.22145
    I1230 20:52:09.599794 27818 caffe.cpp:264] Batch 75, accuracy_c = 0.458333
    I1230 20:52:09.599843 27818 caffe.cpp:264] Batch 75, accuracy_f = 0.316667
    I1230 20:52:09.599853 27818 caffe.cpp:264] Batch 75, loss_c = 1.64695
    I1230 20:52:09.599859 27818 caffe.cpp:264] Batch 75, loss_f = 2.44603
    I1230 20:52:09.893314 27818 caffe.cpp:264] Batch 76, accuracy_c = 0.591667
    I1230 20:52:09.893359 27818 caffe.cpp:264] Batch 76, accuracy_f = 0.416667
    I1230 20:52:09.893367 27818 caffe.cpp:264] Batch 76, loss_c = 1.42194
    I1230 20:52:09.893371 27818 caffe.cpp:264] Batch 76, loss_f = 2.23253
    I1230 20:52:10.163638 27818 caffe.cpp:264] Batch 77, accuracy_c = 0.533333
    I1230 20:52:10.163689 27818 caffe.cpp:264] Batch 77, accuracy_f = 0.416667
    I1230 20:52:10.163697 27818 caffe.cpp:264] Batch 77, loss_c = 1.38299
    I1230 20:52:10.163703 27818 caffe.cpp:264] Batch 77, loss_f = 2.16709
    I1230 20:52:10.439311 27818 caffe.cpp:264] Batch 78, accuracy_c = 0.591667
    I1230 20:52:10.439354 27818 caffe.cpp:264] Batch 78, accuracy_f = 0.441667
    I1230 20:52:10.439362 27818 caffe.cpp:264] Batch 78, loss_c = 1.31485
    I1230 20:52:10.439366 27818 caffe.cpp:264] Batch 78, loss_f = 2.07289
    I1230 20:52:10.710935 27818 caffe.cpp:264] Batch 79, accuracy_c = 0.533333
    I1230 20:52:10.710978 27818 caffe.cpp:264] Batch 79, accuracy_f = 0.458333
    I1230 20:52:10.710985 27818 caffe.cpp:264] Batch 79, loss_c = 1.35614
    I1230 20:52:10.710990 27818 caffe.cpp:264] Batch 79, loss_f = 2.00129
    I1230 20:52:10.976357 27818 caffe.cpp:264] Batch 80, accuracy_c = 0.525
    I1230 20:52:10.976400 27818 caffe.cpp:264] Batch 80, accuracy_f = 0.433333
    I1230 20:52:10.976408 27818 caffe.cpp:264] Batch 80, loss_c = 1.37391
    I1230 20:52:10.976413 27818 caffe.cpp:264] Batch 80, loss_f = 2.131
    I1230 20:52:11.239764 27818 caffe.cpp:264] Batch 81, accuracy_c = 0.583333
    I1230 20:52:11.239805 27818 caffe.cpp:264] Batch 81, accuracy_f = 0.433333
    I1230 20:52:11.239811 27818 caffe.cpp:264] Batch 81, loss_c = 1.33322
    I1230 20:52:11.239817 27818 caffe.cpp:264] Batch 81, loss_f = 2.25288
    I1230 20:52:11.544421 27818 caffe.cpp:264] Batch 82, accuracy_c = 0.508333
    I1230 20:52:11.544461 27818 caffe.cpp:264] Batch 82, accuracy_f = 0.45
    I1230 20:52:11.544469 27818 caffe.cpp:264] Batch 82, loss_c = 1.42291
    I1230 20:52:11.544476 27818 caffe.cpp:264] Batch 82, loss_f = 2.11772
    I1230 20:52:11.544481 27818 caffe.cpp:269] Loss: 3.6363
    I1230 20:52:11.544492 27818 caffe.cpp:281] accuracy_c = 0.549599
    I1230 20:52:11.544498 27818 caffe.cpp:281] accuracy_f = 0.417369
    I1230 20:52:11.544518 27818 caffe.cpp:281] loss_c = 1.43741 (* 1 = 1.43741 loss)
    I1230 20:52:11.544526 27818 caffe.cpp:281] loss_f = 2.1989 (* 1 = 2.1989 loss)
    CPU times: user 144 ms, sys: 24 ms, total: 168 ms
    Wall time: 24.9 s


## The model achieved near 55% accuracy on the 20 coarse labels and 41% accuracy on fine labels.
This means that upon showing the neural network a picture it had never seen, it will correctly classify it in one of the 20 coarse categories 55% of the time or it will classify it correctly in the fine categories 41% of the time right, and ignoring the coarse label. This is amazing, but the neural network for sure could be fine tuned with better solver parameters.

At least, this neural network would be good enough to be listed here: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#494c5356524332303132207461736b2031

Let's convert the notebook to github markdown:


```python
!jupyter nbconvert --to markdown custom-cifar-100.ipynb
!mv custom-cifar-100.md README.md
```

    [NbConvertApp] Converting notebook custom-cifar-10.ipynb to markdown
    [NbConvertApp] Writing 404667 bytes to custom-cifar-10.md
