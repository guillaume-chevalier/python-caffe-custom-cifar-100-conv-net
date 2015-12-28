
# Custom cifar-10 conv net with Caffe in Python (Pycaffe)

Here, I train a custom convnet on the cifar-10 dataset. I did not try to implement any specific published architecture, but to design a new one for learning purposes. It is inspired from the official caffe python ".ipynb" examples available at: https://github.com/BVLC/caffe/tree/master/examples, but not the cifar-10 example itself that was in C++.

Please refer to https://www.cs.toronto.edu/~kriz/cifar.html for more information on the nature of the task and of the dataset on which the convolutional neural network is trained on.

## Dynamically download and convert the cifar-10 dataset to Caffe's HDF5 format using code of another git repo of mine.
More info on the dataset can be found at http://www.cs.toronto.edu/~kriz/cifar.html.


```python
%%time

!rm download-and-convert-cifar-10.py
print("Getting the download script...")
!wget https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py
print("Downloaded script. Will execute to download and convert the cifar-10 dataset:")
!python download-and-convert-cifar-10.py
```

    Getting the download script...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-26 18:49:18--  https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-10.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 199.27.76.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|199.27.76.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3336 (3.3K) [text/plain]
    Saving to: ‘download-and-convert-cifar-10.py’

    100%[======================================>] 3,336       --.-K/s   in 0s      

    2015-12-26 18:49:19 (1.05 GB/s) - ‘download-and-convert-cifar-10.py’ saved [3336/3336]

    Downloaded script. Will execute to download and convert the cifar-10 dataset:

    Downloading...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-26 18:49:19--  http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
    Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 170498071 (163M) [application/x-gzip]
    Saving to: ‘cifar-10-python.tar.gz’

    100%[======================================>] 170,498,071 1.23MB/s   in 2m 15s

    2015-12-26 18:51:34 (1.20 MB/s) - ‘cifar-10-python.tar.gz’ saved [170498071/170498071]

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

    CPU times: user 844 ms, sys: 88 ms, total: 932 ms
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
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)

    n.conv1 = L.Convolution(n.data, kernel_size=4, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, in_place=True)
    n.relu1 = L.ReLU(n.drop1, in_place=True)

    n.conv2 = L.Convolution(n.relu1, kernel_size=4, num_output=42, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    n.drop2 = L.Dropout(n.pool2, in_place=True)
    n.relu2 = L.ReLU(n.drop2, in_place=True)

    n.conv3 = L.Convolution(n.relu2, kernel_size=2, num_output=64, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu3 = L.ReLU(n.pool3, in_place=True)

    n.ip1 = L.InnerProduct(n.relu3, num_output=512, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.sig1, num_output=10, weight_filler=dict(type='xavier'))

    n.accuracy = L.Accuracy(n.ip2, n.label)
    n.loss = L.SoftmaxWithLoss(n.ip2, n.label)
    return n.to_proto()

with open('cnn_train.prototxt', 'w') as f:
    f.write(str(cnn('cifar_10_caffe_hdf5/train.txt', 100)))

with open('cnn_test.prototxt', 'w') as f:
    f.write(str(cnn('cifar_10_caffe_hdf5/test.txt', 120)))
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
     ('label', (100,)),
     ('label_data_1_split_0', (100,)),
     ('label_data_1_split_1', (100,)),
     ('conv1', (100, 32, 29, 29)),
     ('pool1', (100, 32, 14, 14)),
     ('conv2', (100, 42, 11, 11)),
     ('pool2', (100, 42, 5, 5)),
     ('conv3', (100, 64, 4, 4)),
     ('pool3', (100, 64, 2, 2)),
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

    [('conv1', (32, 3, 4, 4)),
     ('conv2', (42, 32, 4, 4)),
     ('conv3', (64, 42, 2, 2)),
     ('ip1', (512, 256)),
     ('ip2', (10, 512))]



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

    snapshot: 25000
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
    I1227 18:26:28.268373  5629 caffe.cpp:184] Using GPUs 0
    I1227 18:26:28.474257  5629 solver.cpp:48] Initializing solver from parameters:
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
    snapshot: 25000
    snapshot_prefix: "cnn_snapshot"
    solver_mode: GPU
    device_id: 0
    rms_decay: 0.98
    type: "RMSProp"
    I1227 18:26:28.474506  5629 solver.cpp:81] Creating training net from train_net file: cnn_train.prototxt
    I1227 18:26:28.474905  5629 net.cpp:49] Initializing net from parameters:
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
        num_output: 32
        kernel_size: 4
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
        pool: AVE
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
        pool: MAX
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
    I1227 18:26:28.475478  5629 layer_factory.hpp:77] Creating layer data
    I1227 18:26:28.475499  5629 net.cpp:106] Creating Layer data
    I1227 18:26:28.475508  5629 net.cpp:411] data -> data
    I1227 18:26:28.475529  5629 net.cpp:411] data -> label
    I1227 18:26:28.475553  5629 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/train.txt
    I1227 18:26:28.475584  5629 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1227 18:26:28.476747  5629 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1227 18:26:30.344238  5629 net.cpp:150] Setting up data
    I1227 18:26:30.344286  5629 net.cpp:157] Top shape: 100 3 32 32 (307200)
    I1227 18:26:30.344296  5629 net.cpp:157] Top shape: 100 (100)
    I1227 18:26:30.344302  5629 net.cpp:165] Memory required for data: 1229200
    I1227 18:26:30.344313  5629 layer_factory.hpp:77] Creating layer label_data_1_split
    I1227 18:26:30.344334  5629 net.cpp:106] Creating Layer label_data_1_split
    I1227 18:26:30.344342  5629 net.cpp:454] label_data_1_split <- label
    I1227 18:26:30.344355  5629 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1227 18:26:30.344368  5629 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1227 18:26:30.344408  5629 net.cpp:150] Setting up label_data_1_split
    I1227 18:26:30.344416  5629 net.cpp:157] Top shape: 100 (100)
    I1227 18:26:30.344424  5629 net.cpp:157] Top shape: 100 (100)
    I1227 18:26:30.344463  5629 net.cpp:165] Memory required for data: 1230000
    I1227 18:26:30.344471  5629 layer_factory.hpp:77] Creating layer conv1
    I1227 18:26:30.344485  5629 net.cpp:106] Creating Layer conv1
    I1227 18:26:30.344492  5629 net.cpp:454] conv1 <- data
    I1227 18:26:30.344501  5629 net.cpp:411] conv1 -> conv1
    I1227 18:26:30.345854  5629 net.cpp:150] Setting up conv1
    I1227 18:26:30.345881  5629 net.cpp:157] Top shape: 100 32 29 29 (2691200)
    I1227 18:26:30.345888  5629 net.cpp:165] Memory required for data: 11994800
    I1227 18:26:30.345902  5629 layer_factory.hpp:77] Creating layer pool1
    I1227 18:26:30.345913  5629 net.cpp:106] Creating Layer pool1
    I1227 18:26:30.345919  5629 net.cpp:454] pool1 <- conv1
    I1227 18:26:30.345927  5629 net.cpp:411] pool1 -> pool1
    I1227 18:26:30.345973  5629 net.cpp:150] Setting up pool1
    I1227 18:26:30.345983  5629 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1227 18:26:30.345988  5629 net.cpp:165] Memory required for data: 14503600
    I1227 18:26:30.345993  5629 layer_factory.hpp:77] Creating layer drop1
    I1227 18:26:30.346004  5629 net.cpp:106] Creating Layer drop1
    I1227 18:26:30.346010  5629 net.cpp:454] drop1 <- pool1
    I1227 18:26:30.346016  5629 net.cpp:397] drop1 -> pool1 (in-place)
    I1227 18:26:30.346040  5629 net.cpp:150] Setting up drop1
    I1227 18:26:30.346048  5629 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1227 18:26:30.346053  5629 net.cpp:165] Memory required for data: 17012400
    I1227 18:26:30.346060  5629 layer_factory.hpp:77] Creating layer relu1
    I1227 18:26:30.346066  5629 net.cpp:106] Creating Layer relu1
    I1227 18:26:30.346071  5629 net.cpp:454] relu1 <- pool1
    I1227 18:26:30.346078  5629 net.cpp:397] relu1 -> pool1 (in-place)
    I1227 18:26:30.346086  5629 net.cpp:150] Setting up relu1
    I1227 18:26:30.346091  5629 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1227 18:26:30.346096  5629 net.cpp:165] Memory required for data: 19521200
    I1227 18:26:30.346101  5629 layer_factory.hpp:77] Creating layer conv2
    I1227 18:26:30.346110  5629 net.cpp:106] Creating Layer conv2
    I1227 18:26:30.346115  5629 net.cpp:454] conv2 <- pool1
    I1227 18:26:30.346122  5629 net.cpp:411] conv2 -> conv2
    I1227 18:26:30.346422  5629 net.cpp:150] Setting up conv2
    I1227 18:26:30.346443  5629 net.cpp:157] Top shape: 100 42 11 11 (508200)
    I1227 18:26:30.346449  5629 net.cpp:165] Memory required for data: 21554000
    I1227 18:26:30.346458  5629 layer_factory.hpp:77] Creating layer pool2
    I1227 18:26:30.346467  5629 net.cpp:106] Creating Layer pool2
    I1227 18:26:30.346472  5629 net.cpp:454] pool2 <- conv2
    I1227 18:26:30.346479  5629 net.cpp:411] pool2 -> pool2
    I1227 18:26:30.346508  5629 net.cpp:150] Setting up pool2
    I1227 18:26:30.346514  5629 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1227 18:26:30.346518  5629 net.cpp:165] Memory required for data: 21974000
    I1227 18:26:30.346524  5629 layer_factory.hpp:77] Creating layer drop2
    I1227 18:26:30.346531  5629 net.cpp:106] Creating Layer drop2
    I1227 18:26:30.346535  5629 net.cpp:454] drop2 <- pool2
    I1227 18:26:30.346541  5629 net.cpp:397] drop2 -> pool2 (in-place)
    I1227 18:26:30.346557  5629 net.cpp:150] Setting up drop2
    I1227 18:26:30.346565  5629 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1227 18:26:30.346570  5629 net.cpp:165] Memory required for data: 22394000
    I1227 18:26:30.346575  5629 layer_factory.hpp:77] Creating layer relu2
    I1227 18:26:30.346580  5629 net.cpp:106] Creating Layer relu2
    I1227 18:26:30.346585  5629 net.cpp:454] relu2 <- pool2
    I1227 18:26:30.346590  5629 net.cpp:397] relu2 -> pool2 (in-place)
    I1227 18:26:30.346597  5629 net.cpp:150] Setting up relu2
    I1227 18:26:30.346602  5629 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1227 18:26:30.346607  5629 net.cpp:165] Memory required for data: 22814000
    I1227 18:26:30.346612  5629 layer_factory.hpp:77] Creating layer conv3
    I1227 18:26:30.346619  5629 net.cpp:106] Creating Layer conv3
    I1227 18:26:30.346634  5629 net.cpp:454] conv3 <- pool2
    I1227 18:26:30.346642  5629 net.cpp:411] conv3 -> conv3
    I1227 18:26:30.347266  5629 net.cpp:150] Setting up conv3
    I1227 18:26:30.347290  5629 net.cpp:157] Top shape: 100 64 4 4 (102400)
    I1227 18:26:30.347311  5629 net.cpp:165] Memory required for data: 23223600
    I1227 18:26:30.347322  5629 layer_factory.hpp:77] Creating layer pool3
    I1227 18:26:30.347340  5629 net.cpp:106] Creating Layer pool3
    I1227 18:26:30.347345  5629 net.cpp:454] pool3 <- conv3
    I1227 18:26:30.347352  5629 net.cpp:411] pool3 -> pool3
    I1227 18:26:30.347380  5629 net.cpp:150] Setting up pool3
    I1227 18:26:30.347388  5629 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1227 18:26:30.347393  5629 net.cpp:165] Memory required for data: 23326000
    I1227 18:26:30.347398  5629 layer_factory.hpp:77] Creating layer relu3
    I1227 18:26:30.347404  5629 net.cpp:106] Creating Layer relu3
    I1227 18:26:30.347409  5629 net.cpp:454] relu3 <- pool3
    I1227 18:26:30.347414  5629 net.cpp:397] relu3 -> pool3 (in-place)
    I1227 18:26:30.347421  5629 net.cpp:150] Setting up relu3
    I1227 18:26:30.347427  5629 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1227 18:26:30.347431  5629 net.cpp:165] Memory required for data: 23428400
    I1227 18:26:30.347436  5629 layer_factory.hpp:77] Creating layer ip1
    I1227 18:26:30.347448  5629 net.cpp:106] Creating Layer ip1
    I1227 18:26:30.347453  5629 net.cpp:454] ip1 <- pool3
    I1227 18:26:30.347460  5629 net.cpp:411] ip1 -> ip1
    I1227 18:26:30.348781  5629 net.cpp:150] Setting up ip1
    I1227 18:26:30.348805  5629 net.cpp:157] Top shape: 100 512 (51200)
    I1227 18:26:30.348811  5629 net.cpp:165] Memory required for data: 23633200
    I1227 18:26:30.348820  5629 layer_factory.hpp:77] Creating layer sig1
    I1227 18:26:30.348829  5629 net.cpp:106] Creating Layer sig1
    I1227 18:26:30.348834  5629 net.cpp:454] sig1 <- ip1
    I1227 18:26:30.348850  5629 net.cpp:397] sig1 -> ip1 (in-place)
    I1227 18:26:30.348857  5629 net.cpp:150] Setting up sig1
    I1227 18:26:30.348863  5629 net.cpp:157] Top shape: 100 512 (51200)
    I1227 18:26:30.348868  5629 net.cpp:165] Memory required for data: 23838000
    I1227 18:26:30.348873  5629 layer_factory.hpp:77] Creating layer ip2
    I1227 18:26:30.348881  5629 net.cpp:106] Creating Layer ip2
    I1227 18:26:30.348886  5629 net.cpp:454] ip2 <- ip1
    I1227 18:26:30.348891  5629 net.cpp:411] ip2 -> ip2
    I1227 18:26:30.349004  5629 net.cpp:150] Setting up ip2
    I1227 18:26:30.349012  5629 net.cpp:157] Top shape: 100 10 (1000)
    I1227 18:26:30.349017  5629 net.cpp:165] Memory required for data: 23842000
    I1227 18:26:30.349037  5629 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1227 18:26:30.349045  5629 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1227 18:26:30.349050  5629 net.cpp:454] ip2_ip2_0_split <- ip2
    I1227 18:26:30.349056  5629 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1227 18:26:30.349063  5629 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1227 18:26:30.349089  5629 net.cpp:150] Setting up ip2_ip2_0_split
    I1227 18:26:30.349097  5629 net.cpp:157] Top shape: 100 10 (1000)
    I1227 18:26:30.349102  5629 net.cpp:157] Top shape: 100 10 (1000)
    I1227 18:26:30.349107  5629 net.cpp:165] Memory required for data: 23850000
    I1227 18:26:30.349112  5629 layer_factory.hpp:77] Creating layer accuracy
    I1227 18:26:30.349119  5629 net.cpp:106] Creating Layer accuracy
    I1227 18:26:30.349124  5629 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1227 18:26:30.349130  5629 net.cpp:454] accuracy <- label_data_1_split_0
    I1227 18:26:30.349136  5629 net.cpp:411] accuracy -> accuracy
    I1227 18:26:30.349145  5629 net.cpp:150] Setting up accuracy
    I1227 18:26:30.349151  5629 net.cpp:157] Top shape: (1)
    I1227 18:26:30.349156  5629 net.cpp:165] Memory required for data: 23850004
    I1227 18:26:30.349161  5629 layer_factory.hpp:77] Creating layer loss
    I1227 18:26:30.349182  5629 net.cpp:106] Creating Layer loss
    I1227 18:26:30.349187  5629 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1227 18:26:30.349194  5629 net.cpp:454] loss <- label_data_1_split_1
    I1227 18:26:30.349200  5629 net.cpp:411] loss -> loss
    I1227 18:26:30.349220  5629 layer_factory.hpp:77] Creating layer loss
    I1227 18:26:30.349284  5629 net.cpp:150] Setting up loss
    I1227 18:26:30.349292  5629 net.cpp:157] Top shape: (1)
    I1227 18:26:30.349298  5629 net.cpp:160]     with loss weight 1
    I1227 18:26:30.349325  5629 net.cpp:165] Memory required for data: 23850008
    I1227 18:26:30.349331  5629 net.cpp:226] loss needs backward computation.
    I1227 18:26:30.349336  5629 net.cpp:228] accuracy does not need backward computation.
    I1227 18:26:30.349342  5629 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1227 18:26:30.349347  5629 net.cpp:226] ip2 needs backward computation.
    I1227 18:26:30.349362  5629 net.cpp:226] sig1 needs backward computation.
    I1227 18:26:30.349367  5629 net.cpp:226] ip1 needs backward computation.
    I1227 18:26:30.349373  5629 net.cpp:226] relu3 needs backward computation.
    I1227 18:26:30.349378  5629 net.cpp:226] pool3 needs backward computation.
    I1227 18:26:30.349385  5629 net.cpp:226] conv3 needs backward computation.
    I1227 18:26:30.349390  5629 net.cpp:226] relu2 needs backward computation.
    I1227 18:26:30.349405  5629 net.cpp:226] drop2 needs backward computation.
    I1227 18:26:30.349409  5629 net.cpp:226] pool2 needs backward computation.
    I1227 18:26:30.349414  5629 net.cpp:226] conv2 needs backward computation.
    I1227 18:26:30.349419  5629 net.cpp:226] relu1 needs backward computation.
    I1227 18:26:30.349424  5629 net.cpp:226] drop1 needs backward computation.
    I1227 18:26:30.349428  5629 net.cpp:226] pool1 needs backward computation.
    I1227 18:26:30.349433  5629 net.cpp:226] conv1 needs backward computation.
    I1227 18:26:30.349438  5629 net.cpp:228] label_data_1_split does not need backward computation.
    I1227 18:26:30.349444  5629 net.cpp:228] data does not need backward computation.
    I1227 18:26:30.349448  5629 net.cpp:270] This network produces output accuracy
    I1227 18:26:30.349454  5629 net.cpp:270] This network produces output loss
    I1227 18:26:30.349467  5629 net.cpp:283] Network initialization done.
    I1227 18:26:30.349707  5629 solver.cpp:181] Creating test net (#0) specified by test_net file: cnn_test.prototxt
    I1227 18:26:30.349815  5629 net.cpp:49] Initializing net from parameters:
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
        num_output: 32
        kernel_size: 4
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
        pool: AVE
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
        pool: MAX
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
    I1227 18:26:30.350246  5629 layer_factory.hpp:77] Creating layer data
    I1227 18:26:30.350256  5629 net.cpp:106] Creating Layer data
    I1227 18:26:30.350272  5629 net.cpp:411] data -> data
    I1227 18:26:30.350281  5629 net.cpp:411] data -> label
    I1227 18:26:30.350289  5629 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1227 18:26:30.350317  5629 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1227 18:26:30.664489  5629 net.cpp:150] Setting up data
    I1227 18:26:30.664525  5629 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1227 18:26:30.664536  5629 net.cpp:157] Top shape: 120 (120)
    I1227 18:26:30.664543  5629 net.cpp:165] Memory required for data: 1475040
    I1227 18:26:30.664552  5629 layer_factory.hpp:77] Creating layer label_data_1_split
    I1227 18:26:30.664566  5629 net.cpp:106] Creating Layer label_data_1_split
    I1227 18:26:30.664574  5629 net.cpp:454] label_data_1_split <- label
    I1227 18:26:30.664603  5629 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1227 18:26:30.664618  5629 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1227 18:26:30.664664  5629 net.cpp:150] Setting up label_data_1_split
    I1227 18:26:30.664674  5629 net.cpp:157] Top shape: 120 (120)
    I1227 18:26:30.664681  5629 net.cpp:157] Top shape: 120 (120)
    I1227 18:26:30.664687  5629 net.cpp:165] Memory required for data: 1476000
    I1227 18:26:30.664693  5629 layer_factory.hpp:77] Creating layer conv1
    I1227 18:26:30.664706  5629 net.cpp:106] Creating Layer conv1
    I1227 18:26:30.664712  5629 net.cpp:454] conv1 <- data
    I1227 18:26:30.664721  5629 net.cpp:411] conv1 -> conv1
    I1227 18:26:30.664921  5629 net.cpp:150] Setting up conv1
    I1227 18:26:30.664934  5629 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1227 18:26:30.664942  5629 net.cpp:165] Memory required for data: 14393760
    I1227 18:26:30.664953  5629 layer_factory.hpp:77] Creating layer pool1
    I1227 18:26:30.664963  5629 net.cpp:106] Creating Layer pool1
    I1227 18:26:30.664970  5629 net.cpp:454] pool1 <- conv1
    I1227 18:26:30.664978  5629 net.cpp:411] pool1 -> pool1
    I1227 18:26:30.665014  5629 net.cpp:150] Setting up pool1
    I1227 18:26:30.665022  5629 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 18:26:30.665029  5629 net.cpp:165] Memory required for data: 17404320
    I1227 18:26:30.665035  5629 layer_factory.hpp:77] Creating layer drop1
    I1227 18:26:30.665045  5629 net.cpp:106] Creating Layer drop1
    I1227 18:26:30.665051  5629 net.cpp:454] drop1 <- pool1
    I1227 18:26:30.665058  5629 net.cpp:397] drop1 -> pool1 (in-place)
    I1227 18:26:30.665079  5629 net.cpp:150] Setting up drop1
    I1227 18:26:30.665088  5629 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 18:26:30.665094  5629 net.cpp:165] Memory required for data: 20414880
    I1227 18:26:30.665101  5629 layer_factory.hpp:77] Creating layer relu1
    I1227 18:26:30.665108  5629 net.cpp:106] Creating Layer relu1
    I1227 18:26:30.665114  5629 net.cpp:454] relu1 <- pool1
    I1227 18:26:30.665122  5629 net.cpp:397] relu1 -> pool1 (in-place)
    I1227 18:26:30.665129  5629 net.cpp:150] Setting up relu1
    I1227 18:26:30.665137  5629 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 18:26:30.665143  5629 net.cpp:165] Memory required for data: 23425440
    I1227 18:26:30.665148  5629 layer_factory.hpp:77] Creating layer conv2
    I1227 18:26:30.665158  5629 net.cpp:106] Creating Layer conv2
    I1227 18:26:30.665163  5629 net.cpp:454] conv2 <- pool1
    I1227 18:26:30.665170  5629 net.cpp:411] conv2 -> conv2
    I1227 18:26:30.665513  5629 net.cpp:150] Setting up conv2
    I1227 18:26:30.665525  5629 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1227 18:26:30.665531  5629 net.cpp:165] Memory required for data: 25864800
    I1227 18:26:30.665542  5629 layer_factory.hpp:77] Creating layer pool2
    I1227 18:26:30.665551  5629 net.cpp:106] Creating Layer pool2
    I1227 18:26:30.665557  5629 net.cpp:454] pool2 <- conv2
    I1227 18:26:30.665565  5629 net.cpp:411] pool2 -> pool2
    I1227 18:26:30.665588  5629 net.cpp:150] Setting up pool2
    I1227 18:26:30.665597  5629 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 18:26:30.665639  5629 net.cpp:165] Memory required for data: 26368800
    I1227 18:26:30.665647  5629 layer_factory.hpp:77] Creating layer drop2
    I1227 18:26:30.665658  5629 net.cpp:106] Creating Layer drop2
    I1227 18:26:30.665665  5629 net.cpp:454] drop2 <- pool2
    I1227 18:26:30.665674  5629 net.cpp:397] drop2 -> pool2 (in-place)
    I1227 18:26:30.665711  5629 net.cpp:150] Setting up drop2
    I1227 18:26:30.665721  5629 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 18:26:30.665727  5629 net.cpp:165] Memory required for data: 26872800
    I1227 18:26:30.665735  5629 layer_factory.hpp:77] Creating layer relu2
    I1227 18:26:30.665743  5629 net.cpp:106] Creating Layer relu2
    I1227 18:26:30.665750  5629 net.cpp:454] relu2 <- pool2
    I1227 18:26:30.665756  5629 net.cpp:397] relu2 -> pool2 (in-place)
    I1227 18:26:30.665765  5629 net.cpp:150] Setting up relu2
    I1227 18:26:30.665771  5629 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 18:26:30.665777  5629 net.cpp:165] Memory required for data: 27376800
    I1227 18:26:30.665783  5629 layer_factory.hpp:77] Creating layer conv3
    I1227 18:26:30.665792  5629 net.cpp:106] Creating Layer conv3
    I1227 18:26:30.665798  5629 net.cpp:454] conv3 <- pool2
    I1227 18:26:30.665807  5629 net.cpp:411] conv3 -> conv3
    I1227 18:26:30.666076  5629 net.cpp:150] Setting up conv3
    I1227 18:26:30.666091  5629 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1227 18:26:30.666108  5629 net.cpp:165] Memory required for data: 27868320
    I1227 18:26:30.666120  5629 layer_factory.hpp:77] Creating layer pool3
    I1227 18:26:30.666128  5629 net.cpp:106] Creating Layer pool3
    I1227 18:26:30.666136  5629 net.cpp:454] pool3 <- conv3
    I1227 18:26:30.666143  5629 net.cpp:411] pool3 -> pool3
    I1227 18:26:30.666177  5629 net.cpp:150] Setting up pool3
    I1227 18:26:30.666185  5629 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 18:26:30.666193  5629 net.cpp:165] Memory required for data: 27991200
    I1227 18:26:30.666198  5629 layer_factory.hpp:77] Creating layer relu3
    I1227 18:26:30.666206  5629 net.cpp:106] Creating Layer relu3
    I1227 18:26:30.666213  5629 net.cpp:454] relu3 <- pool3
    I1227 18:26:30.666219  5629 net.cpp:397] relu3 -> pool3 (in-place)
    I1227 18:26:30.666227  5629 net.cpp:150] Setting up relu3
    I1227 18:26:30.666235  5629 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 18:26:30.666241  5629 net.cpp:165] Memory required for data: 28114080
    I1227 18:26:30.666247  5629 layer_factory.hpp:77] Creating layer ip1
    I1227 18:26:30.666256  5629 net.cpp:106] Creating Layer ip1
    I1227 18:26:30.666262  5629 net.cpp:454] ip1 <- pool3
    I1227 18:26:30.666270  5629 net.cpp:411] ip1 -> ip1
    I1227 18:26:30.667295  5629 net.cpp:150] Setting up ip1
    I1227 18:26:30.667309  5629 net.cpp:157] Top shape: 120 512 (61440)
    I1227 18:26:30.667315  5629 net.cpp:165] Memory required for data: 28359840
    I1227 18:26:30.667323  5629 layer_factory.hpp:77] Creating layer sig1
    I1227 18:26:30.667332  5629 net.cpp:106] Creating Layer sig1
    I1227 18:26:30.667338  5629 net.cpp:454] sig1 <- ip1
    I1227 18:26:30.667346  5629 net.cpp:397] sig1 -> ip1 (in-place)
    I1227 18:26:30.667354  5629 net.cpp:150] Setting up sig1
    I1227 18:26:30.667361  5629 net.cpp:157] Top shape: 120 512 (61440)
    I1227 18:26:30.667366  5629 net.cpp:165] Memory required for data: 28605600
    I1227 18:26:30.667372  5629 layer_factory.hpp:77] Creating layer ip2
    I1227 18:26:30.667381  5629 net.cpp:106] Creating Layer ip2
    I1227 18:26:30.667387  5629 net.cpp:454] ip2 <- ip1
    I1227 18:26:30.667393  5629 net.cpp:411] ip2 -> ip2
    I1227 18:26:30.667513  5629 net.cpp:150] Setting up ip2
    I1227 18:26:30.667523  5629 net.cpp:157] Top shape: 120 10 (1200)
    I1227 18:26:30.667529  5629 net.cpp:165] Memory required for data: 28610400
    I1227 18:26:30.667541  5629 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1227 18:26:30.667549  5629 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1227 18:26:30.667557  5629 net.cpp:454] ip2_ip2_0_split <- ip2
    I1227 18:26:30.667564  5629 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1227 18:26:30.667572  5629 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1227 18:26:30.667603  5629 net.cpp:150] Setting up ip2_ip2_0_split
    I1227 18:26:30.667625  5629 net.cpp:157] Top shape: 120 10 (1200)
    I1227 18:26:30.667634  5629 net.cpp:157] Top shape: 120 10 (1200)
    I1227 18:26:30.667639  5629 net.cpp:165] Memory required for data: 28620000
    I1227 18:26:30.667646  5629 layer_factory.hpp:77] Creating layer accuracy
    I1227 18:26:30.667654  5629 net.cpp:106] Creating Layer accuracy
    I1227 18:26:30.667661  5629 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1227 18:26:30.667668  5629 net.cpp:454] accuracy <- label_data_1_split_0
    I1227 18:26:30.667676  5629 net.cpp:411] accuracy -> accuracy
    I1227 18:26:30.667687  5629 net.cpp:150] Setting up accuracy
    I1227 18:26:30.667695  5629 net.cpp:157] Top shape: (1)
    I1227 18:26:30.667701  5629 net.cpp:165] Memory required for data: 28620004
    I1227 18:26:30.667706  5629 layer_factory.hpp:77] Creating layer loss
    I1227 18:26:30.667716  5629 net.cpp:106] Creating Layer loss
    I1227 18:26:30.667721  5629 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1227 18:26:30.667728  5629 net.cpp:454] loss <- label_data_1_split_1
    I1227 18:26:30.667736  5629 net.cpp:411] loss -> loss
    I1227 18:26:30.667745  5629 layer_factory.hpp:77] Creating layer loss
    I1227 18:26:30.667819  5629 net.cpp:150] Setting up loss
    I1227 18:26:30.667829  5629 net.cpp:157] Top shape: (1)
    I1227 18:26:30.667835  5629 net.cpp:160]     with loss weight 1
    I1227 18:26:30.667848  5629 net.cpp:165] Memory required for data: 28620008
    I1227 18:26:30.667855  5629 net.cpp:226] loss needs backward computation.
    I1227 18:26:30.667862  5629 net.cpp:228] accuracy does not need backward computation.
    I1227 18:26:30.667870  5629 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1227 18:26:30.667876  5629 net.cpp:226] ip2 needs backward computation.
    I1227 18:26:30.667882  5629 net.cpp:226] sig1 needs backward computation.
    I1227 18:26:30.667888  5629 net.cpp:226] ip1 needs backward computation.
    I1227 18:26:30.667894  5629 net.cpp:226] relu3 needs backward computation.
    I1227 18:26:30.667901  5629 net.cpp:226] pool3 needs backward computation.
    I1227 18:26:30.667906  5629 net.cpp:226] conv3 needs backward computation.
    I1227 18:26:30.667913  5629 net.cpp:226] relu2 needs backward computation.
    I1227 18:26:30.667919  5629 net.cpp:226] drop2 needs backward computation.
    I1227 18:26:30.667925  5629 net.cpp:226] pool2 needs backward computation.
    I1227 18:26:30.667932  5629 net.cpp:226] conv2 needs backward computation.
    I1227 18:26:30.667937  5629 net.cpp:226] relu1 needs backward computation.
    I1227 18:26:30.667944  5629 net.cpp:226] drop1 needs backward computation.
    I1227 18:26:30.667949  5629 net.cpp:226] pool1 needs backward computation.
    I1227 18:26:30.667956  5629 net.cpp:226] conv1 needs backward computation.
    I1227 18:26:30.667963  5629 net.cpp:228] label_data_1_split does not need backward computation.
    I1227 18:26:30.667970  5629 net.cpp:228] data does not need backward computation.
    I1227 18:26:30.668256  5629 net.cpp:270] This network produces output accuracy
    I1227 18:26:30.668263  5629 net.cpp:270] This network produces output loss
    I1227 18:26:30.668282  5629 net.cpp:283] Network initialization done.
    I1227 18:26:30.668339  5629 solver.cpp:60] Solver scaffolding done.
    I1227 18:26:30.668747  5629 caffe.cpp:212] Starting Optimization
    I1227 18:26:30.668763  5629 solver.cpp:288] Solving
    I1227 18:26:30.668769  5629 solver.cpp:289] Learning Rate Policy: inv
    I1227 18:26:30.669694  5629 solver.cpp:341] Iteration 0, Testing net (#0)
    I1227 18:26:33.537184  5629 solver.cpp:409]     Test net output #0: accuracy = 0.100333
    I1227 18:26:33.537232  5629 solver.cpp:409]     Test net output #1: loss = 2.42574 (* 1 = 2.42574 loss)
    I1227 18:26:33.615016  5629 solver.cpp:237] Iteration 0, loss = 2.47312
    I1227 18:26:33.615066  5629 solver.cpp:253]     Train net output #0: accuracy = 0.07
    I1227 18:26:33.615078  5629 solver.cpp:253]     Train net output #1: loss = 2.47312 (* 1 = 2.47312 loss)
    I1227 18:26:33.615103  5629 sgd_solver.cpp:106] Iteration 0, lr = 0.0007
    I1227 18:26:41.709278  5629 solver.cpp:237] Iteration 100, loss = 2.28267
    I1227 18:26:41.709323  5629 solver.cpp:253]     Train net output #0: accuracy = 0.14
    I1227 18:26:41.709372  5629 solver.cpp:253]     Train net output #1: loss = 2.28267 (* 1 = 2.28267 loss)
    I1227 18:26:41.709384  5629 sgd_solver.cpp:106] Iteration 100, lr = 0.000694796
    I1227 18:26:48.986471  5629 solver.cpp:237] Iteration 200, loss = 2.31822
    I1227 18:26:48.986515  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:26:48.986526  5629 solver.cpp:253]     Train net output #1: loss = 2.31822 (* 1 = 2.31822 loss)
    I1227 18:26:48.986536  5629 sgd_solver.cpp:106] Iteration 200, lr = 0.00068968
    I1227 18:26:56.608886  5629 solver.cpp:237] Iteration 300, loss = 2.30683
    I1227 18:26:56.608940  5629 solver.cpp:253]     Train net output #0: accuracy = 0.13
    I1227 18:26:56.608957  5629 solver.cpp:253]     Train net output #1: loss = 2.30683 (* 1 = 2.30683 loss)
    I1227 18:26:56.608969  5629 sgd_solver.cpp:106] Iteration 300, lr = 0.000684652
    I1227 18:27:04.842772  5629 solver.cpp:237] Iteration 400, loss = 2.29897
    I1227 18:27:04.842859  5629 solver.cpp:253]     Train net output #0: accuracy = 0.12
    I1227 18:27:04.842875  5629 solver.cpp:253]     Train net output #1: loss = 2.29897 (* 1 = 2.29897 loss)
    I1227 18:27:04.842886  5629 sgd_solver.cpp:106] Iteration 400, lr = 0.000679709
    I1227 18:27:12.360009  5629 solver.cpp:237] Iteration 500, loss = 2.30005
    I1227 18:27:12.360054  5629 solver.cpp:253]     Train net output #0: accuracy = 0.08
    I1227 18:27:12.360067  5629 solver.cpp:253]     Train net output #1: loss = 2.30005 (* 1 = 2.30005 loss)
    I1227 18:27:12.360077  5629 sgd_solver.cpp:106] Iteration 500, lr = 0.000674848
    I1227 18:27:20.149583  5629 solver.cpp:237] Iteration 600, loss = 2.28684
    I1227 18:27:20.149627  5629 solver.cpp:253]     Train net output #0: accuracy = 0.14
    I1227 18:27:20.149639  5629 solver.cpp:253]     Train net output #1: loss = 2.28684 (* 1 = 2.28684 loss)
    I1227 18:27:20.149658  5629 sgd_solver.cpp:106] Iteration 600, lr = 0.000670068
    I1227 18:27:27.218539  5629 solver.cpp:237] Iteration 700, loss = 2.32355
    I1227 18:27:27.218590  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:27:27.218605  5629 solver.cpp:253]     Train net output #1: loss = 2.32355 (* 1 = 2.32355 loss)
    I1227 18:27:27.218616  5629 sgd_solver.cpp:106] Iteration 700, lr = 0.000665365
    I1227 18:27:34.141867  5629 solver.cpp:237] Iteration 800, loss = 2.30506
    I1227 18:27:34.141912  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:27:34.141924  5629 solver.cpp:253]     Train net output #1: loss = 2.30506 (* 1 = 2.30506 loss)
    I1227 18:27:34.141932  5629 sgd_solver.cpp:106] Iteration 800, lr = 0.000660739
    I1227 18:27:41.625886  5629 solver.cpp:237] Iteration 900, loss = 2.30295
    I1227 18:27:41.626041  5629 solver.cpp:253]     Train net output #0: accuracy = 0.1
    I1227 18:27:41.626070  5629 solver.cpp:253]     Train net output #1: loss = 2.30295 (* 1 = 2.30295 loss)
    I1227 18:27:41.626086  5629 sgd_solver.cpp:106] Iteration 900, lr = 0.000656188
    I1227 18:27:48.862304  5629 solver.cpp:341] Iteration 1000, Testing net (#0)
    I1227 18:27:51.902546  5629 solver.cpp:409]     Test net output #0: accuracy = 0.1
    I1227 18:27:51.902595  5629 solver.cpp:409]     Test net output #1: loss = 2.31445 (* 1 = 2.31445 loss)
    I1227 18:27:51.932778  5629 solver.cpp:237] Iteration 1000, loss = 2.3056
    I1227 18:27:51.932812  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:27:51.932827  5629 solver.cpp:253]     Train net output #1: loss = 2.3056 (* 1 = 2.3056 loss)
    I1227 18:27:51.932840  5629 sgd_solver.cpp:106] Iteration 1000, lr = 0.000651709
    I1227 18:27:59.172715  5629 solver.cpp:237] Iteration 1100, loss = 2.27636
    I1227 18:27:59.172760  5629 solver.cpp:253]     Train net output #0: accuracy = 0.14
    I1227 18:27:59.172770  5629 solver.cpp:253]     Train net output #1: loss = 2.27636 (* 1 = 2.27636 loss)
    I1227 18:27:59.172781  5629 sgd_solver.cpp:106] Iteration 1100, lr = 0.0006473
    I1227 18:28:06.471874  5629 solver.cpp:237] Iteration 1200, loss = 2.32595
    I1227 18:28:06.471920  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:28:06.471933  5629 solver.cpp:253]     Train net output #1: loss = 2.32595 (* 1 = 2.32595 loss)
    I1227 18:28:06.471945  5629 sgd_solver.cpp:106] Iteration 1200, lr = 0.000642961
    I1227 18:28:13.825357  5629 solver.cpp:237] Iteration 1300, loss = 2.27547
    I1227 18:28:13.825532  5629 solver.cpp:253]     Train net output #0: accuracy = 0.15
    I1227 18:28:13.825547  5629 solver.cpp:253]     Train net output #1: loss = 2.27547 (* 1 = 2.27547 loss)
    I1227 18:28:13.825556  5629 sgd_solver.cpp:106] Iteration 1300, lr = 0.000638689
    I1227 18:28:21.308553  5629 solver.cpp:237] Iteration 1400, loss = 2.29736
    I1227 18:28:21.308612  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:28:21.308629  5629 solver.cpp:253]     Train net output #1: loss = 2.29736 (* 1 = 2.29736 loss)
    I1227 18:28:21.308647  5629 sgd_solver.cpp:106] Iteration 1400, lr = 0.000634482
    I1227 18:28:28.228771  5629 solver.cpp:237] Iteration 1500, loss = 2.29591
    I1227 18:28:28.228826  5629 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1227 18:28:28.228840  5629 solver.cpp:253]     Train net output #1: loss = 2.29591 (* 1 = 2.29591 loss)
    I1227 18:28:28.228849  5629 sgd_solver.cpp:106] Iteration 1500, lr = 0.00063034
    I1227 18:28:35.171515  5629 solver.cpp:237] Iteration 1600, loss = 2.2464
    I1227 18:28:35.171561  5629 solver.cpp:253]     Train net output #0: accuracy = 0.14
    I1227 18:28:35.171573  5629 solver.cpp:253]     Train net output #1: loss = 2.2464 (* 1 = 2.2464 loss)
    I1227 18:28:35.171581  5629 sgd_solver.cpp:106] Iteration 1600, lr = 0.00062626
    I1227 18:28:42.129457  5629 solver.cpp:237] Iteration 1700, loss = 2.23429
    I1227 18:28:42.129505  5629 solver.cpp:253]     Train net output #0: accuracy = 0.13
    I1227 18:28:42.129518  5629 solver.cpp:253]     Train net output #1: loss = 2.23429 (* 1 = 2.23429 loss)
    I1227 18:28:42.129528  5629 sgd_solver.cpp:106] Iteration 1700, lr = 0.000622241
    I1227 18:28:49.856492  5629 solver.cpp:237] Iteration 1800, loss = 2.15982
    I1227 18:28:49.856647  5629 solver.cpp:253]     Train net output #0: accuracy = 0.16
    I1227 18:28:49.856662  5629 solver.cpp:253]     Train net output #1: loss = 2.15982 (* 1 = 2.15982 loss)
    I1227 18:28:49.856672  5629 sgd_solver.cpp:106] Iteration 1800, lr = 0.000618282
    I1227 18:28:57.206148  5629 solver.cpp:237] Iteration 1900, loss = 2.0232
    I1227 18:28:57.206192  5629 solver.cpp:253]     Train net output #0: accuracy = 0.2
    I1227 18:28:57.206204  5629 solver.cpp:253]     Train net output #1: loss = 2.0232 (* 1 = 2.0232 loss)
    I1227 18:28:57.206213  5629 sgd_solver.cpp:106] Iteration 1900, lr = 0.000614381
    I1227 18:29:04.224725  5629 solver.cpp:341] Iteration 2000, Testing net (#0)
    I1227 18:29:07.009768  5629 solver.cpp:409]     Test net output #0: accuracy = 0.269083
    I1227 18:29:07.009819  5629 solver.cpp:409]     Test net output #1: loss = 1.94307 (* 1 = 1.94307 loss)
    I1227 18:29:07.038086  5629 solver.cpp:237] Iteration 2000, loss = 1.85123
    I1227 18:29:07.038113  5629 solver.cpp:253]     Train net output #0: accuracy = 0.31
    I1227 18:29:07.038125  5629 solver.cpp:253]     Train net output #1: loss = 1.85123 (* 1 = 1.85123 loss)
    I1227 18:29:07.038133  5629 sgd_solver.cpp:106] Iteration 2000, lr = 0.000610537
    I1227 18:29:14.409306  5629 solver.cpp:237] Iteration 2100, loss = 1.95488
    I1227 18:29:14.409350  5629 solver.cpp:253]     Train net output #0: accuracy = 0.27
    I1227 18:29:14.409363  5629 solver.cpp:253]     Train net output #1: loss = 1.95488 (* 1 = 1.95488 loss)
    I1227 18:29:14.409371  5629 sgd_solver.cpp:106] Iteration 2100, lr = 0.000606749
    I1227 18:29:21.733660  5629 solver.cpp:237] Iteration 2200, loss = 1.82716
    I1227 18:29:21.733853  5629 solver.cpp:253]     Train net output #0: accuracy = 0.29
    I1227 18:29:21.733870  5629 solver.cpp:253]     Train net output #1: loss = 1.82716 (* 1 = 1.82716 loss)
    I1227 18:29:21.733880  5629 sgd_solver.cpp:106] Iteration 2200, lr = 0.000603015
    I1227 18:29:29.393339  5629 solver.cpp:237] Iteration 2300, loss = 1.65943
    I1227 18:29:29.393388  5629 solver.cpp:253]     Train net output #0: accuracy = 0.4
    I1227 18:29:29.393403  5629 solver.cpp:253]     Train net output #1: loss = 1.65943 (* 1 = 1.65943 loss)
    I1227 18:29:29.393414  5629 sgd_solver.cpp:106] Iteration 2300, lr = 0.000599334
    I1227 18:29:36.783185  5629 solver.cpp:237] Iteration 2400, loss = 1.62528
    I1227 18:29:36.783229  5629 solver.cpp:253]     Train net output #0: accuracy = 0.4
    I1227 18:29:36.783241  5629 solver.cpp:253]     Train net output #1: loss = 1.62528 (* 1 = 1.62528 loss)
    I1227 18:29:36.783249  5629 sgd_solver.cpp:106] Iteration 2400, lr = 0.000595706
    I1227 18:29:44.162837  5629 solver.cpp:237] Iteration 2500, loss = 1.54416
    I1227 18:29:44.162876  5629 solver.cpp:253]     Train net output #0: accuracy = 0.42
    I1227 18:29:44.162889  5629 solver.cpp:253]     Train net output #1: loss = 1.54416 (* 1 = 1.54416 loss)
    I1227 18:29:44.162899  5629 sgd_solver.cpp:106] Iteration 2500, lr = 0.000592128
    I1227 18:29:51.412714  5629 solver.cpp:237] Iteration 2600, loss = 1.66849
    I1227 18:29:51.412758  5629 solver.cpp:253]     Train net output #0: accuracy = 0.39
    I1227 18:29:51.412775  5629 solver.cpp:253]     Train net output #1: loss = 1.66849 (* 1 = 1.66849 loss)
    I1227 18:29:51.412786  5629 sgd_solver.cpp:106] Iteration 2600, lr = 0.0005886
    I1227 18:29:58.495185  5629 solver.cpp:237] Iteration 2700, loss = 1.53307
    I1227 18:29:58.495321  5629 solver.cpp:253]     Train net output #0: accuracy = 0.38
    I1227 18:29:58.495340  5629 solver.cpp:253]     Train net output #1: loss = 1.53307 (* 1 = 1.53307 loss)
    I1227 18:29:58.495352  5629 sgd_solver.cpp:106] Iteration 2700, lr = 0.00058512
    I1227 18:30:05.732873  5629 solver.cpp:237] Iteration 2800, loss = 1.63277
    I1227 18:30:05.732921  5629 solver.cpp:253]     Train net output #0: accuracy = 0.42
    I1227 18:30:05.732935  5629 solver.cpp:253]     Train net output #1: loss = 1.63277 (* 1 = 1.63277 loss)
    I1227 18:30:05.732946  5629 sgd_solver.cpp:106] Iteration 2800, lr = 0.000581689
    I1227 18:30:13.083770  5629 solver.cpp:237] Iteration 2900, loss = 1.43156
    I1227 18:30:13.083825  5629 solver.cpp:253]     Train net output #0: accuracy = 0.48
    I1227 18:30:13.083842  5629 solver.cpp:253]     Train net output #1: loss = 1.43156 (* 1 = 1.43156 loss)
    I1227 18:30:13.083853  5629 sgd_solver.cpp:106] Iteration 2900, lr = 0.000578303
    I1227 18:30:19.948879  5629 solver.cpp:341] Iteration 3000, Testing net (#0)
    I1227 18:30:22.742386  5629 solver.cpp:409]     Test net output #0: accuracy = 0.450583
    I1227 18:30:22.742434  5629 solver.cpp:409]     Test net output #1: loss = 1.50064 (* 1 = 1.50064 loss)
    I1227 18:30:22.770650  5629 solver.cpp:237] Iteration 3000, loss = 1.48099
    I1227 18:30:22.770676  5629 solver.cpp:253]     Train net output #0: accuracy = 0.42
    I1227 18:30:22.770686  5629 solver.cpp:253]     Train net output #1: loss = 1.48099 (* 1 = 1.48099 loss)
    I1227 18:30:22.770695  5629 sgd_solver.cpp:106] Iteration 3000, lr = 0.000574964
    I1227 18:30:29.705257  5629 solver.cpp:237] Iteration 3100, loss = 1.53706
    I1227 18:30:29.705428  5629 solver.cpp:253]     Train net output #0: accuracy = 0.37
    I1227 18:30:29.705442  5629 solver.cpp:253]     Train net output #1: loss = 1.53706 (* 1 = 1.53706 loss)
    I1227 18:30:29.705452  5629 sgd_solver.cpp:106] Iteration 3100, lr = 0.000571669
    I1227 18:30:36.656234  5629 solver.cpp:237] Iteration 3200, loss = 1.39495
    I1227 18:30:36.656276  5629 solver.cpp:253]     Train net output #0: accuracy = 0.49
    I1227 18:30:36.656289  5629 solver.cpp:253]     Train net output #1: loss = 1.39495 (* 1 = 1.39495 loss)
    I1227 18:30:36.656296  5629 sgd_solver.cpp:106] Iteration 3200, lr = 0.000568418
    I1227 18:30:43.603099  5629 solver.cpp:237] Iteration 3300, loss = 1.38799
    I1227 18:30:43.603135  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:30:43.603147  5629 solver.cpp:253]     Train net output #1: loss = 1.38799 (* 1 = 1.38799 loss)
    I1227 18:30:43.603155  5629 sgd_solver.cpp:106] Iteration 3300, lr = 0.000565209
    I1227 18:30:50.545691  5629 solver.cpp:237] Iteration 3400, loss = 1.29619
    I1227 18:30:50.545727  5629 solver.cpp:253]     Train net output #0: accuracy = 0.52
    I1227 18:30:50.545738  5629 solver.cpp:253]     Train net output #1: loss = 1.29619 (* 1 = 1.29619 loss)
    I1227 18:30:50.545747  5629 sgd_solver.cpp:106] Iteration 3400, lr = 0.000562043
    I1227 18:30:57.535621  5629 solver.cpp:237] Iteration 3500, loss = 1.43779
    I1227 18:30:57.535662  5629 solver.cpp:253]     Train net output #0: accuracy = 0.4
    I1227 18:30:57.535676  5629 solver.cpp:253]     Train net output #1: loss = 1.43779 (* 1 = 1.43779 loss)
    I1227 18:30:57.535687  5629 sgd_solver.cpp:106] Iteration 3500, lr = 0.000558917
    I1227 18:31:04.487952  5629 solver.cpp:237] Iteration 3600, loss = 1.71575
    I1227 18:31:04.488134  5629 solver.cpp:253]     Train net output #0: accuracy = 0.37
    I1227 18:31:04.488159  5629 solver.cpp:253]     Train net output #1: loss = 1.71575 (* 1 = 1.71575 loss)
    I1227 18:31:04.488168  5629 sgd_solver.cpp:106] Iteration 3600, lr = 0.000555832
    I1227 18:31:11.428725  5629 solver.cpp:237] Iteration 3700, loss = 1.43118
    I1227 18:31:11.428768  5629 solver.cpp:253]     Train net output #0: accuracy = 0.49
    I1227 18:31:11.428781  5629 solver.cpp:253]     Train net output #1: loss = 1.43118 (* 1 = 1.43118 loss)
    I1227 18:31:11.428789  5629 sgd_solver.cpp:106] Iteration 3700, lr = 0.000552787
    I1227 18:31:18.365387  5629 solver.cpp:237] Iteration 3800, loss = 1.47022
    I1227 18:31:18.365423  5629 solver.cpp:253]     Train net output #0: accuracy = 0.48
    I1227 18:31:18.365437  5629 solver.cpp:253]     Train net output #1: loss = 1.47022 (* 1 = 1.47022 loss)
    I1227 18:31:18.365444  5629 sgd_solver.cpp:106] Iteration 3800, lr = 0.00054978
    I1227 18:31:25.324193  5629 solver.cpp:237] Iteration 3900, loss = 1.28605
    I1227 18:31:25.324229  5629 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1227 18:31:25.324241  5629 solver.cpp:253]     Train net output #1: loss = 1.28605 (* 1 = 1.28605 loss)
    I1227 18:31:25.324249  5629 sgd_solver.cpp:106] Iteration 3900, lr = 0.000546811
    I1227 18:31:32.730355  5629 solver.cpp:341] Iteration 4000, Testing net (#0)
    I1227 18:31:35.519505  5629 solver.cpp:409]     Test net output #0: accuracy = 0.521917
    I1227 18:31:35.519683  5629 solver.cpp:409]     Test net output #1: loss = 1.3281 (* 1 = 1.3281 loss)
    I1227 18:31:35.548717  5629 solver.cpp:237] Iteration 4000, loss = 1.34292
    I1227 18:31:35.548737  5629 solver.cpp:253]     Train net output #0: accuracy = 0.47
    I1227 18:31:35.548746  5629 solver.cpp:253]     Train net output #1: loss = 1.34292 (* 1 = 1.34292 loss)
    I1227 18:31:35.548756  5629 sgd_solver.cpp:106] Iteration 4000, lr = 0.000543879
    I1227 18:31:42.504854  5629 solver.cpp:237] Iteration 4100, loss = 1.46751
    I1227 18:31:42.504889  5629 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1227 18:31:42.504901  5629 solver.cpp:253]     Train net output #1: loss = 1.46751 (* 1 = 1.46751 loss)
    I1227 18:31:42.504909  5629 sgd_solver.cpp:106] Iteration 4100, lr = 0.000540983
    I1227 18:31:49.492303  5629 solver.cpp:237] Iteration 4200, loss = 1.24058
    I1227 18:31:49.492346  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:31:49.492360  5629 solver.cpp:253]     Train net output #1: loss = 1.24058 (* 1 = 1.24058 loss)
    I1227 18:31:49.492372  5629 sgd_solver.cpp:106] Iteration 4200, lr = 0.000538123
    I1227 18:31:56.440006  5629 solver.cpp:237] Iteration 4300, loss = 1.29628
    I1227 18:31:56.440057  5629 solver.cpp:253]     Train net output #0: accuracy = 0.55
    I1227 18:31:56.440083  5629 solver.cpp:253]     Train net output #1: loss = 1.29628 (* 1 = 1.29628 loss)
    I1227 18:31:56.440094  5629 sgd_solver.cpp:106] Iteration 4300, lr = 0.000535298
    I1227 18:32:03.472745  5629 solver.cpp:237] Iteration 4400, loss = 1.3318
    I1227 18:32:03.472796  5629 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1227 18:32:03.472812  5629 solver.cpp:253]     Train net output #1: loss = 1.3318 (* 1 = 1.3318 loss)
    I1227 18:32:03.472826  5629 sgd_solver.cpp:106] Iteration 4400, lr = 0.000532508
    I1227 18:32:10.746184  5629 solver.cpp:237] Iteration 4500, loss = 1.3102
    I1227 18:32:10.746371  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:32:10.746390  5629 solver.cpp:253]     Train net output #1: loss = 1.3102 (* 1 = 1.3102 loss)
    I1227 18:32:10.746402  5629 sgd_solver.cpp:106] Iteration 4500, lr = 0.000529751
    I1227 18:32:17.958916  5629 solver.cpp:237] Iteration 4600, loss = 1.43517
    I1227 18:32:17.958952  5629 solver.cpp:253]     Train net output #0: accuracy = 0.52
    I1227 18:32:17.958966  5629 solver.cpp:253]     Train net output #1: loss = 1.43517 (* 1 = 1.43517 loss)
    I1227 18:32:17.958976  5629 sgd_solver.cpp:106] Iteration 4600, lr = 0.000527028
    I1227 18:32:25.392035  5629 solver.cpp:237] Iteration 4700, loss = 1.1553
    I1227 18:32:25.392071  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:32:25.392083  5629 solver.cpp:253]     Train net output #1: loss = 1.1553 (* 1 = 1.1553 loss)
    I1227 18:32:25.392093  5629 sgd_solver.cpp:106] Iteration 4700, lr = 0.000524336
    I1227 18:32:33.279791  5629 solver.cpp:237] Iteration 4800, loss = 1.19202
    I1227 18:32:33.279839  5629 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1227 18:32:33.279853  5629 solver.cpp:253]     Train net output #1: loss = 1.19202 (* 1 = 1.19202 loss)
    I1227 18:32:33.279865  5629 sgd_solver.cpp:106] Iteration 4800, lr = 0.000521677
    I1227 18:32:40.990057  5629 solver.cpp:237] Iteration 4900, loss = 1.1872
    I1227 18:32:40.990164  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:32:40.990181  5629 solver.cpp:253]     Train net output #1: loss = 1.1872 (* 1 = 1.1872 loss)
    I1227 18:32:40.990193  5629 sgd_solver.cpp:106] Iteration 4900, lr = 0.000519049
    I1227 18:32:48.107285  5629 solver.cpp:341] Iteration 5000, Testing net (#0)
    I1227 18:32:50.949446  5629 solver.cpp:409]     Test net output #0: accuracy = 0.537
    I1227 18:32:50.949493  5629 solver.cpp:409]     Test net output #1: loss = 1.28985 (* 1 = 1.28985 loss)
    I1227 18:32:50.978063  5629 solver.cpp:237] Iteration 5000, loss = 1.19486
    I1227 18:32:50.978103  5629 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1227 18:32:50.978114  5629 solver.cpp:253]     Train net output #1: loss = 1.19486 (* 1 = 1.19486 loss)
    I1227 18:32:50.978124  5629 sgd_solver.cpp:106] Iteration 5000, lr = 0.000516452
    I1227 18:32:58.285298  5629 solver.cpp:237] Iteration 5100, loss = 1.38452
    I1227 18:32:58.285342  5629 solver.cpp:253]     Train net output #0: accuracy = 0.54
    I1227 18:32:58.285354  5629 solver.cpp:253]     Train net output #1: loss = 1.38452 (* 1 = 1.38452 loss)
    I1227 18:32:58.285362  5629 sgd_solver.cpp:106] Iteration 5100, lr = 0.000513884
    I1227 18:33:06.400564  5629 solver.cpp:237] Iteration 5200, loss = 1.23115
    I1227 18:33:06.400615  5629 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1227 18:33:06.400629  5629 solver.cpp:253]     Train net output #1: loss = 1.23115 (* 1 = 1.23115 loss)
    I1227 18:33:06.400640  5629 sgd_solver.cpp:106] Iteration 5200, lr = 0.000511347
    I1227 18:33:14.159286  5629 solver.cpp:237] Iteration 5300, loss = 1.16903
    I1227 18:33:14.160565  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:33:14.160598  5629 solver.cpp:253]     Train net output #1: loss = 1.16903 (* 1 = 1.16903 loss)
    I1227 18:33:14.160611  5629 sgd_solver.cpp:106] Iteration 5300, lr = 0.000508838
    I1227 18:33:21.629622  5629 solver.cpp:237] Iteration 5400, loss = 1.18602
    I1227 18:33:21.629663  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:33:21.629678  5629 solver.cpp:253]     Train net output #1: loss = 1.18602 (* 1 = 1.18602 loss)
    I1227 18:33:21.629688  5629 sgd_solver.cpp:106] Iteration 5400, lr = 0.000506358
    I1227 18:33:28.920111  5629 solver.cpp:237] Iteration 5500, loss = 1.21692
    I1227 18:33:28.920148  5629 solver.cpp:253]     Train net output #0: accuracy = 0.52
    I1227 18:33:28.920161  5629 solver.cpp:253]     Train net output #1: loss = 1.21692 (* 1 = 1.21692 loss)
    I1227 18:33:28.920169  5629 sgd_solver.cpp:106] Iteration 5500, lr = 0.000503906
    I1227 18:33:36.454668  5629 solver.cpp:237] Iteration 5600, loss = 1.37469
    I1227 18:33:36.454715  5629 solver.cpp:253]     Train net output #0: accuracy = 0.52
    I1227 18:33:36.454730  5629 solver.cpp:253]     Train net output #1: loss = 1.37469 (* 1 = 1.37469 loss)
    I1227 18:33:36.454742  5629 sgd_solver.cpp:106] Iteration 5600, lr = 0.000501481
    I1227 18:33:44.142678  5629 solver.cpp:237] Iteration 5700, loss = 1.09594
    I1227 18:33:44.142727  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:33:44.142740  5629 solver.cpp:253]     Train net output #1: loss = 1.09594 (* 1 = 1.09594 loss)
    I1227 18:33:44.142751  5629 sgd_solver.cpp:106] Iteration 5700, lr = 0.000499084
    I1227 18:33:52.242820  5629 solver.cpp:237] Iteration 5800, loss = 1.25951
    I1227 18:33:52.242976  5629 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1227 18:33:52.243005  5629 solver.cpp:253]     Train net output #1: loss = 1.25951 (* 1 = 1.25951 loss)
    I1227 18:33:52.243016  5629 sgd_solver.cpp:106] Iteration 5800, lr = 0.000496713
    I1227 18:34:00.652976  5629 solver.cpp:237] Iteration 5900, loss = 1.14965
    I1227 18:34:00.653017  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:34:00.653030  5629 solver.cpp:253]     Train net output #1: loss = 1.14965 (* 1 = 1.14965 loss)
    I1227 18:34:00.653041  5629 sgd_solver.cpp:106] Iteration 5900, lr = 0.000494368
    I1227 18:34:09.439059  5629 solver.cpp:341] Iteration 6000, Testing net (#0)
    I1227 18:34:12.713042  5629 solver.cpp:409]     Test net output #0: accuracy = 0.56675
    I1227 18:34:12.713091  5629 solver.cpp:409]     Test net output #1: loss = 1.21743 (* 1 = 1.21743 loss)
    I1227 18:34:12.743311  5629 solver.cpp:237] Iteration 6000, loss = 1.15362
    I1227 18:34:12.743355  5629 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1227 18:34:12.743371  5629 solver.cpp:253]     Train net output #1: loss = 1.15362 (* 1 = 1.15362 loss)
    I1227 18:34:12.743384  5629 sgd_solver.cpp:106] Iteration 6000, lr = 0.000492049
    I1227 18:34:20.001834  5629 solver.cpp:237] Iteration 6100, loss = 1.23098
    I1227 18:34:20.001876  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:34:20.001891  5629 solver.cpp:253]     Train net output #1: loss = 1.23098 (* 1 = 1.23098 loss)
    I1227 18:34:20.001902  5629 sgd_solver.cpp:106] Iteration 6100, lr = 0.000489755
    I1227 18:34:27.364804  5629 solver.cpp:237] Iteration 6200, loss = 1.06282
    I1227 18:34:27.364929  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:34:27.364953  5629 solver.cpp:253]     Train net output #1: loss = 1.06282 (* 1 = 1.06282 loss)
    I1227 18:34:27.364969  5629 sgd_solver.cpp:106] Iteration 6200, lr = 0.000487486
    I1227 18:34:34.755941  5629 solver.cpp:237] Iteration 6300, loss = 1.17344
    I1227 18:34:34.755993  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:34:34.756012  5629 solver.cpp:253]     Train net output #1: loss = 1.17344 (* 1 = 1.17344 loss)
    I1227 18:34:34.756029  5629 sgd_solver.cpp:106] Iteration 6300, lr = 0.000485241
    I1227 18:34:42.891880  5629 solver.cpp:237] Iteration 6400, loss = 1.18384
    I1227 18:34:42.891929  5629 solver.cpp:253]     Train net output #0: accuracy = 0.55
    I1227 18:34:42.891948  5629 solver.cpp:253]     Train net output #1: loss = 1.18384 (* 1 = 1.18384 loss)
    I1227 18:34:42.891963  5629 sgd_solver.cpp:106] Iteration 6400, lr = 0.00048302
    I1227 18:34:50.639109  5629 solver.cpp:237] Iteration 6500, loss = 1.15917
    I1227 18:34:50.639152  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:34:50.639168  5629 solver.cpp:253]     Train net output #1: loss = 1.15917 (* 1 = 1.15917 loss)
    I1227 18:34:50.639178  5629 sgd_solver.cpp:106] Iteration 6500, lr = 0.000480823
    I1227 18:34:57.838558  5629 solver.cpp:237] Iteration 6600, loss = 1.43842
    I1227 18:34:57.838688  5629 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1227 18:34:57.838711  5629 solver.cpp:253]     Train net output #1: loss = 1.43842 (* 1 = 1.43842 loss)
    I1227 18:34:57.838723  5629 sgd_solver.cpp:106] Iteration 6600, lr = 0.000478649
    I1227 18:35:04.822641  5629 solver.cpp:237] Iteration 6700, loss = 1.02904
    I1227 18:35:04.822685  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:35:04.822700  5629 solver.cpp:253]     Train net output #1: loss = 1.02904 (* 1 = 1.02904 loss)
    I1227 18:35:04.822710  5629 sgd_solver.cpp:106] Iteration 6700, lr = 0.000476498
    I1227 18:35:11.787120  5629 solver.cpp:237] Iteration 6800, loss = 1.07065
    I1227 18:35:11.787169  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:35:11.787189  5629 solver.cpp:253]     Train net output #1: loss = 1.07065 (* 1 = 1.07065 loss)
    I1227 18:35:11.787202  5629 sgd_solver.cpp:106] Iteration 6800, lr = 0.000474369
    I1227 18:35:18.727741  5629 solver.cpp:237] Iteration 6900, loss = 1.12404
    I1227 18:35:18.727792  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:35:18.727812  5629 solver.cpp:253]     Train net output #1: loss = 1.12404 (* 1 = 1.12404 loss)
    I1227 18:35:18.727825  5629 sgd_solver.cpp:106] Iteration 6900, lr = 0.000472262
    I1227 18:35:25.608201  5629 solver.cpp:341] Iteration 7000, Testing net (#0)
    I1227 18:35:28.427307  5629 solver.cpp:409]     Test net output #0: accuracy = 0.59775
    I1227 18:35:28.427464  5629 solver.cpp:409]     Test net output #1: loss = 1.13023 (* 1 = 1.13023 loss)
    I1227 18:35:28.460587  5629 solver.cpp:237] Iteration 7000, loss = 1.11345
    I1227 18:35:28.460623  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:35:28.460640  5629 solver.cpp:253]     Train net output #1: loss = 1.11345 (* 1 = 1.11345 loss)
    I1227 18:35:28.460655  5629 sgd_solver.cpp:106] Iteration 7000, lr = 0.000470177
    I1227 18:35:35.424768  5629 solver.cpp:237] Iteration 7100, loss = 1.43503
    I1227 18:35:35.424814  5629 solver.cpp:253]     Train net output #0: accuracy = 0.48
    I1227 18:35:35.424834  5629 solver.cpp:253]     Train net output #1: loss = 1.43503 (* 1 = 1.43503 loss)
    I1227 18:35:35.424849  5629 sgd_solver.cpp:106] Iteration 7100, lr = 0.000468113
    I1227 18:35:42.396363  5629 solver.cpp:237] Iteration 7200, loss = 1.13285
    I1227 18:35:42.396402  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:35:42.396417  5629 solver.cpp:253]     Train net output #1: loss = 1.13285 (* 1 = 1.13285 loss)
    I1227 18:35:42.396427  5629 sgd_solver.cpp:106] Iteration 7200, lr = 0.000466071
    I1227 18:35:49.362485  5629 solver.cpp:237] Iteration 7300, loss = 1.05552
    I1227 18:35:49.362526  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:35:49.362540  5629 solver.cpp:253]     Train net output #1: loss = 1.05552 (* 1 = 1.05552 loss)
    I1227 18:35:49.362551  5629 sgd_solver.cpp:106] Iteration 7300, lr = 0.000464049
    I1227 18:35:56.316444  5629 solver.cpp:237] Iteration 7400, loss = 1.04357
    I1227 18:35:56.316484  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:35:56.316498  5629 solver.cpp:253]     Train net output #1: loss = 1.04357 (* 1 = 1.04357 loss)
    I1227 18:35:56.316509  5629 sgd_solver.cpp:106] Iteration 7400, lr = 0.000462047
    I1227 18:36:03.297530  5629 solver.cpp:237] Iteration 7500, loss = 1.22756
    I1227 18:36:03.297631  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:36:03.297653  5629 solver.cpp:253]     Train net output #1: loss = 1.22756 (* 1 = 1.22756 loss)
    I1227 18:36:03.297668  5629 sgd_solver.cpp:106] Iteration 7500, lr = 0.000460065
    I1227 18:36:10.748872  5629 solver.cpp:237] Iteration 7600, loss = 1.23835
    I1227 18:36:10.748921  5629 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1227 18:36:10.748941  5629 solver.cpp:253]     Train net output #1: loss = 1.23835 (* 1 = 1.23835 loss)
    I1227 18:36:10.748957  5629 sgd_solver.cpp:106] Iteration 7600, lr = 0.000458103
    I1227 18:36:17.871568  5629 solver.cpp:237] Iteration 7700, loss = 1.05636
    I1227 18:36:17.871608  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:36:17.871623  5629 solver.cpp:253]     Train net output #1: loss = 1.05636 (* 1 = 1.05636 loss)
    I1227 18:36:17.871634  5629 sgd_solver.cpp:106] Iteration 7700, lr = 0.000456161
    I1227 18:36:24.829900  5629 solver.cpp:237] Iteration 7800, loss = 1.05922
    I1227 18:36:24.829941  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:36:24.829954  5629 solver.cpp:253]     Train net output #1: loss = 1.05922 (* 1 = 1.05922 loss)
    I1227 18:36:24.829965  5629 sgd_solver.cpp:106] Iteration 7800, lr = 0.000454238
    I1227 18:36:31.772066  5629 solver.cpp:237] Iteration 7900, loss = 1.06271
    I1227 18:36:31.772107  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:36:31.772122  5629 solver.cpp:253]     Train net output #1: loss = 1.06271 (* 1 = 1.06271 loss)
    I1227 18:36:31.772132  5629 sgd_solver.cpp:106] Iteration 7900, lr = 0.000452333
    I1227 18:36:38.669569  5629 solver.cpp:341] Iteration 8000, Testing net (#0)
    I1227 18:36:41.472626  5629 solver.cpp:409]     Test net output #0: accuracy = 0.616
    I1227 18:36:41.472678  5629 solver.cpp:409]     Test net output #1: loss = 1.09174 (* 1 = 1.09174 loss)
    I1227 18:36:41.502913  5629 solver.cpp:237] Iteration 8000, loss = 1.0309
    I1227 18:36:41.502965  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:36:41.502981  5629 solver.cpp:253]     Train net output #1: loss = 1.0309 (* 1 = 1.0309 loss)
    I1227 18:36:41.502993  5629 sgd_solver.cpp:106] Iteration 8000, lr = 0.000450447
    I1227 18:36:48.482734  5629 solver.cpp:237] Iteration 8100, loss = 1.28966
    I1227 18:36:48.482784  5629 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1227 18:36:48.482803  5629 solver.cpp:253]     Train net output #1: loss = 1.28966 (* 1 = 1.28966 loss)
    I1227 18:36:48.482816  5629 sgd_solver.cpp:106] Iteration 8100, lr = 0.000448579
    I1227 18:36:55.449787  5629 solver.cpp:237] Iteration 8200, loss = 0.938865
    I1227 18:36:55.449836  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:36:55.449856  5629 solver.cpp:253]     Train net output #1: loss = 0.938865 (* 1 = 0.938865 loss)
    I1227 18:36:55.449869  5629 sgd_solver.cpp:106] Iteration 8200, lr = 0.000446729
    I1227 18:37:02.415150  5629 solver.cpp:237] Iteration 8300, loss = 1.10277
    I1227 18:37:02.415194  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:37:02.415208  5629 solver.cpp:253]     Train net output #1: loss = 1.10277 (* 1 = 1.10277 loss)
    I1227 18:37:02.415220  5629 sgd_solver.cpp:106] Iteration 8300, lr = 0.000444897
    I1227 18:37:09.376998  5629 solver.cpp:237] Iteration 8400, loss = 1.16051
    I1227 18:37:09.377122  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:37:09.377142  5629 solver.cpp:253]     Train net output #1: loss = 1.16051 (* 1 = 1.16051 loss)
    I1227 18:37:09.377153  5629 sgd_solver.cpp:106] Iteration 8400, lr = 0.000443083
    I1227 18:37:16.328279  5629 solver.cpp:237] Iteration 8500, loss = 1.08278
    I1227 18:37:16.328317  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:37:16.328332  5629 solver.cpp:253]     Train net output #1: loss = 1.08278 (* 1 = 1.08278 loss)
    I1227 18:37:16.328344  5629 sgd_solver.cpp:106] Iteration 8500, lr = 0.000441285
    I1227 18:37:23.296051  5629 solver.cpp:237] Iteration 8600, loss = 1.35329
    I1227 18:37:23.296092  5629 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1227 18:37:23.296106  5629 solver.cpp:253]     Train net output #1: loss = 1.35329 (* 1 = 1.35329 loss)
    I1227 18:37:23.296116  5629 sgd_solver.cpp:106] Iteration 8600, lr = 0.000439505
    I1227 18:37:30.267453  5629 solver.cpp:237] Iteration 8700, loss = 1.06788
    I1227 18:37:30.267495  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:37:30.267510  5629 solver.cpp:253]     Train net output #1: loss = 1.06788 (* 1 = 1.06788 loss)
    I1227 18:37:30.267520  5629 sgd_solver.cpp:106] Iteration 8700, lr = 0.000437741
    I1227 18:37:37.215622  5629 solver.cpp:237] Iteration 8800, loss = 1.09732
    I1227 18:37:37.215663  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:37:37.215678  5629 solver.cpp:253]     Train net output #1: loss = 1.09732 (* 1 = 1.09732 loss)
    I1227 18:37:37.215688  5629 sgd_solver.cpp:106] Iteration 8800, lr = 0.000435993
    I1227 18:37:44.195631  5629 solver.cpp:237] Iteration 8900, loss = 1.06377
    I1227 18:37:44.195755  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:37:44.195776  5629 solver.cpp:253]     Train net output #1: loss = 1.06377 (* 1 = 1.06377 loss)
    I1227 18:37:44.195791  5629 sgd_solver.cpp:106] Iteration 8900, lr = 0.000434262
    I1227 18:37:51.088595  5629 solver.cpp:341] Iteration 9000, Testing net (#0)
    I1227 18:37:54.185873  5629 solver.cpp:409]     Test net output #0: accuracy = 0.600417
    I1227 18:37:54.185919  5629 solver.cpp:409]     Test net output #1: loss = 1.13429 (* 1 = 1.13429 loss)
    I1227 18:37:54.216138  5629 solver.cpp:237] Iteration 9000, loss = 1.04644
    I1227 18:37:54.216192  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:37:54.216208  5629 solver.cpp:253]     Train net output #1: loss = 1.04644 (* 1 = 1.04644 loss)
    I1227 18:37:54.216222  5629 sgd_solver.cpp:106] Iteration 9000, lr = 0.000432547
    I1227 18:38:01.217763  5629 solver.cpp:237] Iteration 9100, loss = 1.24285
    I1227 18:38:01.217805  5629 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1227 18:38:01.217820  5629 solver.cpp:253]     Train net output #1: loss = 1.24285 (* 1 = 1.24285 loss)
    I1227 18:38:01.217830  5629 sgd_solver.cpp:106] Iteration 9100, lr = 0.000430847
    I1227 18:38:08.216835  5629 solver.cpp:237] Iteration 9200, loss = 0.995237
    I1227 18:38:08.216876  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:38:08.216891  5629 solver.cpp:253]     Train net output #1: loss = 0.995237 (* 1 = 0.995237 loss)
    I1227 18:38:08.216902  5629 sgd_solver.cpp:106] Iteration 9200, lr = 0.000429163
    I1227 18:38:15.169459  5629 solver.cpp:237] Iteration 9300, loss = 1.03279
    I1227 18:38:15.169602  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:38:15.169625  5629 solver.cpp:253]     Train net output #1: loss = 1.03279 (* 1 = 1.03279 loss)
    I1227 18:38:15.169641  5629 sgd_solver.cpp:106] Iteration 9300, lr = 0.000427494
    I1227 18:38:22.128520  5629 solver.cpp:237] Iteration 9400, loss = 1.08769
    I1227 18:38:22.128562  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:38:22.128581  5629 solver.cpp:253]     Train net output #1: loss = 1.08769 (* 1 = 1.08769 loss)
    I1227 18:38:22.128592  5629 sgd_solver.cpp:106] Iteration 9400, lr = 0.00042584
    I1227 18:38:29.083577  5629 solver.cpp:237] Iteration 9500, loss = 1.079
    I1227 18:38:29.083619  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:38:29.083633  5629 solver.cpp:253]     Train net output #1: loss = 1.079 (* 1 = 1.079 loss)
    I1227 18:38:29.083645  5629 sgd_solver.cpp:106] Iteration 9500, lr = 0.000424201
    I1227 18:38:36.043649  5629 solver.cpp:237] Iteration 9600, loss = 1.33365
    I1227 18:38:36.043697  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:38:36.043719  5629 solver.cpp:253]     Train net output #1: loss = 1.33365 (* 1 = 1.33365 loss)
    I1227 18:38:36.043732  5629 sgd_solver.cpp:106] Iteration 9600, lr = 0.000422577
    I1227 18:38:43.019053  5629 solver.cpp:237] Iteration 9700, loss = 0.933621
    I1227 18:38:43.019103  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:38:43.019121  5629 solver.cpp:253]     Train net output #1: loss = 0.933621 (* 1 = 0.933621 loss)
    I1227 18:38:43.019136  5629 sgd_solver.cpp:106] Iteration 9700, lr = 0.000420967
    I1227 18:38:49.969519  5629 solver.cpp:237] Iteration 9800, loss = 1.01492
    I1227 18:38:49.969631  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:38:49.969653  5629 solver.cpp:253]     Train net output #1: loss = 1.01492 (* 1 = 1.01492 loss)
    I1227 18:38:49.969667  5629 sgd_solver.cpp:106] Iteration 9800, lr = 0.000419372
    I1227 18:38:56.929126  5629 solver.cpp:237] Iteration 9900, loss = 1.02381
    I1227 18:38:56.929167  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:38:56.929183  5629 solver.cpp:253]     Train net output #1: loss = 1.02381 (* 1 = 1.02381 loss)
    I1227 18:38:56.929193  5629 sgd_solver.cpp:106] Iteration 9900, lr = 0.00041779
    I1227 18:39:03.855936  5629 solver.cpp:341] Iteration 10000, Testing net (#0)
    I1227 18:39:06.654340  5629 solver.cpp:409]     Test net output #0: accuracy = 0.642167
    I1227 18:39:06.654386  5629 solver.cpp:409]     Test net output #1: loss = 1.02581 (* 1 = 1.02581 loss)
    I1227 18:39:06.687206  5629 solver.cpp:237] Iteration 10000, loss = 1.03063
    I1227 18:39:06.687249  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:39:06.687263  5629 solver.cpp:253]     Train net output #1: loss = 1.03063 (* 1 = 1.03063 loss)
    I1227 18:39:06.687275  5629 sgd_solver.cpp:106] Iteration 10000, lr = 0.000416222
    I1227 18:39:13.638186  5629 solver.cpp:237] Iteration 10100, loss = 1.19483
    I1227 18:39:13.638228  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:39:13.638243  5629 solver.cpp:253]     Train net output #1: loss = 1.19483 (* 1 = 1.19483 loss)
    I1227 18:39:13.638254  5629 sgd_solver.cpp:106] Iteration 10100, lr = 0.000414668
    I1227 18:39:20.587836  5629 solver.cpp:237] Iteration 10200, loss = 0.934763
    I1227 18:39:20.587977  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:39:20.587999  5629 solver.cpp:253]     Train net output #1: loss = 0.934763 (* 1 = 0.934763 loss)
    I1227 18:39:20.588014  5629 sgd_solver.cpp:106] Iteration 10200, lr = 0.000413128
    I1227 18:39:27.558254  5629 solver.cpp:237] Iteration 10300, loss = 1.03538
    I1227 18:39:27.558303  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:39:27.558323  5629 solver.cpp:253]     Train net output #1: loss = 1.03538 (* 1 = 1.03538 loss)
    I1227 18:39:27.558337  5629 sgd_solver.cpp:106] Iteration 10300, lr = 0.000411601
    I1227 18:39:34.514817  5629 solver.cpp:237] Iteration 10400, loss = 0.932655
    I1227 18:39:34.514859  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:39:34.514873  5629 solver.cpp:253]     Train net output #1: loss = 0.932655 (* 1 = 0.932655 loss)
    I1227 18:39:34.514884  5629 sgd_solver.cpp:106] Iteration 10400, lr = 0.000410086
    I1227 18:39:41.474575  5629 solver.cpp:237] Iteration 10500, loss = 0.979446
    I1227 18:39:41.474618  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:39:41.474633  5629 solver.cpp:253]     Train net output #1: loss = 0.979446 (* 1 = 0.979446 loss)
    I1227 18:39:41.474644  5629 sgd_solver.cpp:106] Iteration 10500, lr = 0.000408585
    I1227 18:39:48.448271  5629 solver.cpp:237] Iteration 10600, loss = 1.2766
    I1227 18:39:48.448320  5629 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1227 18:39:48.448340  5629 solver.cpp:253]     Train net output #1: loss = 1.2766 (* 1 = 1.2766 loss)
    I1227 18:39:48.448354  5629 sgd_solver.cpp:106] Iteration 10600, lr = 0.000407097
    I1227 18:39:55.417829  5629 solver.cpp:237] Iteration 10700, loss = 0.936306
    I1227 18:39:55.417953  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:39:55.417973  5629 solver.cpp:253]     Train net output #1: loss = 0.936306 (* 1 = 0.936306 loss)
    I1227 18:39:55.417987  5629 sgd_solver.cpp:106] Iteration 10700, lr = 0.000405621
    I1227 18:40:02.396246  5629 solver.cpp:237] Iteration 10800, loss = 0.987383
    I1227 18:40:02.396288  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:40:02.396303  5629 solver.cpp:253]     Train net output #1: loss = 0.987383 (* 1 = 0.987383 loss)
    I1227 18:40:02.396313  5629 sgd_solver.cpp:106] Iteration 10800, lr = 0.000404157
    I1227 18:40:09.359927  5629 solver.cpp:237] Iteration 10900, loss = 1.01774
    I1227 18:40:09.359968  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:40:09.359983  5629 solver.cpp:253]     Train net output #1: loss = 1.01774 (* 1 = 1.01774 loss)
    I1227 18:40:09.359993  5629 sgd_solver.cpp:106] Iteration 10900, lr = 0.000402706
    I1227 18:40:16.257320  5629 solver.cpp:341] Iteration 11000, Testing net (#0)
    I1227 18:40:19.056452  5629 solver.cpp:409]     Test net output #0: accuracy = 0.64375
    I1227 18:40:19.056506  5629 solver.cpp:409]     Test net output #1: loss = 1.01376 (* 1 = 1.01376 loss)
    I1227 18:40:19.089843  5629 solver.cpp:237] Iteration 11000, loss = 0.888766
    I1227 18:40:19.089895  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:40:19.089915  5629 solver.cpp:253]     Train net output #1: loss = 0.888766 (* 1 = 0.888766 loss)
    I1227 18:40:19.089932  5629 sgd_solver.cpp:106] Iteration 11000, lr = 0.000401267
    I1227 18:40:26.036821  5629 solver.cpp:237] Iteration 11100, loss = 1.11712
    I1227 18:40:26.036947  5629 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1227 18:40:26.036963  5629 solver.cpp:253]     Train net output #1: loss = 1.11712 (* 1 = 1.11712 loss)
    I1227 18:40:26.036974  5629 sgd_solver.cpp:106] Iteration 11100, lr = 0.00039984
    I1227 18:40:33.238677  5629 solver.cpp:237] Iteration 11200, loss = 0.930446
    I1227 18:40:33.238719  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:40:33.238735  5629 solver.cpp:253]     Train net output #1: loss = 0.930446 (* 1 = 0.930446 loss)
    I1227 18:40:33.238745  5629 sgd_solver.cpp:106] Iteration 11200, lr = 0.000398425
    I1227 18:40:40.217350  5629 solver.cpp:237] Iteration 11300, loss = 1.00747
    I1227 18:40:40.217397  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:40:40.217417  5629 solver.cpp:253]     Train net output #1: loss = 1.00747 (* 1 = 1.00747 loss)
    I1227 18:40:40.217432  5629 sgd_solver.cpp:106] Iteration 11300, lr = 0.000397021
    I1227 18:40:47.459203  5629 solver.cpp:237] Iteration 11400, loss = 1.03142
    I1227 18:40:47.459251  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:40:47.459271  5629 solver.cpp:253]     Train net output #1: loss = 1.03142 (* 1 = 1.03142 loss)
    I1227 18:40:47.459285  5629 sgd_solver.cpp:106] Iteration 11400, lr = 0.000395629
    I1227 18:40:54.431825  5629 solver.cpp:237] Iteration 11500, loss = 0.957307
    I1227 18:40:54.431874  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:40:54.431891  5629 solver.cpp:253]     Train net output #1: loss = 0.957307 (* 1 = 0.957307 loss)
    I1227 18:40:54.431903  5629 sgd_solver.cpp:106] Iteration 11500, lr = 0.000394248
    I1227 18:41:03.210820  5629 solver.cpp:237] Iteration 11600, loss = 1.2938
    I1227 18:41:03.210918  5629 solver.cpp:253]     Train net output #0: accuracy = 0.5
    I1227 18:41:03.210937  5629 solver.cpp:253]     Train net output #1: loss = 1.2938 (* 1 = 1.2938 loss)
    I1227 18:41:03.210948  5629 sgd_solver.cpp:106] Iteration 11600, lr = 0.000392878
    I1227 18:41:12.500675  5629 solver.cpp:237] Iteration 11700, loss = 0.926621
    I1227 18:41:12.500722  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:41:12.500738  5629 solver.cpp:253]     Train net output #1: loss = 0.926621 (* 1 = 0.926621 loss)
    I1227 18:41:12.500751  5629 sgd_solver.cpp:106] Iteration 11700, lr = 0.000391519
    I1227 18:41:19.574278  5629 solver.cpp:237] Iteration 11800, loss = 0.945665
    I1227 18:41:19.574321  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:41:19.574337  5629 solver.cpp:253]     Train net output #1: loss = 0.945665 (* 1 = 0.945665 loss)
    I1227 18:41:19.574347  5629 sgd_solver.cpp:106] Iteration 11800, lr = 0.000390172
    I1227 18:41:26.774370  5629 solver.cpp:237] Iteration 11900, loss = 1.12068
    I1227 18:41:26.774421  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:41:26.774441  5629 solver.cpp:253]     Train net output #1: loss = 1.12068 (* 1 = 1.12068 loss)
    I1227 18:41:26.774456  5629 sgd_solver.cpp:106] Iteration 11900, lr = 0.000388835
    I1227 18:41:34.533378  5629 solver.cpp:341] Iteration 12000, Testing net (#0)
    I1227 18:41:37.361681  5629 solver.cpp:409]     Test net output #0: accuracy = 0.650917
    I1227 18:41:37.361727  5629 solver.cpp:409]     Test net output #1: loss = 0.993622 (* 1 = 0.993622 loss)
    I1227 18:41:37.391893  5629 solver.cpp:237] Iteration 12000, loss = 0.924779
    I1227 18:41:37.391932  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:41:37.391945  5629 solver.cpp:253]     Train net output #1: loss = 0.924779 (* 1 = 0.924779 loss)
    I1227 18:41:37.391957  5629 sgd_solver.cpp:106] Iteration 12000, lr = 0.000387508
    I1227 18:41:45.505365  5629 solver.cpp:237] Iteration 12100, loss = 1.1112
    I1227 18:41:45.505412  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:41:45.505427  5629 solver.cpp:253]     Train net output #1: loss = 1.1112 (* 1 = 1.1112 loss)
    I1227 18:41:45.505439  5629 sgd_solver.cpp:106] Iteration 12100, lr = 0.000386192
    I1227 18:41:54.132931  5629 solver.cpp:237] Iteration 12200, loss = 0.926275
    I1227 18:41:54.132988  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:41:54.133008  5629 solver.cpp:253]     Train net output #1: loss = 0.926275 (* 1 = 0.926275 loss)
    I1227 18:41:54.133020  5629 sgd_solver.cpp:106] Iteration 12200, lr = 0.000384887
    I1227 18:42:01.656119  5629 solver.cpp:237] Iteration 12300, loss = 0.88426
    I1227 18:42:01.656165  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:42:01.656182  5629 solver.cpp:253]     Train net output #1: loss = 0.88426 (* 1 = 0.88426 loss)
    I1227 18:42:01.656193  5629 sgd_solver.cpp:106] Iteration 12300, lr = 0.000383592
    I1227 18:42:08.625671  5629 solver.cpp:237] Iteration 12400, loss = 0.972771
    I1227 18:42:08.625800  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:42:08.625819  5629 solver.cpp:253]     Train net output #1: loss = 0.972771 (* 1 = 0.972771 loss)
    I1227 18:42:08.625829  5629 sgd_solver.cpp:106] Iteration 12400, lr = 0.000382307
    I1227 18:42:15.743618  5629 solver.cpp:237] Iteration 12500, loss = 0.901366
    I1227 18:42:15.743661  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:42:15.743676  5629 solver.cpp:253]     Train net output #1: loss = 0.901366 (* 1 = 0.901366 loss)
    I1227 18:42:15.743687  5629 sgd_solver.cpp:106] Iteration 12500, lr = 0.000381032
    I1227 18:42:22.702119  5629 solver.cpp:237] Iteration 12600, loss = 1.07301
    I1227 18:42:22.702164  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:42:22.702180  5629 solver.cpp:253]     Train net output #1: loss = 1.07301 (* 1 = 1.07301 loss)
    I1227 18:42:22.702193  5629 sgd_solver.cpp:106] Iteration 12600, lr = 0.000379767
    I1227 18:42:29.651306  5629 solver.cpp:237] Iteration 12700, loss = 0.917217
    I1227 18:42:29.651350  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:42:29.651365  5629 solver.cpp:253]     Train net output #1: loss = 0.917217 (* 1 = 0.917217 loss)
    I1227 18:42:29.651376  5629 sgd_solver.cpp:106] Iteration 12700, lr = 0.000378511
    I1227 18:42:36.588861  5629 solver.cpp:237] Iteration 12800, loss = 0.818937
    I1227 18:42:36.588906  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:42:36.588920  5629 solver.cpp:253]     Train net output #1: loss = 0.818937 (* 1 = 0.818937 loss)
    I1227 18:42:36.588932  5629 sgd_solver.cpp:106] Iteration 12800, lr = 0.000377265
    I1227 18:42:43.538363  5629 solver.cpp:237] Iteration 12900, loss = 1.01924
    I1227 18:42:43.538476  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:42:43.538494  5629 solver.cpp:253]     Train net output #1: loss = 1.01924 (* 1 = 1.01924 loss)
    I1227 18:42:43.538506  5629 sgd_solver.cpp:106] Iteration 12900, lr = 0.000376029
    I1227 18:42:51.461266  5629 solver.cpp:341] Iteration 13000, Testing net (#0)
    I1227 18:42:54.248319  5629 solver.cpp:409]     Test net output #0: accuracy = 0.6515
    I1227 18:42:54.248366  5629 solver.cpp:409]     Test net output #1: loss = 0.975946 (* 1 = 0.975946 loss)
    I1227 18:42:54.278543  5629 solver.cpp:237] Iteration 13000, loss = 0.832988
    I1227 18:42:54.278566  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:42:54.278579  5629 solver.cpp:253]     Train net output #1: loss = 0.832988 (* 1 = 0.832988 loss)
    I1227 18:42:54.278591  5629 sgd_solver.cpp:106] Iteration 13000, lr = 0.000374802
    I1227 18:43:01.376619  5629 solver.cpp:237] Iteration 13100, loss = 1.1786
    I1227 18:43:01.376662  5629 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1227 18:43:01.376677  5629 solver.cpp:253]     Train net output #1: loss = 1.1786 (* 1 = 1.1786 loss)
    I1227 18:43:01.376688  5629 sgd_solver.cpp:106] Iteration 13100, lr = 0.000373585
    I1227 18:43:10.650336  5629 solver.cpp:237] Iteration 13200, loss = 0.928635
    I1227 18:43:10.650382  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:43:10.650398  5629 solver.cpp:253]     Train net output #1: loss = 0.928635 (* 1 = 0.928635 loss)
    I1227 18:43:10.650409  5629 sgd_solver.cpp:106] Iteration 13200, lr = 0.000372376
    I1227 18:43:18.403473  5629 solver.cpp:237] Iteration 13300, loss = 1.09675
    I1227 18:43:18.403615  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:43:18.403631  5629 solver.cpp:253]     Train net output #1: loss = 1.09675 (* 1 = 1.09675 loss)
    I1227 18:43:18.403641  5629 sgd_solver.cpp:106] Iteration 13300, lr = 0.000371177
    I1227 18:43:26.180177  5629 solver.cpp:237] Iteration 13400, loss = 0.956938
    I1227 18:43:26.180232  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:43:26.180246  5629 solver.cpp:253]     Train net output #1: loss = 0.956938 (* 1 = 0.956938 loss)
    I1227 18:43:26.180258  5629 sgd_solver.cpp:106] Iteration 13400, lr = 0.000369987
    I1227 18:43:34.950842  5629 solver.cpp:237] Iteration 13500, loss = 0.876404
    I1227 18:43:34.950893  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:43:34.950907  5629 solver.cpp:253]     Train net output #1: loss = 0.876404 (* 1 = 0.876404 loss)
    I1227 18:43:34.950917  5629 sgd_solver.cpp:106] Iteration 13500, lr = 0.000368805
    I1227 18:43:41.915948  5629 solver.cpp:237] Iteration 13600, loss = 1.13259
    I1227 18:43:41.915992  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:43:41.916007  5629 solver.cpp:253]     Train net output #1: loss = 1.13259 (* 1 = 1.13259 loss)
    I1227 18:43:41.916018  5629 sgd_solver.cpp:106] Iteration 13600, lr = 0.000367633
    I1227 18:43:50.726883  5629 solver.cpp:237] Iteration 13700, loss = 0.93109
    I1227 18:43:50.727058  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:43:50.727088  5629 solver.cpp:253]     Train net output #1: loss = 0.93109 (* 1 = 0.93109 loss)
    I1227 18:43:50.727108  5629 sgd_solver.cpp:106] Iteration 13700, lr = 0.000366469
    I1227 18:43:58.412067  5629 solver.cpp:237] Iteration 13800, loss = 0.967142
    I1227 18:43:58.412111  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:43:58.412125  5629 solver.cpp:253]     Train net output #1: loss = 0.967142 (* 1 = 0.967142 loss)
    I1227 18:43:58.412137  5629 sgd_solver.cpp:106] Iteration 13800, lr = 0.000365313
    I1227 18:44:05.679038  5629 solver.cpp:237] Iteration 13900, loss = 0.96436
    I1227 18:44:05.679087  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:44:05.679100  5629 solver.cpp:253]     Train net output #1: loss = 0.96436 (* 1 = 0.96436 loss)
    I1227 18:44:05.679112  5629 sgd_solver.cpp:106] Iteration 13900, lr = 0.000364166
    I1227 18:44:13.152253  5629 solver.cpp:341] Iteration 14000, Testing net (#0)
    I1227 18:44:15.993198  5629 solver.cpp:409]     Test net output #0: accuracy = 0.663917
    I1227 18:44:15.993242  5629 solver.cpp:409]     Test net output #1: loss = 0.952977 (* 1 = 0.952977 loss)
    I1227 18:44:16.022806  5629 solver.cpp:237] Iteration 14000, loss = 0.881601
    I1227 18:44:16.022855  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:44:16.022868  5629 solver.cpp:253]     Train net output #1: loss = 0.881601 (* 1 = 0.881601 loss)
    I1227 18:44:16.022879  5629 sgd_solver.cpp:106] Iteration 14000, lr = 0.000363028
    I1227 18:44:22.989960  5629 solver.cpp:237] Iteration 14100, loss = 1.11726
    I1227 18:44:22.990068  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:44:22.990095  5629 solver.cpp:253]     Train net output #1: loss = 1.11726 (* 1 = 1.11726 loss)
    I1227 18:44:22.990105  5629 sgd_solver.cpp:106] Iteration 14100, lr = 0.000361897
    I1227 18:44:30.395361  5629 solver.cpp:237] Iteration 14200, loss = 0.954549
    I1227 18:44:30.395416  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:44:30.395437  5629 solver.cpp:253]     Train net output #1: loss = 0.954549 (* 1 = 0.954549 loss)
    I1227 18:44:30.395454  5629 sgd_solver.cpp:106] Iteration 14200, lr = 0.000360775
    I1227 18:44:38.164494  5629 solver.cpp:237] Iteration 14300, loss = 0.829997
    I1227 18:44:38.164542  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:44:38.164556  5629 solver.cpp:253]     Train net output #1: loss = 0.829997 (* 1 = 0.829997 loss)
    I1227 18:44:38.164566  5629 sgd_solver.cpp:106] Iteration 14300, lr = 0.000359661
    I1227 18:44:45.457794  5629 solver.cpp:237] Iteration 14400, loss = 1.0242
    I1227 18:44:45.457849  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:44:45.457870  5629 solver.cpp:253]     Train net output #1: loss = 1.0242 (* 1 = 1.0242 loss)
    I1227 18:44:45.457885  5629 sgd_solver.cpp:106] Iteration 14400, lr = 0.000358555
    I1227 18:44:52.838253  5629 solver.cpp:237] Iteration 14500, loss = 0.894399
    I1227 18:44:52.838294  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:44:52.838307  5629 solver.cpp:253]     Train net output #1: loss = 0.894399 (* 1 = 0.894399 loss)
    I1227 18:44:52.838316  5629 sgd_solver.cpp:106] Iteration 14500, lr = 0.000357457
    I1227 18:45:00.180642  5629 solver.cpp:237] Iteration 14600, loss = 1.12206
    I1227 18:45:00.180809  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:45:00.180836  5629 solver.cpp:253]     Train net output #1: loss = 1.12206 (* 1 = 1.12206 loss)
    I1227 18:45:00.180848  5629 sgd_solver.cpp:106] Iteration 14600, lr = 0.000356366
    I1227 18:45:08.031801  5629 solver.cpp:237] Iteration 14700, loss = 0.895391
    I1227 18:45:08.031842  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:45:08.031857  5629 solver.cpp:253]     Train net output #1: loss = 0.895391 (* 1 = 0.895391 loss)
    I1227 18:45:08.031867  5629 sgd_solver.cpp:106] Iteration 14700, lr = 0.000355284
    I1227 18:45:15.733531  5629 solver.cpp:237] Iteration 14800, loss = 0.803476
    I1227 18:45:15.733599  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:45:15.733616  5629 solver.cpp:253]     Train net output #1: loss = 0.803476 (* 1 = 0.803476 loss)
    I1227 18:45:15.733629  5629 sgd_solver.cpp:106] Iteration 14800, lr = 0.000354209
    I1227 18:45:23.074300  5629 solver.cpp:237] Iteration 14900, loss = 1.0244
    I1227 18:45:23.074342  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:45:23.074355  5629 solver.cpp:253]     Train net output #1: loss = 1.0244 (* 1 = 1.0244 loss)
    I1227 18:45:23.074368  5629 sgd_solver.cpp:106] Iteration 14900, lr = 0.000353141
    I1227 18:45:30.538514  5629 solver.cpp:341] Iteration 15000, Testing net (#0)
    I1227 18:45:33.629137  5629 solver.cpp:409]     Test net output #0: accuracy = 0.677167
    I1227 18:45:33.629189  5629 solver.cpp:409]     Test net output #1: loss = 0.928516 (* 1 = 0.928516 loss)
    I1227 18:45:33.658193  5629 solver.cpp:237] Iteration 15000, loss = 0.857353
    I1227 18:45:33.658237  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:45:33.658251  5629 solver.cpp:253]     Train net output #1: loss = 0.857353 (* 1 = 0.857353 loss)
    I1227 18:45:33.658260  5629 sgd_solver.cpp:106] Iteration 15000, lr = 0.000352081
    I1227 18:45:40.829931  5629 solver.cpp:237] Iteration 15100, loss = 1.16797
    I1227 18:45:40.829988  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:45:40.830009  5629 solver.cpp:253]     Train net output #1: loss = 1.16797 (* 1 = 1.16797 loss)
    I1227 18:45:40.830026  5629 sgd_solver.cpp:106] Iteration 15100, lr = 0.000351029
    I1227 18:45:47.919901  5629 solver.cpp:237] Iteration 15200, loss = 0.873941
    I1227 18:45:47.919941  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:45:47.919955  5629 solver.cpp:253]     Train net output #1: loss = 0.873941 (* 1 = 0.873941 loss)
    I1227 18:45:47.919965  5629 sgd_solver.cpp:106] Iteration 15200, lr = 0.000349984
    I1227 18:45:56.167145  5629 solver.cpp:237] Iteration 15300, loss = 0.896547
    I1227 18:45:56.167186  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:45:56.167199  5629 solver.cpp:253]     Train net output #1: loss = 0.896547 (* 1 = 0.896547 loss)
    I1227 18:45:56.167212  5629 sgd_solver.cpp:106] Iteration 15300, lr = 0.000348946
    I1227 18:46:03.803135  5629 solver.cpp:237] Iteration 15400, loss = 0.955442
    I1227 18:46:03.803256  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:46:03.803272  5629 solver.cpp:253]     Train net output #1: loss = 0.955442 (* 1 = 0.955442 loss)
    I1227 18:46:03.803283  5629 sgd_solver.cpp:106] Iteration 15400, lr = 0.000347915
    I1227 18:46:11.028698  5629 solver.cpp:237] Iteration 15500, loss = 0.852138
    I1227 18:46:11.028764  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:46:11.028787  5629 solver.cpp:253]     Train net output #1: loss = 0.852138 (* 1 = 0.852138 loss)
    I1227 18:46:11.028805  5629 sgd_solver.cpp:106] Iteration 15500, lr = 0.000346891
    I1227 18:46:19.479104  5629 solver.cpp:237] Iteration 15600, loss = 1.00921
    I1227 18:46:19.479146  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:46:19.479161  5629 solver.cpp:253]     Train net output #1: loss = 1.00921 (* 1 = 1.00921 loss)
    I1227 18:46:19.479171  5629 sgd_solver.cpp:106] Iteration 15600, lr = 0.000345874
    I1227 18:46:26.643591  5629 solver.cpp:237] Iteration 15700, loss = 0.891894
    I1227 18:46:26.643642  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:46:26.643661  5629 solver.cpp:253]     Train net output #1: loss = 0.891894 (* 1 = 0.891894 loss)
    I1227 18:46:26.643676  5629 sgd_solver.cpp:106] Iteration 15700, lr = 0.000344864
    I1227 18:46:35.271893  5629 solver.cpp:237] Iteration 15800, loss = 0.810596
    I1227 18:46:35.272012  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:46:35.272029  5629 solver.cpp:253]     Train net output #1: loss = 0.810596 (* 1 = 0.810596 loss)
    I1227 18:46:35.272040  5629 sgd_solver.cpp:106] Iteration 15800, lr = 0.000343861
    I1227 18:46:43.593085  5629 solver.cpp:237] Iteration 15900, loss = 1.00563
    I1227 18:46:43.593153  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:46:43.593178  5629 solver.cpp:253]     Train net output #1: loss = 1.00563 (* 1 = 1.00563 loss)
    I1227 18:46:43.593199  5629 sgd_solver.cpp:106] Iteration 15900, lr = 0.000342865
    I1227 18:46:51.359638  5629 solver.cpp:341] Iteration 16000, Testing net (#0)
    I1227 18:46:54.650388  5629 solver.cpp:409]     Test net output #0: accuracy = 0.667
    I1227 18:46:54.650439  5629 solver.cpp:409]     Test net output #1: loss = 0.945004 (* 1 = 0.945004 loss)
    I1227 18:46:54.684371  5629 solver.cpp:237] Iteration 16000, loss = 0.852168
    I1227 18:46:54.684414  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:46:54.684427  5629 solver.cpp:253]     Train net output #1: loss = 0.852168 (* 1 = 0.852168 loss)
    I1227 18:46:54.684438  5629 sgd_solver.cpp:106] Iteration 16000, lr = 0.000341876
    I1227 18:47:02.532807  5629 solver.cpp:237] Iteration 16100, loss = 1.02771
    I1227 18:47:02.532866  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:47:02.532887  5629 solver.cpp:253]     Train net output #1: loss = 1.02771 (* 1 = 1.02771 loss)
    I1227 18:47:02.532905  5629 sgd_solver.cpp:106] Iteration 16100, lr = 0.000340893
    I1227 18:47:10.326943  5629 solver.cpp:237] Iteration 16200, loss = 0.821154
    I1227 18:47:10.327085  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:47:10.327102  5629 solver.cpp:253]     Train net output #1: loss = 0.821154 (* 1 = 0.821154 loss)
    I1227 18:47:10.327116  5629 sgd_solver.cpp:106] Iteration 16200, lr = 0.000339916
    I1227 18:47:18.205072  5629 solver.cpp:237] Iteration 16300, loss = 0.792728
    I1227 18:47:18.205134  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:47:18.205157  5629 solver.cpp:253]     Train net output #1: loss = 0.792728 (* 1 = 0.792728 loss)
    I1227 18:47:18.205175  5629 sgd_solver.cpp:106] Iteration 16300, lr = 0.000338947
    I1227 18:47:25.992365  5629 solver.cpp:237] Iteration 16400, loss = 1.00245
    I1227 18:47:25.992525  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:47:25.992596  5629 solver.cpp:253]     Train net output #1: loss = 1.00245 (* 1 = 1.00245 loss)
    I1227 18:47:25.992647  5629 sgd_solver.cpp:106] Iteration 16400, lr = 0.000337983
    I1227 18:47:33.839244  5629 solver.cpp:237] Iteration 16500, loss = 0.887327
    I1227 18:47:33.839298  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:47:33.839314  5629 solver.cpp:253]     Train net output #1: loss = 0.887327 (* 1 = 0.887327 loss)
    I1227 18:47:33.839328  5629 sgd_solver.cpp:106] Iteration 16500, lr = 0.000337026
    I1227 18:47:41.820371  5629 solver.cpp:237] Iteration 16600, loss = 0.988782
    I1227 18:47:41.821005  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:47:41.821040  5629 solver.cpp:253]     Train net output #1: loss = 0.988782 (* 1 = 0.988782 loss)
    I1227 18:47:41.821056  5629 sgd_solver.cpp:106] Iteration 16600, lr = 0.000336075
    I1227 18:47:49.702715  5629 solver.cpp:237] Iteration 16700, loss = 0.8081
    I1227 18:47:49.702787  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:47:49.702813  5629 solver.cpp:253]     Train net output #1: loss = 0.8081 (* 1 = 0.8081 loss)
    I1227 18:47:49.702834  5629 sgd_solver.cpp:106] Iteration 16700, lr = 0.000335131
    I1227 18:47:57.575037  5629 solver.cpp:237] Iteration 16800, loss = 0.856198
    I1227 18:47:57.575114  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:47:57.575160  5629 solver.cpp:253]     Train net output #1: loss = 0.856198 (* 1 = 0.856198 loss)
    I1227 18:47:57.575184  5629 sgd_solver.cpp:106] Iteration 16800, lr = 0.000334193
    I1227 18:48:05.442597  5629 solver.cpp:237] Iteration 16900, loss = 0.997125
    I1227 18:48:05.442674  5629 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1227 18:48:05.442703  5629 solver.cpp:253]     Train net output #1: loss = 0.997125 (* 1 = 0.997125 loss)
    I1227 18:48:05.442724  5629 sgd_solver.cpp:106] Iteration 16900, lr = 0.00033326
    I1227 18:48:13.220368  5629 solver.cpp:341] Iteration 17000, Testing net (#0)
    I1227 18:48:16.911269  5629 solver.cpp:409]     Test net output #0: accuracy = 0.682833
    I1227 18:48:16.911353  5629 solver.cpp:409]     Test net output #1: loss = 0.90126 (* 1 = 0.90126 loss)
    I1227 18:48:16.961477  5629 solver.cpp:237] Iteration 17000, loss = 0.820852
    I1227 18:48:16.961573  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:48:16.961618  5629 solver.cpp:253]     Train net output #1: loss = 0.820852 (* 1 = 0.820852 loss)
    I1227 18:48:16.961650  5629 sgd_solver.cpp:106] Iteration 17000, lr = 0.000332334
    I1227 18:48:24.804811  5629 solver.cpp:237] Iteration 17100, loss = 1.05076
    I1227 18:48:24.804883  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:48:24.804924  5629 solver.cpp:253]     Train net output #1: loss = 1.05076 (* 1 = 1.05076 loss)
    I1227 18:48:24.804954  5629 sgd_solver.cpp:106] Iteration 17100, lr = 0.000331414
    I1227 18:48:32.673702  5629 solver.cpp:237] Iteration 17200, loss = 0.837579
    I1227 18:48:32.673799  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:48:32.673840  5629 solver.cpp:253]     Train net output #1: loss = 0.837579 (* 1 = 0.837579 loss)
    I1227 18:48:32.673872  5629 sgd_solver.cpp:106] Iteration 17200, lr = 0.0003305
    I1227 18:48:40.541553  5629 solver.cpp:237] Iteration 17300, loss = 0.805036
    I1227 18:48:40.541630  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:48:40.541657  5629 solver.cpp:253]     Train net output #1: loss = 0.805036 (* 1 = 0.805036 loss)
    I1227 18:48:40.541678  5629 sgd_solver.cpp:106] Iteration 17300, lr = 0.000329592
    I1227 18:48:48.406734  5629 solver.cpp:237] Iteration 17400, loss = 0.902094
    I1227 18:48:48.406941  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:48:48.406977  5629 solver.cpp:253]     Train net output #1: loss = 0.902094 (* 1 = 0.902094 loss)
    I1227 18:48:48.406997  5629 sgd_solver.cpp:106] Iteration 17400, lr = 0.000328689
    I1227 18:48:56.269388  5629 solver.cpp:237] Iteration 17500, loss = 0.796992
    I1227 18:48:56.269462  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:48:56.269490  5629 solver.cpp:253]     Train net output #1: loss = 0.796992 (* 1 = 0.796992 loss)
    I1227 18:48:56.269510  5629 sgd_solver.cpp:106] Iteration 17500, lr = 0.000327792
    I1227 18:49:04.128269  5629 solver.cpp:237] Iteration 17600, loss = 1.0093
    I1227 18:49:04.128341  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:49:04.128367  5629 solver.cpp:253]     Train net output #1: loss = 1.0093 (* 1 = 1.0093 loss)
    I1227 18:49:04.128387  5629 sgd_solver.cpp:106] Iteration 17600, lr = 0.000326901
    I1227 18:49:11.981001  5629 solver.cpp:237] Iteration 17700, loss = 0.814715
    I1227 18:49:11.981073  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:49:11.981098  5629 solver.cpp:253]     Train net output #1: loss = 0.814715 (* 1 = 0.814715 loss)
    I1227 18:49:11.981119  5629 sgd_solver.cpp:106] Iteration 17700, lr = 0.000326015
    I1227 18:49:19.840443  5629 solver.cpp:237] Iteration 17800, loss = 0.70723
    I1227 18:49:19.840723  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:49:19.840757  5629 solver.cpp:253]     Train net output #1: loss = 0.70723 (* 1 = 0.70723 loss)
    I1227 18:49:19.840778  5629 sgd_solver.cpp:106] Iteration 17800, lr = 0.000325136
    I1227 18:49:27.715914  5629 solver.cpp:237] Iteration 17900, loss = 0.831553
    I1227 18:49:27.715993  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:49:27.716019  5629 solver.cpp:253]     Train net output #1: loss = 0.831553 (* 1 = 0.831553 loss)
    I1227 18:49:27.716040  5629 sgd_solver.cpp:106] Iteration 17900, lr = 0.000324261
    I1227 18:49:35.504966  5629 solver.cpp:341] Iteration 18000, Testing net (#0)
    I1227 18:49:39.267355  5629 solver.cpp:409]     Test net output #0: accuracy = 0.685583
    I1227 18:49:39.267432  5629 solver.cpp:409]     Test net output #1: loss = 0.897387 (* 1 = 0.897387 loss)
    I1227 18:49:39.312459  5629 solver.cpp:237] Iteration 18000, loss = 0.811352
    I1227 18:49:39.312522  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:49:39.312547  5629 solver.cpp:253]     Train net output #1: loss = 0.811352 (* 1 = 0.811352 loss)
    I1227 18:49:39.312567  5629 sgd_solver.cpp:106] Iteration 18000, lr = 0.000323392
    I1227 18:49:47.183666  5629 solver.cpp:237] Iteration 18100, loss = 1.04637
    I1227 18:49:47.183748  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:49:47.183775  5629 solver.cpp:253]     Train net output #1: loss = 1.04637 (* 1 = 1.04637 loss)
    I1227 18:49:47.183796  5629 sgd_solver.cpp:106] Iteration 18100, lr = 0.000322529
    I1227 18:49:55.041549  5629 solver.cpp:237] Iteration 18200, loss = 0.847157
    I1227 18:49:55.041751  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:49:55.041790  5629 solver.cpp:253]     Train net output #1: loss = 0.847157 (* 1 = 0.847157 loss)
    I1227 18:49:55.041811  5629 sgd_solver.cpp:106] Iteration 18200, lr = 0.00032167
    I1227 18:50:02.916599  5629 solver.cpp:237] Iteration 18300, loss = 0.76008
    I1227 18:50:02.916669  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:50:02.916695  5629 solver.cpp:253]     Train net output #1: loss = 0.76008 (* 1 = 0.76008 loss)
    I1227 18:50:02.916717  5629 sgd_solver.cpp:106] Iteration 18300, lr = 0.000320818
    I1227 18:50:10.783969  5629 solver.cpp:237] Iteration 18400, loss = 0.932913
    I1227 18:50:10.784044  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:50:10.784073  5629 solver.cpp:253]     Train net output #1: loss = 0.932913 (* 1 = 0.932913 loss)
    I1227 18:50:10.784096  5629 sgd_solver.cpp:106] Iteration 18400, lr = 0.00031997
    I1227 18:50:18.648885  5629 solver.cpp:237] Iteration 18500, loss = 0.728996
    I1227 18:50:18.648960  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:50:18.648988  5629 solver.cpp:253]     Train net output #1: loss = 0.728996 (* 1 = 0.728996 loss)
    I1227 18:50:18.649009  5629 sgd_solver.cpp:106] Iteration 18500, lr = 0.000319128
    I1227 18:50:26.491725  5629 solver.cpp:237] Iteration 18600, loss = 1.16658
    I1227 18:50:26.491936  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:50:26.491971  5629 solver.cpp:253]     Train net output #1: loss = 1.16658 (* 1 = 1.16658 loss)
    I1227 18:50:26.491991  5629 sgd_solver.cpp:106] Iteration 18600, lr = 0.00031829
    I1227 18:50:34.341166  5629 solver.cpp:237] Iteration 18700, loss = 0.832228
    I1227 18:50:34.341236  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:50:34.341264  5629 solver.cpp:253]     Train net output #1: loss = 0.832228 (* 1 = 0.832228 loss)
    I1227 18:50:34.341282  5629 sgd_solver.cpp:106] Iteration 18700, lr = 0.000317458
    I1227 18:50:42.221796  5629 solver.cpp:237] Iteration 18800, loss = 0.773119
    I1227 18:50:42.221874  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:50:42.221905  5629 solver.cpp:253]     Train net output #1: loss = 0.773119 (* 1 = 0.773119 loss)
    I1227 18:50:42.221927  5629 sgd_solver.cpp:106] Iteration 18800, lr = 0.000316631
    I1227 18:50:50.080348  5629 solver.cpp:237] Iteration 18900, loss = 0.973576
    I1227 18:50:50.080420  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:50:50.080446  5629 solver.cpp:253]     Train net output #1: loss = 0.973576 (* 1 = 0.973576 loss)
    I1227 18:50:50.080466  5629 sgd_solver.cpp:106] Iteration 18900, lr = 0.000315809
    I1227 18:50:57.864259  5629 solver.cpp:341] Iteration 19000, Testing net (#0)
    I1227 18:51:01.606524  5629 solver.cpp:409]     Test net output #0: accuracy = 0.683083
    I1227 18:51:01.606608  5629 solver.cpp:409]     Test net output #1: loss = 0.901178 (* 1 = 0.901178 loss)
    I1227 18:51:01.651296  5629 solver.cpp:237] Iteration 19000, loss = 0.706231
    I1227 18:51:01.651370  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 18:51:01.651396  5629 solver.cpp:253]     Train net output #1: loss = 0.706231 (* 1 = 0.706231 loss)
    I1227 18:51:01.651417  5629 sgd_solver.cpp:106] Iteration 19000, lr = 0.000314992
    I1227 18:51:09.514832  5629 solver.cpp:237] Iteration 19100, loss = 0.991204
    I1227 18:51:09.514900  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 18:51:09.514926  5629 solver.cpp:253]     Train net output #1: loss = 0.991204 (* 1 = 0.991204 loss)
    I1227 18:51:09.514947  5629 sgd_solver.cpp:106] Iteration 19100, lr = 0.00031418
    I1227 18:51:17.378159  5629 solver.cpp:237] Iteration 19200, loss = 0.778242
    I1227 18:51:17.378229  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:51:17.378255  5629 solver.cpp:253]     Train net output #1: loss = 0.778242 (* 1 = 0.778242 loss)
    I1227 18:51:17.378275  5629 sgd_solver.cpp:106] Iteration 19200, lr = 0.000313372
    I1227 18:51:25.251886  5629 solver.cpp:237] Iteration 19300, loss = 0.788227
    I1227 18:51:25.251956  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:51:25.251982  5629 solver.cpp:253]     Train net output #1: loss = 0.788227 (* 1 = 0.788227 loss)
    I1227 18:51:25.252003  5629 sgd_solver.cpp:106] Iteration 19300, lr = 0.00031257
    I1227 18:51:33.103922  5629 solver.cpp:237] Iteration 19400, loss = 0.897952
    I1227 18:51:33.104123  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:51:33.104156  5629 solver.cpp:253]     Train net output #1: loss = 0.897952 (* 1 = 0.897952 loss)
    I1227 18:51:33.104176  5629 sgd_solver.cpp:106] Iteration 19400, lr = 0.000311772
    I1227 18:51:41.051877  5629 solver.cpp:237] Iteration 19500, loss = 0.935739
    I1227 18:51:41.051949  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:51:41.051976  5629 solver.cpp:253]     Train net output #1: loss = 0.935739 (* 1 = 0.935739 loss)
    I1227 18:51:41.051997  5629 sgd_solver.cpp:106] Iteration 19500, lr = 0.000310979
    I1227 18:51:48.902154  5629 solver.cpp:237] Iteration 19600, loss = 1.0705
    I1227 18:51:48.902228  5629 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1227 18:51:48.902256  5629 solver.cpp:253]     Train net output #1: loss = 1.0705 (* 1 = 1.0705 loss)
    I1227 18:51:48.902277  5629 sgd_solver.cpp:106] Iteration 19600, lr = 0.000310191
    I1227 18:51:56.764399  5629 solver.cpp:237] Iteration 19700, loss = 0.761653
    I1227 18:51:56.764473  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:51:56.764499  5629 solver.cpp:253]     Train net output #1: loss = 0.761653 (* 1 = 0.761653 loss)
    I1227 18:51:56.764520  5629 sgd_solver.cpp:106] Iteration 19700, lr = 0.000309407
    I1227 18:52:04.632437  5629 solver.cpp:237] Iteration 19800, loss = 0.876549
    I1227 18:52:04.632647  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:52:04.632684  5629 solver.cpp:253]     Train net output #1: loss = 0.876549 (* 1 = 0.876549 loss)
    I1227 18:52:04.632704  5629 sgd_solver.cpp:106] Iteration 19800, lr = 0.000308628
    I1227 18:52:12.485978  5629 solver.cpp:237] Iteration 19900, loss = 0.834914
    I1227 18:52:12.486081  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:52:12.486127  5629 solver.cpp:253]     Train net output #1: loss = 0.834914 (* 1 = 0.834914 loss)
    I1227 18:52:12.486160  5629 sgd_solver.cpp:106] Iteration 19900, lr = 0.000307854
    I1227 18:52:20.258631  5629 solver.cpp:341] Iteration 20000, Testing net (#0)
    I1227 18:52:24.031867  5629 solver.cpp:409]     Test net output #0: accuracy = 0.703583
    I1227 18:52:24.031954  5629 solver.cpp:409]     Test net output #1: loss = 0.851834 (* 1 = 0.851834 loss)
    I1227 18:52:24.076011  5629 solver.cpp:237] Iteration 20000, loss = 0.770437
    I1227 18:52:24.076083  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:52:24.076112  5629 solver.cpp:253]     Train net output #1: loss = 0.770437 (* 1 = 0.770437 loss)
    I1227 18:52:24.076133  5629 sgd_solver.cpp:106] Iteration 20000, lr = 0.000307084
    I1227 18:52:31.943733  5629 solver.cpp:237] Iteration 20100, loss = 1.09498
    I1227 18:52:31.943815  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:52:31.943848  5629 solver.cpp:253]     Train net output #1: loss = 1.09498 (* 1 = 1.09498 loss)
    I1227 18:52:31.943871  5629 sgd_solver.cpp:106] Iteration 20100, lr = 0.000306318
    I1227 18:52:39.789737  5629 solver.cpp:237] Iteration 20200, loss = 0.802985
    I1227 18:52:39.789928  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:52:39.789964  5629 solver.cpp:253]     Train net output #1: loss = 0.802985 (* 1 = 0.802985 loss)
    I1227 18:52:39.789986  5629 sgd_solver.cpp:106] Iteration 20200, lr = 0.000305557
    I1227 18:52:47.648272  5629 solver.cpp:237] Iteration 20300, loss = 0.853947
    I1227 18:52:47.648344  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:52:47.648370  5629 solver.cpp:253]     Train net output #1: loss = 0.853947 (* 1 = 0.853947 loss)
    I1227 18:52:47.648389  5629 sgd_solver.cpp:106] Iteration 20300, lr = 0.000304801
    I1227 18:52:55.490362  5629 solver.cpp:237] Iteration 20400, loss = 0.782386
    I1227 18:52:55.490428  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:52:55.490453  5629 solver.cpp:253]     Train net output #1: loss = 0.782386 (* 1 = 0.782386 loss)
    I1227 18:52:55.490473  5629 sgd_solver.cpp:106] Iteration 20400, lr = 0.000304048
    I1227 18:53:03.429615  5629 solver.cpp:237] Iteration 20500, loss = 0.695779
    I1227 18:53:03.429687  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 18:53:03.429713  5629 solver.cpp:253]     Train net output #1: loss = 0.695779 (* 1 = 0.695779 loss)
    I1227 18:53:03.429733  5629 sgd_solver.cpp:106] Iteration 20500, lr = 0.000303301
    I1227 18:53:11.284884  5629 solver.cpp:237] Iteration 20600, loss = 1.0367
    I1227 18:53:11.285087  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:53:11.285122  5629 solver.cpp:253]     Train net output #1: loss = 1.0367 (* 1 = 1.0367 loss)
    I1227 18:53:11.285142  5629 sgd_solver.cpp:106] Iteration 20600, lr = 0.000302557
    I1227 18:53:19.129637  5629 solver.cpp:237] Iteration 20700, loss = 0.708143
    I1227 18:53:19.129726  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:53:19.129768  5629 solver.cpp:253]     Train net output #1: loss = 0.708143 (* 1 = 0.708143 loss)
    I1227 18:53:19.129801  5629 sgd_solver.cpp:106] Iteration 20700, lr = 0.000301817
    I1227 18:53:26.959581  5629 solver.cpp:237] Iteration 20800, loss = 0.754044
    I1227 18:53:26.959658  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 18:53:26.959684  5629 solver.cpp:253]     Train net output #1: loss = 0.754044 (* 1 = 0.754044 loss)
    I1227 18:53:26.959707  5629 sgd_solver.cpp:106] Iteration 20800, lr = 0.000301082
    I1227 18:53:34.816893  5629 solver.cpp:237] Iteration 20900, loss = 0.951982
    I1227 18:53:34.816963  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 18:53:34.816989  5629 solver.cpp:253]     Train net output #1: loss = 0.951982 (* 1 = 0.951982 loss)
    I1227 18:53:34.817009  5629 sgd_solver.cpp:106] Iteration 20900, lr = 0.000300351
    I1227 18:53:42.607355  5629 solver.cpp:341] Iteration 21000, Testing net (#0)
    I1227 18:53:46.693671  5629 solver.cpp:409]     Test net output #0: accuracy = 0.690583
    I1227 18:53:46.693753  5629 solver.cpp:409]     Test net output #1: loss = 0.886683 (* 1 = 0.886683 loss)
    I1227 18:53:46.742966  5629 solver.cpp:237] Iteration 21000, loss = 0.807031
    I1227 18:53:46.743034  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:53:46.743059  5629 solver.cpp:253]     Train net output #1: loss = 0.807031 (* 1 = 0.807031 loss)
    I1227 18:53:46.743080  5629 sgd_solver.cpp:106] Iteration 21000, lr = 0.000299624
    I1227 18:53:57.213845  5629 solver.cpp:237] Iteration 21100, loss = 0.967185
    I1227 18:53:57.213917  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:53:57.213944  5629 solver.cpp:253]     Train net output #1: loss = 0.967185 (* 1 = 0.967185 loss)
    I1227 18:53:57.213963  5629 sgd_solver.cpp:106] Iteration 21100, lr = 0.000298901
    I1227 18:54:07.567242  5629 solver.cpp:237] Iteration 21200, loss = 0.767976
    I1227 18:54:07.567313  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:54:07.567340  5629 solver.cpp:253]     Train net output #1: loss = 0.767976 (* 1 = 0.767976 loss)
    I1227 18:54:07.567359  5629 sgd_solver.cpp:106] Iteration 21200, lr = 0.000298182
    I1227 18:54:16.498245  5629 solver.cpp:237] Iteration 21300, loss = 0.765502
    I1227 18:54:16.498455  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:54:16.498489  5629 solver.cpp:253]     Train net output #1: loss = 0.765502 (* 1 = 0.765502 loss)
    I1227 18:54:16.498510  5629 sgd_solver.cpp:106] Iteration 21300, lr = 0.000297468
    I1227 18:54:24.362597  5629 solver.cpp:237] Iteration 21400, loss = 0.933979
    I1227 18:54:24.362663  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:54:24.362689  5629 solver.cpp:253]     Train net output #1: loss = 0.933979 (* 1 = 0.933979 loss)
    I1227 18:54:24.362709  5629 sgd_solver.cpp:106] Iteration 21400, lr = 0.000296757
    I1227 18:54:32.719655  5629 solver.cpp:237] Iteration 21500, loss = 0.759488
    I1227 18:54:32.719723  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:54:32.719749  5629 solver.cpp:253]     Train net output #1: loss = 0.759488 (* 1 = 0.759488 loss)
    I1227 18:54:32.719770  5629 sgd_solver.cpp:106] Iteration 21500, lr = 0.00029605
    I1227 18:54:42.046691  5629 solver.cpp:237] Iteration 21600, loss = 1.05636
    I1227 18:54:42.046758  5629 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1227 18:54:42.046784  5629 solver.cpp:253]     Train net output #1: loss = 1.05636 (* 1 = 1.05636 loss)
    I1227 18:54:42.046803  5629 sgd_solver.cpp:106] Iteration 21600, lr = 0.000295347
    I1227 18:54:49.952678  5629 solver.cpp:237] Iteration 21700, loss = 0.748932
    I1227 18:54:49.952822  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:54:49.952852  5629 solver.cpp:253]     Train net output #1: loss = 0.748932 (* 1 = 0.748932 loss)
    I1227 18:54:49.952872  5629 sgd_solver.cpp:106] Iteration 21700, lr = 0.000294648
    I1227 18:54:57.810370  5629 solver.cpp:237] Iteration 21800, loss = 0.82182
    I1227 18:54:57.810437  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:54:57.810463  5629 solver.cpp:253]     Train net output #1: loss = 0.82182 (* 1 = 0.82182 loss)
    I1227 18:54:57.810483  5629 sgd_solver.cpp:106] Iteration 21800, lr = 0.000293953
    I1227 18:55:05.700752  5629 solver.cpp:237] Iteration 21900, loss = 0.829833
    I1227 18:55:05.700822  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:55:05.700850  5629 solver.cpp:253]     Train net output #1: loss = 0.829833 (* 1 = 0.829833 loss)
    I1227 18:55:05.700870  5629 sgd_solver.cpp:106] Iteration 21900, lr = 0.000293261
    I1227 18:55:13.509781  5629 solver.cpp:341] Iteration 22000, Testing net (#0)
    I1227 18:55:17.379748  5629 solver.cpp:409]     Test net output #0: accuracy = 0.686917
    I1227 18:55:17.379823  5629 solver.cpp:409]     Test net output #1: loss = 0.892624 (* 1 = 0.892624 loss)
    I1227 18:55:17.424933  5629 solver.cpp:237] Iteration 22000, loss = 0.793183
    I1227 18:55:17.424998  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:55:17.425022  5629 solver.cpp:253]     Train net output #1: loss = 0.793183 (* 1 = 0.793183 loss)
    I1227 18:55:17.425042  5629 sgd_solver.cpp:106] Iteration 22000, lr = 0.000292574
    I1227 18:55:25.292575  5629 solver.cpp:237] Iteration 22100, loss = 0.903642
    I1227 18:55:25.292773  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:55:25.292805  5629 solver.cpp:253]     Train net output #1: loss = 0.903642 (* 1 = 0.903642 loss)
    I1227 18:55:25.292825  5629 sgd_solver.cpp:106] Iteration 22100, lr = 0.00029189
    I1227 18:55:33.176139  5629 solver.cpp:237] Iteration 22200, loss = 0.793815
    I1227 18:55:33.176208  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 18:55:33.176235  5629 solver.cpp:253]     Train net output #1: loss = 0.793815 (* 1 = 0.793815 loss)
    I1227 18:55:33.176255  5629 sgd_solver.cpp:106] Iteration 22200, lr = 0.00029121
    I1227 18:55:41.123267  5629 solver.cpp:237] Iteration 22300, loss = 0.800039
    I1227 18:55:41.123337  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:55:41.123363  5629 solver.cpp:253]     Train net output #1: loss = 0.800039 (* 1 = 0.800039 loss)
    I1227 18:55:41.123383  5629 sgd_solver.cpp:106] Iteration 22300, lr = 0.000290533
    I1227 18:55:49.010798  5629 solver.cpp:237] Iteration 22400, loss = 0.816254
    I1227 18:55:49.010864  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:55:49.010890  5629 solver.cpp:253]     Train net output #1: loss = 0.816254 (* 1 = 0.816254 loss)
    I1227 18:55:49.010910  5629 sgd_solver.cpp:106] Iteration 22400, lr = 0.000289861
    I1227 18:55:56.911514  5629 solver.cpp:237] Iteration 22500, loss = 0.734895
    I1227 18:55:56.911695  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:55:56.911725  5629 solver.cpp:253]     Train net output #1: loss = 0.734895 (* 1 = 0.734895 loss)
    I1227 18:55:56.911746  5629 sgd_solver.cpp:106] Iteration 22500, lr = 0.000289191
    I1227 18:56:04.843905  5629 solver.cpp:237] Iteration 22600, loss = 0.987214
    I1227 18:56:04.843972  5629 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1227 18:56:04.843997  5629 solver.cpp:253]     Train net output #1: loss = 0.987214 (* 1 = 0.987214 loss)
    I1227 18:56:04.844017  5629 sgd_solver.cpp:106] Iteration 22600, lr = 0.000288526
    I1227 18:56:12.723912  5629 solver.cpp:237] Iteration 22700, loss = 0.741877
    I1227 18:56:12.723979  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 18:56:12.724006  5629 solver.cpp:253]     Train net output #1: loss = 0.741877 (* 1 = 0.741877 loss)
    I1227 18:56:12.724026  5629 sgd_solver.cpp:106] Iteration 22700, lr = 0.000287864
    I1227 18:56:20.611126  5629 solver.cpp:237] Iteration 22800, loss = 0.771346
    I1227 18:56:20.611196  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:56:20.611220  5629 solver.cpp:253]     Train net output #1: loss = 0.771346 (* 1 = 0.771346 loss)
    I1227 18:56:20.611240  5629 sgd_solver.cpp:106] Iteration 22800, lr = 0.000287205
    I1227 18:56:28.498141  5629 solver.cpp:237] Iteration 22900, loss = 0.807561
    I1227 18:56:28.498345  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:56:28.498380  5629 solver.cpp:253]     Train net output #1: loss = 0.807561 (* 1 = 0.807561 loss)
    I1227 18:56:28.498400  5629 sgd_solver.cpp:106] Iteration 22900, lr = 0.00028655
    I1227 18:56:36.308006  5629 solver.cpp:341] Iteration 23000, Testing net (#0)
    I1227 18:56:40.163597  5629 solver.cpp:409]     Test net output #0: accuracy = 0.71675
    I1227 18:56:40.163674  5629 solver.cpp:409]     Test net output #1: loss = 0.813695 (* 1 = 0.813695 loss)
    I1227 18:56:40.207973  5629 solver.cpp:237] Iteration 23000, loss = 0.746901
    I1227 18:56:40.208034  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:56:40.208060  5629 solver.cpp:253]     Train net output #1: loss = 0.746901 (* 1 = 0.746901 loss)
    I1227 18:56:40.208080  5629 sgd_solver.cpp:106] Iteration 23000, lr = 0.000285899
    I1227 18:56:48.099114  5629 solver.cpp:237] Iteration 23100, loss = 0.993612
    I1227 18:56:48.099189  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:56:48.099215  5629 solver.cpp:253]     Train net output #1: loss = 0.993612 (* 1 = 0.993612 loss)
    I1227 18:56:48.099236  5629 sgd_solver.cpp:106] Iteration 23100, lr = 0.000285251
    I1227 18:56:55.977834  5629 solver.cpp:237] Iteration 23200, loss = 0.722179
    I1227 18:56:55.977900  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:56:55.977926  5629 solver.cpp:253]     Train net output #1: loss = 0.722179 (* 1 = 0.722179 loss)
    I1227 18:56:55.977946  5629 sgd_solver.cpp:106] Iteration 23200, lr = 0.000284606
    I1227 18:57:03.885289  5629 solver.cpp:237] Iteration 23300, loss = 0.817224
    I1227 18:57:03.885484  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 18:57:03.885517  5629 solver.cpp:253]     Train net output #1: loss = 0.817224 (* 1 = 0.817224 loss)
    I1227 18:57:03.885536  5629 sgd_solver.cpp:106] Iteration 23300, lr = 0.000283965
    I1227 18:57:11.764001  5629 solver.cpp:237] Iteration 23400, loss = 0.853629
    I1227 18:57:11.764071  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 18:57:11.764097  5629 solver.cpp:253]     Train net output #1: loss = 0.853629 (* 1 = 0.853629 loss)
    I1227 18:57:11.764117  5629 sgd_solver.cpp:106] Iteration 23400, lr = 0.000283327
    I1227 18:57:19.642042  5629 solver.cpp:237] Iteration 23500, loss = 0.786808
    I1227 18:57:19.642110  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 18:57:19.642137  5629 solver.cpp:253]     Train net output #1: loss = 0.786808 (* 1 = 0.786808 loss)
    I1227 18:57:19.642156  5629 sgd_solver.cpp:106] Iteration 23500, lr = 0.000282693
    I1227 18:57:27.545100  5629 solver.cpp:237] Iteration 23600, loss = 0.920236
    I1227 18:57:27.545167  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:57:27.545193  5629 solver.cpp:253]     Train net output #1: loss = 0.920236 (* 1 = 0.920236 loss)
    I1227 18:57:27.545213  5629 sgd_solver.cpp:106] Iteration 23600, lr = 0.000282061
    I1227 18:57:35.410827  5629 solver.cpp:237] Iteration 23700, loss = 0.669755
    I1227 18:57:35.411037  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 18:57:35.411072  5629 solver.cpp:253]     Train net output #1: loss = 0.669755 (* 1 = 0.669755 loss)
    I1227 18:57:35.411092  5629 sgd_solver.cpp:106] Iteration 23700, lr = 0.000281433
    I1227 18:57:43.285007  5629 solver.cpp:237] Iteration 23800, loss = 0.825354
    I1227 18:57:43.285076  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:57:43.285102  5629 solver.cpp:253]     Train net output #1: loss = 0.825354 (* 1 = 0.825354 loss)
    I1227 18:57:43.285121  5629 sgd_solver.cpp:106] Iteration 23800, lr = 0.000280809
    I1227 18:57:51.153368  5629 solver.cpp:237] Iteration 23900, loss = 0.838963
    I1227 18:57:51.153436  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:57:51.153462  5629 solver.cpp:253]     Train net output #1: loss = 0.838963 (* 1 = 0.838963 loss)
    I1227 18:57:51.153482  5629 sgd_solver.cpp:106] Iteration 23900, lr = 0.000280187
    I1227 18:57:58.962714  5629 solver.cpp:341] Iteration 24000, Testing net (#0)
    I1227 18:58:02.846741  5629 solver.cpp:409]     Test net output #0: accuracy = 0.704333
    I1227 18:58:02.846818  5629 solver.cpp:409]     Test net output #1: loss = 0.847211 (* 1 = 0.847211 loss)
    I1227 18:58:02.893584  5629 solver.cpp:237] Iteration 24000, loss = 0.728326
    I1227 18:58:02.893647  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 18:58:02.893672  5629 solver.cpp:253]     Train net output #1: loss = 0.728326 (* 1 = 0.728326 loss)
    I1227 18:58:02.893692  5629 sgd_solver.cpp:106] Iteration 24000, lr = 0.000279569
    I1227 18:58:10.792299  5629 solver.cpp:237] Iteration 24100, loss = 1.00529
    I1227 18:58:10.792487  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 18:58:10.792518  5629 solver.cpp:253]     Train net output #1: loss = 1.00529 (* 1 = 1.00529 loss)
    I1227 18:58:10.792538  5629 sgd_solver.cpp:106] Iteration 24100, lr = 0.000278954
    I1227 18:58:18.676383  5629 solver.cpp:237] Iteration 24200, loss = 0.749831
    I1227 18:58:18.676457  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 18:58:18.676483  5629 solver.cpp:253]     Train net output #1: loss = 0.749831 (* 1 = 0.749831 loss)
    I1227 18:58:18.676503  5629 sgd_solver.cpp:106] Iteration 24200, lr = 0.000278342
    I1227 18:58:26.538525  5629 solver.cpp:237] Iteration 24300, loss = 0.786824
    I1227 18:58:26.538594  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:58:26.538620  5629 solver.cpp:253]     Train net output #1: loss = 0.786824 (* 1 = 0.786824 loss)
    I1227 18:58:26.538640  5629 sgd_solver.cpp:106] Iteration 24300, lr = 0.000277733
    I1227 18:58:34.422039  5629 solver.cpp:237] Iteration 24400, loss = 0.876511
    I1227 18:58:34.422108  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:58:34.422133  5629 solver.cpp:253]     Train net output #1: loss = 0.876511 (* 1 = 0.876511 loss)
    I1227 18:58:34.422153  5629 sgd_solver.cpp:106] Iteration 24400, lr = 0.000277127
    I1227 18:58:42.316421  5629 solver.cpp:237] Iteration 24500, loss = 0.763276
    I1227 18:58:42.316622  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:58:42.316654  5629 solver.cpp:253]     Train net output #1: loss = 0.763276 (* 1 = 0.763276 loss)
    I1227 18:58:42.316674  5629 sgd_solver.cpp:106] Iteration 24500, lr = 0.000276525
    I1227 18:58:50.174897  5629 solver.cpp:237] Iteration 24600, loss = 0.912672
    I1227 18:58:50.174964  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:58:50.174989  5629 solver.cpp:253]     Train net output #1: loss = 0.912672 (* 1 = 0.912672 loss)
    I1227 18:58:50.175009  5629 sgd_solver.cpp:106] Iteration 24600, lr = 0.000275925
    I1227 18:58:58.045966  5629 solver.cpp:237] Iteration 24700, loss = 0.801699
    I1227 18:58:58.046033  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:58:58.046059  5629 solver.cpp:253]     Train net output #1: loss = 0.801699 (* 1 = 0.801699 loss)
    I1227 18:58:58.046078  5629 sgd_solver.cpp:106] Iteration 24700, lr = 0.000275328
    I1227 18:59:05.900960  5629 solver.cpp:237] Iteration 24800, loss = 0.790748
    I1227 18:59:05.901026  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 18:59:05.901051  5629 solver.cpp:253]     Train net output #1: loss = 0.790748 (* 1 = 0.790748 loss)
    I1227 18:59:05.901072  5629 sgd_solver.cpp:106] Iteration 24800, lr = 0.000274735
    I1227 18:59:13.788416  5629 solver.cpp:237] Iteration 24900, loss = 0.819141
    I1227 18:59:13.788602  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 18:59:13.788635  5629 solver.cpp:253]     Train net output #1: loss = 0.819141 (* 1 = 0.819141 loss)
    I1227 18:59:13.788653  5629 sgd_solver.cpp:106] Iteration 24900, lr = 0.000274144
    I1227 18:59:21.603729  5629 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_25000.caffemodel
    I1227 18:59:21.641643  5629 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_25000.solverstate
    I1227 18:59:21.643646  5629 solver.cpp:341] Iteration 25000, Testing net (#0)
    I1227 18:59:25.490546  5629 solver.cpp:409]     Test net output #0: accuracy = 0.705583
    I1227 18:59:25.490623  5629 solver.cpp:409]     Test net output #1: loss = 0.857172 (* 1 = 0.857172 loss)
    I1227 18:59:25.535724  5629 solver.cpp:237] Iteration 25000, loss = 0.76917
    I1227 18:59:25.535789  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 18:59:25.535814  5629 solver.cpp:253]     Train net output #1: loss = 0.76917 (* 1 = 0.76917 loss)
    I1227 18:59:25.535832  5629 sgd_solver.cpp:106] Iteration 25000, lr = 0.000273556
    I1227 18:59:33.428414  5629 solver.cpp:237] Iteration 25100, loss = 0.975552
    I1227 18:59:33.428483  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 18:59:33.428509  5629 solver.cpp:253]     Train net output #1: loss = 0.975552 (* 1 = 0.975552 loss)
    I1227 18:59:33.428529  5629 sgd_solver.cpp:106] Iteration 25100, lr = 0.000272972
    I1227 18:59:41.300588  5629 solver.cpp:237] Iteration 25200, loss = 0.64981
    I1227 18:59:41.300657  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 18:59:41.300684  5629 solver.cpp:253]     Train net output #1: loss = 0.64981 (* 1 = 0.64981 loss)
    I1227 18:59:41.300704  5629 sgd_solver.cpp:106] Iteration 25200, lr = 0.00027239
    I1227 18:59:49.240841  5629 solver.cpp:237] Iteration 25300, loss = 0.759622
    I1227 18:59:49.241055  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 18:59:49.241086  5629 solver.cpp:253]     Train net output #1: loss = 0.759622 (* 1 = 0.759622 loss)
    I1227 18:59:49.241106  5629 sgd_solver.cpp:106] Iteration 25300, lr = 0.000271811
    I1227 18:59:57.112054  5629 solver.cpp:237] Iteration 25400, loss = 0.806375
    I1227 18:59:57.112121  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 18:59:57.112144  5629 solver.cpp:253]     Train net output #1: loss = 0.806375 (* 1 = 0.806375 loss)
    I1227 18:59:57.112164  5629 sgd_solver.cpp:106] Iteration 25400, lr = 0.000271235
    I1227 19:00:04.980684  5629 solver.cpp:237] Iteration 25500, loss = 0.716883
    I1227 19:00:04.980751  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:00:04.980775  5629 solver.cpp:253]     Train net output #1: loss = 0.716883 (* 1 = 0.716883 loss)
    I1227 19:00:04.980793  5629 sgd_solver.cpp:106] Iteration 25500, lr = 0.000270662
    I1227 19:00:12.853349  5629 solver.cpp:237] Iteration 25600, loss = 0.874854
    I1227 19:00:12.853415  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:00:12.853440  5629 solver.cpp:253]     Train net output #1: loss = 0.874854 (* 1 = 0.874854 loss)
    I1227 19:00:12.853461  5629 sgd_solver.cpp:106] Iteration 25600, lr = 0.000270091
    I1227 19:00:20.725332  5629 solver.cpp:237] Iteration 25700, loss = 0.708843
    I1227 19:00:20.725518  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:00:20.725549  5629 solver.cpp:253]     Train net output #1: loss = 0.708843 (* 1 = 0.708843 loss)
    I1227 19:00:20.725569  5629 sgd_solver.cpp:106] Iteration 25700, lr = 0.000269524
    I1227 19:00:28.594749  5629 solver.cpp:237] Iteration 25800, loss = 0.840958
    I1227 19:00:28.594817  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:00:28.594842  5629 solver.cpp:253]     Train net output #1: loss = 0.840958 (* 1 = 0.840958 loss)
    I1227 19:00:28.594861  5629 sgd_solver.cpp:106] Iteration 25800, lr = 0.000268959
    I1227 19:00:36.463641  5629 solver.cpp:237] Iteration 25900, loss = 0.910144
    I1227 19:00:36.463707  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:00:36.463732  5629 solver.cpp:253]     Train net output #1: loss = 0.910144 (* 1 = 0.910144 loss)
    I1227 19:00:36.463752  5629 sgd_solver.cpp:106] Iteration 25900, lr = 0.000268397
    I1227 19:00:44.265758  5629 solver.cpp:341] Iteration 26000, Testing net (#0)
    I1227 19:00:48.126871  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7035
    I1227 19:00:48.126947  5629 solver.cpp:409]     Test net output #1: loss = 0.851747 (* 1 = 0.851747 loss)
    I1227 19:00:48.171423  5629 solver.cpp:237] Iteration 26000, loss = 0.764115
    I1227 19:00:48.171489  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:00:48.171514  5629 solver.cpp:253]     Train net output #1: loss = 0.764115 (* 1 = 0.764115 loss)
    I1227 19:00:48.171532  5629 sgd_solver.cpp:106] Iteration 26000, lr = 0.000267837
    I1227 19:00:56.057242  5629 solver.cpp:237] Iteration 26100, loss = 0.981128
    I1227 19:00:56.057456  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 19:00:56.057489  5629 solver.cpp:253]     Train net output #1: loss = 0.981128 (* 1 = 0.981128 loss)
    I1227 19:00:56.057510  5629 sgd_solver.cpp:106] Iteration 26100, lr = 0.000267281
    I1227 19:01:04.005139  5629 solver.cpp:237] Iteration 26200, loss = 0.746684
    I1227 19:01:04.005208  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:01:04.005234  5629 solver.cpp:253]     Train net output #1: loss = 0.746684 (* 1 = 0.746684 loss)
    I1227 19:01:04.005252  5629 sgd_solver.cpp:106] Iteration 26200, lr = 0.000266727
    I1227 19:01:11.874272  5629 solver.cpp:237] Iteration 26300, loss = 0.797054
    I1227 19:01:11.874341  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:01:11.874366  5629 solver.cpp:253]     Train net output #1: loss = 0.797054 (* 1 = 0.797054 loss)
    I1227 19:01:11.874384  5629 sgd_solver.cpp:106] Iteration 26300, lr = 0.000266175
    I1227 19:01:19.773408  5629 solver.cpp:237] Iteration 26400, loss = 0.796608
    I1227 19:01:19.773478  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:01:19.773504  5629 solver.cpp:253]     Train net output #1: loss = 0.796608 (* 1 = 0.796608 loss)
    I1227 19:01:19.773524  5629 sgd_solver.cpp:106] Iteration 26400, lr = 0.000265627
    I1227 19:01:27.637965  5629 solver.cpp:237] Iteration 26500, loss = 0.677011
    I1227 19:01:27.638224  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:01:27.638260  5629 solver.cpp:253]     Train net output #1: loss = 0.677011 (* 1 = 0.677011 loss)
    I1227 19:01:27.638280  5629 sgd_solver.cpp:106] Iteration 26500, lr = 0.000265081
    I1227 19:01:35.523555  5629 solver.cpp:237] Iteration 26600, loss = 0.880859
    I1227 19:01:35.523622  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:01:35.523645  5629 solver.cpp:253]     Train net output #1: loss = 0.880859 (* 1 = 0.880859 loss)
    I1227 19:01:35.523666  5629 sgd_solver.cpp:106] Iteration 26600, lr = 0.000264537
    I1227 19:01:43.424953  5629 solver.cpp:237] Iteration 26700, loss = 0.680198
    I1227 19:01:43.425016  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:01:43.425042  5629 solver.cpp:253]     Train net output #1: loss = 0.680198 (* 1 = 0.680198 loss)
    I1227 19:01:43.425062  5629 sgd_solver.cpp:106] Iteration 26700, lr = 0.000263997
    I1227 19:01:51.340698  5629 solver.cpp:237] Iteration 26800, loss = 0.740398
    I1227 19:01:51.340766  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:01:51.340792  5629 solver.cpp:253]     Train net output #1: loss = 0.740398 (* 1 = 0.740398 loss)
    I1227 19:01:51.340812  5629 sgd_solver.cpp:106] Iteration 26800, lr = 0.000263458
    I1227 19:01:59.229207  5629 solver.cpp:237] Iteration 26900, loss = 0.910497
    I1227 19:01:59.229398  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:01:59.229431  5629 solver.cpp:253]     Train net output #1: loss = 0.910497 (* 1 = 0.910497 loss)
    I1227 19:01:59.229449  5629 sgd_solver.cpp:106] Iteration 26900, lr = 0.000262923
    I1227 19:02:07.033440  5629 solver.cpp:341] Iteration 27000, Testing net (#0)
    I1227 19:02:10.934165  5629 solver.cpp:409]     Test net output #0: accuracy = 0.71175
    I1227 19:02:10.934242  5629 solver.cpp:409]     Test net output #1: loss = 0.827186 (* 1 = 0.827186 loss)
    I1227 19:02:10.979370  5629 solver.cpp:237] Iteration 27000, loss = 0.716645
    I1227 19:02:10.979435  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:02:10.979460  5629 solver.cpp:253]     Train net output #1: loss = 0.716645 (* 1 = 0.716645 loss)
    I1227 19:02:10.979478  5629 sgd_solver.cpp:106] Iteration 27000, lr = 0.00026239
    I1227 19:02:18.861476  5629 solver.cpp:237] Iteration 27100, loss = 0.858413
    I1227 19:02:18.861542  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:02:18.861567  5629 solver.cpp:253]     Train net output #1: loss = 0.858413 (* 1 = 0.858413 loss)
    I1227 19:02:18.861588  5629 sgd_solver.cpp:106] Iteration 27100, lr = 0.000261859
    I1227 19:02:26.749259  5629 solver.cpp:237] Iteration 27200, loss = 0.751103
    I1227 19:02:26.749330  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:02:26.749353  5629 solver.cpp:253]     Train net output #1: loss = 0.751103 (* 1 = 0.751103 loss)
    I1227 19:02:26.749373  5629 sgd_solver.cpp:106] Iteration 27200, lr = 0.000261331
    I1227 19:02:35.294100  5629 solver.cpp:237] Iteration 27300, loss = 0.815372
    I1227 19:02:35.294311  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:02:35.294343  5629 solver.cpp:253]     Train net output #1: loss = 0.815372 (* 1 = 0.815372 loss)
    I1227 19:02:35.294363  5629 sgd_solver.cpp:106] Iteration 27300, lr = 0.000260805
    I1227 19:02:43.414989  5629 solver.cpp:237] Iteration 27400, loss = 0.822175
    I1227 19:02:43.415061  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:02:43.415086  5629 solver.cpp:253]     Train net output #1: loss = 0.822175 (* 1 = 0.822175 loss)
    I1227 19:02:43.415107  5629 sgd_solver.cpp:106] Iteration 27400, lr = 0.000260282
    I1227 19:02:51.291157  5629 solver.cpp:237] Iteration 27500, loss = 0.782257
    I1227 19:02:51.291225  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 19:02:51.291252  5629 solver.cpp:253]     Train net output #1: loss = 0.782257 (* 1 = 0.782257 loss)
    I1227 19:02:51.291270  5629 sgd_solver.cpp:106] Iteration 27500, lr = 0.000259761
    I1227 19:02:59.177605  5629 solver.cpp:237] Iteration 27600, loss = 0.791964
    I1227 19:02:59.177671  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:02:59.177696  5629 solver.cpp:253]     Train net output #1: loss = 0.791964 (* 1 = 0.791964 loss)
    I1227 19:02:59.177714  5629 sgd_solver.cpp:106] Iteration 27600, lr = 0.000259243
    I1227 19:03:06.805414  5629 solver.cpp:237] Iteration 27700, loss = 0.726823
    I1227 19:03:06.805619  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:03:06.805655  5629 solver.cpp:253]     Train net output #1: loss = 0.726823 (* 1 = 0.726823 loss)
    I1227 19:03:06.805672  5629 sgd_solver.cpp:106] Iteration 27700, lr = 0.000258727
    I1227 19:03:13.758534  5629 solver.cpp:237] Iteration 27800, loss = 0.695004
    I1227 19:03:13.758600  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:03:13.758625  5629 solver.cpp:253]     Train net output #1: loss = 0.695004 (* 1 = 0.695004 loss)
    I1227 19:03:13.758644  5629 sgd_solver.cpp:106] Iteration 27800, lr = 0.000258214
    I1227 19:03:20.704604  5629 solver.cpp:237] Iteration 27900, loss = 0.90323
    I1227 19:03:20.704670  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:03:20.704695  5629 solver.cpp:253]     Train net output #1: loss = 0.90323 (* 1 = 0.90323 loss)
    I1227 19:03:20.704716  5629 sgd_solver.cpp:106] Iteration 27900, lr = 0.000257702
    I1227 19:03:27.592294  5629 solver.cpp:341] Iteration 28000, Testing net (#0)
    I1227 19:03:30.431048  5629 solver.cpp:409]     Test net output #0: accuracy = 0.703667
    I1227 19:03:30.431118  5629 solver.cpp:409]     Test net output #1: loss = 0.846985 (* 1 = 0.846985 loss)
    I1227 19:03:30.465804  5629 solver.cpp:237] Iteration 28000, loss = 0.758457
    I1227 19:03:30.465844  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:03:30.465867  5629 solver.cpp:253]     Train net output #1: loss = 0.758457 (* 1 = 0.758457 loss)
    I1227 19:03:30.465885  5629 sgd_solver.cpp:106] Iteration 28000, lr = 0.000257194
    I1227 19:03:37.418241  5629 solver.cpp:237] Iteration 28100, loss = 0.98977
    I1227 19:03:37.418424  5629 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1227 19:03:37.418455  5629 solver.cpp:253]     Train net output #1: loss = 0.98977 (* 1 = 0.98977 loss)
    I1227 19:03:37.418474  5629 sgd_solver.cpp:106] Iteration 28100, lr = 0.000256687
    I1227 19:03:44.405465  5629 solver.cpp:237] Iteration 28200, loss = 0.687545
    I1227 19:03:44.405527  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:03:44.405552  5629 solver.cpp:253]     Train net output #1: loss = 0.687545 (* 1 = 0.687545 loss)
    I1227 19:03:44.405570  5629 sgd_solver.cpp:106] Iteration 28200, lr = 0.000256183
    I1227 19:03:51.348600  5629 solver.cpp:237] Iteration 28300, loss = 0.716402
    I1227 19:03:51.348664  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:03:51.348690  5629 solver.cpp:253]     Train net output #1: loss = 0.716402 (* 1 = 0.716402 loss)
    I1227 19:03:51.348707  5629 sgd_solver.cpp:106] Iteration 28300, lr = 0.000255681
    I1227 19:03:58.287973  5629 solver.cpp:237] Iteration 28400, loss = 0.810449
    I1227 19:03:58.288039  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:03:58.288064  5629 solver.cpp:253]     Train net output #1: loss = 0.810449 (* 1 = 0.810449 loss)
    I1227 19:03:58.288081  5629 sgd_solver.cpp:106] Iteration 28400, lr = 0.000255182
    I1227 19:04:05.242106  5629 solver.cpp:237] Iteration 28500, loss = 0.747401
    I1227 19:04:05.242151  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:04:05.242166  5629 solver.cpp:253]     Train net output #1: loss = 0.747401 (* 1 = 0.747401 loss)
    I1227 19:04:05.242177  5629 sgd_solver.cpp:106] Iteration 28500, lr = 0.000254684
    I1227 19:04:12.155640  5629 solver.cpp:237] Iteration 28600, loss = 0.827042
    I1227 19:04:12.155782  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:04:12.155803  5629 solver.cpp:253]     Train net output #1: loss = 0.827042 (* 1 = 0.827042 loss)
    I1227 19:04:12.155814  5629 sgd_solver.cpp:106] Iteration 28600, lr = 0.000254189
    I1227 19:04:19.037061  5629 solver.cpp:237] Iteration 28700, loss = 0.708155
    I1227 19:04:19.037101  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:04:19.037117  5629 solver.cpp:253]     Train net output #1: loss = 0.708155 (* 1 = 0.708155 loss)
    I1227 19:04:19.037129  5629 sgd_solver.cpp:106] Iteration 28700, lr = 0.000253697
    I1227 19:04:26.083559  5629 solver.cpp:237] Iteration 28800, loss = 0.728975
    I1227 19:04:26.083605  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:04:26.083619  5629 solver.cpp:253]     Train net output #1: loss = 0.728975 (* 1 = 0.728975 loss)
    I1227 19:04:26.083632  5629 sgd_solver.cpp:106] Iteration 28800, lr = 0.000253206
    I1227 19:04:32.993656  5629 solver.cpp:237] Iteration 28900, loss = 0.827726
    I1227 19:04:32.993700  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:04:32.993716  5629 solver.cpp:253]     Train net output #1: loss = 0.827726 (* 1 = 0.827726 loss)
    I1227 19:04:32.993727  5629 sgd_solver.cpp:106] Iteration 28900, lr = 0.000252718
    I1227 19:04:39.812068  5629 solver.cpp:341] Iteration 29000, Testing net (#0)
    I1227 19:04:42.586786  5629 solver.cpp:409]     Test net output #0: accuracy = 0.707083
    I1227 19:04:42.586913  5629 solver.cpp:409]     Test net output #1: loss = 0.841125 (* 1 = 0.841125 loss)
    I1227 19:04:42.617177  5629 solver.cpp:237] Iteration 29000, loss = 0.622441
    I1227 19:04:42.617221  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:04:42.617235  5629 solver.cpp:253]     Train net output #1: loss = 0.622441 (* 1 = 0.622441 loss)
    I1227 19:04:42.617249  5629 sgd_solver.cpp:106] Iteration 29000, lr = 0.000252232
    I1227 19:04:49.508186  5629 solver.cpp:237] Iteration 29100, loss = 0.763395
    I1227 19:04:49.508231  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:04:49.508246  5629 solver.cpp:253]     Train net output #1: loss = 0.763395 (* 1 = 0.763395 loss)
    I1227 19:04:49.508257  5629 sgd_solver.cpp:106] Iteration 29100, lr = 0.000251748
    I1227 19:04:56.410923  5629 solver.cpp:237] Iteration 29200, loss = 0.711492
    I1227 19:04:56.410969  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:04:56.410984  5629 solver.cpp:253]     Train net output #1: loss = 0.711492 (* 1 = 0.711492 loss)
    I1227 19:04:56.410995  5629 sgd_solver.cpp:106] Iteration 29200, lr = 0.000251266
    I1227 19:05:03.328713  5629 solver.cpp:237] Iteration 29300, loss = 0.823618
    I1227 19:05:03.328758  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:05:03.328773  5629 solver.cpp:253]     Train net output #1: loss = 0.823618 (* 1 = 0.823618 loss)
    I1227 19:05:03.328783  5629 sgd_solver.cpp:106] Iteration 29300, lr = 0.000250786
    I1227 19:05:10.212697  5629 solver.cpp:237] Iteration 29400, loss = 0.745692
    I1227 19:05:10.212740  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:05:10.212756  5629 solver.cpp:253]     Train net output #1: loss = 0.745692 (* 1 = 0.745692 loss)
    I1227 19:05:10.212767  5629 sgd_solver.cpp:106] Iteration 29400, lr = 0.000250309
    I1227 19:05:17.111980  5629 solver.cpp:237] Iteration 29500, loss = 0.613819
    I1227 19:05:17.112104  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:05:17.112123  5629 solver.cpp:253]     Train net output #1: loss = 0.613819 (* 1 = 0.613819 loss)
    I1227 19:05:17.112134  5629 sgd_solver.cpp:106] Iteration 29500, lr = 0.000249833
    I1227 19:05:24.003011  5629 solver.cpp:237] Iteration 29600, loss = 0.927575
    I1227 19:05:24.003054  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 19:05:24.003070  5629 solver.cpp:253]     Train net output #1: loss = 0.927575 (* 1 = 0.927575 loss)
    I1227 19:05:24.003082  5629 sgd_solver.cpp:106] Iteration 29600, lr = 0.00024936
    I1227 19:05:30.898430  5629 solver.cpp:237] Iteration 29700, loss = 0.59237
    I1227 19:05:30.898473  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:05:30.898488  5629 solver.cpp:253]     Train net output #1: loss = 0.59237 (* 1 = 0.59237 loss)
    I1227 19:05:30.898500  5629 sgd_solver.cpp:106] Iteration 29700, lr = 0.000248889
    I1227 19:05:37.782774  5629 solver.cpp:237] Iteration 29800, loss = 0.689745
    I1227 19:05:37.782819  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:05:37.782835  5629 solver.cpp:253]     Train net output #1: loss = 0.689745 (* 1 = 0.689745 loss)
    I1227 19:05:37.782847  5629 sgd_solver.cpp:106] Iteration 29800, lr = 0.00024842
    I1227 19:05:44.698515  5629 solver.cpp:237] Iteration 29900, loss = 0.822962
    I1227 19:05:44.698561  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:05:44.698576  5629 solver.cpp:253]     Train net output #1: loss = 0.822962 (* 1 = 0.822962 loss)
    I1227 19:05:44.698588  5629 sgd_solver.cpp:106] Iteration 29900, lr = 0.000247952
    I1227 19:05:51.517719  5629 solver.cpp:341] Iteration 30000, Testing net (#0)
    I1227 19:05:54.286739  5629 solver.cpp:409]     Test net output #0: accuracy = 0.718667
    I1227 19:05:54.286787  5629 solver.cpp:409]     Test net output #1: loss = 0.804514 (* 1 = 0.804514 loss)
    I1227 19:05:54.317051  5629 solver.cpp:237] Iteration 30000, loss = 0.611743
    I1227 19:05:54.317075  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:05:54.317086  5629 solver.cpp:253]     Train net output #1: loss = 0.611743 (* 1 = 0.611743 loss)
    I1227 19:05:54.317097  5629 sgd_solver.cpp:106] Iteration 30000, lr = 0.000247487
    I1227 19:06:01.225153  5629 solver.cpp:237] Iteration 30100, loss = 0.937098
    I1227 19:06:01.225196  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:06:01.225213  5629 solver.cpp:253]     Train net output #1: loss = 0.937098 (* 1 = 0.937098 loss)
    I1227 19:06:01.225224  5629 sgd_solver.cpp:106] Iteration 30100, lr = 0.000247024
    I1227 19:06:08.096993  5629 solver.cpp:237] Iteration 30200, loss = 0.66935
    I1227 19:06:08.097049  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:06:08.097067  5629 solver.cpp:253]     Train net output #1: loss = 0.66935 (* 1 = 0.66935 loss)
    I1227 19:06:08.097079  5629 sgd_solver.cpp:106] Iteration 30200, lr = 0.000246563
    I1227 19:06:14.983747  5629 solver.cpp:237] Iteration 30300, loss = 0.814323
    I1227 19:06:14.983789  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:06:14.983804  5629 solver.cpp:253]     Train net output #1: loss = 0.814323 (* 1 = 0.814323 loss)
    I1227 19:06:14.983816  5629 sgd_solver.cpp:106] Iteration 30300, lr = 0.000246104
    I1227 19:06:21.870299  5629 solver.cpp:237] Iteration 30400, loss = 0.79303
    I1227 19:06:21.870415  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:06:21.870434  5629 solver.cpp:253]     Train net output #1: loss = 0.79303 (* 1 = 0.79303 loss)
    I1227 19:06:21.870445  5629 sgd_solver.cpp:106] Iteration 30400, lr = 0.000245647
    I1227 19:06:28.748989  5629 solver.cpp:237] Iteration 30500, loss = 0.704173
    I1227 19:06:28.749034  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:06:28.749050  5629 solver.cpp:253]     Train net output #1: loss = 0.704173 (* 1 = 0.704173 loss)
    I1227 19:06:28.749061  5629 sgd_solver.cpp:106] Iteration 30500, lr = 0.000245192
    I1227 19:06:36.039489  5629 solver.cpp:237] Iteration 30600, loss = 0.87505
    I1227 19:06:36.039535  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:06:36.039549  5629 solver.cpp:253]     Train net output #1: loss = 0.87505 (* 1 = 0.87505 loss)
    I1227 19:06:36.039559  5629 sgd_solver.cpp:106] Iteration 30600, lr = 0.000244739
    I1227 19:06:42.951978  5629 solver.cpp:237] Iteration 30700, loss = 0.774247
    I1227 19:06:42.952028  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:06:42.952041  5629 solver.cpp:253]     Train net output #1: loss = 0.774247 (* 1 = 0.774247 loss)
    I1227 19:06:42.952050  5629 sgd_solver.cpp:106] Iteration 30700, lr = 0.000244288
    I1227 19:06:49.836627  5629 solver.cpp:237] Iteration 30800, loss = 0.792455
    I1227 19:06:49.836675  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:06:49.836689  5629 solver.cpp:253]     Train net output #1: loss = 0.792455 (* 1 = 0.792455 loss)
    I1227 19:06:49.836699  5629 sgd_solver.cpp:106] Iteration 30800, lr = 0.000243839
    I1227 19:06:56.709508  5629 solver.cpp:237] Iteration 30900, loss = 0.722618
    I1227 19:06:56.709691  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:06:56.709718  5629 solver.cpp:253]     Train net output #1: loss = 0.722618 (* 1 = 0.722618 loss)
    I1227 19:06:56.709729  5629 sgd_solver.cpp:106] Iteration 30900, lr = 0.000243392
    I1227 19:07:03.538597  5629 solver.cpp:341] Iteration 31000, Testing net (#0)
    I1227 19:07:06.320504  5629 solver.cpp:409]     Test net output #0: accuracy = 0.715167
    I1227 19:07:06.320554  5629 solver.cpp:409]     Test net output #1: loss = 0.817767 (* 1 = 0.817767 loss)
    I1227 19:07:06.350746  5629 solver.cpp:237] Iteration 31000, loss = 0.8023
    I1227 19:07:06.350788  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:07:06.350802  5629 solver.cpp:253]     Train net output #1: loss = 0.8023 (* 1 = 0.8023 loss)
    I1227 19:07:06.350814  5629 sgd_solver.cpp:106] Iteration 31000, lr = 0.000242946
    I1227 19:07:13.239653  5629 solver.cpp:237] Iteration 31100, loss = 0.845654
    I1227 19:07:13.239697  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:07:13.239712  5629 solver.cpp:253]     Train net output #1: loss = 0.845654 (* 1 = 0.845654 loss)
    I1227 19:07:13.239722  5629 sgd_solver.cpp:106] Iteration 31100, lr = 0.000242503
    I1227 19:07:20.115597  5629 solver.cpp:237] Iteration 31200, loss = 0.776606
    I1227 19:07:20.115646  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:07:20.115659  5629 solver.cpp:253]     Train net output #1: loss = 0.776606 (* 1 = 0.776606 loss)
    I1227 19:07:20.115669  5629 sgd_solver.cpp:106] Iteration 31200, lr = 0.000242061
    I1227 19:07:27.010061  5629 solver.cpp:237] Iteration 31300, loss = 0.738415
    I1227 19:07:27.010185  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:07:27.010202  5629 solver.cpp:253]     Train net output #1: loss = 0.738415 (* 1 = 0.738415 loss)
    I1227 19:07:27.010212  5629 sgd_solver.cpp:106] Iteration 31300, lr = 0.000241621
    I1227 19:07:33.885646  5629 solver.cpp:237] Iteration 31400, loss = 0.784622
    I1227 19:07:33.885692  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:07:33.885706  5629 solver.cpp:253]     Train net output #1: loss = 0.784622 (* 1 = 0.784622 loss)
    I1227 19:07:33.885717  5629 sgd_solver.cpp:106] Iteration 31400, lr = 0.000241184
    I1227 19:07:40.764142  5629 solver.cpp:237] Iteration 31500, loss = 0.646482
    I1227 19:07:40.764192  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:07:40.764206  5629 solver.cpp:253]     Train net output #1: loss = 0.646482 (* 1 = 0.646482 loss)
    I1227 19:07:40.764216  5629 sgd_solver.cpp:106] Iteration 31500, lr = 0.000240748
    I1227 19:07:47.676785  5629 solver.cpp:237] Iteration 31600, loss = 0.896327
    I1227 19:07:47.676841  5629 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1227 19:07:47.676856  5629 solver.cpp:253]     Train net output #1: loss = 0.896327 (* 1 = 0.896327 loss)
    I1227 19:07:47.676867  5629 sgd_solver.cpp:106] Iteration 31600, lr = 0.000240313
    I1227 19:07:54.540748  5629 solver.cpp:237] Iteration 31700, loss = 0.711265
    I1227 19:07:54.540798  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:07:54.540812  5629 solver.cpp:253]     Train net output #1: loss = 0.711265 (* 1 = 0.711265 loss)
    I1227 19:07:54.540822  5629 sgd_solver.cpp:106] Iteration 31700, lr = 0.000239881
    I1227 19:08:01.452529  5629 solver.cpp:237] Iteration 31800, loss = 0.795953
    I1227 19:08:01.452708  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:08:01.452739  5629 solver.cpp:253]     Train net output #1: loss = 0.795953 (* 1 = 0.795953 loss)
    I1227 19:08:01.452755  5629 sgd_solver.cpp:106] Iteration 31800, lr = 0.000239451
    I1227 19:08:08.319073  5629 solver.cpp:237] Iteration 31900, loss = 0.810264
    I1227 19:08:08.319114  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:08:08.319128  5629 solver.cpp:253]     Train net output #1: loss = 0.810264 (* 1 = 0.810264 loss)
    I1227 19:08:08.319139  5629 sgd_solver.cpp:106] Iteration 31900, lr = 0.000239022
    I1227 19:08:15.182726  5629 solver.cpp:341] Iteration 32000, Testing net (#0)
    I1227 19:08:17.964781  5629 solver.cpp:409]     Test net output #0: accuracy = 0.72475
    I1227 19:08:17.964828  5629 solver.cpp:409]     Test net output #1: loss = 0.786237 (* 1 = 0.786237 loss)
    I1227 19:08:17.995113  5629 solver.cpp:237] Iteration 32000, loss = 0.666535
    I1227 19:08:17.995156  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:08:17.995168  5629 solver.cpp:253]     Train net output #1: loss = 0.666535 (* 1 = 0.666535 loss)
    I1227 19:08:17.995180  5629 sgd_solver.cpp:106] Iteration 32000, lr = 0.000238595
    I1227 19:08:24.863718  5629 solver.cpp:237] Iteration 32100, loss = 0.8803
    I1227 19:08:24.863762  5629 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1227 19:08:24.863777  5629 solver.cpp:253]     Train net output #1: loss = 0.8803 (* 1 = 0.8803 loss)
    I1227 19:08:24.863788  5629 sgd_solver.cpp:106] Iteration 32100, lr = 0.00023817
    I1227 19:08:31.741971  5629 solver.cpp:237] Iteration 32200, loss = 0.629453
    I1227 19:08:31.742162  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:08:31.742182  5629 solver.cpp:253]     Train net output #1: loss = 0.629453 (* 1 = 0.629453 loss)
    I1227 19:08:31.742193  5629 sgd_solver.cpp:106] Iteration 32200, lr = 0.000237746
    I1227 19:08:38.615578  5629 solver.cpp:237] Iteration 32300, loss = 0.799327
    I1227 19:08:38.615635  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:08:38.615658  5629 solver.cpp:253]     Train net output #1: loss = 0.799327 (* 1 = 0.799327 loss)
    I1227 19:08:38.615672  5629 sgd_solver.cpp:106] Iteration 32300, lr = 0.000237325
    I1227 19:08:45.507345  5629 solver.cpp:237] Iteration 32400, loss = 0.75682
    I1227 19:08:45.507386  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:08:45.507400  5629 solver.cpp:253]     Train net output #1: loss = 0.75682 (* 1 = 0.75682 loss)
    I1227 19:08:45.507412  5629 sgd_solver.cpp:106] Iteration 32400, lr = 0.000236905
    I1227 19:08:52.405038  5629 solver.cpp:237] Iteration 32500, loss = 0.720766
    I1227 19:08:52.405103  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:08:52.405128  5629 solver.cpp:253]     Train net output #1: loss = 0.720766 (* 1 = 0.720766 loss)
    I1227 19:08:52.405146  5629 sgd_solver.cpp:106] Iteration 32500, lr = 0.000236486
    I1227 19:08:59.314369  5629 solver.cpp:237] Iteration 32600, loss = 0.884789
    I1227 19:08:59.314415  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:08:59.314430  5629 solver.cpp:253]     Train net output #1: loss = 0.884789 (* 1 = 0.884789 loss)
    I1227 19:08:59.314442  5629 sgd_solver.cpp:106] Iteration 32600, lr = 0.00023607
    I1227 19:09:06.187587  5629 solver.cpp:237] Iteration 32700, loss = 0.670424
    I1227 19:09:06.187724  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:09:06.187741  5629 solver.cpp:253]     Train net output #1: loss = 0.670424 (* 1 = 0.670424 loss)
    I1227 19:09:06.187750  5629 sgd_solver.cpp:106] Iteration 32700, lr = 0.000235655
    I1227 19:09:13.086197  5629 solver.cpp:237] Iteration 32800, loss = 0.668934
    I1227 19:09:13.086256  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:09:13.086278  5629 solver.cpp:253]     Train net output #1: loss = 0.668934 (* 1 = 0.668934 loss)
    I1227 19:09:13.086295  5629 sgd_solver.cpp:106] Iteration 32800, lr = 0.000235242
    I1227 19:09:19.955364  5629 solver.cpp:237] Iteration 32900, loss = 0.894811
    I1227 19:09:19.955405  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:09:19.955420  5629 solver.cpp:253]     Train net output #1: loss = 0.894811 (* 1 = 0.894811 loss)
    I1227 19:09:19.955430  5629 sgd_solver.cpp:106] Iteration 32900, lr = 0.000234831
    I1227 19:09:26.778898  5629 solver.cpp:341] Iteration 33000, Testing net (#0)
    I1227 19:09:29.534232  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7155
    I1227 19:09:29.534327  5629 solver.cpp:409]     Test net output #1: loss = 0.812503 (* 1 = 0.812503 loss)
    I1227 19:09:29.570801  5629 solver.cpp:237] Iteration 33000, loss = 0.832681
    I1227 19:09:29.570878  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:09:29.570914  5629 solver.cpp:253]     Train net output #1: loss = 0.832681 (* 1 = 0.832681 loss)
    I1227 19:09:29.570935  5629 sgd_solver.cpp:106] Iteration 33000, lr = 0.000234421
    I1227 19:09:36.466317  5629 solver.cpp:237] Iteration 33100, loss = 0.825722
    I1227 19:09:36.466464  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:09:36.466481  5629 solver.cpp:253]     Train net output #1: loss = 0.825722 (* 1 = 0.825722 loss)
    I1227 19:09:36.466490  5629 sgd_solver.cpp:106] Iteration 33100, lr = 0.000234013
    I1227 19:09:43.371079  5629 solver.cpp:237] Iteration 33200, loss = 0.670654
    I1227 19:09:43.371145  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:09:43.371170  5629 solver.cpp:253]     Train net output #1: loss = 0.670654 (* 1 = 0.670654 loss)
    I1227 19:09:43.371187  5629 sgd_solver.cpp:106] Iteration 33200, lr = 0.000233607
    I1227 19:09:50.291955  5629 solver.cpp:237] Iteration 33300, loss = 0.713114
    I1227 19:09:50.291999  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:09:50.292016  5629 solver.cpp:253]     Train net output #1: loss = 0.713114 (* 1 = 0.713114 loss)
    I1227 19:09:50.292027  5629 sgd_solver.cpp:106] Iteration 33300, lr = 0.000233202
    I1227 19:09:57.163204  5629 solver.cpp:237] Iteration 33400, loss = 0.867551
    I1227 19:09:57.163256  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:09:57.163272  5629 solver.cpp:253]     Train net output #1: loss = 0.867551 (* 1 = 0.867551 loss)
    I1227 19:09:57.163283  5629 sgd_solver.cpp:106] Iteration 33400, lr = 0.000232799
    I1227 19:10:04.046386  5629 solver.cpp:237] Iteration 33500, loss = 0.672346
    I1227 19:10:04.046443  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:10:04.046465  5629 solver.cpp:253]     Train net output #1: loss = 0.672346 (* 1 = 0.672346 loss)
    I1227 19:10:04.046481  5629 sgd_solver.cpp:106] Iteration 33500, lr = 0.000232397
    I1227 19:10:10.923980  5629 solver.cpp:237] Iteration 33600, loss = 0.825779
    I1227 19:10:10.924075  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:10:10.924091  5629 solver.cpp:253]     Train net output #1: loss = 0.825779 (* 1 = 0.825779 loss)
    I1227 19:10:10.924103  5629 sgd_solver.cpp:106] Iteration 33600, lr = 0.000231997
    I1227 19:10:17.811378  5629 solver.cpp:237] Iteration 33700, loss = 0.688157
    I1227 19:10:17.811431  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:10:17.811453  5629 solver.cpp:253]     Train net output #1: loss = 0.688157 (* 1 = 0.688157 loss)
    I1227 19:10:17.811468  5629 sgd_solver.cpp:106] Iteration 33700, lr = 0.000231599
    I1227 19:10:24.699586  5629 solver.cpp:237] Iteration 33800, loss = 0.749791
    I1227 19:10:24.699631  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:10:24.699647  5629 solver.cpp:253]     Train net output #1: loss = 0.749791 (* 1 = 0.749791 loss)
    I1227 19:10:24.699658  5629 sgd_solver.cpp:106] Iteration 33800, lr = 0.000231202
    I1227 19:10:32.291950  5629 solver.cpp:237] Iteration 33900, loss = 0.767718
    I1227 19:10:32.292021  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:10:32.292047  5629 solver.cpp:253]     Train net output #1: loss = 0.767718 (* 1 = 0.767718 loss)
    I1227 19:10:32.292065  5629 sgd_solver.cpp:106] Iteration 33900, lr = 0.000230807
    I1227 19:10:40.046248  5629 solver.cpp:341] Iteration 34000, Testing net (#0)
    I1227 19:10:43.354154  5629 solver.cpp:409]     Test net output #0: accuracy = 0.722
    I1227 19:10:43.354316  5629 solver.cpp:409]     Test net output #1: loss = 0.7917 (* 1 = 0.7917 loss)
    I1227 19:10:43.389086  5629 solver.cpp:237] Iteration 34000, loss = 0.659058
    I1227 19:10:43.389129  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:10:43.389143  5629 solver.cpp:253]     Train net output #1: loss = 0.659058 (* 1 = 0.659058 loss)
    I1227 19:10:43.389155  5629 sgd_solver.cpp:106] Iteration 34000, lr = 0.000230414
    I1227 19:10:51.258060  5629 solver.cpp:237] Iteration 34100, loss = 0.836421
    I1227 19:10:51.258121  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:10:51.258143  5629 solver.cpp:253]     Train net output #1: loss = 0.836421 (* 1 = 0.836421 loss)
    I1227 19:10:51.258160  5629 sgd_solver.cpp:106] Iteration 34100, lr = 0.000230022
    I1227 19:10:59.091292  5629 solver.cpp:237] Iteration 34200, loss = 0.669387
    I1227 19:10:59.091351  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:10:59.091366  5629 solver.cpp:253]     Train net output #1: loss = 0.669387 (* 1 = 0.669387 loss)
    I1227 19:10:59.091377  5629 sgd_solver.cpp:106] Iteration 34200, lr = 0.000229631
    I1227 19:11:06.956313  5629 solver.cpp:237] Iteration 34300, loss = 0.634052
    I1227 19:11:06.956377  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:11:06.956398  5629 solver.cpp:253]     Train net output #1: loss = 0.634052 (* 1 = 0.634052 loss)
    I1227 19:11:06.956415  5629 sgd_solver.cpp:106] Iteration 34300, lr = 0.000229243
    I1227 19:11:14.787616  5629 solver.cpp:237] Iteration 34400, loss = 0.737396
    I1227 19:11:14.787721  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:11:14.787739  5629 solver.cpp:253]     Train net output #1: loss = 0.737396 (* 1 = 0.737396 loss)
    I1227 19:11:14.787751  5629 sgd_solver.cpp:106] Iteration 34400, lr = 0.000228855
    I1227 19:11:22.651757  5629 solver.cpp:237] Iteration 34500, loss = 0.689755
    I1227 19:11:22.651819  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:11:22.651841  5629 solver.cpp:253]     Train net output #1: loss = 0.689755 (* 1 = 0.689755 loss)
    I1227 19:11:22.651859  5629 sgd_solver.cpp:106] Iteration 34500, lr = 0.000228469
    I1227 19:11:30.485668  5629 solver.cpp:237] Iteration 34600, loss = 0.838093
    I1227 19:11:30.485728  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:11:30.485743  5629 solver.cpp:253]     Train net output #1: loss = 0.838093 (* 1 = 0.838093 loss)
    I1227 19:11:30.485754  5629 sgd_solver.cpp:106] Iteration 34600, lr = 0.000228085
    I1227 19:11:38.350848  5629 solver.cpp:237] Iteration 34700, loss = 0.67513
    I1227 19:11:38.350909  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:11:38.350931  5629 solver.cpp:253]     Train net output #1: loss = 0.67513 (* 1 = 0.67513 loss)
    I1227 19:11:38.350949  5629 sgd_solver.cpp:106] Iteration 34700, lr = 0.000227702
    I1227 19:11:46.144913  5629 solver.cpp:237] Iteration 34800, loss = 0.709137
    I1227 19:11:46.145058  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:11:46.145078  5629 solver.cpp:253]     Train net output #1: loss = 0.709137 (* 1 = 0.709137 loss)
    I1227 19:11:46.145090  5629 sgd_solver.cpp:106] Iteration 34800, lr = 0.000227321
    I1227 19:11:54.027878  5629 solver.cpp:237] Iteration 34900, loss = 0.7251
    I1227 19:11:54.027986  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:11:54.028038  5629 solver.cpp:253]     Train net output #1: loss = 0.7251 (* 1 = 0.7251 loss)
    I1227 19:11:54.028075  5629 sgd_solver.cpp:106] Iteration 34900, lr = 0.000226941
    I1227 19:12:01.766640  5629 solver.cpp:341] Iteration 35000, Testing net (#0)
    I1227 19:12:05.008061  5629 solver.cpp:409]     Test net output #0: accuracy = 0.730083
    I1227 19:12:05.008136  5629 solver.cpp:409]     Test net output #1: loss = 0.776993 (* 1 = 0.776993 loss)
    I1227 19:12:05.043311  5629 solver.cpp:237] Iteration 35000, loss = 0.674332
    I1227 19:12:05.043364  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:12:05.043380  5629 solver.cpp:253]     Train net output #1: loss = 0.674332 (* 1 = 0.674332 loss)
    I1227 19:12:05.043391  5629 sgd_solver.cpp:106] Iteration 35000, lr = 0.000226563
    I1227 19:12:12.910696  5629 solver.cpp:237] Iteration 35100, loss = 0.89045
    I1227 19:12:12.910758  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 19:12:12.910781  5629 solver.cpp:253]     Train net output #1: loss = 0.89045 (* 1 = 0.89045 loss)
    I1227 19:12:12.910797  5629 sgd_solver.cpp:106] Iteration 35100, lr = 0.000226186
    I1227 19:12:20.645503  5629 solver.cpp:237] Iteration 35200, loss = 0.629995
    I1227 19:12:20.645648  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:12:20.645664  5629 solver.cpp:253]     Train net output #1: loss = 0.629995 (* 1 = 0.629995 loss)
    I1227 19:12:20.645675  5629 sgd_solver.cpp:106] Iteration 35200, lr = 0.000225811
    I1227 19:12:28.491904  5629 solver.cpp:237] Iteration 35300, loss = 0.694779
    I1227 19:12:28.491964  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:12:28.491986  5629 solver.cpp:253]     Train net output #1: loss = 0.694779 (* 1 = 0.694779 loss)
    I1227 19:12:28.492007  5629 sgd_solver.cpp:106] Iteration 35300, lr = 0.000225437
    I1227 19:12:36.313154  5629 solver.cpp:237] Iteration 35400, loss = 0.794013
    I1227 19:12:36.313207  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:12:36.313222  5629 solver.cpp:253]     Train net output #1: loss = 0.794013 (* 1 = 0.794013 loss)
    I1227 19:12:36.313235  5629 sgd_solver.cpp:106] Iteration 35400, lr = 0.000225064
    I1227 19:12:44.194612  5629 solver.cpp:237] Iteration 35500, loss = 0.646995
    I1227 19:12:44.194671  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:12:44.194694  5629 solver.cpp:253]     Train net output #1: loss = 0.646995 (* 1 = 0.646995 loss)
    I1227 19:12:44.194710  5629 sgd_solver.cpp:106] Iteration 35500, lr = 0.000224693
    I1227 19:12:52.026505  5629 solver.cpp:237] Iteration 35600, loss = 0.811971
    I1227 19:12:52.026671  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:12:52.026691  5629 solver.cpp:253]     Train net output #1: loss = 0.811971 (* 1 = 0.811971 loss)
    I1227 19:12:52.026702  5629 sgd_solver.cpp:106] Iteration 35600, lr = 0.000224323
    I1227 19:12:59.863366  5629 solver.cpp:237] Iteration 35700, loss = 0.653134
    I1227 19:12:59.863426  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:12:59.863448  5629 solver.cpp:253]     Train net output #1: loss = 0.653134 (* 1 = 0.653134 loss)
    I1227 19:12:59.863466  5629 sgd_solver.cpp:106] Iteration 35700, lr = 0.000223955
    I1227 19:13:07.699357  5629 solver.cpp:237] Iteration 35800, loss = 0.65875
    I1227 19:13:07.699409  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:13:07.699424  5629 solver.cpp:253]     Train net output #1: loss = 0.65875 (* 1 = 0.65875 loss)
    I1227 19:13:07.699436  5629 sgd_solver.cpp:106] Iteration 35800, lr = 0.000223588
    I1227 19:13:15.547570  5629 solver.cpp:237] Iteration 35900, loss = 0.6156
    I1227 19:13:15.547638  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:13:15.547663  5629 solver.cpp:253]     Train net output #1: loss = 0.6156 (* 1 = 0.6156 loss)
    I1227 19:13:15.547682  5629 sgd_solver.cpp:106] Iteration 35900, lr = 0.000223223
    I1227 19:13:23.364027  5629 solver.cpp:341] Iteration 36000, Testing net (#0)
    I1227 19:13:27.244792  5629 solver.cpp:409]     Test net output #0: accuracy = 0.725583
    I1227 19:13:27.244868  5629 solver.cpp:409]     Test net output #1: loss = 0.78479 (* 1 = 0.78479 loss)
    I1227 19:13:27.290297  5629 solver.cpp:237] Iteration 36000, loss = 0.754879
    I1227 19:13:27.290359  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:13:27.290383  5629 solver.cpp:253]     Train net output #1: loss = 0.754879 (* 1 = 0.754879 loss)
    I1227 19:13:27.290403  5629 sgd_solver.cpp:106] Iteration 36000, lr = 0.000222859
    I1227 19:13:35.182875  5629 solver.cpp:237] Iteration 36100, loss = 0.921111
    I1227 19:13:35.182955  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:13:35.182981  5629 solver.cpp:253]     Train net output #1: loss = 0.921111 (* 1 = 0.921111 loss)
    I1227 19:13:35.183001  5629 sgd_solver.cpp:106] Iteration 36100, lr = 0.000222496
    I1227 19:13:43.067360  5629 solver.cpp:237] Iteration 36200, loss = 0.662612
    I1227 19:13:43.067425  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:13:43.067451  5629 solver.cpp:253]     Train net output #1: loss = 0.662612 (* 1 = 0.662612 loss)
    I1227 19:13:43.067468  5629 sgd_solver.cpp:106] Iteration 36200, lr = 0.000222135
    I1227 19:13:50.957602  5629 solver.cpp:237] Iteration 36300, loss = 0.670059
    I1227 19:13:50.957669  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:13:50.957693  5629 solver.cpp:253]     Train net output #1: loss = 0.670059 (* 1 = 0.670059 loss)
    I1227 19:13:50.957711  5629 sgd_solver.cpp:106] Iteration 36300, lr = 0.000221775
    I1227 19:13:58.838747  5629 solver.cpp:237] Iteration 36400, loss = 0.732406
    I1227 19:13:58.838943  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:13:58.838973  5629 solver.cpp:253]     Train net output #1: loss = 0.732406 (* 1 = 0.732406 loss)
    I1227 19:13:58.838994  5629 sgd_solver.cpp:106] Iteration 36400, lr = 0.000221416
    I1227 19:14:06.712234  5629 solver.cpp:237] Iteration 36500, loss = 0.740457
    I1227 19:14:06.712306  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:14:06.712344  5629 solver.cpp:253]     Train net output #1: loss = 0.740457 (* 1 = 0.740457 loss)
    I1227 19:14:06.712363  5629 sgd_solver.cpp:106] Iteration 36500, lr = 0.000221059
    I1227 19:14:14.582355  5629 solver.cpp:237] Iteration 36600, loss = 0.752963
    I1227 19:14:14.582422  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:14:14.582447  5629 solver.cpp:253]     Train net output #1: loss = 0.752963 (* 1 = 0.752963 loss)
    I1227 19:14:14.582465  5629 sgd_solver.cpp:106] Iteration 36600, lr = 0.000220703
    I1227 19:14:22.443888  5629 solver.cpp:237] Iteration 36700, loss = 0.634662
    I1227 19:14:22.443956  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:14:22.443981  5629 solver.cpp:253]     Train net output #1: loss = 0.634662 (* 1 = 0.634662 loss)
    I1227 19:14:22.444000  5629 sgd_solver.cpp:106] Iteration 36700, lr = 0.000220349
    I1227 19:14:30.324038  5629 solver.cpp:237] Iteration 36800, loss = 0.633993
    I1227 19:14:30.324200  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:14:30.324229  5629 solver.cpp:253]     Train net output #1: loss = 0.633993 (* 1 = 0.633993 loss)
    I1227 19:14:30.324249  5629 sgd_solver.cpp:106] Iteration 36800, lr = 0.000219995
    I1227 19:14:38.222034  5629 solver.cpp:237] Iteration 36900, loss = 0.832897
    I1227 19:14:38.222105  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:14:38.222131  5629 solver.cpp:253]     Train net output #1: loss = 0.832897 (* 1 = 0.832897 loss)
    I1227 19:14:38.222149  5629 sgd_solver.cpp:106] Iteration 36900, lr = 0.000219644
    I1227 19:14:46.021267  5629 solver.cpp:341] Iteration 37000, Testing net (#0)
    I1227 19:14:49.905037  5629 solver.cpp:409]     Test net output #0: accuracy = 0.723667
    I1227 19:14:49.905117  5629 solver.cpp:409]     Test net output #1: loss = 0.795503 (* 1 = 0.795503 loss)
    I1227 19:14:49.941962  5629 solver.cpp:237] Iteration 37000, loss = 0.723934
    I1227 19:14:49.942026  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:14:49.942050  5629 solver.cpp:253]     Train net output #1: loss = 0.723934 (* 1 = 0.723934 loss)
    I1227 19:14:49.942068  5629 sgd_solver.cpp:106] Iteration 37000, lr = 0.000219293
    I1227 19:14:57.771998  5629 solver.cpp:237] Iteration 37100, loss = 0.868884
    I1227 19:14:57.772064  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:14:57.772089  5629 solver.cpp:253]     Train net output #1: loss = 0.868884 (* 1 = 0.868884 loss)
    I1227 19:14:57.772107  5629 sgd_solver.cpp:106] Iteration 37100, lr = 0.000218944
    I1227 19:15:05.693009  5629 solver.cpp:237] Iteration 37200, loss = 0.651797
    I1227 19:15:05.693200  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:15:05.693233  5629 solver.cpp:253]     Train net output #1: loss = 0.651797 (* 1 = 0.651797 loss)
    I1227 19:15:05.693253  5629 sgd_solver.cpp:106] Iteration 37200, lr = 0.000218596
    I1227 19:15:13.566500  5629 solver.cpp:237] Iteration 37300, loss = 0.687449
    I1227 19:15:13.566575  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:15:13.566601  5629 solver.cpp:253]     Train net output #1: loss = 0.687449 (* 1 = 0.687449 loss)
    I1227 19:15:13.566620  5629 sgd_solver.cpp:106] Iteration 37300, lr = 0.000218249
    I1227 19:15:21.431413  5629 solver.cpp:237] Iteration 37400, loss = 0.757121
    I1227 19:15:21.431480  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:15:21.431505  5629 solver.cpp:253]     Train net output #1: loss = 0.757121 (* 1 = 0.757121 loss)
    I1227 19:15:21.431525  5629 sgd_solver.cpp:106] Iteration 37400, lr = 0.000217904
    I1227 19:15:29.340894  5629 solver.cpp:237] Iteration 37500, loss = 0.740825
    I1227 19:15:29.340962  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:15:29.340987  5629 solver.cpp:253]     Train net output #1: loss = 0.740825 (* 1 = 0.740825 loss)
    I1227 19:15:29.341006  5629 sgd_solver.cpp:106] Iteration 37500, lr = 0.000217559
    I1227 19:15:37.229995  5629 solver.cpp:237] Iteration 37600, loss = 0.822014
    I1227 19:15:37.230151  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:15:37.230181  5629 solver.cpp:253]     Train net output #1: loss = 0.822014 (* 1 = 0.822014 loss)
    I1227 19:15:37.230200  5629 sgd_solver.cpp:106] Iteration 37600, lr = 0.000217216
    I1227 19:15:45.103309  5629 solver.cpp:237] Iteration 37700, loss = 0.786153
    I1227 19:15:45.103381  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:15:45.103406  5629 solver.cpp:253]     Train net output #1: loss = 0.786153 (* 1 = 0.786153 loss)
    I1227 19:15:45.103425  5629 sgd_solver.cpp:106] Iteration 37700, lr = 0.000216875
    I1227 19:15:53.052253  5629 solver.cpp:237] Iteration 37800, loss = 0.613725
    I1227 19:15:53.052321  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:15:53.052348  5629 solver.cpp:253]     Train net output #1: loss = 0.613725 (* 1 = 0.613725 loss)
    I1227 19:15:53.052366  5629 sgd_solver.cpp:106] Iteration 37800, lr = 0.000216535
    I1227 19:16:00.952841  5629 solver.cpp:237] Iteration 37900, loss = 0.888936
    I1227 19:16:00.952910  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:16:00.952936  5629 solver.cpp:253]     Train net output #1: loss = 0.888936 (* 1 = 0.888936 loss)
    I1227 19:16:00.952956  5629 sgd_solver.cpp:106] Iteration 37900, lr = 0.000216195
    I1227 19:16:08.753762  5629 solver.cpp:341] Iteration 38000, Testing net (#0)
    I1227 19:16:12.640424  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7355
    I1227 19:16:12.640497  5629 solver.cpp:409]     Test net output #1: loss = 0.772539 (* 1 = 0.772539 loss)
    I1227 19:16:12.684836  5629 solver.cpp:237] Iteration 38000, loss = 0.699533
    I1227 19:16:12.684897  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:16:12.684921  5629 solver.cpp:253]     Train net output #1: loss = 0.699533 (* 1 = 0.699533 loss)
    I1227 19:16:12.684941  5629 sgd_solver.cpp:106] Iteration 38000, lr = 0.000215857
    I1227 19:16:20.548604  5629 solver.cpp:237] Iteration 38100, loss = 0.775749
    I1227 19:16:20.548671  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:16:20.548696  5629 solver.cpp:253]     Train net output #1: loss = 0.775749 (* 1 = 0.775749 loss)
    I1227 19:16:20.548715  5629 sgd_solver.cpp:106] Iteration 38100, lr = 0.000215521
    I1227 19:16:28.446470  5629 solver.cpp:237] Iteration 38200, loss = 0.640908
    I1227 19:16:28.446537  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:16:28.446560  5629 solver.cpp:253]     Train net output #1: loss = 0.640908 (* 1 = 0.640908 loss)
    I1227 19:16:28.446580  5629 sgd_solver.cpp:106] Iteration 38200, lr = 0.000215185
    I1227 19:16:36.315423  5629 solver.cpp:237] Iteration 38300, loss = 0.646644
    I1227 19:16:36.315493  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:16:36.315518  5629 solver.cpp:253]     Train net output #1: loss = 0.646644 (* 1 = 0.646644 loss)
    I1227 19:16:36.315538  5629 sgd_solver.cpp:106] Iteration 38300, lr = 0.000214851
    I1227 19:16:44.204912  5629 solver.cpp:237] Iteration 38400, loss = 0.852382
    I1227 19:16:44.205099  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:16:44.205129  5629 solver.cpp:253]     Train net output #1: loss = 0.852382 (* 1 = 0.852382 loss)
    I1227 19:16:44.205148  5629 sgd_solver.cpp:106] Iteration 38400, lr = 0.000214518
    I1227 19:16:52.106429  5629 solver.cpp:237] Iteration 38500, loss = 0.6952
    I1227 19:16:52.106498  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:16:52.106523  5629 solver.cpp:253]     Train net output #1: loss = 0.6952 (* 1 = 0.6952 loss)
    I1227 19:16:52.106540  5629 sgd_solver.cpp:106] Iteration 38500, lr = 0.000214186
    I1227 19:16:59.986012  5629 solver.cpp:237] Iteration 38600, loss = 0.876096
    I1227 19:16:59.986078  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:16:59.986104  5629 solver.cpp:253]     Train net output #1: loss = 0.876096 (* 1 = 0.876096 loss)
    I1227 19:16:59.986124  5629 sgd_solver.cpp:106] Iteration 38600, lr = 0.000213856
    I1227 19:17:07.875133  5629 solver.cpp:237] Iteration 38700, loss = 0.663463
    I1227 19:17:07.875200  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:17:07.875224  5629 solver.cpp:253]     Train net output #1: loss = 0.663463 (* 1 = 0.663463 loss)
    I1227 19:17:07.875243  5629 sgd_solver.cpp:106] Iteration 38700, lr = 0.000213526
    I1227 19:17:15.748582  5629 solver.cpp:237] Iteration 38800, loss = 0.743163
    I1227 19:17:15.748739  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:17:15.748770  5629 solver.cpp:253]     Train net output #1: loss = 0.743163 (* 1 = 0.743163 loss)
    I1227 19:17:15.748790  5629 sgd_solver.cpp:106] Iteration 38800, lr = 0.000213198
    I1227 19:17:23.627467  5629 solver.cpp:237] Iteration 38900, loss = 0.780641
    I1227 19:17:23.627537  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:17:23.627562  5629 solver.cpp:253]     Train net output #1: loss = 0.780641 (* 1 = 0.780641 loss)
    I1227 19:17:23.627580  5629 sgd_solver.cpp:106] Iteration 38900, lr = 0.000212871
    I1227 19:17:31.427171  5629 solver.cpp:341] Iteration 39000, Testing net (#0)
    I1227 19:17:35.273723  5629 solver.cpp:409]     Test net output #0: accuracy = 0.732667
    I1227 19:17:35.273828  5629 solver.cpp:409]     Test net output #1: loss = 0.774453 (* 1 = 0.774453 loss)
    I1227 19:17:35.319754  5629 solver.cpp:237] Iteration 39000, loss = 0.739432
    I1227 19:17:35.319823  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:17:35.319849  5629 solver.cpp:253]     Train net output #1: loss = 0.739432 (* 1 = 0.739432 loss)
    I1227 19:17:35.319869  5629 sgd_solver.cpp:106] Iteration 39000, lr = 0.000212545
    I1227 19:17:43.229491  5629 solver.cpp:237] Iteration 39100, loss = 0.941508
    I1227 19:17:43.229563  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:17:43.229589  5629 solver.cpp:253]     Train net output #1: loss = 0.941508 (* 1 = 0.941508 loss)
    I1227 19:17:43.229609  5629 sgd_solver.cpp:106] Iteration 39100, lr = 0.00021222
    I1227 19:17:51.120103  5629 solver.cpp:237] Iteration 39200, loss = 0.631492
    I1227 19:17:51.120286  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:17:51.120317  5629 solver.cpp:253]     Train net output #1: loss = 0.631492 (* 1 = 0.631492 loss)
    I1227 19:17:51.120337  5629 sgd_solver.cpp:106] Iteration 39200, lr = 0.000211897
    I1227 19:17:58.986712  5629 solver.cpp:237] Iteration 39300, loss = 0.739046
    I1227 19:17:58.986778  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:17:58.986804  5629 solver.cpp:253]     Train net output #1: loss = 0.739046 (* 1 = 0.739046 loss)
    I1227 19:17:58.986824  5629 sgd_solver.cpp:106] Iteration 39300, lr = 0.000211574
    I1227 19:18:06.865783  5629 solver.cpp:237] Iteration 39400, loss = 0.809162
    I1227 19:18:06.865855  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:18:06.865880  5629 solver.cpp:253]     Train net output #1: loss = 0.809162 (* 1 = 0.809162 loss)
    I1227 19:18:06.865900  5629 sgd_solver.cpp:106] Iteration 39400, lr = 0.000211253
    I1227 19:18:14.732743  5629 solver.cpp:237] Iteration 39500, loss = 0.633003
    I1227 19:18:14.732810  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:18:14.732834  5629 solver.cpp:253]     Train net output #1: loss = 0.633003 (* 1 = 0.633003 loss)
    I1227 19:18:14.732853  5629 sgd_solver.cpp:106] Iteration 39500, lr = 0.000210933
    I1227 19:18:22.604977  5629 solver.cpp:237] Iteration 39600, loss = 0.781198
    I1227 19:18:22.605183  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:18:22.605216  5629 solver.cpp:253]     Train net output #1: loss = 0.781198 (* 1 = 0.781198 loss)
    I1227 19:18:22.605234  5629 sgd_solver.cpp:106] Iteration 39600, lr = 0.000210614
    I1227 19:18:30.505702  5629 solver.cpp:237] Iteration 39700, loss = 0.605052
    I1227 19:18:30.505766  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:18:30.505791  5629 solver.cpp:253]     Train net output #1: loss = 0.605052 (* 1 = 0.605052 loss)
    I1227 19:18:30.505810  5629 sgd_solver.cpp:106] Iteration 39700, lr = 0.000210296
    I1227 19:18:38.398485  5629 solver.cpp:237] Iteration 39800, loss = 0.591872
    I1227 19:18:38.398553  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:18:38.398579  5629 solver.cpp:253]     Train net output #1: loss = 0.591872 (* 1 = 0.591872 loss)
    I1227 19:18:38.398599  5629 sgd_solver.cpp:106] Iteration 39800, lr = 0.000209979
    I1227 19:18:46.274224  5629 solver.cpp:237] Iteration 39900, loss = 0.760476
    I1227 19:18:46.274288  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:18:46.274312  5629 solver.cpp:253]     Train net output #1: loss = 0.760476 (* 1 = 0.760476 loss)
    I1227 19:18:46.274333  5629 sgd_solver.cpp:106] Iteration 39900, lr = 0.000209663
    I1227 19:18:54.082079  5629 solver.cpp:341] Iteration 40000, Testing net (#0)
    I1227 19:18:57.941939  5629 solver.cpp:409]     Test net output #0: accuracy = 0.72075
    I1227 19:18:57.942042  5629 solver.cpp:409]     Test net output #1: loss = 0.798005 (* 1 = 0.798005 loss)
    I1227 19:18:57.986677  5629 solver.cpp:237] Iteration 40000, loss = 0.640817
    I1227 19:18:57.986757  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:18:57.986795  5629 solver.cpp:253]     Train net output #1: loss = 0.640817 (* 1 = 0.640817 loss)
    I1227 19:18:57.986815  5629 sgd_solver.cpp:106] Iteration 40000, lr = 0.000209349
    I1227 19:19:05.894341  5629 solver.cpp:237] Iteration 40100, loss = 0.781815
    I1227 19:19:05.894408  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:19:05.894434  5629 solver.cpp:253]     Train net output #1: loss = 0.781815 (* 1 = 0.781815 loss)
    I1227 19:19:05.894454  5629 sgd_solver.cpp:106] Iteration 40100, lr = 0.000209035
    I1227 19:19:13.777036  5629 solver.cpp:237] Iteration 40200, loss = 0.625023
    I1227 19:19:13.777102  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:19:13.777127  5629 solver.cpp:253]     Train net output #1: loss = 0.625023 (* 1 = 0.625023 loss)
    I1227 19:19:13.777146  5629 sgd_solver.cpp:106] Iteration 40200, lr = 0.000208723
    I1227 19:19:21.654808  5629 solver.cpp:237] Iteration 40300, loss = 0.601373
    I1227 19:19:21.654876  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:19:21.654901  5629 solver.cpp:253]     Train net output #1: loss = 0.601373 (* 1 = 0.601373 loss)
    I1227 19:19:21.654920  5629 sgd_solver.cpp:106] Iteration 40300, lr = 0.000208412
    I1227 19:19:29.548940  5629 solver.cpp:237] Iteration 40400, loss = 0.797915
    I1227 19:19:29.549131  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:19:29.549161  5629 solver.cpp:253]     Train net output #1: loss = 0.797915 (* 1 = 0.797915 loss)
    I1227 19:19:29.549180  5629 sgd_solver.cpp:106] Iteration 40400, lr = 0.000208101
    I1227 19:19:37.404359  5629 solver.cpp:237] Iteration 40500, loss = 0.612293
    I1227 19:19:37.404430  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:19:37.404455  5629 solver.cpp:253]     Train net output #1: loss = 0.612293 (* 1 = 0.612293 loss)
    I1227 19:19:37.404474  5629 sgd_solver.cpp:106] Iteration 40500, lr = 0.000207792
    I1227 19:19:45.296057  5629 solver.cpp:237] Iteration 40600, loss = 0.752782
    I1227 19:19:45.296128  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:19:45.296154  5629 solver.cpp:253]     Train net output #1: loss = 0.752782 (* 1 = 0.752782 loss)
    I1227 19:19:45.296174  5629 sgd_solver.cpp:106] Iteration 40600, lr = 0.000207484
    I1227 19:19:53.169472  5629 solver.cpp:237] Iteration 40700, loss = 0.601384
    I1227 19:19:53.169541  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:19:53.169567  5629 solver.cpp:253]     Train net output #1: loss = 0.601384 (* 1 = 0.601384 loss)
    I1227 19:19:53.169586  5629 sgd_solver.cpp:106] Iteration 40700, lr = 0.000207177
    I1227 19:20:01.036056  5629 solver.cpp:237] Iteration 40800, loss = 0.709791
    I1227 19:20:01.036257  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:20:01.036290  5629 solver.cpp:253]     Train net output #1: loss = 0.709791 (* 1 = 0.709791 loss)
    I1227 19:20:01.036310  5629 sgd_solver.cpp:106] Iteration 40800, lr = 0.000206871
    I1227 19:20:08.908018  5629 solver.cpp:237] Iteration 40900, loss = 0.683971
    I1227 19:20:08.908095  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:20:08.908121  5629 solver.cpp:253]     Train net output #1: loss = 0.683971 (* 1 = 0.683971 loss)
    I1227 19:20:08.908141  5629 sgd_solver.cpp:106] Iteration 40900, lr = 0.000206566
    I1227 19:20:16.721406  5629 solver.cpp:341] Iteration 41000, Testing net (#0)
    I1227 19:20:20.606771  5629 solver.cpp:409]     Test net output #0: accuracy = 0.734667
    I1227 19:20:20.606847  5629 solver.cpp:409]     Test net output #1: loss = 0.76511 (* 1 = 0.76511 loss)
    I1227 19:20:20.651396  5629 solver.cpp:237] Iteration 41000, loss = 0.669187
    I1227 19:20:20.651463  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:20:20.651486  5629 solver.cpp:253]     Train net output #1: loss = 0.669187 (* 1 = 0.669187 loss)
    I1227 19:20:20.651506  5629 sgd_solver.cpp:106] Iteration 41000, lr = 0.000206263
    I1227 19:20:28.517985  5629 solver.cpp:237] Iteration 41100, loss = 0.810521
    I1227 19:20:28.518054  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:20:28.518079  5629 solver.cpp:253]     Train net output #1: loss = 0.810521 (* 1 = 0.810521 loss)
    I1227 19:20:28.518098  5629 sgd_solver.cpp:106] Iteration 41100, lr = 0.00020596
    I1227 19:20:36.398721  5629 solver.cpp:237] Iteration 41200, loss = 0.716567
    I1227 19:20:36.398912  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:20:36.398943  5629 solver.cpp:253]     Train net output #1: loss = 0.716567 (* 1 = 0.716567 loss)
    I1227 19:20:36.398962  5629 sgd_solver.cpp:106] Iteration 41200, lr = 0.000205658
    I1227 19:20:44.285955  5629 solver.cpp:237] Iteration 41300, loss = 0.681617
    I1227 19:20:44.286023  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:20:44.286048  5629 solver.cpp:253]     Train net output #1: loss = 0.681617 (* 1 = 0.681617 loss)
    I1227 19:20:44.286068  5629 sgd_solver.cpp:106] Iteration 41300, lr = 0.000205357
    I1227 19:20:52.166404  5629 solver.cpp:237] Iteration 41400, loss = 0.682773
    I1227 19:20:52.166473  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:20:52.166499  5629 solver.cpp:253]     Train net output #1: loss = 0.682773 (* 1 = 0.682773 loss)
    I1227 19:20:52.166519  5629 sgd_solver.cpp:106] Iteration 41400, lr = 0.000205058
    I1227 19:21:00.028183  5629 solver.cpp:237] Iteration 41500, loss = 0.627834
    I1227 19:21:00.028249  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:21:00.028275  5629 solver.cpp:253]     Train net output #1: loss = 0.627834 (* 1 = 0.627834 loss)
    I1227 19:21:00.028295  5629 sgd_solver.cpp:106] Iteration 41500, lr = 0.000204759
    I1227 19:21:07.893319  5629 solver.cpp:237] Iteration 41600, loss = 0.70616
    I1227 19:21:07.893499  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:21:07.893529  5629 solver.cpp:253]     Train net output #1: loss = 0.70616 (* 1 = 0.70616 loss)
    I1227 19:21:07.893548  5629 sgd_solver.cpp:106] Iteration 41600, lr = 0.000204461
    I1227 19:21:15.782202  5629 solver.cpp:237] Iteration 41700, loss = 0.662334
    I1227 19:21:15.782266  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:21:15.782291  5629 solver.cpp:253]     Train net output #1: loss = 0.662334 (* 1 = 0.662334 loss)
    I1227 19:21:15.782310  5629 sgd_solver.cpp:106] Iteration 41700, lr = 0.000204164
    I1227 19:21:23.644342  5629 solver.cpp:237] Iteration 41800, loss = 0.747335
    I1227 19:21:23.644410  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:21:23.644434  5629 solver.cpp:253]     Train net output #1: loss = 0.747335 (* 1 = 0.747335 loss)
    I1227 19:21:23.644454  5629 sgd_solver.cpp:106] Iteration 41800, lr = 0.000203869
    I1227 19:21:31.524055  5629 solver.cpp:237] Iteration 41900, loss = 0.70612
    I1227 19:21:31.524122  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:21:31.524148  5629 solver.cpp:253]     Train net output #1: loss = 0.70612 (* 1 = 0.70612 loss)
    I1227 19:21:31.524168  5629 sgd_solver.cpp:106] Iteration 41900, lr = 0.000203574
    I1227 19:21:39.306557  5629 solver.cpp:341] Iteration 42000, Testing net (#0)
    I1227 19:21:43.183233  5629 solver.cpp:409]     Test net output #0: accuracy = 0.730667
    I1227 19:21:43.183305  5629 solver.cpp:409]     Test net output #1: loss = 0.781169 (* 1 = 0.781169 loss)
    I1227 19:21:43.227742  5629 solver.cpp:237] Iteration 42000, loss = 0.669507
    I1227 19:21:43.227802  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:21:43.227825  5629 solver.cpp:253]     Train net output #1: loss = 0.669507 (* 1 = 0.669507 loss)
    I1227 19:21:43.227844  5629 sgd_solver.cpp:106] Iteration 42000, lr = 0.00020328
    I1227 19:21:51.117065  5629 solver.cpp:237] Iteration 42100, loss = 0.81619
    I1227 19:21:51.117132  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:21:51.117158  5629 solver.cpp:253]     Train net output #1: loss = 0.81619 (* 1 = 0.81619 loss)
    I1227 19:21:51.117177  5629 sgd_solver.cpp:106] Iteration 42100, lr = 0.000202988
    I1227 19:21:58.972724  5629 solver.cpp:237] Iteration 42200, loss = 0.635761
    I1227 19:21:58.972792  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:21:58.972818  5629 solver.cpp:253]     Train net output #1: loss = 0.635761 (* 1 = 0.635761 loss)
    I1227 19:21:58.972837  5629 sgd_solver.cpp:106] Iteration 42200, lr = 0.000202696
    I1227 19:22:06.860569  5629 solver.cpp:237] Iteration 42300, loss = 0.781373
    I1227 19:22:06.860637  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:22:06.860662  5629 solver.cpp:253]     Train net output #1: loss = 0.781373 (* 1 = 0.781373 loss)
    I1227 19:22:06.860682  5629 sgd_solver.cpp:106] Iteration 42300, lr = 0.000202405
    I1227 19:22:14.719480  5629 solver.cpp:237] Iteration 42400, loss = 0.759061
    I1227 19:22:14.719637  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:22:14.719666  5629 solver.cpp:253]     Train net output #1: loss = 0.759061 (* 1 = 0.759061 loss)
    I1227 19:22:14.719684  5629 sgd_solver.cpp:106] Iteration 42400, lr = 0.000202115
    I1227 19:22:22.577955  5629 solver.cpp:237] Iteration 42500, loss = 0.602044
    I1227 19:22:22.578022  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:22:22.578047  5629 solver.cpp:253]     Train net output #1: loss = 0.602044 (* 1 = 0.602044 loss)
    I1227 19:22:22.578064  5629 sgd_solver.cpp:106] Iteration 42500, lr = 0.000201827
    I1227 19:22:30.469877  5629 solver.cpp:237] Iteration 42600, loss = 0.779066
    I1227 19:22:30.469944  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:22:30.469969  5629 solver.cpp:253]     Train net output #1: loss = 0.779066 (* 1 = 0.779066 loss)
    I1227 19:22:30.469990  5629 sgd_solver.cpp:106] Iteration 42600, lr = 0.000201539
    I1227 19:22:38.318867  5629 solver.cpp:237] Iteration 42700, loss = 0.666527
    I1227 19:22:38.318931  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:22:38.318956  5629 solver.cpp:253]     Train net output #1: loss = 0.666527 (* 1 = 0.666527 loss)
    I1227 19:22:38.318974  5629 sgd_solver.cpp:106] Iteration 42700, lr = 0.000201252
    I1227 19:22:46.199802  5629 solver.cpp:237] Iteration 42800, loss = 0.702061
    I1227 19:22:46.200004  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:22:46.200036  5629 solver.cpp:253]     Train net output #1: loss = 0.702061 (* 1 = 0.702061 loss)
    I1227 19:22:46.200054  5629 sgd_solver.cpp:106] Iteration 42800, lr = 0.000200966
    I1227 19:22:54.075716  5629 solver.cpp:237] Iteration 42900, loss = 0.759696
    I1227 19:22:54.075778  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:22:54.075801  5629 solver.cpp:253]     Train net output #1: loss = 0.759696 (* 1 = 0.759696 loss)
    I1227 19:22:54.075820  5629 sgd_solver.cpp:106] Iteration 42900, lr = 0.000200681
    I1227 19:23:01.874371  5629 solver.cpp:341] Iteration 43000, Testing net (#0)
    I1227 19:23:05.700530  5629 solver.cpp:409]     Test net output #0: accuracy = 0.73875
    I1227 19:23:05.700613  5629 solver.cpp:409]     Test net output #1: loss = 0.754963 (* 1 = 0.754963 loss)
    I1227 19:23:05.743774  5629 solver.cpp:237] Iteration 43000, loss = 0.630615
    I1227 19:23:05.743839  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:23:05.743863  5629 solver.cpp:253]     Train net output #1: loss = 0.630615 (* 1 = 0.630615 loss)
    I1227 19:23:05.743885  5629 sgd_solver.cpp:106] Iteration 43000, lr = 0.000200397
    I1227 19:23:13.602597  5629 solver.cpp:237] Iteration 43100, loss = 0.726531
    I1227 19:23:13.602663  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:23:13.602687  5629 solver.cpp:253]     Train net output #1: loss = 0.726531 (* 1 = 0.726531 loss)
    I1227 19:23:13.602706  5629 sgd_solver.cpp:106] Iteration 43100, lr = 0.000200114
    I1227 19:23:21.483057  5629 solver.cpp:237] Iteration 43200, loss = 0.639133
    I1227 19:23:21.483259  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:23:21.483294  5629 solver.cpp:253]     Train net output #1: loss = 0.639133 (* 1 = 0.639133 loss)
    I1227 19:23:21.483314  5629 sgd_solver.cpp:106] Iteration 43200, lr = 0.000199832
    I1227 19:23:29.353603  5629 solver.cpp:237] Iteration 43300, loss = 0.717183
    I1227 19:23:29.353670  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:23:29.353694  5629 solver.cpp:253]     Train net output #1: loss = 0.717183 (* 1 = 0.717183 loss)
    I1227 19:23:29.353713  5629 sgd_solver.cpp:106] Iteration 43300, lr = 0.00019955
    I1227 19:23:37.231379  5629 solver.cpp:237] Iteration 43400, loss = 0.70372
    I1227 19:23:37.231444  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:23:37.231469  5629 solver.cpp:253]     Train net output #1: loss = 0.70372 (* 1 = 0.70372 loss)
    I1227 19:23:37.231487  5629 sgd_solver.cpp:106] Iteration 43400, lr = 0.00019927
    I1227 19:23:45.140101  5629 solver.cpp:237] Iteration 43500, loss = 0.724212
    I1227 19:23:45.140166  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:23:45.140190  5629 solver.cpp:253]     Train net output #1: loss = 0.724212 (* 1 = 0.724212 loss)
    I1227 19:23:45.140210  5629 sgd_solver.cpp:106] Iteration 43500, lr = 0.000198991
    I1227 19:23:53.014721  5629 solver.cpp:237] Iteration 43600, loss = 0.794098
    I1227 19:23:53.014922  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:23:53.014951  5629 solver.cpp:253]     Train net output #1: loss = 0.794098 (* 1 = 0.794098 loss)
    I1227 19:23:53.014971  5629 sgd_solver.cpp:106] Iteration 43600, lr = 0.000198712
    I1227 19:24:00.899709  5629 solver.cpp:237] Iteration 43700, loss = 0.612365
    I1227 19:24:00.899775  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:24:00.899801  5629 solver.cpp:253]     Train net output #1: loss = 0.612365 (* 1 = 0.612365 loss)
    I1227 19:24:00.899819  5629 sgd_solver.cpp:106] Iteration 43700, lr = 0.000198435
    I1227 19:24:08.782966  5629 solver.cpp:237] Iteration 43800, loss = 0.704936
    I1227 19:24:08.783031  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:24:08.783056  5629 solver.cpp:253]     Train net output #1: loss = 0.704936 (* 1 = 0.704936 loss)
    I1227 19:24:08.783076  5629 sgd_solver.cpp:106] Iteration 43800, lr = 0.000198158
    I1227 19:24:16.619256  5629 solver.cpp:237] Iteration 43900, loss = 0.76238
    I1227 19:24:16.619328  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:24:16.619354  5629 solver.cpp:253]     Train net output #1: loss = 0.76238 (* 1 = 0.76238 loss)
    I1227 19:24:16.619372  5629 sgd_solver.cpp:106] Iteration 43900, lr = 0.000197882
    I1227 19:24:24.373172  5629 solver.cpp:341] Iteration 44000, Testing net (#0)
    I1227 19:24:28.168866  5629 solver.cpp:409]     Test net output #0: accuracy = 0.734583
    I1227 19:24:28.168941  5629 solver.cpp:409]     Test net output #1: loss = 0.762369 (* 1 = 0.762369 loss)
    I1227 19:24:28.214247  5629 solver.cpp:237] Iteration 44000, loss = 0.621862
    I1227 19:24:28.214309  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:24:28.214334  5629 solver.cpp:253]     Train net output #1: loss = 0.621862 (* 1 = 0.621862 loss)
    I1227 19:24:28.214352  5629 sgd_solver.cpp:106] Iteration 44000, lr = 0.000197607
    I1227 19:24:36.134528  5629 solver.cpp:237] Iteration 44100, loss = 0.803305
    I1227 19:24:36.134595  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:24:36.134619  5629 solver.cpp:253]     Train net output #1: loss = 0.803305 (* 1 = 0.803305 loss)
    I1227 19:24:36.134639  5629 sgd_solver.cpp:106] Iteration 44100, lr = 0.000197333
    I1227 19:24:44.017745  5629 solver.cpp:237] Iteration 44200, loss = 0.626528
    I1227 19:24:44.017809  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:24:44.017834  5629 solver.cpp:253]     Train net output #1: loss = 0.626528 (* 1 = 0.626528 loss)
    I1227 19:24:44.017854  5629 sgd_solver.cpp:106] Iteration 44200, lr = 0.00019706
    I1227 19:24:51.908865  5629 solver.cpp:237] Iteration 44300, loss = 0.618802
    I1227 19:24:51.908929  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:24:51.908953  5629 solver.cpp:253]     Train net output #1: loss = 0.618802 (* 1 = 0.618802 loss)
    I1227 19:24:51.908973  5629 sgd_solver.cpp:106] Iteration 44300, lr = 0.000196788
    I1227 19:24:59.807692  5629 solver.cpp:237] Iteration 44400, loss = 0.723382
    I1227 19:24:59.807852  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:24:59.807879  5629 solver.cpp:253]     Train net output #1: loss = 0.723382 (* 1 = 0.723382 loss)
    I1227 19:24:59.807898  5629 sgd_solver.cpp:106] Iteration 44400, lr = 0.000196516
    I1227 19:25:07.694928  5629 solver.cpp:237] Iteration 44500, loss = 0.62186
    I1227 19:25:07.694994  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:25:07.695019  5629 solver.cpp:253]     Train net output #1: loss = 0.62186 (* 1 = 0.62186 loss)
    I1227 19:25:07.695037  5629 sgd_solver.cpp:106] Iteration 44500, lr = 0.000196246
    I1227 19:25:15.542748  5629 solver.cpp:237] Iteration 44600, loss = 0.796996
    I1227 19:25:15.542814  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:25:15.542840  5629 solver.cpp:253]     Train net output #1: loss = 0.796996 (* 1 = 0.796996 loss)
    I1227 19:25:15.542857  5629 sgd_solver.cpp:106] Iteration 44600, lr = 0.000195976
    I1227 19:25:23.417446  5629 solver.cpp:237] Iteration 44700, loss = 0.647933
    I1227 19:25:23.417510  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:25:23.417534  5629 solver.cpp:253]     Train net output #1: loss = 0.647933 (* 1 = 0.647933 loss)
    I1227 19:25:23.417554  5629 sgd_solver.cpp:106] Iteration 44700, lr = 0.000195708
    I1227 19:25:31.279021  5629 solver.cpp:237] Iteration 44800, loss = 0.753293
    I1227 19:25:31.279189  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:25:31.279218  5629 solver.cpp:253]     Train net output #1: loss = 0.753293 (* 1 = 0.753293 loss)
    I1227 19:25:31.279237  5629 sgd_solver.cpp:106] Iteration 44800, lr = 0.00019544
    I1227 19:25:39.156158  5629 solver.cpp:237] Iteration 44900, loss = 0.710091
    I1227 19:25:39.156225  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:25:39.156250  5629 solver.cpp:253]     Train net output #1: loss = 0.710091 (* 1 = 0.710091 loss)
    I1227 19:25:39.156270  5629 sgd_solver.cpp:106] Iteration 44900, lr = 0.000195173
    I1227 19:25:46.931641  5629 solver.cpp:341] Iteration 45000, Testing net (#0)
    I1227 19:25:50.776352  5629 solver.cpp:409]     Test net output #0: accuracy = 0.73725
    I1227 19:25:50.776428  5629 solver.cpp:409]     Test net output #1: loss = 0.755565 (* 1 = 0.755565 loss)
    I1227 19:25:50.821238  5629 solver.cpp:237] Iteration 45000, loss = 0.693214
    I1227 19:25:50.821302  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:25:50.821326  5629 solver.cpp:253]     Train net output #1: loss = 0.693214 (* 1 = 0.693214 loss)
    I1227 19:25:50.821346  5629 sgd_solver.cpp:106] Iteration 45000, lr = 0.000194906
    I1227 19:25:58.697368  5629 solver.cpp:237] Iteration 45100, loss = 0.827528
    I1227 19:25:58.697435  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:25:58.697460  5629 solver.cpp:253]     Train net output #1: loss = 0.827528 (* 1 = 0.827528 loss)
    I1227 19:25:58.697479  5629 sgd_solver.cpp:106] Iteration 45100, lr = 0.000194641
    I1227 19:26:06.576812  5629 solver.cpp:237] Iteration 45200, loss = 0.643979
    I1227 19:26:06.577026  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:26:06.577061  5629 solver.cpp:253]     Train net output #1: loss = 0.643979 (* 1 = 0.643979 loss)
    I1227 19:26:06.577081  5629 sgd_solver.cpp:106] Iteration 45200, lr = 0.000194376
    I1227 19:26:14.488456  5629 solver.cpp:237] Iteration 45300, loss = 0.646529
    I1227 19:26:14.488520  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:26:14.488546  5629 solver.cpp:253]     Train net output #1: loss = 0.646529 (* 1 = 0.646529 loss)
    I1227 19:26:14.488566  5629 sgd_solver.cpp:106] Iteration 45300, lr = 0.000194113
    I1227 19:26:22.340754  5629 solver.cpp:237] Iteration 45400, loss = 0.836886
    I1227 19:26:22.340818  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:26:22.340844  5629 solver.cpp:253]     Train net output #1: loss = 0.836886 (* 1 = 0.836886 loss)
    I1227 19:26:22.340863  5629 sgd_solver.cpp:106] Iteration 45400, lr = 0.00019385
    I1227 19:26:30.218794  5629 solver.cpp:237] Iteration 45500, loss = 0.717558
    I1227 19:26:30.218864  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:26:30.218889  5629 solver.cpp:253]     Train net output #1: loss = 0.717558 (* 1 = 0.717558 loss)
    I1227 19:26:30.218909  5629 sgd_solver.cpp:106] Iteration 45500, lr = 0.000193588
    I1227 19:26:38.095867  5629 solver.cpp:237] Iteration 45600, loss = 0.797945
    I1227 19:26:38.096084  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:26:38.096119  5629 solver.cpp:253]     Train net output #1: loss = 0.797945 (* 1 = 0.797945 loss)
    I1227 19:26:38.096139  5629 sgd_solver.cpp:106] Iteration 45600, lr = 0.000193327
    I1227 19:26:45.976294  5629 solver.cpp:237] Iteration 45700, loss = 0.687948
    I1227 19:26:45.976363  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:26:45.976387  5629 solver.cpp:253]     Train net output #1: loss = 0.687948 (* 1 = 0.687948 loss)
    I1227 19:26:45.976407  5629 sgd_solver.cpp:106] Iteration 45700, lr = 0.000193066
    I1227 19:26:53.853889  5629 solver.cpp:237] Iteration 45800, loss = 0.739976
    I1227 19:26:53.853955  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:26:53.853978  5629 solver.cpp:253]     Train net output #1: loss = 0.739976 (* 1 = 0.739976 loss)
    I1227 19:26:53.853999  5629 sgd_solver.cpp:106] Iteration 45800, lr = 0.000192807
    I1227 19:27:01.739068  5629 solver.cpp:237] Iteration 45900, loss = 0.671548
    I1227 19:27:01.739135  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:27:01.739161  5629 solver.cpp:253]     Train net output #1: loss = 0.671548 (* 1 = 0.671548 loss)
    I1227 19:27:01.739179  5629 sgd_solver.cpp:106] Iteration 45900, lr = 0.000192548
    I1227 19:27:09.526141  5629 solver.cpp:341] Iteration 46000, Testing net (#0)
    I1227 19:27:13.390812  5629 solver.cpp:409]     Test net output #0: accuracy = 0.729333
    I1227 19:27:13.390892  5629 solver.cpp:409]     Test net output #1: loss = 0.780365 (* 1 = 0.780365 loss)
    I1227 19:27:13.435550  5629 solver.cpp:237] Iteration 46000, loss = 0.577577
    I1227 19:27:13.435613  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:27:13.435637  5629 solver.cpp:253]     Train net output #1: loss = 0.577577 (* 1 = 0.577577 loss)
    I1227 19:27:13.435658  5629 sgd_solver.cpp:106] Iteration 46000, lr = 0.00019229
    I1227 19:27:21.316927  5629 solver.cpp:237] Iteration 46100, loss = 0.791862
    I1227 19:27:21.316997  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:27:21.317021  5629 solver.cpp:253]     Train net output #1: loss = 0.791862 (* 1 = 0.791862 loss)
    I1227 19:27:21.317040  5629 sgd_solver.cpp:106] Iteration 46100, lr = 0.000192033
    I1227 19:27:29.194573  5629 solver.cpp:237] Iteration 46200, loss = 0.650285
    I1227 19:27:29.194638  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:27:29.194663  5629 solver.cpp:253]     Train net output #1: loss = 0.650285 (* 1 = 0.650285 loss)
    I1227 19:27:29.194681  5629 sgd_solver.cpp:106] Iteration 46200, lr = 0.000191777
    I1227 19:27:37.093590  5629 solver.cpp:237] Iteration 46300, loss = 0.737288
    I1227 19:27:37.093657  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:27:37.093682  5629 solver.cpp:253]     Train net output #1: loss = 0.737288 (* 1 = 0.737288 loss)
    I1227 19:27:37.093701  5629 sgd_solver.cpp:106] Iteration 46300, lr = 0.000191521
    I1227 19:27:44.964987  5629 solver.cpp:237] Iteration 46400, loss = 0.869715
    I1227 19:27:44.965178  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:27:44.965209  5629 solver.cpp:253]     Train net output #1: loss = 0.869715 (* 1 = 0.869715 loss)
    I1227 19:27:44.965229  5629 sgd_solver.cpp:106] Iteration 46400, lr = 0.000191266
    I1227 19:27:52.824735  5629 solver.cpp:237] Iteration 46500, loss = 0.628857
    I1227 19:27:52.824802  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:27:52.824827  5629 solver.cpp:253]     Train net output #1: loss = 0.628857 (* 1 = 0.628857 loss)
    I1227 19:27:52.824846  5629 sgd_solver.cpp:106] Iteration 46500, lr = 0.000191012
    I1227 19:28:00.695497  5629 solver.cpp:237] Iteration 46600, loss = 0.847775
    I1227 19:28:00.695567  5629 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1227 19:28:00.695592  5629 solver.cpp:253]     Train net output #1: loss = 0.847775 (* 1 = 0.847775 loss)
    I1227 19:28:00.695612  5629 sgd_solver.cpp:106] Iteration 46600, lr = 0.000190759
    I1227 19:28:08.576433  5629 solver.cpp:237] Iteration 46700, loss = 0.701016
    I1227 19:28:08.576503  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:28:08.576529  5629 solver.cpp:253]     Train net output #1: loss = 0.701016 (* 1 = 0.701016 loss)
    I1227 19:28:08.576547  5629 sgd_solver.cpp:106] Iteration 46700, lr = 0.000190507
    I1227 19:28:16.470142  5629 solver.cpp:237] Iteration 46800, loss = 0.755109
    I1227 19:28:16.470324  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:28:16.470356  5629 solver.cpp:253]     Train net output #1: loss = 0.755109 (* 1 = 0.755109 loss)
    I1227 19:28:16.470377  5629 sgd_solver.cpp:106] Iteration 46800, lr = 0.000190255
    I1227 19:28:24.346917  5629 solver.cpp:237] Iteration 46900, loss = 0.768113
    I1227 19:28:24.346983  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:28:24.347007  5629 solver.cpp:253]     Train net output #1: loss = 0.768113 (* 1 = 0.768113 loss)
    I1227 19:28:24.347025  5629 sgd_solver.cpp:106] Iteration 46900, lr = 0.000190004
    I1227 19:28:32.158215  5629 solver.cpp:341] Iteration 47000, Testing net (#0)
    I1227 19:28:36.019790  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7365
    I1227 19:28:36.019865  5629 solver.cpp:409]     Test net output #1: loss = 0.749705 (* 1 = 0.749705 loss)
    I1227 19:28:36.064330  5629 solver.cpp:237] Iteration 47000, loss = 0.627489
    I1227 19:28:36.064394  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:28:36.064419  5629 solver.cpp:253]     Train net output #1: loss = 0.627489 (* 1 = 0.627489 loss)
    I1227 19:28:36.064437  5629 sgd_solver.cpp:106] Iteration 47000, lr = 0.000189754
    I1227 19:28:43.949857  5629 solver.cpp:237] Iteration 47100, loss = 0.753908
    I1227 19:28:43.949926  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:28:43.949951  5629 solver.cpp:253]     Train net output #1: loss = 0.753908 (* 1 = 0.753908 loss)
    I1227 19:28:43.949970  5629 sgd_solver.cpp:106] Iteration 47100, lr = 0.000189505
    I1227 19:28:51.801384  5629 solver.cpp:237] Iteration 47200, loss = 0.558527
    I1227 19:28:51.801575  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:28:51.801605  5629 solver.cpp:253]     Train net output #1: loss = 0.558527 (* 1 = 0.558527 loss)
    I1227 19:28:51.801625  5629 sgd_solver.cpp:106] Iteration 47200, lr = 0.000189257
    I1227 19:28:59.664652  5629 solver.cpp:237] Iteration 47300, loss = 0.677518
    I1227 19:28:59.664722  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:28:59.664747  5629 solver.cpp:253]     Train net output #1: loss = 0.677518 (* 1 = 0.677518 loss)
    I1227 19:28:59.664767  5629 sgd_solver.cpp:106] Iteration 47300, lr = 0.000189009
    I1227 19:29:07.539906  5629 solver.cpp:237] Iteration 47400, loss = 0.701609
    I1227 19:29:07.539976  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:29:07.540002  5629 solver.cpp:253]     Train net output #1: loss = 0.701609 (* 1 = 0.701609 loss)
    I1227 19:29:07.540020  5629 sgd_solver.cpp:106] Iteration 47400, lr = 0.000188762
    I1227 19:29:15.418875  5629 solver.cpp:237] Iteration 47500, loss = 0.682579
    I1227 19:29:15.418942  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:29:15.418967  5629 solver.cpp:253]     Train net output #1: loss = 0.682579 (* 1 = 0.682579 loss)
    I1227 19:29:15.418987  5629 sgd_solver.cpp:106] Iteration 47500, lr = 0.000188516
    I1227 19:29:23.188863  5629 solver.cpp:237] Iteration 47600, loss = 0.760918
    I1227 19:29:23.189057  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:29:23.189087  5629 solver.cpp:253]     Train net output #1: loss = 0.760918 (* 1 = 0.760918 loss)
    I1227 19:29:23.189107  5629 sgd_solver.cpp:106] Iteration 47600, lr = 0.00018827
    I1227 19:29:31.272074  5629 solver.cpp:237] Iteration 47700, loss = 0.60145
    I1227 19:29:31.272150  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:29:31.272176  5629 solver.cpp:253]     Train net output #1: loss = 0.60145 (* 1 = 0.60145 loss)
    I1227 19:29:31.272195  5629 sgd_solver.cpp:106] Iteration 47700, lr = 0.000188025
    I1227 19:29:40.005762  5629 solver.cpp:237] Iteration 47800, loss = 0.587554
    I1227 19:29:40.005828  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 19:29:40.005853  5629 solver.cpp:253]     Train net output #1: loss = 0.587554 (* 1 = 0.587554 loss)
    I1227 19:29:40.005872  5629 sgd_solver.cpp:106] Iteration 47800, lr = 0.000187781
    I1227 19:29:47.854125  5629 solver.cpp:237] Iteration 47900, loss = 0.804377
    I1227 19:29:47.854190  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:29:47.854215  5629 solver.cpp:253]     Train net output #1: loss = 0.804377 (* 1 = 0.804377 loss)
    I1227 19:29:47.854233  5629 sgd_solver.cpp:106] Iteration 47900, lr = 0.000187538
    I1227 19:29:54.803817  5629 solver.cpp:341] Iteration 48000, Testing net (#0)
    I1227 19:29:57.741731  5629 solver.cpp:409]     Test net output #0: accuracy = 0.741833
    I1227 19:29:57.741804  5629 solver.cpp:409]     Test net output #1: loss = 0.73145 (* 1 = 0.73145 loss)
    I1227 19:29:57.776412  5629 solver.cpp:237] Iteration 48000, loss = 0.557017
    I1227 19:29:57.776475  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 19:29:57.776499  5629 solver.cpp:253]     Train net output #1: loss = 0.557017 (* 1 = 0.557017 loss)
    I1227 19:29:57.776516  5629 sgd_solver.cpp:106] Iteration 48000, lr = 0.000187295
    I1227 19:30:06.227632  5629 solver.cpp:237] Iteration 48100, loss = 0.723213
    I1227 19:30:06.227702  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:30:06.227728  5629 solver.cpp:253]     Train net output #1: loss = 0.723213 (* 1 = 0.723213 loss)
    I1227 19:30:06.227747  5629 sgd_solver.cpp:106] Iteration 48100, lr = 0.000187054
    I1227 19:30:14.125638  5629 solver.cpp:237] Iteration 48200, loss = 0.591965
    I1227 19:30:14.125704  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:30:14.125727  5629 solver.cpp:253]     Train net output #1: loss = 0.591965 (* 1 = 0.591965 loss)
    I1227 19:30:14.125746  5629 sgd_solver.cpp:106] Iteration 48200, lr = 0.000186812
    I1227 19:30:22.022936  5629 solver.cpp:237] Iteration 48300, loss = 0.572824
    I1227 19:30:22.023003  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:30:22.023028  5629 solver.cpp:253]     Train net output #1: loss = 0.572824 (* 1 = 0.572824 loss)
    I1227 19:30:22.023047  5629 sgd_solver.cpp:106] Iteration 48300, lr = 0.000186572
    I1227 19:30:29.896359  5629 solver.cpp:237] Iteration 48400, loss = 0.697205
    I1227 19:30:29.896517  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:30:29.896545  5629 solver.cpp:253]     Train net output #1: loss = 0.697205 (* 1 = 0.697205 loss)
    I1227 19:30:29.896565  5629 sgd_solver.cpp:106] Iteration 48400, lr = 0.000186332
    I1227 19:30:37.789683  5629 solver.cpp:237] Iteration 48500, loss = 0.643388
    I1227 19:30:37.789758  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:30:37.789783  5629 solver.cpp:253]     Train net output #1: loss = 0.643388 (* 1 = 0.643388 loss)
    I1227 19:30:37.789803  5629 sgd_solver.cpp:106] Iteration 48500, lr = 0.000186093
    I1227 19:30:45.688328  5629 solver.cpp:237] Iteration 48600, loss = 0.707872
    I1227 19:30:45.688392  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:30:45.688416  5629 solver.cpp:253]     Train net output #1: loss = 0.707872 (* 1 = 0.707872 loss)
    I1227 19:30:45.688436  5629 sgd_solver.cpp:106] Iteration 48600, lr = 0.000185855
    I1227 19:30:55.087946  5629 solver.cpp:237] Iteration 48700, loss = 0.606469
    I1227 19:30:55.088023  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:30:55.088055  5629 solver.cpp:253]     Train net output #1: loss = 0.606469 (* 1 = 0.606469 loss)
    I1227 19:30:55.088079  5629 sgd_solver.cpp:106] Iteration 48700, lr = 0.000185618
    I1227 19:31:02.086333  5629 solver.cpp:237] Iteration 48800, loss = 0.601651
    I1227 19:31:02.086508  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:31:02.086536  5629 solver.cpp:253]     Train net output #1: loss = 0.601651 (* 1 = 0.601651 loss)
    I1227 19:31:02.086555  5629 sgd_solver.cpp:106] Iteration 48800, lr = 0.000185381
    I1227 19:31:10.497280  5629 solver.cpp:237] Iteration 48900, loss = 0.814895
    I1227 19:31:10.497349  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:31:10.497376  5629 solver.cpp:253]     Train net output #1: loss = 0.814895 (* 1 = 0.814895 loss)
    I1227 19:31:10.497398  5629 sgd_solver.cpp:106] Iteration 48900, lr = 0.000185145
    I1227 19:31:18.493043  5629 solver.cpp:341] Iteration 49000, Testing net (#0)
    I1227 19:31:21.478979  5629 solver.cpp:409]     Test net output #0: accuracy = 0.726333
    I1227 19:31:21.479048  5629 solver.cpp:409]     Test net output #1: loss = 0.792708 (* 1 = 0.792708 loss)
    I1227 19:31:21.513986  5629 solver.cpp:237] Iteration 49000, loss = 0.593368
    I1227 19:31:21.514025  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:31:21.514047  5629 solver.cpp:253]     Train net output #1: loss = 0.593368 (* 1 = 0.593368 loss)
    I1227 19:31:21.514065  5629 sgd_solver.cpp:106] Iteration 49000, lr = 0.000184909
    I1227 19:31:29.102046  5629 solver.cpp:237] Iteration 49100, loss = 0.719564
    I1227 19:31:29.102113  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:31:29.102138  5629 solver.cpp:253]     Train net output #1: loss = 0.719564 (* 1 = 0.719564 loss)
    I1227 19:31:29.102157  5629 sgd_solver.cpp:106] Iteration 49100, lr = 0.000184675
    I1227 19:31:36.843999  5629 solver.cpp:237] Iteration 49200, loss = 0.63344
    I1227 19:31:36.844194  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:31:36.844224  5629 solver.cpp:253]     Train net output #1: loss = 0.63344 (* 1 = 0.63344 loss)
    I1227 19:31:36.844244  5629 sgd_solver.cpp:106] Iteration 49200, lr = 0.000184441
    I1227 19:31:44.178143  5629 solver.cpp:237] Iteration 49300, loss = 0.663307
    I1227 19:31:44.178243  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:31:44.178282  5629 solver.cpp:253]     Train net output #1: loss = 0.663307 (* 1 = 0.663307 loss)
    I1227 19:31:44.178311  5629 sgd_solver.cpp:106] Iteration 49300, lr = 0.000184207
    I1227 19:31:52.370764  5629 solver.cpp:237] Iteration 49400, loss = 0.747239
    I1227 19:31:52.370826  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:31:52.370849  5629 solver.cpp:253]     Train net output #1: loss = 0.747239 (* 1 = 0.747239 loss)
    I1227 19:31:52.370867  5629 sgd_solver.cpp:106] Iteration 49400, lr = 0.000183975
    I1227 19:31:59.539448  5629 solver.cpp:237] Iteration 49500, loss = 0.62774
    I1227 19:31:59.539489  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:31:59.539504  5629 solver.cpp:253]     Train net output #1: loss = 0.62774 (* 1 = 0.62774 loss)
    I1227 19:31:59.539515  5629 sgd_solver.cpp:106] Iteration 49500, lr = 0.000183743
    I1227 19:32:06.913229  5629 solver.cpp:237] Iteration 49600, loss = 0.755655
    I1227 19:32:06.913354  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:32:06.913370  5629 solver.cpp:253]     Train net output #1: loss = 0.755655 (* 1 = 0.755655 loss)
    I1227 19:32:06.913379  5629 sgd_solver.cpp:106] Iteration 49600, lr = 0.000183512
    I1227 19:32:14.578825  5629 solver.cpp:237] Iteration 49700, loss = 0.563807
    I1227 19:32:14.578871  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:32:14.578886  5629 solver.cpp:253]     Train net output #1: loss = 0.563807 (* 1 = 0.563807 loss)
    I1227 19:32:14.578896  5629 sgd_solver.cpp:106] Iteration 49700, lr = 0.000183281
    I1227 19:32:22.239750  5629 solver.cpp:237] Iteration 49800, loss = 0.568208
    I1227 19:32:22.239794  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:32:22.239807  5629 solver.cpp:253]     Train net output #1: loss = 0.568208 (* 1 = 0.568208 loss)
    I1227 19:32:22.239816  5629 sgd_solver.cpp:106] Iteration 49800, lr = 0.000183051
    I1227 19:32:30.256500  5629 solver.cpp:237] Iteration 49900, loss = 0.77977
    I1227 19:32:30.256538  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:32:30.256551  5629 solver.cpp:253]     Train net output #1: loss = 0.77977 (* 1 = 0.77977 loss)
    I1227 19:32:30.256561  5629 sgd_solver.cpp:106] Iteration 49900, lr = 0.000182822
    I1227 19:32:37.623147  5629 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_50000.caffemodel
    I1227 19:32:37.663816  5629 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_50000.solverstate
    I1227 19:32:37.664991  5629 solver.cpp:341] Iteration 50000, Testing net (#0)
    I1227 19:32:40.479421  5629 solver.cpp:409]     Test net output #0: accuracy = 0.74
    I1227 19:32:40.479477  5629 solver.cpp:409]     Test net output #1: loss = 0.753742 (* 1 = 0.753742 loss)
    I1227 19:32:40.510429  5629 solver.cpp:237] Iteration 50000, loss = 0.674941
    I1227 19:32:40.510460  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:32:40.510475  5629 solver.cpp:253]     Train net output #1: loss = 0.674941 (* 1 = 0.674941 loss)
    I1227 19:32:40.510489  5629 sgd_solver.cpp:106] Iteration 50000, lr = 0.000182593
    I1227 19:32:47.815603  5629 solver.cpp:237] Iteration 50100, loss = 0.753069
    I1227 19:32:47.815660  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:32:47.815681  5629 solver.cpp:253]     Train net output #1: loss = 0.753069 (* 1 = 0.753069 loss)
    I1227 19:32:47.815697  5629 sgd_solver.cpp:106] Iteration 50100, lr = 0.000182365
    I1227 19:32:54.828044  5629 solver.cpp:237] Iteration 50200, loss = 0.579235
    I1227 19:32:54.828086  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:32:54.828101  5629 solver.cpp:253]     Train net output #1: loss = 0.579235 (* 1 = 0.579235 loss)
    I1227 19:32:54.828112  5629 sgd_solver.cpp:106] Iteration 50200, lr = 0.000182138
    I1227 19:33:01.795956  5629 solver.cpp:237] Iteration 50300, loss = 0.570466
    I1227 19:33:01.796006  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:33:01.796020  5629 solver.cpp:253]     Train net output #1: loss = 0.570466 (* 1 = 0.570466 loss)
    I1227 19:33:01.796030  5629 sgd_solver.cpp:106] Iteration 50300, lr = 0.000181911
    I1227 19:33:08.992175  5629 solver.cpp:237] Iteration 50400, loss = 0.638005
    I1227 19:33:08.992305  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:33:08.992321  5629 solver.cpp:253]     Train net output #1: loss = 0.638005 (* 1 = 0.638005 loss)
    I1227 19:33:08.992329  5629 sgd_solver.cpp:106] Iteration 50400, lr = 0.000181686
    I1227 19:33:16.190467  5629 solver.cpp:237] Iteration 50500, loss = 0.59863
    I1227 19:33:16.190515  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:33:16.190528  5629 solver.cpp:253]     Train net output #1: loss = 0.59863 (* 1 = 0.59863 loss)
    I1227 19:33:16.190536  5629 sgd_solver.cpp:106] Iteration 50500, lr = 0.00018146
    I1227 19:33:23.233084  5629 solver.cpp:237] Iteration 50600, loss = 0.670807
    I1227 19:33:23.233140  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:33:23.233161  5629 solver.cpp:253]     Train net output #1: loss = 0.670807 (* 1 = 0.670807 loss)
    I1227 19:33:23.233177  5629 sgd_solver.cpp:106] Iteration 50600, lr = 0.000181236
    I1227 19:33:30.211746  5629 solver.cpp:237] Iteration 50700, loss = 0.585332
    I1227 19:33:30.211788  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:33:30.211803  5629 solver.cpp:253]     Train net output #1: loss = 0.585332 (* 1 = 0.585332 loss)
    I1227 19:33:30.211813  5629 sgd_solver.cpp:106] Iteration 50700, lr = 0.000181012
    I1227 19:33:37.169888  5629 solver.cpp:237] Iteration 50800, loss = 0.688031
    I1227 19:33:37.169934  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:33:37.169947  5629 solver.cpp:253]     Train net output #1: loss = 0.688031 (* 1 = 0.688031 loss)
    I1227 19:33:37.169956  5629 sgd_solver.cpp:106] Iteration 50800, lr = 0.000180788
    I1227 19:33:44.161763  5629 solver.cpp:237] Iteration 50900, loss = 0.806396
    I1227 19:33:44.161886  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:33:44.161903  5629 solver.cpp:253]     Train net output #1: loss = 0.806396 (* 1 = 0.806396 loss)
    I1227 19:33:44.161914  5629 sgd_solver.cpp:106] Iteration 50900, lr = 0.000180566
    I1227 19:33:51.358366  5629 solver.cpp:341] Iteration 51000, Testing net (#0)
    I1227 19:33:55.253546  5629 solver.cpp:409]     Test net output #0: accuracy = 0.733333
    I1227 19:33:55.253620  5629 solver.cpp:409]     Test net output #1: loss = 0.760979 (* 1 = 0.760979 loss)
    I1227 19:33:55.300897  5629 solver.cpp:237] Iteration 51000, loss = 0.604574
    I1227 19:33:55.300981  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:33:55.301026  5629 solver.cpp:253]     Train net output #1: loss = 0.604574 (* 1 = 0.604574 loss)
    I1227 19:33:55.301053  5629 sgd_solver.cpp:106] Iteration 51000, lr = 0.000180344
    I1227 19:34:03.042052  5629 solver.cpp:237] Iteration 51100, loss = 0.745568
    I1227 19:34:03.042107  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:34:03.042122  5629 solver.cpp:253]     Train net output #1: loss = 0.745568 (* 1 = 0.745568 loss)
    I1227 19:34:03.042134  5629 sgd_solver.cpp:106] Iteration 51100, lr = 0.000180122
    I1227 19:34:10.877146  5629 solver.cpp:237] Iteration 51200, loss = 0.624799
    I1227 19:34:10.877207  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:34:10.877228  5629 solver.cpp:253]     Train net output #1: loss = 0.624799 (* 1 = 0.624799 loss)
    I1227 19:34:10.877246  5629 sgd_solver.cpp:106] Iteration 51200, lr = 0.000179901
    I1227 19:34:19.198127  5629 solver.cpp:237] Iteration 51300, loss = 0.64955
    I1227 19:34:19.198246  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:34:19.198263  5629 solver.cpp:253]     Train net output #1: loss = 0.64955 (* 1 = 0.64955 loss)
    I1227 19:34:19.198274  5629 sgd_solver.cpp:106] Iteration 51300, lr = 0.000179681
    I1227 19:34:27.474334  5629 solver.cpp:237] Iteration 51400, loss = 0.715364
    I1227 19:34:27.474396  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:34:27.474419  5629 solver.cpp:253]     Train net output #1: loss = 0.715364 (* 1 = 0.715364 loss)
    I1227 19:34:27.474436  5629 sgd_solver.cpp:106] Iteration 51400, lr = 0.000179462
    I1227 19:34:35.378273  5629 solver.cpp:237] Iteration 51500, loss = 0.635343
    I1227 19:34:35.378322  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:34:35.378339  5629 solver.cpp:253]     Train net output #1: loss = 0.635343 (* 1 = 0.635343 loss)
    I1227 19:34:35.378352  5629 sgd_solver.cpp:106] Iteration 51500, lr = 0.000179243
    I1227 19:34:43.203594  5629 solver.cpp:237] Iteration 51600, loss = 0.779334
    I1227 19:34:43.203656  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:34:43.203680  5629 solver.cpp:253]     Train net output #1: loss = 0.779334 (* 1 = 0.779334 loss)
    I1227 19:34:43.203696  5629 sgd_solver.cpp:106] Iteration 51600, lr = 0.000179025
    I1227 19:34:50.993469  5629 solver.cpp:237] Iteration 51700, loss = 0.585119
    I1227 19:34:50.993597  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:34:50.993613  5629 solver.cpp:253]     Train net output #1: loss = 0.585119 (* 1 = 0.585119 loss)
    I1227 19:34:50.993623  5629 sgd_solver.cpp:106] Iteration 51700, lr = 0.000178807
    I1227 19:34:58.778724  5629 solver.cpp:237] Iteration 51800, loss = 0.698524
    I1227 19:34:58.778786  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:34:58.778808  5629 solver.cpp:253]     Train net output #1: loss = 0.698524 (* 1 = 0.698524 loss)
    I1227 19:34:58.778825  5629 sgd_solver.cpp:106] Iteration 51800, lr = 0.00017859
    I1227 19:35:06.574228  5629 solver.cpp:237] Iteration 51900, loss = 0.758862
    I1227 19:35:06.574285  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:35:06.574300  5629 solver.cpp:253]     Train net output #1: loss = 0.758862 (* 1 = 0.758862 loss)
    I1227 19:35:06.574311  5629 sgd_solver.cpp:106] Iteration 51900, lr = 0.000178373
    I1227 19:35:14.352392  5629 solver.cpp:341] Iteration 52000, Testing net (#0)
    I1227 19:35:17.931625  5629 solver.cpp:409]     Test net output #0: accuracy = 0.737
    I1227 19:35:17.931694  5629 solver.cpp:409]     Test net output #1: loss = 0.752272 (* 1 = 0.752272 loss)
    I1227 19:35:17.966315  5629 solver.cpp:237] Iteration 52000, loss = 0.676413
    I1227 19:35:17.966370  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:35:17.966390  5629 solver.cpp:253]     Train net output #1: loss = 0.676413 (* 1 = 0.676413 loss)
    I1227 19:35:17.966408  5629 sgd_solver.cpp:106] Iteration 52000, lr = 0.000178158
    I1227 19:35:25.804481  5629 solver.cpp:237] Iteration 52100, loss = 0.735433
    I1227 19:35:25.804597  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 19:35:25.804615  5629 solver.cpp:253]     Train net output #1: loss = 0.735433 (* 1 = 0.735433 loss)
    I1227 19:35:25.804627  5629 sgd_solver.cpp:106] Iteration 52100, lr = 0.000177942
    I1227 19:35:33.526423  5629 solver.cpp:237] Iteration 52200, loss = 0.656925
    I1227 19:35:33.526487  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:35:33.526511  5629 solver.cpp:253]     Train net output #1: loss = 0.656925 (* 1 = 0.656925 loss)
    I1227 19:35:33.526528  5629 sgd_solver.cpp:106] Iteration 52200, lr = 0.000177728
    I1227 19:35:41.333845  5629 solver.cpp:237] Iteration 52300, loss = 0.726041
    I1227 19:35:41.333899  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:35:41.333912  5629 solver.cpp:253]     Train net output #1: loss = 0.726041 (* 1 = 0.726041 loss)
    I1227 19:35:41.333925  5629 sgd_solver.cpp:106] Iteration 52300, lr = 0.000177514
    I1227 19:35:48.581503  5629 solver.cpp:237] Iteration 52400, loss = 0.73314
    I1227 19:35:48.581565  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:35:48.581588  5629 solver.cpp:253]     Train net output #1: loss = 0.73314 (* 1 = 0.73314 loss)
    I1227 19:35:48.581604  5629 sgd_solver.cpp:106] Iteration 52400, lr = 0.0001773
    I1227 19:35:56.378674  5629 solver.cpp:237] Iteration 52500, loss = 0.620999
    I1227 19:35:56.378803  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:35:56.378823  5629 solver.cpp:253]     Train net output #1: loss = 0.620999 (* 1 = 0.620999 loss)
    I1227 19:35:56.378835  5629 sgd_solver.cpp:106] Iteration 52500, lr = 0.000177088
    I1227 19:36:04.161936  5629 solver.cpp:237] Iteration 52600, loss = 0.794102
    I1227 19:36:04.161995  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:36:04.162017  5629 solver.cpp:253]     Train net output #1: loss = 0.794102 (* 1 = 0.794102 loss)
    I1227 19:36:04.162034  5629 sgd_solver.cpp:106] Iteration 52600, lr = 0.000176875
    I1227 19:36:11.898129  5629 solver.cpp:237] Iteration 52700, loss = 0.538473
    I1227 19:36:11.898172  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:36:11.898186  5629 solver.cpp:253]     Train net output #1: loss = 0.538473 (* 1 = 0.538473 loss)
    I1227 19:36:11.898198  5629 sgd_solver.cpp:106] Iteration 52700, lr = 0.000176664
    I1227 19:36:19.826474  5629 solver.cpp:237] Iteration 52800, loss = 0.647684
    I1227 19:36:19.826531  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:36:19.826553  5629 solver.cpp:253]     Train net output #1: loss = 0.647684 (* 1 = 0.647684 loss)
    I1227 19:36:19.826570  5629 sgd_solver.cpp:106] Iteration 52800, lr = 0.000176453
    I1227 19:36:28.926452  5629 solver.cpp:237] Iteration 52900, loss = 0.746832
    I1227 19:36:28.926564  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:36:28.926594  5629 solver.cpp:253]     Train net output #1: loss = 0.746832 (* 1 = 0.746832 loss)
    I1227 19:36:28.926604  5629 sgd_solver.cpp:106] Iteration 52900, lr = 0.000176242
    I1227 19:36:39.368723  5629 solver.cpp:341] Iteration 53000, Testing net (#0)
    I1227 19:36:43.575484  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7425
    I1227 19:36:43.575531  5629 solver.cpp:409]     Test net output #1: loss = 0.73686 (* 1 = 0.73686 loss)
    I1227 19:36:43.619235  5629 solver.cpp:237] Iteration 53000, loss = 0.634829
    I1227 19:36:43.619313  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:36:43.619339  5629 solver.cpp:253]     Train net output #1: loss = 0.634829 (* 1 = 0.634829 loss)
    I1227 19:36:43.619354  5629 sgd_solver.cpp:106] Iteration 53000, lr = 0.000176032
    I1227 19:36:53.958668  5629 solver.cpp:237] Iteration 53100, loss = 0.796962
    I1227 19:36:53.958730  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:36:53.958752  5629 solver.cpp:253]     Train net output #1: loss = 0.796962 (* 1 = 0.796962 loss)
    I1227 19:36:53.958770  5629 sgd_solver.cpp:106] Iteration 53100, lr = 0.000175823
    I1227 19:37:04.538101  5629 solver.cpp:237] Iteration 53200, loss = 0.617837
    I1227 19:37:04.538290  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:37:04.538322  5629 solver.cpp:253]     Train net output #1: loss = 0.617837 (* 1 = 0.617837 loss)
    I1227 19:37:04.538341  5629 sgd_solver.cpp:106] Iteration 53200, lr = 0.000175614
    I1227 19:37:14.976755  5629 solver.cpp:237] Iteration 53300, loss = 0.639357
    I1227 19:37:14.976804  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:37:14.976820  5629 solver.cpp:253]     Train net output #1: loss = 0.639357 (* 1 = 0.639357 loss)
    I1227 19:37:14.976835  5629 sgd_solver.cpp:106] Iteration 53300, lr = 0.000175406
    I1227 19:37:25.270412  5629 solver.cpp:237] Iteration 53400, loss = 0.700502
    I1227 19:37:25.270473  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:37:25.270498  5629 solver.cpp:253]     Train net output #1: loss = 0.700502 (* 1 = 0.700502 loss)
    I1227 19:37:25.270515  5629 sgd_solver.cpp:106] Iteration 53400, lr = 0.000175199
    I1227 19:37:35.745061  5629 solver.cpp:237] Iteration 53500, loss = 0.665933
    I1227 19:37:35.745271  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:37:35.745304  5629 solver.cpp:253]     Train net output #1: loss = 0.665933 (* 1 = 0.665933 loss)
    I1227 19:37:35.745317  5629 sgd_solver.cpp:106] Iteration 53500, lr = 0.000174992
    I1227 19:37:46.214974  5629 solver.cpp:237] Iteration 53600, loss = 0.805797
    I1227 19:37:46.215026  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:37:46.215044  5629 solver.cpp:253]     Train net output #1: loss = 0.805797 (* 1 = 0.805797 loss)
    I1227 19:37:46.215056  5629 sgd_solver.cpp:106] Iteration 53600, lr = 0.000174785
    I1227 19:37:56.954641  5629 solver.cpp:237] Iteration 53700, loss = 0.579543
    I1227 19:37:56.954742  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:37:56.954771  5629 solver.cpp:253]     Train net output #1: loss = 0.579543 (* 1 = 0.579543 loss)
    I1227 19:37:56.954803  5629 sgd_solver.cpp:106] Iteration 53700, lr = 0.00017458
    I1227 19:38:07.521626  5629 solver.cpp:237] Iteration 53800, loss = 0.585301
    I1227 19:38:07.521795  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:38:07.521831  5629 solver.cpp:253]     Train net output #1: loss = 0.585301 (* 1 = 0.585301 loss)
    I1227 19:38:07.521850  5629 sgd_solver.cpp:106] Iteration 53800, lr = 0.000174374
    I1227 19:38:18.125530  5629 solver.cpp:237] Iteration 53900, loss = 0.73242
    I1227 19:38:18.125577  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:38:18.125596  5629 solver.cpp:253]     Train net output #1: loss = 0.73242 (* 1 = 0.73242 loss)
    I1227 19:38:18.125608  5629 sgd_solver.cpp:106] Iteration 53900, lr = 0.00017417
    I1227 19:38:28.499130  5629 solver.cpp:341] Iteration 54000, Testing net (#0)
    I1227 19:38:32.826045  5629 solver.cpp:409]     Test net output #0: accuracy = 0.74125
    I1227 19:38:32.826095  5629 solver.cpp:409]     Test net output #1: loss = 0.74449 (* 1 = 0.74449 loss)
    I1227 19:38:32.867409  5629 solver.cpp:237] Iteration 54000, loss = 0.610373
    I1227 19:38:32.867492  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:38:32.867523  5629 solver.cpp:253]     Train net output #1: loss = 0.610373 (* 1 = 0.610373 loss)
    I1227 19:38:32.867545  5629 sgd_solver.cpp:106] Iteration 54000, lr = 0.000173965
    I1227 19:38:43.384132  5629 solver.cpp:237] Iteration 54100, loss = 0.836369
    I1227 19:38:43.384328  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:38:43.384362  5629 solver.cpp:253]     Train net output #1: loss = 0.836369 (* 1 = 0.836369 loss)
    I1227 19:38:43.384380  5629 sgd_solver.cpp:106] Iteration 54100, lr = 0.000173762
    I1227 19:38:53.803511  5629 solver.cpp:237] Iteration 54200, loss = 0.731077
    I1227 19:38:53.803558  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:38:53.803573  5629 solver.cpp:253]     Train net output #1: loss = 0.731077 (* 1 = 0.731077 loss)
    I1227 19:38:53.803585  5629 sgd_solver.cpp:106] Iteration 54200, lr = 0.000173559
    I1227 19:39:04.188477  5629 solver.cpp:237] Iteration 54300, loss = 0.66418
    I1227 19:39:04.188521  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:39:04.188536  5629 solver.cpp:253]     Train net output #1: loss = 0.66418 (* 1 = 0.66418 loss)
    I1227 19:39:04.188549  5629 sgd_solver.cpp:106] Iteration 54300, lr = 0.000173356
    I1227 19:39:14.669400  5629 solver.cpp:237] Iteration 54400, loss = 0.749439
    I1227 19:39:14.669589  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:39:14.669617  5629 solver.cpp:253]     Train net output #1: loss = 0.749439 (* 1 = 0.749439 loss)
    I1227 19:39:14.669633  5629 sgd_solver.cpp:106] Iteration 54400, lr = 0.000173154
    I1227 19:39:24.972945  5629 solver.cpp:237] Iteration 54500, loss = 0.679103
    I1227 19:39:24.972987  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:39:24.973002  5629 solver.cpp:253]     Train net output #1: loss = 0.679103 (* 1 = 0.679103 loss)
    I1227 19:39:24.973014  5629 sgd_solver.cpp:106] Iteration 54500, lr = 0.000172953
    I1227 19:39:35.493111  5629 solver.cpp:237] Iteration 54600, loss = 0.726539
    I1227 19:39:35.493162  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:39:35.493182  5629 solver.cpp:253]     Train net output #1: loss = 0.726539 (* 1 = 0.726539 loss)
    I1227 19:39:35.493197  5629 sgd_solver.cpp:106] Iteration 54600, lr = 0.000172752
    I1227 19:39:45.976770  5629 solver.cpp:237] Iteration 54700, loss = 0.576295
    I1227 19:39:45.976933  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:39:45.976966  5629 solver.cpp:253]     Train net output #1: loss = 0.576295 (* 1 = 0.576295 loss)
    I1227 19:39:45.976987  5629 sgd_solver.cpp:106] Iteration 54700, lr = 0.000172552
    I1227 19:39:56.476877  5629 solver.cpp:237] Iteration 54800, loss = 0.590107
    I1227 19:39:56.476920  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:39:56.476933  5629 solver.cpp:253]     Train net output #1: loss = 0.590107 (* 1 = 0.590107 loss)
    I1227 19:39:56.476944  5629 sgd_solver.cpp:106] Iteration 54800, lr = 0.000172352
    I1227 19:40:07.040750  5629 solver.cpp:237] Iteration 54900, loss = 0.63438
    I1227 19:40:07.040791  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:40:07.040803  5629 solver.cpp:253]     Train net output #1: loss = 0.63438 (* 1 = 0.63438 loss)
    I1227 19:40:07.040814  5629 sgd_solver.cpp:106] Iteration 54900, lr = 0.000172153
    I1227 19:40:14.877115  5629 solver.cpp:341] Iteration 55000, Testing net (#0)
    I1227 19:40:18.840178  5629 solver.cpp:409]     Test net output #0: accuracy = 0.738917
    I1227 19:40:18.840309  5629 solver.cpp:409]     Test net output #1: loss = 0.758141 (* 1 = 0.758141 loss)
    I1227 19:40:18.875509  5629 solver.cpp:237] Iteration 55000, loss = 0.603077
    I1227 19:40:18.875569  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:40:18.875594  5629 solver.cpp:253]     Train net output #1: loss = 0.603077 (* 1 = 0.603077 loss)
    I1227 19:40:18.875614  5629 sgd_solver.cpp:106] Iteration 55000, lr = 0.000171954
    I1227 19:40:25.888993  5629 solver.cpp:237] Iteration 55100, loss = 0.748106
    I1227 19:40:25.889056  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:40:25.889081  5629 solver.cpp:253]     Train net output #1: loss = 0.748106 (* 1 = 0.748106 loss)
    I1227 19:40:25.889098  5629 sgd_solver.cpp:106] Iteration 55100, lr = 0.000171756
    I1227 19:40:33.339890  5629 solver.cpp:237] Iteration 55200, loss = 0.616243
    I1227 19:40:33.339956  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:40:33.339980  5629 solver.cpp:253]     Train net output #1: loss = 0.616243 (* 1 = 0.616243 loss)
    I1227 19:40:33.339998  5629 sgd_solver.cpp:106] Iteration 55200, lr = 0.000171559
    I1227 19:40:40.559684  5629 solver.cpp:237] Iteration 55300, loss = 0.570887
    I1227 19:40:40.559751  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:40:40.559777  5629 solver.cpp:253]     Train net output #1: loss = 0.570887 (* 1 = 0.570887 loss)
    I1227 19:40:40.559797  5629 sgd_solver.cpp:106] Iteration 55300, lr = 0.000171361
    I1227 19:40:48.312109  5629 solver.cpp:237] Iteration 55400, loss = 0.675353
    I1227 19:40:48.312183  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:40:48.312209  5629 solver.cpp:253]     Train net output #1: loss = 0.675353 (* 1 = 0.675353 loss)
    I1227 19:40:48.312229  5629 sgd_solver.cpp:106] Iteration 55400, lr = 0.000171165
    I1227 19:40:56.800907  5629 solver.cpp:237] Iteration 55500, loss = 0.611131
    I1227 19:40:56.801151  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:40:56.801197  5629 solver.cpp:253]     Train net output #1: loss = 0.611131 (* 1 = 0.611131 loss)
    I1227 19:40:56.801225  5629 sgd_solver.cpp:106] Iteration 55500, lr = 0.000170969
    I1227 19:41:04.774154  5629 solver.cpp:237] Iteration 55600, loss = 0.712402
    I1227 19:41:04.774226  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:41:04.774252  5629 solver.cpp:253]     Train net output #1: loss = 0.712402 (* 1 = 0.712402 loss)
    I1227 19:41:04.774272  5629 sgd_solver.cpp:106] Iteration 55600, lr = 0.000170773
    I1227 19:41:12.546140  5629 solver.cpp:237] Iteration 55700, loss = 0.570026
    I1227 19:41:12.546227  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:41:12.546268  5629 solver.cpp:253]     Train net output #1: loss = 0.570026 (* 1 = 0.570026 loss)
    I1227 19:41:12.546303  5629 sgd_solver.cpp:106] Iteration 55700, lr = 0.000170578
    I1227 19:41:20.394431  5629 solver.cpp:237] Iteration 55800, loss = 0.648717
    I1227 19:41:20.394505  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:41:20.394531  5629 solver.cpp:253]     Train net output #1: loss = 0.648717 (* 1 = 0.648717 loss)
    I1227 19:41:20.394551  5629 sgd_solver.cpp:106] Iteration 55800, lr = 0.000170384
    I1227 19:41:29.465559  5629 solver.cpp:237] Iteration 55900, loss = 0.672707
    I1227 19:41:29.465785  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:41:29.465834  5629 solver.cpp:253]     Train net output #1: loss = 0.672707 (* 1 = 0.672707 loss)
    I1227 19:41:29.465867  5629 sgd_solver.cpp:106] Iteration 55900, lr = 0.00017019
    I1227 19:41:40.114506  5629 solver.cpp:341] Iteration 56000, Testing net (#0)
    I1227 19:41:44.561426  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7445
    I1227 19:41:44.561511  5629 solver.cpp:409]     Test net output #1: loss = 0.732816 (* 1 = 0.732816 loss)
    I1227 19:41:44.613765  5629 solver.cpp:237] Iteration 56000, loss = 0.56675
    I1227 19:41:44.613829  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:41:44.613853  5629 solver.cpp:253]     Train net output #1: loss = 0.56675 (* 1 = 0.56675 loss)
    I1227 19:41:44.613873  5629 sgd_solver.cpp:106] Iteration 56000, lr = 0.000169997
    I1227 19:41:55.100076  5629 solver.cpp:237] Iteration 56100, loss = 0.756722
    I1227 19:41:55.100158  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:41:55.100184  5629 solver.cpp:253]     Train net output #1: loss = 0.756722 (* 1 = 0.756722 loss)
    I1227 19:41:55.100205  5629 sgd_solver.cpp:106] Iteration 56100, lr = 0.000169804
    I1227 19:42:02.769377  5629 solver.cpp:237] Iteration 56200, loss = 0.665332
    I1227 19:42:02.769605  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:42:02.769660  5629 solver.cpp:253]     Train net output #1: loss = 0.665332 (* 1 = 0.665332 loss)
    I1227 19:42:02.769696  5629 sgd_solver.cpp:106] Iteration 56200, lr = 0.000169611
    I1227 19:42:10.282855  5629 solver.cpp:237] Iteration 56300, loss = 0.624394
    I1227 19:42:10.282928  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:42:10.282954  5629 solver.cpp:253]     Train net output #1: loss = 0.624394 (* 1 = 0.624394 loss)
    I1227 19:42:10.282974  5629 sgd_solver.cpp:106] Iteration 56300, lr = 0.000169419
    I1227 19:42:19.573281  5629 solver.cpp:237] Iteration 56400, loss = 0.743866
    I1227 19:42:19.573374  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:42:19.573401  5629 solver.cpp:253]     Train net output #1: loss = 0.743866 (* 1 = 0.743866 loss)
    I1227 19:42:19.573422  5629 sgd_solver.cpp:106] Iteration 56400, lr = 0.000169228
    I1227 19:42:30.146278  5629 solver.cpp:237] Iteration 56500, loss = 0.6296
    I1227 19:42:30.146349  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:42:30.146375  5629 solver.cpp:253]     Train net output #1: loss = 0.6296 (* 1 = 0.6296 loss)
    I1227 19:42:30.146395  5629 sgd_solver.cpp:106] Iteration 56500, lr = 0.000169037
    I1227 19:42:40.688908  5629 solver.cpp:237] Iteration 56600, loss = 0.767375
    I1227 19:42:40.689141  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:42:40.689173  5629 solver.cpp:253]     Train net output #1: loss = 0.767375 (* 1 = 0.767375 loss)
    I1227 19:42:40.689195  5629 sgd_solver.cpp:106] Iteration 56600, lr = 0.000168847
    I1227 19:42:51.303117  5629 solver.cpp:237] Iteration 56700, loss = 0.698018
    I1227 19:42:51.303194  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:42:51.303220  5629 solver.cpp:253]     Train net output #1: loss = 0.698018 (* 1 = 0.698018 loss)
    I1227 19:42:51.303239  5629 sgd_solver.cpp:106] Iteration 56700, lr = 0.000168657
    I1227 19:43:01.907763  5629 solver.cpp:237] Iteration 56800, loss = 0.60007
    I1227 19:43:01.907838  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:43:01.907866  5629 solver.cpp:253]     Train net output #1: loss = 0.60007 (* 1 = 0.60007 loss)
    I1227 19:43:01.907886  5629 sgd_solver.cpp:106] Iteration 56800, lr = 0.000168467
    I1227 19:43:12.426719  5629 solver.cpp:237] Iteration 56900, loss = 0.707877
    I1227 19:43:12.426898  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:43:12.426929  5629 solver.cpp:253]     Train net output #1: loss = 0.707877 (* 1 = 0.707877 loss)
    I1227 19:43:12.426949  5629 sgd_solver.cpp:106] Iteration 56900, lr = 0.000168278
    I1227 19:43:22.911716  5629 solver.cpp:341] Iteration 57000, Testing net (#0)
    I1227 19:43:27.340744  5629 solver.cpp:409]     Test net output #0: accuracy = 0.738083
    I1227 19:43:27.340826  5629 solver.cpp:409]     Test net output #1: loss = 0.755277 (* 1 = 0.755277 loss)
    I1227 19:43:27.390550  5629 solver.cpp:237] Iteration 57000, loss = 0.687348
    I1227 19:43:27.390619  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:43:27.390643  5629 solver.cpp:253]     Train net output #1: loss = 0.687348 (* 1 = 0.687348 loss)
    I1227 19:43:27.390663  5629 sgd_solver.cpp:106] Iteration 57000, lr = 0.00016809
    I1227 19:43:38.004794  5629 solver.cpp:237] Iteration 57100, loss = 0.742086
    I1227 19:43:38.004889  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 19:43:38.004935  5629 solver.cpp:253]     Train net output #1: loss = 0.742086 (* 1 = 0.742086 loss)
    I1227 19:43:38.004966  5629 sgd_solver.cpp:106] Iteration 57100, lr = 0.000167902
    I1227 19:43:48.779939  5629 solver.cpp:237] Iteration 57200, loss = 0.594533
    I1227 19:43:48.780104  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:43:48.780135  5629 solver.cpp:253]     Train net output #1: loss = 0.594533 (* 1 = 0.594533 loss)
    I1227 19:43:48.780156  5629 sgd_solver.cpp:106] Iteration 57200, lr = 0.000167715
    I1227 19:43:59.326431  5629 solver.cpp:237] Iteration 57300, loss = 0.707763
    I1227 19:43:59.326498  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:43:59.326524  5629 solver.cpp:253]     Train net output #1: loss = 0.707763 (* 1 = 0.707763 loss)
    I1227 19:43:59.326542  5629 sgd_solver.cpp:106] Iteration 57300, lr = 0.000167528
    I1227 19:44:09.931157  5629 solver.cpp:237] Iteration 57400, loss = 0.711449
    I1227 19:44:09.931226  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:44:09.931251  5629 solver.cpp:253]     Train net output #1: loss = 0.711449 (* 1 = 0.711449 loss)
    I1227 19:44:09.931269  5629 sgd_solver.cpp:106] Iteration 57400, lr = 0.000167341
    I1227 19:44:20.513032  5629 solver.cpp:237] Iteration 57500, loss = 0.565954
    I1227 19:44:20.513226  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:44:20.513257  5629 solver.cpp:253]     Train net output #1: loss = 0.565954 (* 1 = 0.565954 loss)
    I1227 19:44:20.513275  5629 sgd_solver.cpp:106] Iteration 57500, lr = 0.000167155
    I1227 19:44:31.070927  5629 solver.cpp:237] Iteration 57600, loss = 0.728666
    I1227 19:44:31.071009  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:44:31.071038  5629 solver.cpp:253]     Train net output #1: loss = 0.728666 (* 1 = 0.728666 loss)
    I1227 19:44:31.071058  5629 sgd_solver.cpp:106] Iteration 57600, lr = 0.00016697
    I1227 19:44:41.663529  5629 solver.cpp:237] Iteration 57700, loss = 0.599252
    I1227 19:44:41.663599  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:44:41.663625  5629 solver.cpp:253]     Train net output #1: loss = 0.599252 (* 1 = 0.599252 loss)
    I1227 19:44:41.663643  5629 sgd_solver.cpp:106] Iteration 57700, lr = 0.000166785
    I1227 19:44:52.281811  5629 solver.cpp:237] Iteration 57800, loss = 0.666472
    I1227 19:44:52.282039  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:44:52.282073  5629 solver.cpp:253]     Train net output #1: loss = 0.666472 (* 1 = 0.666472 loss)
    I1227 19:44:52.282094  5629 sgd_solver.cpp:106] Iteration 57800, lr = 0.0001666
    I1227 19:45:02.955653  5629 solver.cpp:237] Iteration 57900, loss = 0.69526
    I1227 19:45:02.955735  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:45:02.955761  5629 solver.cpp:253]     Train net output #1: loss = 0.69526 (* 1 = 0.69526 loss)
    I1227 19:45:02.955781  5629 sgd_solver.cpp:106] Iteration 57900, lr = 0.000166416
    I1227 19:45:13.387048  5629 solver.cpp:341] Iteration 58000, Testing net (#0)
    I1227 19:45:17.812448  5629 solver.cpp:409]     Test net output #0: accuracy = 0.744667
    I1227 19:45:17.812556  5629 solver.cpp:409]     Test net output #1: loss = 0.736753 (* 1 = 0.736753 loss)
    I1227 19:45:17.859808  5629 solver.cpp:237] Iteration 58000, loss = 0.684805
    I1227 19:45:17.859874  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:45:17.859899  5629 solver.cpp:253]     Train net output #1: loss = 0.684805 (* 1 = 0.684805 loss)
    I1227 19:45:17.859918  5629 sgd_solver.cpp:106] Iteration 58000, lr = 0.000166233
    I1227 19:45:28.508129  5629 solver.cpp:237] Iteration 58100, loss = 0.701051
    I1227 19:45:28.508319  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:45:28.508352  5629 solver.cpp:253]     Train net output #1: loss = 0.701051 (* 1 = 0.701051 loss)
    I1227 19:45:28.508370  5629 sgd_solver.cpp:106] Iteration 58100, lr = 0.00016605
    I1227 19:45:39.217631  5629 solver.cpp:237] Iteration 58200, loss = 0.646794
    I1227 19:45:39.217710  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:45:39.217736  5629 solver.cpp:253]     Train net output #1: loss = 0.646794 (* 1 = 0.646794 loss)
    I1227 19:45:39.217756  5629 sgd_solver.cpp:106] Iteration 58200, lr = 0.000165867
    I1227 19:45:49.976227  5629 solver.cpp:237] Iteration 58300, loss = 0.706762
    I1227 19:45:49.976300  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:45:49.976325  5629 solver.cpp:253]     Train net output #1: loss = 0.706762 (* 1 = 0.706762 loss)
    I1227 19:45:49.976344  5629 sgd_solver.cpp:106] Iteration 58300, lr = 0.000165685
    I1227 19:46:00.615376  5629 solver.cpp:237] Iteration 58400, loss = 0.746303
    I1227 19:46:00.615700  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:46:00.615733  5629 solver.cpp:253]     Train net output #1: loss = 0.746303 (* 1 = 0.746303 loss)
    I1227 19:46:00.615753  5629 sgd_solver.cpp:106] Iteration 58400, lr = 0.000165503
    I1227 19:46:11.087332  5629 solver.cpp:237] Iteration 58500, loss = 0.704109
    I1227 19:46:11.087402  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:46:11.087429  5629 solver.cpp:253]     Train net output #1: loss = 0.704109 (* 1 = 0.704109 loss)
    I1227 19:46:11.087447  5629 sgd_solver.cpp:106] Iteration 58500, lr = 0.000165322
    I1227 19:46:21.609719  5629 solver.cpp:237] Iteration 58600, loss = 0.730227
    I1227 19:46:21.609787  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:46:21.609812  5629 solver.cpp:253]     Train net output #1: loss = 0.730227 (* 1 = 0.730227 loss)
    I1227 19:46:21.609832  5629 sgd_solver.cpp:106] Iteration 58600, lr = 0.000165141
    I1227 19:46:32.184937  5629 solver.cpp:237] Iteration 58700, loss = 0.646886
    I1227 19:46:32.196689  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:46:32.196743  5629 solver.cpp:253]     Train net output #1: loss = 0.646886 (* 1 = 0.646886 loss)
    I1227 19:46:32.196765  5629 sgd_solver.cpp:106] Iteration 58700, lr = 0.000164961
    I1227 19:46:42.715909  5629 solver.cpp:237] Iteration 58800, loss = 0.655873
    I1227 19:46:42.715980  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:46:42.716006  5629 solver.cpp:253]     Train net output #1: loss = 0.655873 (* 1 = 0.655873 loss)
    I1227 19:46:42.716027  5629 sgd_solver.cpp:106] Iteration 58800, lr = 0.000164781
    I1227 19:46:53.272264  5629 solver.cpp:237] Iteration 58900, loss = 0.654592
    I1227 19:46:53.272377  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:46:53.272404  5629 solver.cpp:253]     Train net output #1: loss = 0.654592 (* 1 = 0.654592 loss)
    I1227 19:46:53.272425  5629 sgd_solver.cpp:106] Iteration 58900, lr = 0.000164601
    I1227 19:47:03.598645  5629 solver.cpp:341] Iteration 59000, Testing net (#0)
    I1227 19:47:07.940879  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7415
    I1227 19:47:07.940961  5629 solver.cpp:409]     Test net output #1: loss = 0.744447 (* 1 = 0.744447 loss)
    I1227 19:47:07.990027  5629 solver.cpp:237] Iteration 59000, loss = 0.526773
    I1227 19:47:07.990092  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:47:07.990118  5629 solver.cpp:253]     Train net output #1: loss = 0.526773 (* 1 = 0.526773 loss)
    I1227 19:47:07.990136  5629 sgd_solver.cpp:106] Iteration 59000, lr = 0.000164422
    I1227 19:47:18.531327  5629 solver.cpp:237] Iteration 59100, loss = 0.671553
    I1227 19:47:18.531399  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:47:18.531426  5629 solver.cpp:253]     Train net output #1: loss = 0.671553 (* 1 = 0.671553 loss)
    I1227 19:47:18.531445  5629 sgd_solver.cpp:106] Iteration 59100, lr = 0.000164244
    I1227 19:47:29.068526  5629 solver.cpp:237] Iteration 59200, loss = 0.582251
    I1227 19:47:29.068604  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:47:29.068631  5629 solver.cpp:253]     Train net output #1: loss = 0.582251 (* 1 = 0.582251 loss)
    I1227 19:47:29.068661  5629 sgd_solver.cpp:106] Iteration 59200, lr = 0.000164066
    I1227 19:47:39.613015  5629 solver.cpp:237] Iteration 59300, loss = 0.677414
    I1227 19:47:39.613198  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:47:39.613246  5629 solver.cpp:253]     Train net output #1: loss = 0.677414 (* 1 = 0.677414 loss)
    I1227 19:47:39.613277  5629 sgd_solver.cpp:106] Iteration 59300, lr = 0.000163888
    I1227 19:47:50.052032  5629 solver.cpp:237] Iteration 59400, loss = 0.771153
    I1227 19:47:50.052098  5629 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1227 19:47:50.052124  5629 solver.cpp:253]     Train net output #1: loss = 0.771153 (* 1 = 0.771153 loss)
    I1227 19:47:50.052142  5629 sgd_solver.cpp:106] Iteration 59400, lr = 0.000163711
    I1227 19:48:00.517278  5629 solver.cpp:237] Iteration 59500, loss = 0.627284
    I1227 19:48:00.517351  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:48:00.517376  5629 solver.cpp:253]     Train net output #1: loss = 0.627284 (* 1 = 0.627284 loss)
    I1227 19:48:00.517395  5629 sgd_solver.cpp:106] Iteration 59500, lr = 0.000163535
    I1227 19:48:09.579365  5629 solver.cpp:237] Iteration 59600, loss = 0.724893
    I1227 19:48:09.579433  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:48:09.579459  5629 solver.cpp:253]     Train net output #1: loss = 0.724893 (* 1 = 0.724893 loss)
    I1227 19:48:09.579478  5629 sgd_solver.cpp:106] Iteration 59600, lr = 0.000163358
    I1227 19:48:18.009320  5629 solver.cpp:237] Iteration 59700, loss = 0.605761
    I1227 19:48:18.009481  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:48:18.009510  5629 solver.cpp:253]     Train net output #1: loss = 0.605761 (* 1 = 0.605761 loss)
    I1227 19:48:18.009528  5629 sgd_solver.cpp:106] Iteration 59700, lr = 0.000163182
    I1227 19:48:25.220772  5629 solver.cpp:237] Iteration 59800, loss = 0.609044
    I1227 19:48:25.220834  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:48:25.220857  5629 solver.cpp:253]     Train net output #1: loss = 0.609044 (* 1 = 0.609044 loss)
    I1227 19:48:25.220875  5629 sgd_solver.cpp:106] Iteration 59800, lr = 0.000163007
    I1227 19:48:32.820731  5629 solver.cpp:237] Iteration 59900, loss = 0.673802
    I1227 19:48:32.820798  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:48:32.820822  5629 solver.cpp:253]     Train net output #1: loss = 0.673802 (* 1 = 0.673802 loss)
    I1227 19:48:32.820842  5629 sgd_solver.cpp:106] Iteration 59900, lr = 0.000162832
    I1227 19:48:40.435590  5629 solver.cpp:341] Iteration 60000, Testing net (#0)
    I1227 19:48:43.341547  5629 solver.cpp:409]     Test net output #0: accuracy = 0.744
    I1227 19:48:43.341624  5629 solver.cpp:409]     Test net output #1: loss = 0.741479 (* 1 = 0.741479 loss)
    I1227 19:48:43.376335  5629 solver.cpp:237] Iteration 60000, loss = 0.69577
    I1227 19:48:43.376407  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:48:43.376432  5629 solver.cpp:253]     Train net output #1: loss = 0.69577 (* 1 = 0.69577 loss)
    I1227 19:48:43.376454  5629 sgd_solver.cpp:106] Iteration 60000, lr = 0.000162658
    I1227 19:48:51.324308  5629 solver.cpp:237] Iteration 60100, loss = 0.834628
    I1227 19:48:51.324512  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:48:51.324542  5629 solver.cpp:253]     Train net output #1: loss = 0.834628 (* 1 = 0.834628 loss)
    I1227 19:48:51.324561  5629 sgd_solver.cpp:106] Iteration 60100, lr = 0.000162484
    I1227 19:48:58.583361  5629 solver.cpp:237] Iteration 60200, loss = 0.648372
    I1227 19:48:58.583423  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:48:58.583446  5629 solver.cpp:253]     Train net output #1: loss = 0.648372 (* 1 = 0.648372 loss)
    I1227 19:48:58.583464  5629 sgd_solver.cpp:106] Iteration 60200, lr = 0.00016231
    I1227 19:49:05.866951  5629 solver.cpp:237] Iteration 60300, loss = 0.586412
    I1227 19:49:05.867015  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:49:05.867039  5629 solver.cpp:253]     Train net output #1: loss = 0.586412 (* 1 = 0.586412 loss)
    I1227 19:49:05.867058  5629 sgd_solver.cpp:106] Iteration 60300, lr = 0.000162137
    I1227 19:49:12.842231  5629 solver.cpp:237] Iteration 60400, loss = 0.698905
    I1227 19:49:12.842294  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:49:12.842319  5629 solver.cpp:253]     Train net output #1: loss = 0.698905 (* 1 = 0.698905 loss)
    I1227 19:49:12.842335  5629 sgd_solver.cpp:106] Iteration 60400, lr = 0.000161964
    I1227 19:49:20.235677  5629 solver.cpp:237] Iteration 60500, loss = 0.598593
    I1227 19:49:20.235740  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:49:20.235764  5629 solver.cpp:253]     Train net output #1: loss = 0.598593 (* 1 = 0.598593 loss)
    I1227 19:49:20.235783  5629 sgd_solver.cpp:106] Iteration 60500, lr = 0.000161792
    I1227 19:49:27.785053  5629 solver.cpp:237] Iteration 60600, loss = 0.692095
    I1227 19:49:27.785249  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:49:27.785284  5629 solver.cpp:253]     Train net output #1: loss = 0.692095 (* 1 = 0.692095 loss)
    I1227 19:49:27.785302  5629 sgd_solver.cpp:106] Iteration 60600, lr = 0.00016162
    I1227 19:49:35.068506  5629 solver.cpp:237] Iteration 60700, loss = 0.623507
    I1227 19:49:35.068570  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:49:35.068603  5629 solver.cpp:253]     Train net output #1: loss = 0.623507 (* 1 = 0.623507 loss)
    I1227 19:49:35.068622  5629 sgd_solver.cpp:106] Iteration 60700, lr = 0.000161448
    I1227 19:49:44.133934  5629 solver.cpp:237] Iteration 60800, loss = 0.53765
    I1227 19:49:44.134001  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:49:44.134027  5629 solver.cpp:253]     Train net output #1: loss = 0.53765 (* 1 = 0.53765 loss)
    I1227 19:49:44.134047  5629 sgd_solver.cpp:106] Iteration 60800, lr = 0.000161277
    I1227 19:49:52.550890  5629 solver.cpp:237] Iteration 60900, loss = 0.772604
    I1227 19:49:52.550952  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:49:52.550976  5629 solver.cpp:253]     Train net output #1: loss = 0.772604 (* 1 = 0.772604 loss)
    I1227 19:49:52.550994  5629 sgd_solver.cpp:106] Iteration 60900, lr = 0.000161107
    I1227 19:50:00.974493  5629 solver.cpp:341] Iteration 61000, Testing net (#0)
    I1227 19:50:04.017818  5629 solver.cpp:409]     Test net output #0: accuracy = 0.742
    I1227 19:50:04.017870  5629 solver.cpp:409]     Test net output #1: loss = 0.73294 (* 1 = 0.73294 loss)
    I1227 19:50:04.076560  5629 solver.cpp:237] Iteration 61000, loss = 0.540405
    I1227 19:50:04.076606  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:50:04.076620  5629 solver.cpp:253]     Train net output #1: loss = 0.540405 (* 1 = 0.540405 loss)
    I1227 19:50:04.076632  5629 sgd_solver.cpp:106] Iteration 61000, lr = 0.000160936
    I1227 19:50:11.931001  5629 solver.cpp:237] Iteration 61100, loss = 0.659327
    I1227 19:50:11.931043  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:50:11.931059  5629 solver.cpp:253]     Train net output #1: loss = 0.659327 (* 1 = 0.659327 loss)
    I1227 19:50:11.931071  5629 sgd_solver.cpp:106] Iteration 61100, lr = 0.000160767
    I1227 19:50:19.035156  5629 solver.cpp:237] Iteration 61200, loss = 0.626115
    I1227 19:50:19.035202  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:50:19.035212  5629 solver.cpp:253]     Train net output #1: loss = 0.626115 (* 1 = 0.626115 loss)
    I1227 19:50:19.035220  5629 sgd_solver.cpp:106] Iteration 61200, lr = 0.000160597
    I1227 19:50:25.989542  5629 solver.cpp:237] Iteration 61300, loss = 0.534735
    I1227 19:50:25.989586  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 19:50:25.989598  5629 solver.cpp:253]     Train net output #1: loss = 0.534735 (* 1 = 0.534735 loss)
    I1227 19:50:25.989605  5629 sgd_solver.cpp:106] Iteration 61300, lr = 0.000160428
    I1227 19:50:32.939923  5629 solver.cpp:237] Iteration 61400, loss = 0.709561
    I1227 19:50:32.940124  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:50:32.940141  5629 solver.cpp:253]     Train net output #1: loss = 0.709561 (* 1 = 0.709561 loss)
    I1227 19:50:32.940150  5629 sgd_solver.cpp:106] Iteration 61400, lr = 0.00016026
    I1227 19:50:39.899909  5629 solver.cpp:237] Iteration 61500, loss = 0.633325
    I1227 19:50:39.899965  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:50:39.899986  5629 solver.cpp:253]     Train net output #1: loss = 0.633325 (* 1 = 0.633325 loss)
    I1227 19:50:39.900002  5629 sgd_solver.cpp:106] Iteration 61500, lr = 0.000160092
    I1227 19:50:47.516083  5629 solver.cpp:237] Iteration 61600, loss = 0.801253
    I1227 19:50:47.516119  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:50:47.516130  5629 solver.cpp:253]     Train net output #1: loss = 0.801253 (* 1 = 0.801253 loss)
    I1227 19:50:47.516139  5629 sgd_solver.cpp:106] Iteration 61600, lr = 0.000159924
    I1227 19:50:54.472517  5629 solver.cpp:237] Iteration 61700, loss = 0.580038
    I1227 19:50:54.472555  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:50:54.472568  5629 solver.cpp:253]     Train net output #1: loss = 0.580038 (* 1 = 0.580038 loss)
    I1227 19:50:54.472581  5629 sgd_solver.cpp:106] Iteration 61700, lr = 0.000159757
    I1227 19:51:01.426849  5629 solver.cpp:237] Iteration 61800, loss = 0.66493
    I1227 19:51:01.426887  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:51:01.426898  5629 solver.cpp:253]     Train net output #1: loss = 0.66493 (* 1 = 0.66493 loss)
    I1227 19:51:01.426906  5629 sgd_solver.cpp:106] Iteration 61800, lr = 0.00015959
    I1227 19:51:08.367287  5629 solver.cpp:237] Iteration 61900, loss = 0.738099
    I1227 19:51:08.367487  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:51:08.367504  5629 solver.cpp:253]     Train net output #1: loss = 0.738099 (* 1 = 0.738099 loss)
    I1227 19:51:08.367512  5629 sgd_solver.cpp:106] Iteration 61900, lr = 0.000159423
    I1227 19:51:15.220847  5629 solver.cpp:341] Iteration 62000, Testing net (#0)
    I1227 19:51:18.055307  5629 solver.cpp:409]     Test net output #0: accuracy = 0.737583
    I1227 19:51:18.055366  5629 solver.cpp:409]     Test net output #1: loss = 0.757291 (* 1 = 0.757291 loss)
    I1227 19:51:18.088722  5629 solver.cpp:237] Iteration 62000, loss = 0.599697
    I1227 19:51:18.088768  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:51:18.088788  5629 solver.cpp:253]     Train net output #1: loss = 0.599697 (* 1 = 0.599697 loss)
    I1227 19:51:18.088805  5629 sgd_solver.cpp:106] Iteration 62000, lr = 0.000159257
    I1227 19:51:25.265235  5629 solver.cpp:237] Iteration 62100, loss = 0.607052
    I1227 19:51:25.265272  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:51:25.265285  5629 solver.cpp:253]     Train net output #1: loss = 0.607052 (* 1 = 0.607052 loss)
    I1227 19:51:25.265293  5629 sgd_solver.cpp:106] Iteration 62100, lr = 0.000159091
    I1227 19:51:32.747189  5629 solver.cpp:237] Iteration 62200, loss = 0.592857
    I1227 19:51:32.747236  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:51:32.747248  5629 solver.cpp:253]     Train net output #1: loss = 0.592857 (* 1 = 0.592857 loss)
    I1227 19:51:32.747257  5629 sgd_solver.cpp:106] Iteration 62200, lr = 0.000158926
    I1227 19:51:40.462662  5629 solver.cpp:237] Iteration 62300, loss = 0.567296
    I1227 19:51:40.462854  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:51:40.462872  5629 solver.cpp:253]     Train net output #1: loss = 0.567296 (* 1 = 0.567296 loss)
    I1227 19:51:40.462883  5629 sgd_solver.cpp:106] Iteration 62300, lr = 0.000158761
    I1227 19:51:48.339110  5629 solver.cpp:237] Iteration 62400, loss = 0.640197
    I1227 19:51:48.339166  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:51:48.339189  5629 solver.cpp:253]     Train net output #1: loss = 0.640197 (* 1 = 0.640197 loss)
    I1227 19:51:48.339203  5629 sgd_solver.cpp:106] Iteration 62400, lr = 0.000158597
    I1227 19:51:55.414228  5629 solver.cpp:237] Iteration 62500, loss = 0.547246
    I1227 19:51:55.414273  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:51:55.414288  5629 solver.cpp:253]     Train net output #1: loss = 0.547246 (* 1 = 0.547246 loss)
    I1227 19:51:55.414301  5629 sgd_solver.cpp:106] Iteration 62500, lr = 0.000158433
    I1227 19:52:02.588263  5629 solver.cpp:237] Iteration 62600, loss = 0.761087
    I1227 19:52:02.588306  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:52:02.588321  5629 solver.cpp:253]     Train net output #1: loss = 0.761087 (* 1 = 0.761087 loss)
    I1227 19:52:02.588333  5629 sgd_solver.cpp:106] Iteration 62600, lr = 0.000158269
    I1227 19:52:10.089166  5629 solver.cpp:237] Iteration 62700, loss = 0.528602
    I1227 19:52:10.089213  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:52:10.089226  5629 solver.cpp:253]     Train net output #1: loss = 0.528602 (* 1 = 0.528602 loss)
    I1227 19:52:10.089234  5629 sgd_solver.cpp:106] Iteration 62700, lr = 0.000158106
    I1227 19:52:17.279583  5629 solver.cpp:237] Iteration 62800, loss = 0.70108
    I1227 19:52:17.279743  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:52:17.279755  5629 solver.cpp:253]     Train net output #1: loss = 0.70108 (* 1 = 0.70108 loss)
    I1227 19:52:17.279762  5629 sgd_solver.cpp:106] Iteration 62800, lr = 0.000157943
    I1227 19:52:24.600814  5629 solver.cpp:237] Iteration 62900, loss = 0.752184
    I1227 19:52:24.600867  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:52:24.600889  5629 solver.cpp:253]     Train net output #1: loss = 0.752184 (* 1 = 0.752184 loss)
    I1227 19:52:24.600905  5629 sgd_solver.cpp:106] Iteration 62900, lr = 0.00015778
    I1227 19:52:31.723731  5629 solver.cpp:341] Iteration 63000, Testing net (#0)
    I1227 19:52:34.930711  5629 solver.cpp:409]     Test net output #0: accuracy = 0.739667
    I1227 19:52:34.930754  5629 solver.cpp:409]     Test net output #1: loss = 0.735417 (* 1 = 0.735417 loss)
    I1227 19:52:34.960906  5629 solver.cpp:237] Iteration 63000, loss = 0.644017
    I1227 19:52:34.960929  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:52:34.960942  5629 solver.cpp:253]     Train net output #1: loss = 0.644017 (* 1 = 0.644017 loss)
    I1227 19:52:34.960953  5629 sgd_solver.cpp:106] Iteration 63000, lr = 0.000157618
    I1227 19:52:42.090340  5629 solver.cpp:237] Iteration 63100, loss = 0.720503
    I1227 19:52:42.090383  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:52:42.090397  5629 solver.cpp:253]     Train net output #1: loss = 0.720503 (* 1 = 0.720503 loss)
    I1227 19:52:42.090409  5629 sgd_solver.cpp:106] Iteration 63100, lr = 0.000157456
    I1227 19:52:49.606909  5629 solver.cpp:237] Iteration 63200, loss = 0.546203
    I1227 19:52:49.607033  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:52:49.607050  5629 solver.cpp:253]     Train net output #1: loss = 0.546203 (* 1 = 0.546203 loss)
    I1227 19:52:49.607058  5629 sgd_solver.cpp:106] Iteration 63200, lr = 0.000157295
    I1227 19:52:56.830983  5629 solver.cpp:237] Iteration 63300, loss = 0.587528
    I1227 19:52:56.831028  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:52:56.831040  5629 solver.cpp:253]     Train net output #1: loss = 0.587528 (* 1 = 0.587528 loss)
    I1227 19:52:56.831049  5629 sgd_solver.cpp:106] Iteration 63300, lr = 0.000157134
    I1227 19:53:03.920692  5629 solver.cpp:237] Iteration 63400, loss = 0.702867
    I1227 19:53:03.920738  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:53:03.920750  5629 solver.cpp:253]     Train net output #1: loss = 0.702867 (* 1 = 0.702867 loss)
    I1227 19:53:03.920758  5629 sgd_solver.cpp:106] Iteration 63400, lr = 0.000156973
    I1227 19:53:11.497887  5629 solver.cpp:237] Iteration 63500, loss = 0.635053
    I1227 19:53:11.497927  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:53:11.497938  5629 solver.cpp:253]     Train net output #1: loss = 0.635053 (* 1 = 0.635053 loss)
    I1227 19:53:11.497948  5629 sgd_solver.cpp:106] Iteration 63500, lr = 0.000156813
    I1227 19:53:18.880406  5629 solver.cpp:237] Iteration 63600, loss = 0.638512
    I1227 19:53:18.880451  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:53:18.880467  5629 solver.cpp:253]     Train net output #1: loss = 0.638512 (* 1 = 0.638512 loss)
    I1227 19:53:18.880478  5629 sgd_solver.cpp:106] Iteration 63600, lr = 0.000156653
    I1227 19:53:26.085942  5629 solver.cpp:237] Iteration 63700, loss = 0.539402
    I1227 19:53:26.086081  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 19:53:26.086097  5629 solver.cpp:253]     Train net output #1: loss = 0.539402 (* 1 = 0.539402 loss)
    I1227 19:53:26.086107  5629 sgd_solver.cpp:106] Iteration 63700, lr = 0.000156494
    I1227 19:53:33.365537  5629 solver.cpp:237] Iteration 63800, loss = 0.618935
    I1227 19:53:33.365576  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:53:33.365588  5629 solver.cpp:253]     Train net output #1: loss = 0.618935 (* 1 = 0.618935 loss)
    I1227 19:53:33.365597  5629 sgd_solver.cpp:106] Iteration 63800, lr = 0.000156335
    I1227 19:53:41.294345  5629 solver.cpp:237] Iteration 63900, loss = 0.834216
    I1227 19:53:41.294384  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:53:41.294397  5629 solver.cpp:253]     Train net output #1: loss = 0.834216 (* 1 = 0.834216 loss)
    I1227 19:53:41.294406  5629 sgd_solver.cpp:106] Iteration 63900, lr = 0.000156176
    I1227 19:53:49.779320  5629 solver.cpp:341] Iteration 64000, Testing net (#0)
    I1227 19:53:52.966955  5629 solver.cpp:409]     Test net output #0: accuracy = 0.73625
    I1227 19:53:52.967005  5629 solver.cpp:409]     Test net output #1: loss = 0.756526 (* 1 = 0.756526 loss)
    I1227 19:53:52.998309  5629 solver.cpp:237] Iteration 64000, loss = 0.603562
    I1227 19:53:52.998369  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:53:52.998388  5629 solver.cpp:253]     Train net output #1: loss = 0.603562 (* 1 = 0.603562 loss)
    I1227 19:53:52.998402  5629 sgd_solver.cpp:106] Iteration 64000, lr = 0.000156018
    I1227 19:54:00.351531  5629 solver.cpp:237] Iteration 64100, loss = 0.716854
    I1227 19:54:00.351662  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:54:00.351686  5629 solver.cpp:253]     Train net output #1: loss = 0.716854 (* 1 = 0.716854 loss)
    I1227 19:54:00.351691  5629 sgd_solver.cpp:106] Iteration 64100, lr = 0.00015586
    I1227 19:54:07.948369  5629 solver.cpp:237] Iteration 64200, loss = 0.502305
    I1227 19:54:07.948426  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:54:07.948446  5629 solver.cpp:253]     Train net output #1: loss = 0.502305 (* 1 = 0.502305 loss)
    I1227 19:54:07.948464  5629 sgd_solver.cpp:106] Iteration 64200, lr = 0.000155702
    I1227 19:54:15.619809  5629 solver.cpp:237] Iteration 64300, loss = 0.626447
    I1227 19:54:15.619861  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:54:15.619882  5629 solver.cpp:253]     Train net output #1: loss = 0.626447 (* 1 = 0.626447 loss)
    I1227 19:54:15.619899  5629 sgd_solver.cpp:106] Iteration 64300, lr = 0.000155545
    I1227 19:54:23.141885  5629 solver.cpp:237] Iteration 64400, loss = 0.714884
    I1227 19:54:23.141944  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:54:23.141966  5629 solver.cpp:253]     Train net output #1: loss = 0.714884 (* 1 = 0.714884 loss)
    I1227 19:54:23.141983  5629 sgd_solver.cpp:106] Iteration 64400, lr = 0.000155388
    I1227 19:54:30.640296  5629 solver.cpp:237] Iteration 64500, loss = 0.57515
    I1227 19:54:30.640473  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:54:30.640502  5629 solver.cpp:253]     Train net output #1: loss = 0.57515 (* 1 = 0.57515 loss)
    I1227 19:54:30.640517  5629 sgd_solver.cpp:106] Iteration 64500, lr = 0.000155232
    I1227 19:54:37.897773  5629 solver.cpp:237] Iteration 64600, loss = 0.66746
    I1227 19:54:37.897827  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:54:37.897848  5629 solver.cpp:253]     Train net output #1: loss = 0.66746 (* 1 = 0.66746 loss)
    I1227 19:54:37.897866  5629 sgd_solver.cpp:106] Iteration 64600, lr = 0.000155076
    I1227 19:54:45.591450  5629 solver.cpp:237] Iteration 64700, loss = 0.596854
    I1227 19:54:45.591529  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:54:45.591564  5629 solver.cpp:253]     Train net output #1: loss = 0.596854 (* 1 = 0.596854 loss)
    I1227 19:54:45.591590  5629 sgd_solver.cpp:106] Iteration 64700, lr = 0.00015492
    I1227 19:54:53.993319  5629 solver.cpp:237] Iteration 64800, loss = 0.630774
    I1227 19:54:53.993374  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:54:53.993396  5629 solver.cpp:253]     Train net output #1: loss = 0.630774 (* 1 = 0.630774 loss)
    I1227 19:54:53.993412  5629 sgd_solver.cpp:106] Iteration 64800, lr = 0.000154765
    I1227 19:55:02.400888  5629 solver.cpp:237] Iteration 64900, loss = 0.740716
    I1227 19:55:02.401051  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:55:02.401077  5629 solver.cpp:253]     Train net output #1: loss = 0.740716 (* 1 = 0.740716 loss)
    I1227 19:55:02.401093  5629 sgd_solver.cpp:106] Iteration 64900, lr = 0.00015461
    I1227 19:55:09.669940  5629 solver.cpp:341] Iteration 65000, Testing net (#0)
    I1227 19:55:12.616829  5629 solver.cpp:409]     Test net output #0: accuracy = 0.7465
    I1227 19:55:12.616888  5629 solver.cpp:409]     Test net output #1: loss = 0.727205 (* 1 = 0.727205 loss)
    I1227 19:55:12.650215  5629 solver.cpp:237] Iteration 65000, loss = 0.563503
    I1227 19:55:12.650249  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:55:12.650267  5629 solver.cpp:253]     Train net output #1: loss = 0.563503 (* 1 = 0.563503 loss)
    I1227 19:55:12.650284  5629 sgd_solver.cpp:106] Iteration 65000, lr = 0.000154455
    I1227 19:55:20.431922  5629 solver.cpp:237] Iteration 65100, loss = 0.771739
    I1227 19:55:20.431978  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:55:20.432000  5629 solver.cpp:253]     Train net output #1: loss = 0.771739 (* 1 = 0.771739 loss)
    I1227 19:55:20.432018  5629 sgd_solver.cpp:106] Iteration 65100, lr = 0.000154301
    I1227 19:55:28.017828  5629 solver.cpp:237] Iteration 65200, loss = 0.590788
    I1227 19:55:28.017889  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 19:55:28.017912  5629 solver.cpp:253]     Train net output #1: loss = 0.590788 (* 1 = 0.590788 loss)
    I1227 19:55:28.017930  5629 sgd_solver.cpp:106] Iteration 65200, lr = 0.000154147
    I1227 19:55:37.389633  5629 solver.cpp:237] Iteration 65300, loss = 0.529101
    I1227 19:55:37.389806  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:55:37.389832  5629 solver.cpp:253]     Train net output #1: loss = 0.529101 (* 1 = 0.529101 loss)
    I1227 19:55:37.389842  5629 sgd_solver.cpp:106] Iteration 65300, lr = 0.000153993
    I1227 19:55:45.592525  5629 solver.cpp:237] Iteration 65400, loss = 0.703231
    I1227 19:55:45.592591  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:55:45.592614  5629 solver.cpp:253]     Train net output #1: loss = 0.703231 (* 1 = 0.703231 loss)
    I1227 19:55:45.592633  5629 sgd_solver.cpp:106] Iteration 65400, lr = 0.00015384
    I1227 19:55:54.557360  5629 solver.cpp:237] Iteration 65500, loss = 0.517666
    I1227 19:55:54.557416  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:55:54.557437  5629 solver.cpp:253]     Train net output #1: loss = 0.517666 (* 1 = 0.517666 loss)
    I1227 19:55:54.557452  5629 sgd_solver.cpp:106] Iteration 65500, lr = 0.000153687
    I1227 19:56:01.690898  5629 solver.cpp:237] Iteration 65600, loss = 0.690726
    I1227 19:56:01.690955  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:56:01.690978  5629 solver.cpp:253]     Train net output #1: loss = 0.690726 (* 1 = 0.690726 loss)
    I1227 19:56:01.690994  5629 sgd_solver.cpp:106] Iteration 65600, lr = 0.000153535
    I1227 19:56:08.968863  5629 solver.cpp:237] Iteration 65700, loss = 0.516869
    I1227 19:56:08.969107  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 19:56:08.969133  5629 solver.cpp:253]     Train net output #1: loss = 0.516869 (* 1 = 0.516869 loss)
    I1227 19:56:08.969146  5629 sgd_solver.cpp:106] Iteration 65700, lr = 0.000153383
    I1227 19:56:16.111277  5629 solver.cpp:237] Iteration 65800, loss = 0.666892
    I1227 19:56:16.111322  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:56:16.111338  5629 solver.cpp:253]     Train net output #1: loss = 0.666892 (* 1 = 0.666892 loss)
    I1227 19:56:16.111349  5629 sgd_solver.cpp:106] Iteration 65800, lr = 0.000153231
    I1227 19:56:23.664686  5629 solver.cpp:237] Iteration 65900, loss = 0.783968
    I1227 19:56:23.664746  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:56:23.664760  5629 solver.cpp:253]     Train net output #1: loss = 0.783968 (* 1 = 0.783968 loss)
    I1227 19:56:23.664772  5629 sgd_solver.cpp:106] Iteration 65900, lr = 0.000153079
    I1227 19:56:30.636368  5629 solver.cpp:341] Iteration 66000, Testing net (#0)
    I1227 19:56:33.476778  5629 solver.cpp:409]     Test net output #0: accuracy = 0.74675
    I1227 19:56:33.476826  5629 solver.cpp:409]     Test net output #1: loss = 0.733207 (* 1 = 0.733207 loss)
    I1227 19:56:33.508688  5629 solver.cpp:237] Iteration 66000, loss = 0.593925
    I1227 19:56:33.508716  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:56:33.508726  5629 solver.cpp:253]     Train net output #1: loss = 0.593925 (* 1 = 0.593925 loss)
    I1227 19:56:33.508736  5629 sgd_solver.cpp:106] Iteration 66000, lr = 0.000152928
    I1227 19:56:40.808328  5629 solver.cpp:237] Iteration 66100, loss = 0.665527
    I1227 19:56:40.808488  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 19:56:40.808514  5629 solver.cpp:253]     Train net output #1: loss = 0.665527 (* 1 = 0.665527 loss)
    I1227 19:56:40.808524  5629 sgd_solver.cpp:106] Iteration 66100, lr = 0.000152778
    I1227 19:56:48.141840  5629 solver.cpp:237] Iteration 66200, loss = 0.532533
    I1227 19:56:48.141897  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:56:48.141919  5629 solver.cpp:253]     Train net output #1: loss = 0.532533 (* 1 = 0.532533 loss)
    I1227 19:56:48.141937  5629 sgd_solver.cpp:106] Iteration 66200, lr = 0.000152627
    I1227 19:56:55.240571  5629 solver.cpp:237] Iteration 66300, loss = 0.677477
    I1227 19:56:55.240615  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:56:55.240630  5629 solver.cpp:253]     Train net output #1: loss = 0.677477 (* 1 = 0.677477 loss)
    I1227 19:56:55.240640  5629 sgd_solver.cpp:106] Iteration 66300, lr = 0.000152477
    I1227 19:57:02.318475  5629 solver.cpp:237] Iteration 66400, loss = 0.731443
    I1227 19:57:02.318519  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:57:02.318534  5629 solver.cpp:253]     Train net output #1: loss = 0.731443 (* 1 = 0.731443 loss)
    I1227 19:57:02.318545  5629 sgd_solver.cpp:106] Iteration 66400, lr = 0.000152327
    I1227 19:57:09.329959  5629 solver.cpp:237] Iteration 66500, loss = 0.600773
    I1227 19:57:09.330004  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:57:09.330016  5629 solver.cpp:253]     Train net output #1: loss = 0.600773 (* 1 = 0.600773 loss)
    I1227 19:57:09.330024  5629 sgd_solver.cpp:106] Iteration 66500, lr = 0.000152178
    I1227 19:57:16.394593  5629 solver.cpp:237] Iteration 66600, loss = 0.723669
    I1227 19:57:16.394742  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:57:16.394755  5629 solver.cpp:253]     Train net output #1: loss = 0.723669 (* 1 = 0.723669 loss)
    I1227 19:57:16.394760  5629 sgd_solver.cpp:106] Iteration 66600, lr = 0.000152029
    I1227 19:57:23.423346  5629 solver.cpp:237] Iteration 66700, loss = 0.528765
    I1227 19:57:23.423393  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:57:23.423404  5629 solver.cpp:253]     Train net output #1: loss = 0.528765 (* 1 = 0.528765 loss)
    I1227 19:57:23.423413  5629 sgd_solver.cpp:106] Iteration 66700, lr = 0.00015188
    I1227 19:57:30.439990  5629 solver.cpp:237] Iteration 66800, loss = 0.541601
    I1227 19:57:30.440037  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:57:30.440049  5629 solver.cpp:253]     Train net output #1: loss = 0.541601 (* 1 = 0.541601 loss)
    I1227 19:57:30.440057  5629 sgd_solver.cpp:106] Iteration 66800, lr = 0.000151732
    I1227 19:57:37.505388  5629 solver.cpp:237] Iteration 66900, loss = 0.738646
    I1227 19:57:37.505445  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:57:37.505467  5629 solver.cpp:253]     Train net output #1: loss = 0.738646 (* 1 = 0.738646 loss)
    I1227 19:57:37.505483  5629 sgd_solver.cpp:106] Iteration 66900, lr = 0.000151584
    I1227 19:57:44.601781  5629 solver.cpp:341] Iteration 67000, Testing net (#0)
    I1227 19:57:47.509343  5629 solver.cpp:409]     Test net output #0: accuracy = 0.751167
    I1227 19:57:47.509508  5629 solver.cpp:409]     Test net output #1: loss = 0.71589 (* 1 = 0.71589 loss)
    I1227 19:57:47.537783  5629 solver.cpp:237] Iteration 67000, loss = 0.693053
    I1227 19:57:47.537811  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 19:57:47.537822  5629 solver.cpp:253]     Train net output #1: loss = 0.693053 (* 1 = 0.693053 loss)
    I1227 19:57:47.537832  5629 sgd_solver.cpp:106] Iteration 67000, lr = 0.000151436
    I1227 19:57:54.684890  5629 solver.cpp:237] Iteration 67100, loss = 0.675778
    I1227 19:57:54.684939  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:57:54.684952  5629 solver.cpp:253]     Train net output #1: loss = 0.675778 (* 1 = 0.675778 loss)
    I1227 19:57:54.684962  5629 sgd_solver.cpp:106] Iteration 67100, lr = 0.000151289
    I1227 19:58:01.765048  5629 solver.cpp:237] Iteration 67200, loss = 0.56699
    I1227 19:58:01.765103  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:58:01.765116  5629 solver.cpp:253]     Train net output #1: loss = 0.56699 (* 1 = 0.56699 loss)
    I1227 19:58:01.765127  5629 sgd_solver.cpp:106] Iteration 67200, lr = 0.000151142
    I1227 19:58:08.876281  5629 solver.cpp:237] Iteration 67300, loss = 0.676598
    I1227 19:58:08.876330  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:58:08.876349  5629 solver.cpp:253]     Train net output #1: loss = 0.676598 (* 1 = 0.676598 loss)
    I1227 19:58:08.876363  5629 sgd_solver.cpp:106] Iteration 67300, lr = 0.000150995
    I1227 19:58:16.694358  5629 solver.cpp:237] Iteration 67400, loss = 0.682068
    I1227 19:58:16.694407  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 19:58:16.694422  5629 solver.cpp:253]     Train net output #1: loss = 0.682068 (* 1 = 0.682068 loss)
    I1227 19:58:16.694432  5629 sgd_solver.cpp:106] Iteration 67400, lr = 0.000150849
    I1227 19:58:23.816725  5629 solver.cpp:237] Iteration 67500, loss = 0.598698
    I1227 19:58:23.816882  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 19:58:23.816907  5629 solver.cpp:253]     Train net output #1: loss = 0.598698 (* 1 = 0.598698 loss)
    I1227 19:58:23.816917  5629 sgd_solver.cpp:106] Iteration 67500, lr = 0.000150703
    I1227 19:58:30.857421  5629 solver.cpp:237] Iteration 67600, loss = 0.764198
    I1227 19:58:30.857457  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 19:58:30.857470  5629 solver.cpp:253]     Train net output #1: loss = 0.764198 (* 1 = 0.764198 loss)
    I1227 19:58:30.857478  5629 sgd_solver.cpp:106] Iteration 67600, lr = 0.000150557
    I1227 19:58:37.901163  5629 solver.cpp:237] Iteration 67700, loss = 0.585307
    I1227 19:58:37.901216  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 19:58:37.901237  5629 solver.cpp:253]     Train net output #1: loss = 0.585307 (* 1 = 0.585307 loss)
    I1227 19:58:37.901252  5629 sgd_solver.cpp:106] Iteration 67700, lr = 0.000150412
    I1227 19:58:44.999086  5629 solver.cpp:237] Iteration 67800, loss = 0.675998
    I1227 19:58:44.999130  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 19:58:44.999146  5629 solver.cpp:253]     Train net output #1: loss = 0.675998 (* 1 = 0.675998 loss)
    I1227 19:58:44.999157  5629 sgd_solver.cpp:106] Iteration 67800, lr = 0.000150267
    I1227 19:58:53.542202  5629 solver.cpp:237] Iteration 67900, loss = 0.600841
    I1227 19:58:53.542255  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 19:58:53.542271  5629 solver.cpp:253]     Train net output #1: loss = 0.600841 (* 1 = 0.600841 loss)
    I1227 19:58:53.542284  5629 sgd_solver.cpp:106] Iteration 67900, lr = 0.000150122
    I1227 19:59:00.720686  5629 solver.cpp:341] Iteration 68000, Testing net (#0)
    I1227 19:59:03.593958  5629 solver.cpp:409]     Test net output #0: accuracy = 0.742667
    I1227 19:59:03.594007  5629 solver.cpp:409]     Test net output #1: loss = 0.736194 (* 1 = 0.736194 loss)
    I1227 19:59:03.624157  5629 solver.cpp:237] Iteration 68000, loss = 0.551664
    I1227 19:59:03.624187  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:59:03.624200  5629 solver.cpp:253]     Train net output #1: loss = 0.551664 (* 1 = 0.551664 loss)
    I1227 19:59:03.624212  5629 sgd_solver.cpp:106] Iteration 68000, lr = 0.000149978
    I1227 19:59:11.366571  5629 solver.cpp:237] Iteration 68100, loss = 0.724034
    I1227 19:59:11.366614  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 19:59:11.366631  5629 solver.cpp:253]     Train net output #1: loss = 0.724034 (* 1 = 0.724034 loss)
    I1227 19:59:11.366641  5629 sgd_solver.cpp:106] Iteration 68100, lr = 0.000149834
    I1227 19:59:19.091246  5629 solver.cpp:237] Iteration 68200, loss = 0.496309
    I1227 19:59:19.091286  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 19:59:19.091298  5629 solver.cpp:253]     Train net output #1: loss = 0.496309 (* 1 = 0.496309 loss)
    I1227 19:59:19.091308  5629 sgd_solver.cpp:106] Iteration 68200, lr = 0.00014969
    I1227 19:59:26.195684  5629 solver.cpp:237] Iteration 68300, loss = 0.657453
    I1227 19:59:26.195725  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:59:26.195740  5629 solver.cpp:253]     Train net output #1: loss = 0.657453 (* 1 = 0.657453 loss)
    I1227 19:59:26.195751  5629 sgd_solver.cpp:106] Iteration 68300, lr = 0.000149547
    I1227 19:59:33.335399  5629 solver.cpp:237] Iteration 68400, loss = 0.635681
    I1227 19:59:33.335605  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:59:33.335620  5629 solver.cpp:253]     Train net output #1: loss = 0.635681 (* 1 = 0.635681 loss)
    I1227 19:59:33.335629  5629 sgd_solver.cpp:106] Iteration 68400, lr = 0.000149404
    I1227 19:59:40.526710  5629 solver.cpp:237] Iteration 68500, loss = 0.575234
    I1227 19:59:40.526768  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 19:59:40.526789  5629 solver.cpp:253]     Train net output #1: loss = 0.575234 (* 1 = 0.575234 loss)
    I1227 19:59:40.526805  5629 sgd_solver.cpp:106] Iteration 68500, lr = 0.000149261
    I1227 19:59:47.687153  5629 solver.cpp:237] Iteration 68600, loss = 0.652993
    I1227 19:59:47.687203  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 19:59:47.687217  5629 solver.cpp:253]     Train net output #1: loss = 0.652993 (* 1 = 0.652993 loss)
    I1227 19:59:47.687228  5629 sgd_solver.cpp:106] Iteration 68600, lr = 0.000149118
    I1227 19:59:54.638937  5629 solver.cpp:237] Iteration 68700, loss = 0.595109
    I1227 19:59:54.638975  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 19:59:54.638988  5629 solver.cpp:253]     Train net output #1: loss = 0.595109 (* 1 = 0.595109 loss)
    I1227 19:59:54.638996  5629 sgd_solver.cpp:106] Iteration 68700, lr = 0.000148976
    I1227 20:00:01.822990  5629 solver.cpp:237] Iteration 68800, loss = 0.707943
    I1227 20:00:01.823034  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:00:01.823047  5629 solver.cpp:253]     Train net output #1: loss = 0.707943 (* 1 = 0.707943 loss)
    I1227 20:00:01.823058  5629 sgd_solver.cpp:106] Iteration 68800, lr = 0.000148834
    I1227 20:00:09.262101  5629 solver.cpp:237] Iteration 68900, loss = 0.679011
    I1227 20:00:09.262270  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:00:09.262286  5629 solver.cpp:253]     Train net output #1: loss = 0.679011 (* 1 = 0.679011 loss)
    I1227 20:00:09.262295  5629 sgd_solver.cpp:106] Iteration 68900, lr = 0.000148693
    I1227 20:00:16.208148  5629 solver.cpp:341] Iteration 69000, Testing net (#0)
    I1227 20:00:19.086225  5629 solver.cpp:409]     Test net output #0: accuracy = 0.749583
    I1227 20:00:19.086269  5629 solver.cpp:409]     Test net output #1: loss = 0.728686 (* 1 = 0.728686 loss)
    I1227 20:00:19.119135  5629 solver.cpp:237] Iteration 69000, loss = 0.624287
    I1227 20:00:19.119174  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:00:19.119186  5629 solver.cpp:253]     Train net output #1: loss = 0.624287 (* 1 = 0.624287 loss)
    I1227 20:00:19.119199  5629 sgd_solver.cpp:106] Iteration 69000, lr = 0.000148552
    I1227 20:00:26.179267  5629 solver.cpp:237] Iteration 69100, loss = 0.691214
    I1227 20:00:26.179303  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:00:26.179314  5629 solver.cpp:253]     Train net output #1: loss = 0.691214 (* 1 = 0.691214 loss)
    I1227 20:00:26.179324  5629 sgd_solver.cpp:106] Iteration 69100, lr = 0.000148411
    I1227 20:00:33.294009  5629 solver.cpp:237] Iteration 69200, loss = 0.543676
    I1227 20:00:33.294049  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:00:33.294064  5629 solver.cpp:253]     Train net output #1: loss = 0.543676 (* 1 = 0.543676 loss)
    I1227 20:00:33.294073  5629 sgd_solver.cpp:106] Iteration 69200, lr = 0.00014827
    I1227 20:00:40.398380  5629 solver.cpp:237] Iteration 69300, loss = 0.65418
    I1227 20:00:40.398490  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:00:40.398507  5629 solver.cpp:253]     Train net output #1: loss = 0.65418 (* 1 = 0.65418 loss)
    I1227 20:00:40.398515  5629 sgd_solver.cpp:106] Iteration 69300, lr = 0.00014813
    I1227 20:00:47.616724  5629 solver.cpp:237] Iteration 69400, loss = 0.64154
    I1227 20:00:47.616768  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:00:47.616785  5629 solver.cpp:253]     Train net output #1: loss = 0.64154 (* 1 = 0.64154 loss)
    I1227 20:00:47.616796  5629 sgd_solver.cpp:106] Iteration 69400, lr = 0.00014799
    I1227 20:00:54.765390  5629 solver.cpp:237] Iteration 69500, loss = 0.517314
    I1227 20:00:54.765446  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:00:54.765467  5629 solver.cpp:253]     Train net output #1: loss = 0.517314 (* 1 = 0.517314 loss)
    I1227 20:00:54.765485  5629 sgd_solver.cpp:106] Iteration 69500, lr = 0.00014785
    I1227 20:01:02.026211  5629 solver.cpp:237] Iteration 69600, loss = 0.809697
    I1227 20:01:02.026247  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 20:01:02.026257  5629 solver.cpp:253]     Train net output #1: loss = 0.809697 (* 1 = 0.809697 loss)
    I1227 20:01:02.026265  5629 sgd_solver.cpp:106] Iteration 69600, lr = 0.000147711
    I1227 20:01:09.061050  5629 solver.cpp:237] Iteration 69700, loss = 0.603081
    I1227 20:01:09.061103  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:01:09.061126  5629 solver.cpp:253]     Train net output #1: loss = 0.603081 (* 1 = 0.603081 loss)
    I1227 20:01:09.061143  5629 sgd_solver.cpp:106] Iteration 69700, lr = 0.000147572
    I1227 20:01:16.049564  5629 solver.cpp:237] Iteration 69800, loss = 0.625251
    I1227 20:01:16.049727  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:01:16.049751  5629 solver.cpp:253]     Train net output #1: loss = 0.625251 (* 1 = 0.625251 loss)
    I1227 20:01:16.049758  5629 sgd_solver.cpp:106] Iteration 69800, lr = 0.000147433
    I1227 20:01:23.000764  5629 solver.cpp:237] Iteration 69900, loss = 0.74363
    I1227 20:01:23.000810  5629 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1227 20:01:23.000823  5629 solver.cpp:253]     Train net output #1: loss = 0.74363 (* 1 = 0.74363 loss)
    I1227 20:01:23.000830  5629 sgd_solver.cpp:106] Iteration 69900, lr = 0.000147295
    I1227 20:01:29.880537  5629 solver.cpp:341] Iteration 70000, Testing net (#0)
    I1227 20:01:32.661494  5629 solver.cpp:409]     Test net output #0: accuracy = 0.745333
    I1227 20:01:32.661543  5629 solver.cpp:409]     Test net output #1: loss = 0.735784 (* 1 = 0.735784 loss)
    I1227 20:01:32.700717  5629 solver.cpp:237] Iteration 70000, loss = 0.52412
    I1227 20:01:32.700770  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:01:32.700795  5629 solver.cpp:253]     Train net output #1: loss = 0.52412 (* 1 = 0.52412 loss)
    I1227 20:01:32.700806  5629 sgd_solver.cpp:106] Iteration 70000, lr = 0.000147157
    I1227 20:01:39.665957  5629 solver.cpp:237] Iteration 70100, loss = 0.705539
    I1227 20:01:39.666019  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:01:39.666043  5629 solver.cpp:253]     Train net output #1: loss = 0.705539 (* 1 = 0.705539 loss)
    I1227 20:01:39.666061  5629 sgd_solver.cpp:106] Iteration 70100, lr = 0.000147019
    I1227 20:01:46.635668  5629 solver.cpp:237] Iteration 70200, loss = 0.655451
    I1227 20:01:46.635819  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:01:46.635833  5629 solver.cpp:253]     Train net output #1: loss = 0.655451 (* 1 = 0.655451 loss)
    I1227 20:01:46.635841  5629 sgd_solver.cpp:106] Iteration 70200, lr = 0.000146882
    I1227 20:01:53.593823  5629 solver.cpp:237] Iteration 70300, loss = 0.631775
    I1227 20:01:53.593860  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:01:53.593873  5629 solver.cpp:253]     Train net output #1: loss = 0.631775 (* 1 = 0.631775 loss)
    I1227 20:01:53.593880  5629 sgd_solver.cpp:106] Iteration 70300, lr = 0.000146744
    I1227 20:02:00.588232  5629 solver.cpp:237] Iteration 70400, loss = 0.732161
    I1227 20:02:00.588275  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 20:02:00.588289  5629 solver.cpp:253]     Train net output #1: loss = 0.732161 (* 1 = 0.732161 loss)
    I1227 20:02:00.588299  5629 sgd_solver.cpp:106] Iteration 70400, lr = 0.000146607
    I1227 20:02:07.827831  5629 solver.cpp:237] Iteration 70500, loss = 0.644525
    I1227 20:02:07.827879  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:02:07.827893  5629 solver.cpp:253]     Train net output #1: loss = 0.644525 (* 1 = 0.644525 loss)
    I1227 20:02:07.827903  5629 sgd_solver.cpp:106] Iteration 70500, lr = 0.000146471
    I1227 20:02:14.783620  5629 solver.cpp:237] Iteration 70600, loss = 0.801652
    I1227 20:02:14.783671  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:02:14.783686  5629 solver.cpp:253]     Train net output #1: loss = 0.801652 (* 1 = 0.801652 loss)
    I1227 20:02:14.783697  5629 sgd_solver.cpp:106] Iteration 70600, lr = 0.000146335
    I1227 20:02:21.739218  5629 solver.cpp:237] Iteration 70700, loss = 0.621801
    I1227 20:02:21.739423  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:02:21.739436  5629 solver.cpp:253]     Train net output #1: loss = 0.621801 (* 1 = 0.621801 loss)
    I1227 20:02:21.739444  5629 sgd_solver.cpp:106] Iteration 70700, lr = 0.000146198
    I1227 20:02:28.739986  5629 solver.cpp:237] Iteration 70800, loss = 0.578368
    I1227 20:02:28.740031  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:02:28.740046  5629 solver.cpp:253]     Train net output #1: loss = 0.578368 (* 1 = 0.578368 loss)
    I1227 20:02:28.740057  5629 sgd_solver.cpp:106] Iteration 70800, lr = 0.000146063
    I1227 20:02:35.651536  5629 solver.cpp:237] Iteration 70900, loss = 0.626066
    I1227 20:02:35.651576  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:02:35.651587  5629 solver.cpp:253]     Train net output #1: loss = 0.626066 (* 1 = 0.626066 loss)
    I1227 20:02:35.651595  5629 sgd_solver.cpp:106] Iteration 70900, lr = 0.000145927
    I1227 20:02:43.067284  5629 solver.cpp:341] Iteration 71000, Testing net (#0)
    I1227 20:02:45.894212  5629 solver.cpp:409]     Test net output #0: accuracy = 0.740167
    I1227 20:02:45.894259  5629 solver.cpp:409]     Test net output #1: loss = 0.754898 (* 1 = 0.754898 loss)
    I1227 20:02:45.924548  5629 solver.cpp:237] Iteration 71000, loss = 0.646314
    I1227 20:02:45.924592  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:02:45.924608  5629 solver.cpp:253]     Train net output #1: loss = 0.646314 (* 1 = 0.646314 loss)
    I1227 20:02:45.924619  5629 sgd_solver.cpp:106] Iteration 71000, lr = 0.000145792
    I1227 20:02:52.937188  5629 solver.cpp:237] Iteration 71100, loss = 0.683216
    I1227 20:02:52.937348  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:02:52.937362  5629 solver.cpp:253]     Train net output #1: loss = 0.683216 (* 1 = 0.683216 loss)
    I1227 20:02:52.937371  5629 sgd_solver.cpp:106] Iteration 71100, lr = 0.000145657
    I1227 20:02:59.887727  5629 solver.cpp:237] Iteration 71200, loss = 0.539836
    I1227 20:02:59.887763  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:02:59.887773  5629 solver.cpp:253]     Train net output #1: loss = 0.539836 (* 1 = 0.539836 loss)
    I1227 20:02:59.887783  5629 sgd_solver.cpp:106] Iteration 71200, lr = 0.000145523
    I1227 20:03:06.867563  5629 solver.cpp:237] Iteration 71300, loss = 0.589436
    I1227 20:03:06.867619  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:03:06.867642  5629 solver.cpp:253]     Train net output #1: loss = 0.589436 (* 1 = 0.589436 loss)
    I1227 20:03:06.867660  5629 sgd_solver.cpp:106] Iteration 71300, lr = 0.000145389
    I1227 20:03:14.022579  5629 solver.cpp:237] Iteration 71400, loss = 0.675991
    I1227 20:03:14.022617  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:03:14.022629  5629 solver.cpp:253]     Train net output #1: loss = 0.675991 (* 1 = 0.675991 loss)
    I1227 20:03:14.022639  5629 sgd_solver.cpp:106] Iteration 71400, lr = 0.000145255
    I1227 20:03:21.705348  5629 solver.cpp:237] Iteration 71500, loss = 0.614437
    I1227 20:03:21.705406  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:03:21.705427  5629 solver.cpp:253]     Train net output #1: loss = 0.614437 (* 1 = 0.614437 loss)
    I1227 20:03:21.705443  5629 sgd_solver.cpp:106] Iteration 71500, lr = 0.000145121
    I1227 20:03:29.117116  5629 solver.cpp:237] Iteration 71600, loss = 0.784929
    I1227 20:03:29.117280  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:03:29.117295  5629 solver.cpp:253]     Train net output #1: loss = 0.784929 (* 1 = 0.784929 loss)
    I1227 20:03:29.117305  5629 sgd_solver.cpp:106] Iteration 71600, lr = 0.000144987
    I1227 20:03:36.061420  5629 solver.cpp:237] Iteration 71700, loss = 0.570549
    I1227 20:03:36.061475  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:03:36.061496  5629 solver.cpp:253]     Train net output #1: loss = 0.570549 (* 1 = 0.570549 loss)
    I1227 20:03:36.061511  5629 sgd_solver.cpp:106] Iteration 71700, lr = 0.000144854
    I1227 20:03:43.048979  5629 solver.cpp:237] Iteration 71800, loss = 0.635068
    I1227 20:03:43.049018  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:03:43.049032  5629 solver.cpp:253]     Train net output #1: loss = 0.635068 (* 1 = 0.635068 loss)
    I1227 20:03:43.049041  5629 sgd_solver.cpp:106] Iteration 71800, lr = 0.000144721
    I1227 20:03:49.991736  5629 solver.cpp:237] Iteration 71900, loss = 0.68737
    I1227 20:03:49.991775  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:03:49.991786  5629 solver.cpp:253]     Train net output #1: loss = 0.68737 (* 1 = 0.68737 loss)
    I1227 20:03:49.991794  5629 sgd_solver.cpp:106] Iteration 71900, lr = 0.000144589
    I1227 20:03:56.906538  5629 solver.cpp:341] Iteration 72000, Testing net (#0)
    I1227 20:03:59.739899  5629 solver.cpp:409]     Test net output #0: accuracy = 0.747667
    I1227 20:03:59.740051  5629 solver.cpp:409]     Test net output #1: loss = 0.730646 (* 1 = 0.730646 loss)
    I1227 20:03:59.777058  5629 solver.cpp:237] Iteration 72000, loss = 0.525569
    I1227 20:03:59.777107  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:03:59.777120  5629 solver.cpp:253]     Train net output #1: loss = 0.525569 (* 1 = 0.525569 loss)
    I1227 20:03:59.777132  5629 sgd_solver.cpp:106] Iteration 72000, lr = 0.000144457
    I1227 20:04:07.494115  5629 solver.cpp:237] Iteration 72100, loss = 0.683711
    I1227 20:04:07.494156  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:04:07.494170  5629 solver.cpp:253]     Train net output #1: loss = 0.683711 (* 1 = 0.683711 loss)
    I1227 20:04:07.494181  5629 sgd_solver.cpp:106] Iteration 72100, lr = 0.000144325
    I1227 20:04:15.034901  5629 solver.cpp:237] Iteration 72200, loss = 0.525558
    I1227 20:04:15.034947  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:04:15.034958  5629 solver.cpp:253]     Train net output #1: loss = 0.525558 (* 1 = 0.525558 loss)
    I1227 20:04:15.034967  5629 sgd_solver.cpp:106] Iteration 72200, lr = 0.000144193
    I1227 20:04:22.390717  5629 solver.cpp:237] Iteration 72300, loss = 0.600052
    I1227 20:04:22.390771  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:04:22.390794  5629 solver.cpp:253]     Train net output #1: loss = 0.600052 (* 1 = 0.600052 loss)
    I1227 20:04:22.390810  5629 sgd_solver.cpp:106] Iteration 72300, lr = 0.000144062
    I1227 20:04:29.628526  5629 solver.cpp:237] Iteration 72400, loss = 0.677865
    I1227 20:04:29.628571  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:04:29.628592  5629 solver.cpp:253]     Train net output #1: loss = 0.677865 (* 1 = 0.677865 loss)
    I1227 20:04:29.628602  5629 sgd_solver.cpp:106] Iteration 72400, lr = 0.00014393
    I1227 20:04:36.590615  5629 solver.cpp:237] Iteration 72500, loss = 0.567246
    I1227 20:04:36.590736  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:04:36.590771  5629 solver.cpp:253]     Train net output #1: loss = 0.567246 (* 1 = 0.567246 loss)
    I1227 20:04:36.590780  5629 sgd_solver.cpp:106] Iteration 72500, lr = 0.0001438
    I1227 20:04:43.540735  5629 solver.cpp:237] Iteration 72600, loss = 0.804911
    I1227 20:04:43.540783  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:04:43.540797  5629 solver.cpp:253]     Train net output #1: loss = 0.804911 (* 1 = 0.804911 loss)
    I1227 20:04:43.540807  5629 sgd_solver.cpp:106] Iteration 72600, lr = 0.000143669
    I1227 20:04:50.538408  5629 solver.cpp:237] Iteration 72700, loss = 0.615146
    I1227 20:04:50.538462  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:04:50.538483  5629 solver.cpp:253]     Train net output #1: loss = 0.615146 (* 1 = 0.615146 loss)
    I1227 20:04:50.538498  5629 sgd_solver.cpp:106] Iteration 72700, lr = 0.000143539
    I1227 20:04:57.523092  5629 solver.cpp:237] Iteration 72800, loss = 0.571264
    I1227 20:04:57.523134  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:04:57.523149  5629 solver.cpp:253]     Train net output #1: loss = 0.571264 (* 1 = 0.571264 loss)
    I1227 20:04:57.523157  5629 sgd_solver.cpp:106] Iteration 72800, lr = 0.000143409
    I1227 20:05:04.461470  5629 solver.cpp:237] Iteration 72900, loss = 0.584033
    I1227 20:05:04.461508  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:05:04.461519  5629 solver.cpp:253]     Train net output #1: loss = 0.584033 (* 1 = 0.584033 loss)
    I1227 20:05:04.461527  5629 sgd_solver.cpp:106] Iteration 72900, lr = 0.000143279
    I1227 20:05:11.326091  5629 solver.cpp:341] Iteration 73000, Testing net (#0)
    I1227 20:05:14.166074  5629 solver.cpp:409]     Test net output #0: accuracy = 0.745417
    I1227 20:05:14.166165  5629 solver.cpp:409]     Test net output #1: loss = 0.72636 (* 1 = 0.72636 loss)
    I1227 20:05:14.212450  5629 solver.cpp:237] Iteration 73000, loss = 0.523751
    I1227 20:05:14.212508  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:05:14.212529  5629 solver.cpp:253]     Train net output #1: loss = 0.523751 (* 1 = 0.523751 loss)
    I1227 20:05:14.212546  5629 sgd_solver.cpp:106] Iteration 73000, lr = 0.000143149
    I1227 20:05:21.599761  5629 solver.cpp:237] Iteration 73100, loss = 0.671678
    I1227 20:05:21.599803  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:05:21.599817  5629 solver.cpp:253]     Train net output #1: loss = 0.671678 (* 1 = 0.671678 loss)
    I1227 20:05:21.599827  5629 sgd_solver.cpp:106] Iteration 73100, lr = 0.00014302
    I1227 20:05:29.029847  5629 solver.cpp:237] Iteration 73200, loss = 0.587577
    I1227 20:05:29.029904  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:05:29.029925  5629 solver.cpp:253]     Train net output #1: loss = 0.587577 (* 1 = 0.587577 loss)
    I1227 20:05:29.029942  5629 sgd_solver.cpp:106] Iteration 73200, lr = 0.000142891
    I1227 20:05:36.245818  5629 solver.cpp:237] Iteration 73300, loss = 0.55883
    I1227 20:05:36.245867  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:05:36.245887  5629 solver.cpp:253]     Train net output #1: loss = 0.55883 (* 1 = 0.55883 loss)
    I1227 20:05:36.245899  5629 sgd_solver.cpp:106] Iteration 73300, lr = 0.000142763
    I1227 20:05:43.230377  5629 solver.cpp:237] Iteration 73400, loss = 0.728044
    I1227 20:05:43.230504  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 20:05:43.230522  5629 solver.cpp:253]     Train net output #1: loss = 0.728044 (* 1 = 0.728044 loss)
    I1227 20:05:43.230533  5629 sgd_solver.cpp:106] Iteration 73400, lr = 0.000142634
    I1227 20:05:50.194506  5629 solver.cpp:237] Iteration 73500, loss = 0.596377
    I1227 20:05:50.194546  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:05:50.194561  5629 solver.cpp:253]     Train net output #1: loss = 0.596377 (* 1 = 0.596377 loss)
    I1227 20:05:50.194571  5629 sgd_solver.cpp:106] Iteration 73500, lr = 0.000142506
    I1227 20:05:57.748703  5629 solver.cpp:237] Iteration 73600, loss = 0.730154
    I1227 20:05:57.748744  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:05:57.748757  5629 solver.cpp:253]     Train net output #1: loss = 0.730154 (* 1 = 0.730154 loss)
    I1227 20:05:57.748769  5629 sgd_solver.cpp:106] Iteration 73600, lr = 0.000142378
    I1227 20:06:05.727432  5629 solver.cpp:237] Iteration 73700, loss = 0.573324
    I1227 20:06:05.727475  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:06:05.727490  5629 solver.cpp:253]     Train net output #1: loss = 0.573324 (* 1 = 0.573324 loss)
    I1227 20:06:05.727502  5629 sgd_solver.cpp:106] Iteration 73700, lr = 0.000142251
    I1227 20:06:13.489032  5629 solver.cpp:237] Iteration 73800, loss = 0.559025
    I1227 20:06:13.489178  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:06:13.489204  5629 solver.cpp:253]     Train net output #1: loss = 0.559025 (* 1 = 0.559025 loss)
    I1227 20:06:13.489223  5629 sgd_solver.cpp:106] Iteration 73800, lr = 0.000142123
    I1227 20:06:21.643368  5629 solver.cpp:237] Iteration 73900, loss = 0.745782
    I1227 20:06:21.643404  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:06:21.643416  5629 solver.cpp:253]     Train net output #1: loss = 0.745782 (* 1 = 0.745782 loss)
    I1227 20:06:21.643424  5629 sgd_solver.cpp:106] Iteration 73900, lr = 0.000141996
    I1227 20:06:28.906141  5629 solver.cpp:341] Iteration 74000, Testing net (#0)
    I1227 20:06:32.024027  5629 solver.cpp:409]     Test net output #0: accuracy = 0.747167
    I1227 20:06:32.024078  5629 solver.cpp:409]     Test net output #1: loss = 0.731086 (* 1 = 0.731086 loss)
    I1227 20:06:32.064792  5629 solver.cpp:237] Iteration 74000, loss = 0.583938
    I1227 20:06:32.064836  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:06:32.064851  5629 solver.cpp:253]     Train net output #1: loss = 0.583938 (* 1 = 0.583938 loss)
    I1227 20:06:32.064863  5629 sgd_solver.cpp:106] Iteration 74000, lr = 0.000141869
    I1227 20:06:39.415920  5629 solver.cpp:237] Iteration 74100, loss = 0.67295
    I1227 20:06:39.415966  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:06:39.415979  5629 solver.cpp:253]     Train net output #1: loss = 0.67295 (* 1 = 0.67295 loss)
    I1227 20:06:39.415990  5629 sgd_solver.cpp:106] Iteration 74100, lr = 0.000141743
    I1227 20:06:46.613808  5629 solver.cpp:237] Iteration 74200, loss = 0.510806
    I1227 20:06:46.613977  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:06:46.614001  5629 solver.cpp:253]     Train net output #1: loss = 0.510806 (* 1 = 0.510806 loss)
    I1227 20:06:46.614012  5629 sgd_solver.cpp:106] Iteration 74200, lr = 0.000141617
    I1227 20:06:54.714278  5629 solver.cpp:237] Iteration 74300, loss = 0.646379
    I1227 20:06:54.714318  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:06:54.714330  5629 solver.cpp:253]     Train net output #1: loss = 0.646379 (* 1 = 0.646379 loss)
    I1227 20:06:54.714340  5629 sgd_solver.cpp:106] Iteration 74300, lr = 0.000141491
    I1227 20:07:02.199156  5629 solver.cpp:237] Iteration 74400, loss = 0.615259
    I1227 20:07:02.199216  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:07:02.199239  5629 solver.cpp:253]     Train net output #1: loss = 0.615259 (* 1 = 0.615259 loss)
    I1227 20:07:02.199255  5629 sgd_solver.cpp:106] Iteration 74400, lr = 0.000141365
    I1227 20:07:09.271543  5629 solver.cpp:237] Iteration 74500, loss = 0.500891
    I1227 20:07:09.271586  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:07:09.271601  5629 solver.cpp:253]     Train net output #1: loss = 0.500891 (* 1 = 0.500891 loss)
    I1227 20:07:09.271611  5629 sgd_solver.cpp:106] Iteration 74500, lr = 0.000141239
    I1227 20:07:16.343994  5629 solver.cpp:237] Iteration 74600, loss = 0.694381
    I1227 20:07:16.344040  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:07:16.344053  5629 solver.cpp:253]     Train net output #1: loss = 0.694381 (* 1 = 0.694381 loss)
    I1227 20:07:16.344061  5629 sgd_solver.cpp:106] Iteration 74600, lr = 0.000141114
    I1227 20:07:23.356066  5629 solver.cpp:237] Iteration 74700, loss = 0.614548
    I1227 20:07:23.356225  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:07:23.356238  5629 solver.cpp:253]     Train net output #1: loss = 0.614548 (* 1 = 0.614548 loss)
    I1227 20:07:23.356243  5629 sgd_solver.cpp:106] Iteration 74700, lr = 0.000140989
    I1227 20:07:30.476001  5629 solver.cpp:237] Iteration 74800, loss = 0.551361
    I1227 20:07:30.476039  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:07:30.476052  5629 solver.cpp:253]     Train net output #1: loss = 0.551361 (* 1 = 0.551361 loss)
    I1227 20:07:30.476060  5629 sgd_solver.cpp:106] Iteration 74800, lr = 0.000140864
    I1227 20:07:37.831610  5629 solver.cpp:237] Iteration 74900, loss = 0.738645
    I1227 20:07:37.831655  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:07:37.831670  5629 solver.cpp:253]     Train net output #1: loss = 0.738645 (* 1 = 0.738645 loss)
    I1227 20:07:37.831681  5629 sgd_solver.cpp:106] Iteration 74900, lr = 0.00014074
    I1227 20:07:45.367727  5629 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_75000.caffemodel
    I1227 20:07:45.408464  5629 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_75000.solverstate
    I1227 20:07:45.409482  5629 solver.cpp:341] Iteration 75000, Testing net (#0)
    I1227 20:07:48.352877  5629 solver.cpp:409]     Test net output #0: accuracy = 0.747167
    I1227 20:07:48.352923  5629 solver.cpp:409]     Test net output #1: loss = 0.731318 (* 1 = 0.731318 loss)
    I1227 20:07:48.383126  5629 solver.cpp:237] Iteration 75000, loss = 0.581774
    I1227 20:07:48.383172  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:07:48.383184  5629 solver.cpp:253]     Train net output #1: loss = 0.581774 (* 1 = 0.581774 loss)
    I1227 20:07:48.383196  5629 sgd_solver.cpp:106] Iteration 75000, lr = 0.000140616
    I1227 20:07:57.304600  5629 solver.cpp:237] Iteration 75100, loss = 0.783733
    I1227 20:07:57.304795  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 20:07:57.304811  5629 solver.cpp:253]     Train net output #1: loss = 0.783733 (* 1 = 0.783733 loss)
    I1227 20:07:57.304822  5629 sgd_solver.cpp:106] Iteration 75100, lr = 0.000140492
    I1227 20:08:05.475255  5629 solver.cpp:237] Iteration 75200, loss = 0.523408
    I1227 20:08:05.475317  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:08:05.475342  5629 solver.cpp:253]     Train net output #1: loss = 0.523408 (* 1 = 0.523408 loss)
    I1227 20:08:05.475358  5629 sgd_solver.cpp:106] Iteration 75200, lr = 0.000140368
    I1227 20:08:13.316980  5629 solver.cpp:237] Iteration 75300, loss = 0.570156
    I1227 20:08:13.317023  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:08:13.317039  5629 solver.cpp:253]     Train net output #1: loss = 0.570156 (* 1 = 0.570156 loss)
    I1227 20:08:13.317051  5629 sgd_solver.cpp:106] Iteration 75300, lr = 0.000140245
    I1227 20:08:21.173467  5629 solver.cpp:237] Iteration 75400, loss = 0.629364
    I1227 20:08:21.173514  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:08:21.173529  5629 solver.cpp:253]     Train net output #1: loss = 0.629364 (* 1 = 0.629364 loss)
    I1227 20:08:21.173542  5629 sgd_solver.cpp:106] Iteration 75400, lr = 0.000140121
    I1227 20:08:29.991467  5629 solver.cpp:237] Iteration 75500, loss = 0.668579
    I1227 20:08:29.991588  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 20:08:29.991605  5629 solver.cpp:253]     Train net output #1: loss = 0.668579 (* 1 = 0.668579 loss)
    I1227 20:08:29.991618  5629 sgd_solver.cpp:106] Iteration 75500, lr = 0.000139999
    I1227 20:08:37.836016  5629 solver.cpp:237] Iteration 75600, loss = 0.7255
    I1227 20:08:37.836061  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:08:37.836074  5629 solver.cpp:253]     Train net output #1: loss = 0.7255 (* 1 = 0.7255 loss)
    I1227 20:08:37.836086  5629 sgd_solver.cpp:106] Iteration 75600, lr = 0.000139876
    I1227 20:08:45.701449  5629 solver.cpp:237] Iteration 75700, loss = 0.563233
    I1227 20:08:45.701493  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:08:45.701508  5629 solver.cpp:253]     Train net output #1: loss = 0.563233 (* 1 = 0.563233 loss)
    I1227 20:08:45.701520  5629 sgd_solver.cpp:106] Iteration 75700, lr = 0.000139753
    I1227 20:08:53.534647  5629 solver.cpp:237] Iteration 75800, loss = 0.553508
    I1227 20:08:53.534693  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:08:53.534708  5629 solver.cpp:253]     Train net output #1: loss = 0.553508 (* 1 = 0.553508 loss)
    I1227 20:08:53.534719  5629 sgd_solver.cpp:106] Iteration 75800, lr = 0.000139631
    I1227 20:09:01.410353  5629 solver.cpp:237] Iteration 75900, loss = 0.683392
    I1227 20:09:01.410447  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:09:01.410465  5629 solver.cpp:253]     Train net output #1: loss = 0.683392 (* 1 = 0.683392 loss)
    I1227 20:09:01.410476  5629 sgd_solver.cpp:106] Iteration 75900, lr = 0.000139509
    I1227 20:09:09.183576  5629 solver.cpp:341] Iteration 76000, Testing net (#0)
    I1227 20:09:12.459349  5629 solver.cpp:409]     Test net output #0: accuracy = 0.742833
    I1227 20:09:12.459406  5629 solver.cpp:409]     Test net output #1: loss = 0.731547 (* 1 = 0.731547 loss)
    I1227 20:09:12.494153  5629 solver.cpp:237] Iteration 76000, loss = 0.538651
    I1227 20:09:12.494196  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:09:12.494210  5629 solver.cpp:253]     Train net output #1: loss = 0.538651 (* 1 = 0.538651 loss)
    I1227 20:09:12.494222  5629 sgd_solver.cpp:106] Iteration 76000, lr = 0.000139388
    I1227 20:09:20.348199  5629 solver.cpp:237] Iteration 76100, loss = 0.635132
    I1227 20:09:20.348256  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:09:20.348276  5629 solver.cpp:253]     Train net output #1: loss = 0.635132 (* 1 = 0.635132 loss)
    I1227 20:09:20.348292  5629 sgd_solver.cpp:106] Iteration 76100, lr = 0.000139266
    I1227 20:09:28.211810  5629 solver.cpp:237] Iteration 76200, loss = 0.616369
    I1227 20:09:28.211856  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:09:28.211871  5629 solver.cpp:253]     Train net output #1: loss = 0.616369 (* 1 = 0.616369 loss)
    I1227 20:09:28.211884  5629 sgd_solver.cpp:106] Iteration 76200, lr = 0.000139145
    I1227 20:09:36.057468  5629 solver.cpp:237] Iteration 76300, loss = 0.681161
    I1227 20:09:36.057597  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:09:36.057615  5629 solver.cpp:253]     Train net output #1: loss = 0.681161 (* 1 = 0.681161 loss)
    I1227 20:09:36.057624  5629 sgd_solver.cpp:106] Iteration 76300, lr = 0.000139024
    I1227 20:09:43.931064  5629 solver.cpp:237] Iteration 76400, loss = 0.675492
    I1227 20:09:43.931123  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:09:43.931146  5629 solver.cpp:253]     Train net output #1: loss = 0.675492 (* 1 = 0.675492 loss)
    I1227 20:09:43.931164  5629 sgd_solver.cpp:106] Iteration 76400, lr = 0.000138903
    I1227 20:09:51.809111  5629 solver.cpp:237] Iteration 76500, loss = 0.558075
    I1227 20:09:51.809167  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:09:51.809185  5629 solver.cpp:253]     Train net output #1: loss = 0.558075 (* 1 = 0.558075 loss)
    I1227 20:09:51.809197  5629 sgd_solver.cpp:106] Iteration 76500, lr = 0.000138783
    I1227 20:09:59.665971  5629 solver.cpp:237] Iteration 76600, loss = 0.682576
    I1227 20:09:59.666030  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:09:59.666054  5629 solver.cpp:253]     Train net output #1: loss = 0.682576 (* 1 = 0.682576 loss)
    I1227 20:09:59.666070  5629 sgd_solver.cpp:106] Iteration 76600, lr = 0.000138663
    I1227 20:10:07.527732  5629 solver.cpp:237] Iteration 76700, loss = 0.549511
    I1227 20:10:07.527834  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:10:07.527850  5629 solver.cpp:253]     Train net output #1: loss = 0.549511 (* 1 = 0.549511 loss)
    I1227 20:10:07.527863  5629 sgd_solver.cpp:106] Iteration 76700, lr = 0.000138543
    I1227 20:10:15.409728  5629 solver.cpp:237] Iteration 76800, loss = 0.573056
    I1227 20:10:15.409786  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:10:15.409809  5629 solver.cpp:253]     Train net output #1: loss = 0.573056 (* 1 = 0.573056 loss)
    I1227 20:10:15.409826  5629 sgd_solver.cpp:106] Iteration 76800, lr = 0.000138423
    I1227 20:10:23.269114  5629 solver.cpp:237] Iteration 76900, loss = 0.670908
    I1227 20:10:23.269161  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:10:23.269177  5629 solver.cpp:253]     Train net output #1: loss = 0.670908 (* 1 = 0.670908 loss)
    I1227 20:10:23.269189  5629 sgd_solver.cpp:106] Iteration 76900, lr = 0.000138304
    I1227 20:10:31.042948  5629 solver.cpp:341] Iteration 77000, Testing net (#0)
    I1227 20:10:34.480892  5629 solver.cpp:409]     Test net output #0: accuracy = 0.750334
    I1227 20:10:34.480944  5629 solver.cpp:409]     Test net output #1: loss = 0.71559 (* 1 = 0.71559 loss)
    I1227 20:10:34.515432  5629 solver.cpp:237] Iteration 77000, loss = 0.600038
    I1227 20:10:34.515491  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:10:34.515506  5629 solver.cpp:253]     Train net output #1: loss = 0.600038 (* 1 = 0.600038 loss)
    I1227 20:10:34.515518  5629 sgd_solver.cpp:106] Iteration 77000, lr = 0.000138184
    I1227 20:10:42.401698  5629 solver.cpp:237] Iteration 77100, loss = 0.644348
    I1227 20:10:42.401900  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:10:42.401931  5629 solver.cpp:253]     Train net output #1: loss = 0.644348 (* 1 = 0.644348 loss)
    I1227 20:10:42.401950  5629 sgd_solver.cpp:106] Iteration 77100, lr = 0.000138065
    I1227 20:10:50.256275  5629 solver.cpp:237] Iteration 77200, loss = 0.587258
    I1227 20:10:50.256319  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:10:50.256332  5629 solver.cpp:253]     Train net output #1: loss = 0.587258 (* 1 = 0.587258 loss)
    I1227 20:10:50.256343  5629 sgd_solver.cpp:106] Iteration 77200, lr = 0.000137946
    I1227 20:10:58.137274  5629 solver.cpp:237] Iteration 77300, loss = 0.518637
    I1227 20:10:58.137339  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:10:58.137364  5629 solver.cpp:253]     Train net output #1: loss = 0.518637 (* 1 = 0.518637 loss)
    I1227 20:10:58.137384  5629 sgd_solver.cpp:106] Iteration 77300, lr = 0.000137828
    I1227 20:11:06.002595  5629 solver.cpp:237] Iteration 77400, loss = 0.666619
    I1227 20:11:06.002656  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:11:06.002671  5629 solver.cpp:253]     Train net output #1: loss = 0.666619 (* 1 = 0.666619 loss)
    I1227 20:11:06.002683  5629 sgd_solver.cpp:106] Iteration 77400, lr = 0.00013771
    I1227 20:11:13.869761  5629 solver.cpp:237] Iteration 77500, loss = 0.635998
    I1227 20:11:13.869933  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:11:13.869961  5629 solver.cpp:253]     Train net output #1: loss = 0.635998 (* 1 = 0.635998 loss)
    I1227 20:11:13.869973  5629 sgd_solver.cpp:106] Iteration 77500, lr = 0.000137592
    I1227 20:11:21.714103  5629 solver.cpp:237] Iteration 77600, loss = 0.773953
    I1227 20:11:21.714150  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:11:21.714164  5629 solver.cpp:253]     Train net output #1: loss = 0.773953 (* 1 = 0.773953 loss)
    I1227 20:11:21.714175  5629 sgd_solver.cpp:106] Iteration 77600, lr = 0.000137474
    I1227 20:11:29.569003  5629 solver.cpp:237] Iteration 77700, loss = 0.477348
    I1227 20:11:29.569069  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:11:29.569093  5629 solver.cpp:253]     Train net output #1: loss = 0.477348 (* 1 = 0.477348 loss)
    I1227 20:11:29.569113  5629 sgd_solver.cpp:106] Iteration 77700, lr = 0.000137356
    I1227 20:11:37.403952  5629 solver.cpp:237] Iteration 77800, loss = 0.566705
    I1227 20:11:37.404000  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:11:37.404016  5629 solver.cpp:253]     Train net output #1: loss = 0.566705 (* 1 = 0.566705 loss)
    I1227 20:11:37.404027  5629 sgd_solver.cpp:106] Iteration 77800, lr = 0.000137239
    I1227 20:11:45.282512  5629 solver.cpp:237] Iteration 77900, loss = 0.657462
    I1227 20:11:45.282668  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:11:45.282696  5629 solver.cpp:253]     Train net output #1: loss = 0.657462 (* 1 = 0.657462 loss)
    I1227 20:11:45.282708  5629 sgd_solver.cpp:106] Iteration 77900, lr = 0.000137122
    I1227 20:11:53.066519  5629 solver.cpp:341] Iteration 78000, Testing net (#0)
    I1227 20:11:56.356353  5629 solver.cpp:409]     Test net output #0: accuracy = 0.748417
    I1227 20:11:56.356407  5629 solver.cpp:409]     Test net output #1: loss = 0.721598 (* 1 = 0.721598 loss)
    I1227 20:11:56.391059  5629 solver.cpp:237] Iteration 78000, loss = 0.654017
    I1227 20:11:56.391116  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:11:56.391132  5629 solver.cpp:253]     Train net output #1: loss = 0.654017 (* 1 = 0.654017 loss)
    I1227 20:11:56.391146  5629 sgd_solver.cpp:106] Iteration 78000, lr = 0.000137005
    I1227 20:12:04.275291  5629 solver.cpp:237] Iteration 78100, loss = 0.776344
    I1227 20:12:04.275355  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:12:04.275378  5629 solver.cpp:253]     Train net output #1: loss = 0.776344 (* 1 = 0.776344 loss)
    I1227 20:12:04.275395  5629 sgd_solver.cpp:106] Iteration 78100, lr = 0.000136888
    I1227 20:12:12.118149  5629 solver.cpp:237] Iteration 78200, loss = 0.51194
    I1227 20:12:12.118192  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:12:12.118206  5629 solver.cpp:253]     Train net output #1: loss = 0.51194 (* 1 = 0.51194 loss)
    I1227 20:12:12.118216  5629 sgd_solver.cpp:106] Iteration 78200, lr = 0.000136772
    I1227 20:12:20.004876  5629 solver.cpp:237] Iteration 78300, loss = 0.587434
    I1227 20:12:20.005077  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:12:20.005105  5629 solver.cpp:253]     Train net output #1: loss = 0.587434 (* 1 = 0.587434 loss)
    I1227 20:12:20.005118  5629 sgd_solver.cpp:106] Iteration 78300, lr = 0.000136656
    I1227 20:12:27.858713  5629 solver.cpp:237] Iteration 78400, loss = 0.666731
    I1227 20:12:27.858772  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:12:27.858788  5629 solver.cpp:253]     Train net output #1: loss = 0.666731 (* 1 = 0.666731 loss)
    I1227 20:12:27.858800  5629 sgd_solver.cpp:106] Iteration 78400, lr = 0.00013654
    I1227 20:12:35.752758  5629 solver.cpp:237] Iteration 78500, loss = 0.505551
    I1227 20:12:35.752825  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:12:35.752849  5629 solver.cpp:253]     Train net output #1: loss = 0.505551 (* 1 = 0.505551 loss)
    I1227 20:12:35.752868  5629 sgd_solver.cpp:106] Iteration 78500, lr = 0.000136424
    I1227 20:12:43.608752  5629 solver.cpp:237] Iteration 78600, loss = 0.689825
    I1227 20:12:43.608799  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:12:43.608814  5629 solver.cpp:253]     Train net output #1: loss = 0.689825 (* 1 = 0.689825 loss)
    I1227 20:12:43.608826  5629 sgd_solver.cpp:106] Iteration 78600, lr = 0.000136308
    I1227 20:12:51.471446  5629 solver.cpp:237] Iteration 78700, loss = 0.626977
    I1227 20:12:51.471607  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:12:51.471627  5629 solver.cpp:253]     Train net output #1: loss = 0.626977 (* 1 = 0.626977 loss)
    I1227 20:12:51.471638  5629 sgd_solver.cpp:106] Iteration 78700, lr = 0.000136193
    I1227 20:12:59.310753  5629 solver.cpp:237] Iteration 78800, loss = 0.619717
    I1227 20:12:59.310817  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:12:59.310833  5629 solver.cpp:253]     Train net output #1: loss = 0.619717 (* 1 = 0.619717 loss)
    I1227 20:12:59.310856  5629 sgd_solver.cpp:106] Iteration 78800, lr = 0.000136078
    I1227 20:13:07.183418  5629 solver.cpp:237] Iteration 78900, loss = 0.758568
    I1227 20:13:07.183490  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 20:13:07.183514  5629 solver.cpp:253]     Train net output #1: loss = 0.758568 (* 1 = 0.758568 loss)
    I1227 20:13:07.183533  5629 sgd_solver.cpp:106] Iteration 78900, lr = 0.000135963
    I1227 20:13:14.943851  5629 solver.cpp:341] Iteration 79000, Testing net (#0)
    I1227 20:13:18.227412  5629 solver.cpp:409]     Test net output #0: accuracy = 0.743833
    I1227 20:13:18.227479  5629 solver.cpp:409]     Test net output #1: loss = 0.731689 (* 1 = 0.731689 loss)
    I1227 20:13:18.262925  5629 solver.cpp:237] Iteration 79000, loss = 0.540461
    I1227 20:13:18.262979  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:13:18.262998  5629 solver.cpp:253]     Train net output #1: loss = 0.540461 (* 1 = 0.540461 loss)
    I1227 20:13:18.263013  5629 sgd_solver.cpp:106] Iteration 79000, lr = 0.000135849
    I1227 20:13:26.145642  5629 solver.cpp:237] Iteration 79100, loss = 0.679985
    I1227 20:13:26.145809  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:13:26.145836  5629 solver.cpp:253]     Train net output #1: loss = 0.679985 (* 1 = 0.679985 loss)
    I1227 20:13:26.145853  5629 sgd_solver.cpp:106] Iteration 79100, lr = 0.000135734
    I1227 20:13:34.000550  5629 solver.cpp:237] Iteration 79200, loss = 0.52566
    I1227 20:13:34.000603  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:13:34.000619  5629 solver.cpp:253]     Train net output #1: loss = 0.52566 (* 1 = 0.52566 loss)
    I1227 20:13:34.000633  5629 sgd_solver.cpp:106] Iteration 79200, lr = 0.00013562
    I1227 20:13:41.844943  5629 solver.cpp:237] Iteration 79300, loss = 0.564274
    I1227 20:13:41.845005  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:13:41.845027  5629 solver.cpp:253]     Train net output #1: loss = 0.564274 (* 1 = 0.564274 loss)
    I1227 20:13:41.845044  5629 sgd_solver.cpp:106] Iteration 79300, lr = 0.000135506
    I1227 20:13:49.705761  5629 solver.cpp:237] Iteration 79400, loss = 0.684729
    I1227 20:13:49.705811  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:13:49.705827  5629 solver.cpp:253]     Train net output #1: loss = 0.684729 (* 1 = 0.684729 loss)
    I1227 20:13:49.705840  5629 sgd_solver.cpp:106] Iteration 79400, lr = 0.000135393
    I1227 20:13:57.593303  5629 solver.cpp:237] Iteration 79500, loss = 0.594313
    I1227 20:13:57.593469  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:13:57.593497  5629 solver.cpp:253]     Train net output #1: loss = 0.594313 (* 1 = 0.594313 loss)
    I1227 20:13:57.593514  5629 sgd_solver.cpp:106] Iteration 79500, lr = 0.000135279
    I1227 20:14:05.458379  5629 solver.cpp:237] Iteration 79600, loss = 0.7213
    I1227 20:14:05.458422  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:14:05.458437  5629 solver.cpp:253]     Train net output #1: loss = 0.7213 (* 1 = 0.7213 loss)
    I1227 20:14:05.458448  5629 sgd_solver.cpp:106] Iteration 79600, lr = 0.000135166
    I1227 20:14:13.328871  5629 solver.cpp:237] Iteration 79700, loss = 0.486574
    I1227 20:14:13.328934  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:14:13.328956  5629 solver.cpp:253]     Train net output #1: loss = 0.486574 (* 1 = 0.486574 loss)
    I1227 20:14:13.328974  5629 sgd_solver.cpp:106] Iteration 79700, lr = 0.000135053
    I1227 20:14:21.209800  5629 solver.cpp:237] Iteration 79800, loss = 0.57263
    I1227 20:14:21.209853  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:14:21.209867  5629 solver.cpp:253]     Train net output #1: loss = 0.57263 (* 1 = 0.57263 loss)
    I1227 20:14:21.209879  5629 sgd_solver.cpp:106] Iteration 79800, lr = 0.00013494
    I1227 20:14:29.073807  5629 solver.cpp:237] Iteration 79900, loss = 0.720653
    I1227 20:14:29.073973  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 20:14:29.074004  5629 solver.cpp:253]     Train net output #1: loss = 0.720653 (* 1 = 0.720653 loss)
    I1227 20:14:29.074021  5629 sgd_solver.cpp:106] Iteration 79900, lr = 0.000134827
    I1227 20:14:36.230636  5629 solver.cpp:341] Iteration 80000, Testing net (#0)
    I1227 20:14:39.016954  5629 solver.cpp:409]     Test net output #0: accuracy = 0.743083
    I1227 20:14:39.016999  5629 solver.cpp:409]     Test net output #1: loss = 0.741454 (* 1 = 0.741454 loss)
    I1227 20:14:39.046557  5629 solver.cpp:237] Iteration 80000, loss = 0.607643
    I1227 20:14:39.046602  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:14:39.046614  5629 solver.cpp:253]     Train net output #1: loss = 0.607643 (* 1 = 0.607643 loss)
    I1227 20:14:39.046627  5629 sgd_solver.cpp:106] Iteration 80000, lr = 0.000134715
    I1227 20:14:45.968572  5629 solver.cpp:237] Iteration 80100, loss = 0.653295
    I1227 20:14:45.968613  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:14:45.968626  5629 solver.cpp:253]     Train net output #1: loss = 0.653295 (* 1 = 0.653295 loss)
    I1227 20:14:45.968634  5629 sgd_solver.cpp:106] Iteration 80100, lr = 0.000134603
    I1227 20:14:52.874310  5629 solver.cpp:237] Iteration 80200, loss = 0.464384
    I1227 20:14:52.874352  5629 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1227 20:14:52.874366  5629 solver.cpp:253]     Train net output #1: loss = 0.464384 (* 1 = 0.464384 loss)
    I1227 20:14:52.874377  5629 sgd_solver.cpp:106] Iteration 80200, lr = 0.000134491
    I1227 20:14:59.773905  5629 solver.cpp:237] Iteration 80300, loss = 0.565773
    I1227 20:14:59.774039  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:14:59.774062  5629 solver.cpp:253]     Train net output #1: loss = 0.565773 (* 1 = 0.565773 loss)
    I1227 20:14:59.774068  5629 sgd_solver.cpp:106] Iteration 80300, lr = 0.000134379
    I1227 20:15:06.677520  5629 solver.cpp:237] Iteration 80400, loss = 0.679536
    I1227 20:15:06.677628  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:15:06.677665  5629 solver.cpp:253]     Train net output #1: loss = 0.679535 (* 1 = 0.679535 loss)
    I1227 20:15:06.677690  5629 sgd_solver.cpp:106] Iteration 80400, lr = 0.000134268
    I1227 20:15:13.587581  5629 solver.cpp:237] Iteration 80500, loss = 0.533282
    I1227 20:15:13.587621  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:15:13.587635  5629 solver.cpp:253]     Train net output #1: loss = 0.533282 (* 1 = 0.533282 loss)
    I1227 20:15:13.587646  5629 sgd_solver.cpp:106] Iteration 80500, lr = 0.000134156
    I1227 20:15:20.496003  5629 solver.cpp:237] Iteration 80600, loss = 0.634363
    I1227 20:15:20.496049  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:15:20.496068  5629 solver.cpp:253]     Train net output #1: loss = 0.634363 (* 1 = 0.634363 loss)
    I1227 20:15:20.496083  5629 sgd_solver.cpp:106] Iteration 80600, lr = 0.000134045
    I1227 20:15:27.407817  5629 solver.cpp:237] Iteration 80700, loss = 0.523612
    I1227 20:15:27.407857  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:15:27.407871  5629 solver.cpp:253]     Train net output #1: loss = 0.523612 (* 1 = 0.523612 loss)
    I1227 20:15:27.407881  5629 sgd_solver.cpp:106] Iteration 80700, lr = 0.000133935
    I1227 20:15:34.307291  5629 solver.cpp:237] Iteration 80800, loss = 0.629115
    I1227 20:15:34.307427  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:15:34.307443  5629 solver.cpp:253]     Train net output #1: loss = 0.629115 (* 1 = 0.629115 loss)
    I1227 20:15:34.307451  5629 sgd_solver.cpp:106] Iteration 80800, lr = 0.000133824
    I1227 20:15:41.223517  5629 solver.cpp:237] Iteration 80900, loss = 0.640756
    I1227 20:15:41.223577  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:15:41.223598  5629 solver.cpp:253]     Train net output #1: loss = 0.640756 (* 1 = 0.640756 loss)
    I1227 20:15:41.223614  5629 sgd_solver.cpp:106] Iteration 80900, lr = 0.000133713
    I1227 20:15:48.108968  5629 solver.cpp:341] Iteration 81000, Testing net (#0)
    I1227 20:15:50.867162  5629 solver.cpp:409]     Test net output #0: accuracy = 0.745167
    I1227 20:15:50.867204  5629 solver.cpp:409]     Test net output #1: loss = 0.736918 (* 1 = 0.736918 loss)
    I1227 20:15:50.896127  5629 solver.cpp:237] Iteration 81000, loss = 0.521446
    I1227 20:15:50.896147  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:15:50.896157  5629 solver.cpp:253]     Train net output #1: loss = 0.521446 (* 1 = 0.521446 loss)
    I1227 20:15:50.896167  5629 sgd_solver.cpp:106] Iteration 81000, lr = 0.000133603
    I1227 20:15:57.815316  5629 solver.cpp:237] Iteration 81100, loss = 0.624991
    I1227 20:15:57.815373  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:15:57.815395  5629 solver.cpp:253]     Train net output #1: loss = 0.624991 (* 1 = 0.624991 loss)
    I1227 20:15:57.815412  5629 sgd_solver.cpp:106] Iteration 81100, lr = 0.000133493
    I1227 20:16:04.721756  5629 solver.cpp:237] Iteration 81200, loss = 0.555279
    I1227 20:16:04.721886  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:16:04.721911  5629 solver.cpp:253]     Train net output #1: loss = 0.555279 (* 1 = 0.555279 loss)
    I1227 20:16:04.721918  5629 sgd_solver.cpp:106] Iteration 81200, lr = 0.000133383
    I1227 20:16:11.604214  5629 solver.cpp:237] Iteration 81300, loss = 0.629002
    I1227 20:16:11.604252  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:16:11.604264  5629 solver.cpp:253]     Train net output #1: loss = 0.629002 (* 1 = 0.629002 loss)
    I1227 20:16:11.604274  5629 sgd_solver.cpp:106] Iteration 81300, lr = 0.000133274
    I1227 20:16:18.517915  5629 solver.cpp:237] Iteration 81400, loss = 0.743154
    I1227 20:16:18.517969  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:16:18.517988  5629 solver.cpp:253]     Train net output #1: loss = 0.743154 (* 1 = 0.743154 loss)
    I1227 20:16:18.517998  5629 sgd_solver.cpp:106] Iteration 81400, lr = 0.000133164
    I1227 20:16:25.417691  5629 solver.cpp:237] Iteration 81500, loss = 0.624319
    I1227 20:16:25.417729  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:16:25.417742  5629 solver.cpp:253]     Train net output #1: loss = 0.624319 (* 1 = 0.624319 loss)
    I1227 20:16:25.417749  5629 sgd_solver.cpp:106] Iteration 81500, lr = 0.000133055
    I1227 20:16:32.327518  5629 solver.cpp:237] Iteration 81600, loss = 0.697523
    I1227 20:16:32.327590  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:16:32.327632  5629 solver.cpp:253]     Train net output #1: loss = 0.697523 (* 1 = 0.697523 loss)
    I1227 20:16:32.327658  5629 sgd_solver.cpp:106] Iteration 81600, lr = 0.000132946
    I1227 20:16:39.788388  5629 solver.cpp:237] Iteration 81700, loss = 0.486829
    I1227 20:16:39.788553  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:16:39.788594  5629 solver.cpp:253]     Train net output #1: loss = 0.486829 (* 1 = 0.486829 loss)
    I1227 20:16:39.788601  5629 sgd_solver.cpp:106] Iteration 81700, lr = 0.000132838
    I1227 20:16:48.112323  5629 solver.cpp:237] Iteration 81800, loss = 0.58344
    I1227 20:16:48.112362  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:16:48.112375  5629 solver.cpp:253]     Train net output #1: loss = 0.58344 (* 1 = 0.58344 loss)
    I1227 20:16:48.112385  5629 sgd_solver.cpp:106] Iteration 81800, lr = 0.000132729
    I1227 20:16:56.175402  5629 solver.cpp:237] Iteration 81900, loss = 0.592487
    I1227 20:16:56.175451  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:16:56.175465  5629 solver.cpp:253]     Train net output #1: loss = 0.592487 (* 1 = 0.592487 loss)
    I1227 20:16:56.175477  5629 sgd_solver.cpp:106] Iteration 81900, lr = 0.000132621
    I1227 20:17:03.717279  5629 solver.cpp:341] Iteration 82000, Testing net (#0)
    I1227 20:17:06.984768  5629 solver.cpp:409]     Test net output #0: accuracy = 0.751417
    I1227 20:17:06.984833  5629 solver.cpp:409]     Test net output #1: loss = 0.708929 (* 1 = 0.708929 loss)
    I1227 20:17:07.018256  5629 solver.cpp:237] Iteration 82000, loss = 0.600745
    I1227 20:17:07.018311  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:17:07.018333  5629 solver.cpp:253]     Train net output #1: loss = 0.600745 (* 1 = 0.600745 loss)
    I1227 20:17:07.018350  5629 sgd_solver.cpp:106] Iteration 82000, lr = 0.000132513
    I1227 20:17:14.248872  5629 solver.cpp:237] Iteration 82100, loss = 0.703191
    I1227 20:17:14.248987  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:17:14.249012  5629 solver.cpp:253]     Train net output #1: loss = 0.70319 (* 1 = 0.70319 loss)
    I1227 20:17:14.249024  5629 sgd_solver.cpp:106] Iteration 82100, lr = 0.000132405
    I1227 20:17:21.577214  5629 solver.cpp:237] Iteration 82200, loss = 0.510338
    I1227 20:17:21.577270  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:17:21.577291  5629 solver.cpp:253]     Train net output #1: loss = 0.510337 (* 1 = 0.510337 loss)
    I1227 20:17:21.577306  5629 sgd_solver.cpp:106] Iteration 82200, lr = 0.000132297
    I1227 20:17:28.864828  5629 solver.cpp:237] Iteration 82300, loss = 0.506707
    I1227 20:17:28.864866  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:17:28.864877  5629 solver.cpp:253]     Train net output #1: loss = 0.506707 (* 1 = 0.506707 loss)
    I1227 20:17:28.864887  5629 sgd_solver.cpp:106] Iteration 82300, lr = 0.000132189
    I1227 20:17:36.010666  5629 solver.cpp:237] Iteration 82400, loss = 0.725278
    I1227 20:17:36.010704  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:17:36.010715  5629 solver.cpp:253]     Train net output #1: loss = 0.725278 (* 1 = 0.725278 loss)
    I1227 20:17:36.010725  5629 sgd_solver.cpp:106] Iteration 82400, lr = 0.000132082
    I1227 20:17:43.597398  5629 solver.cpp:237] Iteration 82500, loss = 0.511864
    I1227 20:17:43.597436  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:17:43.597450  5629 solver.cpp:253]     Train net output #1: loss = 0.511864 (* 1 = 0.511864 loss)
    I1227 20:17:43.597460  5629 sgd_solver.cpp:106] Iteration 82500, lr = 0.000131975
    I1227 20:17:50.732640  5629 solver.cpp:237] Iteration 82600, loss = 0.728187
    I1227 20:17:50.732815  5629 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1227 20:17:50.732828  5629 solver.cpp:253]     Train net output #1: loss = 0.728187 (* 1 = 0.728187 loss)
    I1227 20:17:50.732838  5629 sgd_solver.cpp:106] Iteration 82600, lr = 0.000131868
    I1227 20:17:58.113236  5629 solver.cpp:237] Iteration 82700, loss = 0.482765
    I1227 20:17:58.113277  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:17:58.113291  5629 solver.cpp:253]     Train net output #1: loss = 0.482765 (* 1 = 0.482765 loss)
    I1227 20:17:58.113301  5629 sgd_solver.cpp:106] Iteration 82700, lr = 0.000131761
    I1227 20:18:05.360086  5629 solver.cpp:237] Iteration 82800, loss = 0.584379
    I1227 20:18:05.360151  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:18:05.360175  5629 solver.cpp:253]     Train net output #1: loss = 0.584379 (* 1 = 0.584379 loss)
    I1227 20:18:05.360193  5629 sgd_solver.cpp:106] Iteration 82800, lr = 0.000131655
    I1227 20:18:12.545982  5629 solver.cpp:237] Iteration 82900, loss = 0.675513
    I1227 20:18:12.546036  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:18:12.546057  5629 solver.cpp:253]     Train net output #1: loss = 0.675513 (* 1 = 0.675513 loss)
    I1227 20:18:12.546072  5629 sgd_solver.cpp:106] Iteration 82900, lr = 0.000131549
    I1227 20:18:19.608271  5629 solver.cpp:341] Iteration 83000, Testing net (#0)
    I1227 20:18:22.473176  5629 solver.cpp:409]     Test net output #0: accuracy = 0.741666
    I1227 20:18:22.473348  5629 solver.cpp:409]     Test net output #1: loss = 0.728734 (* 1 = 0.728734 loss)
    I1227 20:18:22.502439  5629 solver.cpp:237] Iteration 83000, loss = 0.528874
    I1227 20:18:22.502488  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:18:22.502501  5629 solver.cpp:253]     Train net output #1: loss = 0.528874 (* 1 = 0.528874 loss)
    I1227 20:18:22.502511  5629 sgd_solver.cpp:106] Iteration 83000, lr = 0.000131443
    I1227 20:18:29.839934  5629 solver.cpp:237] Iteration 83100, loss = 0.740278
    I1227 20:18:29.839978  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:18:29.839993  5629 solver.cpp:253]     Train net output #1: loss = 0.740278 (* 1 = 0.740278 loss)
    I1227 20:18:29.840005  5629 sgd_solver.cpp:106] Iteration 83100, lr = 0.000131337
    I1227 20:18:36.995585  5629 solver.cpp:237] Iteration 83200, loss = 0.480516
    I1227 20:18:36.995622  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:18:36.995633  5629 solver.cpp:253]     Train net output #1: loss = 0.480516 (* 1 = 0.480516 loss)
    I1227 20:18:36.995642  5629 sgd_solver.cpp:106] Iteration 83200, lr = 0.000131231
    I1227 20:18:44.129856  5629 solver.cpp:237] Iteration 83300, loss = 0.560365
    I1227 20:18:44.129910  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:18:44.129933  5629 solver.cpp:253]     Train net output #1: loss = 0.560364 (* 1 = 0.560364 loss)
    I1227 20:18:44.129950  5629 sgd_solver.cpp:106] Iteration 83300, lr = 0.000131125
    I1227 20:18:51.320053  5629 solver.cpp:237] Iteration 83400, loss = 0.644783
    I1227 20:18:51.320089  5629 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1227 20:18:51.320101  5629 solver.cpp:253]     Train net output #1: loss = 0.644783 (* 1 = 0.644783 loss)
    I1227 20:18:51.320111  5629 sgd_solver.cpp:106] Iteration 83400, lr = 0.00013102
    I1227 20:18:58.569960  5629 solver.cpp:237] Iteration 83500, loss = 0.60254
    I1227 20:18:58.570127  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:18:58.570153  5629 solver.cpp:253]     Train net output #1: loss = 0.60254 (* 1 = 0.60254 loss)
    I1227 20:18:58.570168  5629 sgd_solver.cpp:106] Iteration 83500, lr = 0.000130915
    I1227 20:19:05.714087  5629 solver.cpp:237] Iteration 83600, loss = 0.713353
    I1227 20:19:05.714124  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:19:05.714136  5629 solver.cpp:253]     Train net output #1: loss = 0.713353 (* 1 = 0.713353 loss)
    I1227 20:19:05.714146  5629 sgd_solver.cpp:106] Iteration 83600, lr = 0.00013081
    I1227 20:19:12.727519  5629 solver.cpp:237] Iteration 83700, loss = 0.586055
    I1227 20:19:12.727560  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:19:12.727573  5629 solver.cpp:253]     Train net output #1: loss = 0.586054 (* 1 = 0.586054 loss)
    I1227 20:19:12.727586  5629 sgd_solver.cpp:106] Iteration 83700, lr = 0.000130705
    I1227 20:19:19.851375  5629 solver.cpp:237] Iteration 83800, loss = 0.577146
    I1227 20:19:19.851413  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:19:19.851426  5629 solver.cpp:253]     Train net output #1: loss = 0.577146 (* 1 = 0.577146 loss)
    I1227 20:19:19.851436  5629 sgd_solver.cpp:106] Iteration 83800, lr = 0.000130601
    I1227 20:19:26.993464  5629 solver.cpp:237] Iteration 83900, loss = 0.648074
    I1227 20:19:26.993500  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:19:26.993511  5629 solver.cpp:253]     Train net output #1: loss = 0.648074 (* 1 = 0.648074 loss)
    I1227 20:19:26.993520  5629 sgd_solver.cpp:106] Iteration 83900, lr = 0.000130496
    I1227 20:19:33.992725  5629 solver.cpp:341] Iteration 84000, Testing net (#0)
    I1227 20:19:36.853391  5629 solver.cpp:409]     Test net output #0: accuracy = 0.742167
    I1227 20:19:36.853433  5629 solver.cpp:409]     Test net output #1: loss = 0.740123 (* 1 = 0.740123 loss)
    I1227 20:19:36.882279  5629 solver.cpp:237] Iteration 84000, loss = 0.576068
    I1227 20:19:36.882299  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:19:36.882309  5629 solver.cpp:253]     Train net output #1: loss = 0.576068 (* 1 = 0.576068 loss)
    I1227 20:19:36.882319  5629 sgd_solver.cpp:106] Iteration 84000, lr = 0.000130392
    I1227 20:19:44.021446  5629 solver.cpp:237] Iteration 84100, loss = 0.747343
    I1227 20:19:44.021494  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:19:44.021507  5629 solver.cpp:253]     Train net output #1: loss = 0.747343 (* 1 = 0.747343 loss)
    I1227 20:19:44.021517  5629 sgd_solver.cpp:106] Iteration 84100, lr = 0.000130288
    I1227 20:19:51.301839  5629 solver.cpp:237] Iteration 84200, loss = 0.477027
    I1227 20:19:51.301903  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:19:51.301928  5629 solver.cpp:253]     Train net output #1: loss = 0.477027 (* 1 = 0.477027 loss)
    I1227 20:19:51.301945  5629 sgd_solver.cpp:106] Iteration 84200, lr = 0.000130185
    I1227 20:19:58.637306  5629 solver.cpp:237] Iteration 84300, loss = 0.50595
    I1227 20:19:58.637341  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:19:58.637353  5629 solver.cpp:253]     Train net output #1: loss = 0.50595 (* 1 = 0.50595 loss)
    I1227 20:19:58.637362  5629 sgd_solver.cpp:106] Iteration 84300, lr = 0.000130081
    I1227 20:20:05.831851  5629 solver.cpp:237] Iteration 84400, loss = 0.729073
    I1227 20:20:05.832018  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:20:05.832032  5629 solver.cpp:253]     Train net output #1: loss = 0.729073 (* 1 = 0.729073 loss)
    I1227 20:20:05.832039  5629 sgd_solver.cpp:106] Iteration 84400, lr = 0.000129978
    I1227 20:20:13.057363  5629 solver.cpp:237] Iteration 84500, loss = 0.478549
    I1227 20:20:13.057407  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:20:13.057422  5629 solver.cpp:253]     Train net output #1: loss = 0.478549 (* 1 = 0.478549 loss)
    I1227 20:20:13.057433  5629 sgd_solver.cpp:106] Iteration 84500, lr = 0.000129875
    I1227 20:20:20.297049  5629 solver.cpp:237] Iteration 84600, loss = 0.592749
    I1227 20:20:20.297087  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:20:20.297099  5629 solver.cpp:253]     Train net output #1: loss = 0.592749 (* 1 = 0.592749 loss)
    I1227 20:20:20.297109  5629 sgd_solver.cpp:106] Iteration 84600, lr = 0.000129772
    I1227 20:20:27.440353  5629 solver.cpp:237] Iteration 84700, loss = 0.514236
    I1227 20:20:27.440393  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:20:27.440405  5629 solver.cpp:253]     Train net output #1: loss = 0.514236 (* 1 = 0.514236 loss)
    I1227 20:20:27.440414  5629 sgd_solver.cpp:106] Iteration 84700, lr = 0.000129669
    I1227 20:20:34.748543  5629 solver.cpp:237] Iteration 84800, loss = 0.644858
    I1227 20:20:34.748602  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:20:34.748620  5629 solver.cpp:253]     Train net output #1: loss = 0.644858 (* 1 = 0.644858 loss)
    I1227 20:20:34.748631  5629 sgd_solver.cpp:106] Iteration 84800, lr = 0.000129566
    I1227 20:20:41.946115  5629 solver.cpp:237] Iteration 84900, loss = 0.755768
    I1227 20:20:41.946239  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:20:41.946257  5629 solver.cpp:253]     Train net output #1: loss = 0.755768 (* 1 = 0.755768 loss)
    I1227 20:20:41.946269  5629 sgd_solver.cpp:106] Iteration 84900, lr = 0.000129464
    I1227 20:20:48.937958  5629 solver.cpp:341] Iteration 85000, Testing net (#0)
    I1227 20:20:51.741101  5629 solver.cpp:409]     Test net output #0: accuracy = 0.743916
    I1227 20:20:51.741142  5629 solver.cpp:409]     Test net output #1: loss = 0.72525 (* 1 = 0.72525 loss)
    I1227 20:20:51.773640  5629 solver.cpp:237] Iteration 85000, loss = 0.508513
    I1227 20:20:51.773663  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:20:51.773674  5629 solver.cpp:253]     Train net output #1: loss = 0.508512 (* 1 = 0.508512 loss)
    I1227 20:20:51.773684  5629 sgd_solver.cpp:106] Iteration 85000, lr = 0.000129362
    I1227 20:20:59.001284  5629 solver.cpp:237] Iteration 85100, loss = 0.640181
    I1227 20:20:59.001329  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:20:59.001345  5629 solver.cpp:253]     Train net output #1: loss = 0.64018 (* 1 = 0.64018 loss)
    I1227 20:20:59.001358  5629 sgd_solver.cpp:106] Iteration 85100, lr = 0.00012926
    I1227 20:21:06.128265  5629 solver.cpp:237] Iteration 85200, loss = 0.473794
    I1227 20:21:06.128314  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:21:06.128324  5629 solver.cpp:253]     Train net output #1: loss = 0.473794 (* 1 = 0.473794 loss)
    I1227 20:21:06.128334  5629 sgd_solver.cpp:106] Iteration 85200, lr = 0.000129158
    I1227 20:21:13.344157  5629 solver.cpp:237] Iteration 85300, loss = 0.60269
    I1227 20:21:13.344342  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:21:13.344368  5629 solver.cpp:253]     Train net output #1: loss = 0.60269 (* 1 = 0.60269 loss)
    I1227 20:21:13.344379  5629 sgd_solver.cpp:106] Iteration 85300, lr = 0.000129056
    I1227 20:21:20.402806  5629 solver.cpp:237] Iteration 85400, loss = 0.671639
    I1227 20:21:20.402851  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:21:20.402868  5629 solver.cpp:253]     Train net output #1: loss = 0.671638 (* 1 = 0.671638 loss)
    I1227 20:21:20.402878  5629 sgd_solver.cpp:106] Iteration 85400, lr = 0.000128955
    I1227 20:21:27.495105  5629 solver.cpp:237] Iteration 85500, loss = 0.537355
    I1227 20:21:27.495151  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:21:27.495163  5629 solver.cpp:253]     Train net output #1: loss = 0.537355 (* 1 = 0.537355 loss)
    I1227 20:21:27.495172  5629 sgd_solver.cpp:106] Iteration 85500, lr = 0.000128853
    I1227 20:21:34.659063  5629 solver.cpp:237] Iteration 85600, loss = 0.628978
    I1227 20:21:34.659119  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:21:34.659140  5629 solver.cpp:253]     Train net output #1: loss = 0.628978 (* 1 = 0.628978 loss)
    I1227 20:21:34.659157  5629 sgd_solver.cpp:106] Iteration 85600, lr = 0.000128752
    I1227 20:21:41.804432  5629 solver.cpp:237] Iteration 85700, loss = 0.555629
    I1227 20:21:41.804471  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:21:41.804481  5629 solver.cpp:253]     Train net output #1: loss = 0.555629 (* 1 = 0.555629 loss)
    I1227 20:21:41.804491  5629 sgd_solver.cpp:106] Iteration 85700, lr = 0.000128651
    I1227 20:21:49.010491  5629 solver.cpp:237] Iteration 85800, loss = 0.558537
    I1227 20:21:49.010665  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:21:49.010680  5629 solver.cpp:253]     Train net output #1: loss = 0.558537 (* 1 = 0.558537 loss)
    I1227 20:21:49.010689  5629 sgd_solver.cpp:106] Iteration 85800, lr = 0.000128551
    I1227 20:21:56.236799  5629 solver.cpp:237] Iteration 85900, loss = 0.612753
    I1227 20:21:56.236853  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:21:56.236874  5629 solver.cpp:253]     Train net output #1: loss = 0.612752 (* 1 = 0.612752 loss)
    I1227 20:21:56.236891  5629 sgd_solver.cpp:106] Iteration 85900, lr = 0.00012845
    I1227 20:22:03.356724  5629 solver.cpp:341] Iteration 86000, Testing net (#0)
    I1227 20:22:06.214275  5629 solver.cpp:409]     Test net output #0: accuracy = 0.74825
    I1227 20:22:06.214323  5629 solver.cpp:409]     Test net output #1: loss = 0.722414 (* 1 = 0.722414 loss)
    I1227 20:22:06.243088  5629 solver.cpp:237] Iteration 86000, loss = 0.604499
    I1227 20:22:06.243125  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:22:06.243135  5629 solver.cpp:253]     Train net output #1: loss = 0.604498 (* 1 = 0.604498 loss)
    I1227 20:22:06.243145  5629 sgd_solver.cpp:106] Iteration 86000, lr = 0.00012835
    I1227 20:22:13.354048  5629 solver.cpp:237] Iteration 86100, loss = 0.674641
    I1227 20:22:13.354092  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:22:13.354115  5629 solver.cpp:253]     Train net output #1: loss = 0.674641 (* 1 = 0.674641 loss)
    I1227 20:22:13.354125  5629 sgd_solver.cpp:106] Iteration 86100, lr = 0.000128249
    I1227 20:22:20.468098  5629 solver.cpp:237] Iteration 86200, loss = 0.587271
    I1227 20:22:20.468253  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:22:20.468267  5629 solver.cpp:253]     Train net output #1: loss = 0.58727 (* 1 = 0.58727 loss)
    I1227 20:22:20.468276  5629 sgd_solver.cpp:106] Iteration 86200, lr = 0.000128149
    I1227 20:22:27.732235  5629 solver.cpp:237] Iteration 86300, loss = 0.562004
    I1227 20:22:27.732281  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:22:27.732295  5629 solver.cpp:253]     Train net output #1: loss = 0.562004 (* 1 = 0.562004 loss)
    I1227 20:22:27.732308  5629 sgd_solver.cpp:106] Iteration 86300, lr = 0.00012805
    I1227 20:22:34.882019  5629 solver.cpp:237] Iteration 86400, loss = 0.601669
    I1227 20:22:34.882063  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:22:34.882074  5629 solver.cpp:253]     Train net output #1: loss = 0.601669 (* 1 = 0.601669 loss)
    I1227 20:22:34.882084  5629 sgd_solver.cpp:106] Iteration 86400, lr = 0.00012795
    I1227 20:22:41.986416  5629 solver.cpp:237] Iteration 86500, loss = 0.534506
    I1227 20:22:41.986474  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:22:41.986496  5629 solver.cpp:253]     Train net output #1: loss = 0.534506 (* 1 = 0.534506 loss)
    I1227 20:22:41.986513  5629 sgd_solver.cpp:106] Iteration 86500, lr = 0.000127851
    I1227 20:22:49.139734  5629 solver.cpp:237] Iteration 86600, loss = 0.667847
    I1227 20:22:49.139773  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:22:49.139786  5629 solver.cpp:253]     Train net output #1: loss = 0.667847 (* 1 = 0.667847 loss)
    I1227 20:22:49.139796  5629 sgd_solver.cpp:106] Iteration 86600, lr = 0.000127751
    I1227 20:22:56.440822  5629 solver.cpp:237] Iteration 86700, loss = 0.462007
    I1227 20:22:56.440976  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:22:56.440990  5629 solver.cpp:253]     Train net output #1: loss = 0.462007 (* 1 = 0.462007 loss)
    I1227 20:22:56.440997  5629 sgd_solver.cpp:106] Iteration 86700, lr = 0.000127652
    I1227 20:23:03.666944  5629 solver.cpp:237] Iteration 86800, loss = 0.583754
    I1227 20:23:03.666990  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:23:03.667004  5629 solver.cpp:253]     Train net output #1: loss = 0.583754 (* 1 = 0.583754 loss)
    I1227 20:23:03.667016  5629 sgd_solver.cpp:106] Iteration 86800, lr = 0.000127553
    I1227 20:23:10.698112  5629 solver.cpp:237] Iteration 86900, loss = 0.703874
    I1227 20:23:10.698159  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:23:10.698173  5629 solver.cpp:253]     Train net output #1: loss = 0.703874 (* 1 = 0.703874 loss)
    I1227 20:23:10.698184  5629 sgd_solver.cpp:106] Iteration 86900, lr = 0.000127455
    I1227 20:23:17.662960  5629 solver.cpp:341] Iteration 87000, Testing net (#0)
    I1227 20:23:20.546766  5629 solver.cpp:409]     Test net output #0: accuracy = 0.749333
    I1227 20:23:20.546831  5629 solver.cpp:409]     Test net output #1: loss = 0.724976 (* 1 = 0.724976 loss)
    I1227 20:23:20.581082  5629 solver.cpp:237] Iteration 87000, loss = 0.516124
    I1227 20:23:20.581132  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:23:20.581154  5629 solver.cpp:253]     Train net output #1: loss = 0.516124 (* 1 = 0.516124 loss)
    I1227 20:23:20.581172  5629 sgd_solver.cpp:106] Iteration 87000, lr = 0.000127356
    I1227 20:23:27.792870  5629 solver.cpp:237] Iteration 87100, loss = 0.615155
    I1227 20:23:27.793045  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:23:27.793059  5629 solver.cpp:253]     Train net output #1: loss = 0.615155 (* 1 = 0.615155 loss)
    I1227 20:23:27.793068  5629 sgd_solver.cpp:106] Iteration 87100, lr = 0.000127258
    I1227 20:23:35.043596  5629 solver.cpp:237] Iteration 87200, loss = 0.545073
    I1227 20:23:35.043653  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:23:35.043674  5629 solver.cpp:253]     Train net output #1: loss = 0.545073 (* 1 = 0.545073 loss)
    I1227 20:23:35.043691  5629 sgd_solver.cpp:106] Iteration 87200, lr = 0.000127159
    I1227 20:23:42.104029  5629 solver.cpp:237] Iteration 87300, loss = 0.531102
    I1227 20:23:42.104070  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:23:42.104086  5629 solver.cpp:253]     Train net output #1: loss = 0.531101 (* 1 = 0.531101 loss)
    I1227 20:23:42.104099  5629 sgd_solver.cpp:106] Iteration 87300, lr = 0.000127061
    I1227 20:23:49.342300  5629 solver.cpp:237] Iteration 87400, loss = 0.681034
    I1227 20:23:49.342339  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:23:49.342351  5629 solver.cpp:253]     Train net output #1: loss = 0.681034 (* 1 = 0.681034 loss)
    I1227 20:23:49.342361  5629 sgd_solver.cpp:106] Iteration 87400, lr = 0.000126963
    I1227 20:23:56.367517  5629 solver.cpp:237] Iteration 87500, loss = 0.573417
    I1227 20:23:56.367554  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:23:56.367565  5629 solver.cpp:253]     Train net output #1: loss = 0.573416 (* 1 = 0.573416 loss)
    I1227 20:23:56.367573  5629 sgd_solver.cpp:106] Iteration 87500, lr = 0.000126866
    I1227 20:24:03.441491  5629 solver.cpp:237] Iteration 87600, loss = 0.64737
    I1227 20:24:03.441614  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:24:03.441629  5629 solver.cpp:253]     Train net output #1: loss = 0.64737 (* 1 = 0.64737 loss)
    I1227 20:24:03.441639  5629 sgd_solver.cpp:106] Iteration 87600, lr = 0.000126768
    I1227 20:24:10.451340  5629 solver.cpp:237] Iteration 87700, loss = 0.505202
    I1227 20:24:10.451376  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:24:10.451388  5629 solver.cpp:253]     Train net output #1: loss = 0.505202 (* 1 = 0.505202 loss)
    I1227 20:24:10.451397  5629 sgd_solver.cpp:106] Iteration 87700, lr = 0.000126671
    I1227 20:24:17.489382  5629 solver.cpp:237] Iteration 87800, loss = 0.504873
    I1227 20:24:17.489425  5629 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1227 20:24:17.489439  5629 solver.cpp:253]     Train net output #1: loss = 0.504873 (* 1 = 0.504873 loss)
    I1227 20:24:17.489450  5629 sgd_solver.cpp:106] Iteration 87800, lr = 0.000126574
    I1227 20:24:24.583083  5629 solver.cpp:237] Iteration 87900, loss = 0.69145
    I1227 20:24:24.583132  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:24:24.583147  5629 solver.cpp:253]     Train net output #1: loss = 0.69145 (* 1 = 0.69145 loss)
    I1227 20:24:24.583156  5629 sgd_solver.cpp:106] Iteration 87900, lr = 0.000126477
    I1227 20:24:31.701236  5629 solver.cpp:341] Iteration 88000, Testing net (#0)
    I1227 20:24:34.724222  5629 solver.cpp:409]     Test net output #0: accuracy = 0.749667
    I1227 20:24:34.724367  5629 solver.cpp:409]     Test net output #1: loss = 0.711815 (* 1 = 0.711815 loss)
    I1227 20:24:34.753943  5629 solver.cpp:237] Iteration 88000, loss = 0.523153
    I1227 20:24:34.753983  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:24:34.753996  5629 solver.cpp:253]     Train net output #1: loss = 0.523153 (* 1 = 0.523153 loss)
    I1227 20:24:34.754006  5629 sgd_solver.cpp:106] Iteration 88000, lr = 0.00012638
    I1227 20:24:42.345026  5629 solver.cpp:237] Iteration 88100, loss = 0.757426
    I1227 20:24:42.345060  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:24:42.345072  5629 solver.cpp:253]     Train net output #1: loss = 0.757426 (* 1 = 0.757426 loss)
    I1227 20:24:42.345080  5629 sgd_solver.cpp:106] Iteration 88100, lr = 0.000126283
    I1227 20:24:49.603183  5629 solver.cpp:237] Iteration 88200, loss = 0.483596
    I1227 20:24:49.603234  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:24:49.603255  5629 solver.cpp:253]     Train net output #1: loss = 0.483596 (* 1 = 0.483596 loss)
    I1227 20:24:49.603271  5629 sgd_solver.cpp:106] Iteration 88200, lr = 0.000126187
    I1227 20:24:57.123780  5629 solver.cpp:237] Iteration 88300, loss = 0.600096
    I1227 20:24:57.123821  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:24:57.123832  5629 solver.cpp:253]     Train net output #1: loss = 0.600096 (* 1 = 0.600096 loss)
    I1227 20:24:57.123843  5629 sgd_solver.cpp:106] Iteration 88300, lr = 0.000126091
    I1227 20:25:04.411190  5629 solver.cpp:237] Iteration 88400, loss = 0.720678
    I1227 20:25:04.411232  5629 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1227 20:25:04.411248  5629 solver.cpp:253]     Train net output #1: loss = 0.720678 (* 1 = 0.720678 loss)
    I1227 20:25:04.411259  5629 sgd_solver.cpp:106] Iteration 88400, lr = 0.000125995
    I1227 20:25:11.491220  5629 solver.cpp:237] Iteration 88500, loss = 0.514168
    I1227 20:25:11.491363  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:25:11.491386  5629 solver.cpp:253]     Train net output #1: loss = 0.514167 (* 1 = 0.514167 loss)
    I1227 20:25:11.491396  5629 sgd_solver.cpp:106] Iteration 88500, lr = 0.000125899
    I1227 20:25:18.586447  5629 solver.cpp:237] Iteration 88600, loss = 0.618565
    I1227 20:25:18.586491  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:25:18.586506  5629 solver.cpp:253]     Train net output #1: loss = 0.618565 (* 1 = 0.618565 loss)
    I1227 20:25:18.586519  5629 sgd_solver.cpp:106] Iteration 88600, lr = 0.000125803
    I1227 20:25:25.628823  5629 solver.cpp:237] Iteration 88700, loss = 0.579955
    I1227 20:25:25.628865  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:25:25.628880  5629 solver.cpp:253]     Train net output #1: loss = 0.579955 (* 1 = 0.579955 loss)
    I1227 20:25:25.628890  5629 sgd_solver.cpp:106] Iteration 88700, lr = 0.000125707
    I1227 20:25:32.651036  5629 solver.cpp:237] Iteration 88800, loss = 0.639898
    I1227 20:25:32.651078  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:25:32.651093  5629 solver.cpp:253]     Train net output #1: loss = 0.639898 (* 1 = 0.639898 loss)
    I1227 20:25:32.651103  5629 sgd_solver.cpp:106] Iteration 88800, lr = 0.000125612
    I1227 20:25:39.681969  5629 solver.cpp:237] Iteration 88900, loss = 0.696811
    I1227 20:25:39.682016  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:25:39.682031  5629 solver.cpp:253]     Train net output #1: loss = 0.696811 (* 1 = 0.696811 loss)
    I1227 20:25:39.682042  5629 sgd_solver.cpp:106] Iteration 88900, lr = 0.000125516
    I1227 20:25:46.633956  5629 solver.cpp:341] Iteration 89000, Testing net (#0)
    I1227 20:25:49.465301  5629 solver.cpp:409]     Test net output #0: accuracy = 0.741333
    I1227 20:25:49.465348  5629 solver.cpp:409]     Test net output #1: loss = 0.73832 (* 1 = 0.73832 loss)
    I1227 20:25:49.495546  5629 solver.cpp:237] Iteration 89000, loss = 0.575896
    I1227 20:25:49.495590  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:25:49.495604  5629 solver.cpp:253]     Train net output #1: loss = 0.575896 (* 1 = 0.575896 loss)
    I1227 20:25:49.495615  5629 sgd_solver.cpp:106] Iteration 89000, lr = 0.000125421
    I1227 20:25:56.535022  5629 solver.cpp:237] Iteration 89100, loss = 0.768368
    I1227 20:25:56.535063  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:25:56.535079  5629 solver.cpp:253]     Train net output #1: loss = 0.768368 (* 1 = 0.768368 loss)
    I1227 20:25:56.535089  5629 sgd_solver.cpp:106] Iteration 89100, lr = 0.000125326
    I1227 20:26:03.578600  5629 solver.cpp:237] Iteration 89200, loss = 0.636189
    I1227 20:26:03.578644  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:26:03.578660  5629 solver.cpp:253]     Train net output #1: loss = 0.636189 (* 1 = 0.636189 loss)
    I1227 20:26:03.578671  5629 sgd_solver.cpp:106] Iteration 89200, lr = 0.000125232
    I1227 20:26:10.604954  5629 solver.cpp:237] Iteration 89300, loss = 0.603446
    I1227 20:26:10.604995  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:26:10.605010  5629 solver.cpp:253]     Train net output #1: loss = 0.603446 (* 1 = 0.603446 loss)
    I1227 20:26:10.605020  5629 sgd_solver.cpp:106] Iteration 89300, lr = 0.000125137
    I1227 20:26:17.653290  5629 solver.cpp:237] Iteration 89400, loss = 0.668347
    I1227 20:26:17.653427  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:26:17.653445  5629 solver.cpp:253]     Train net output #1: loss = 0.668347 (* 1 = 0.668347 loss)
    I1227 20:26:17.653456  5629 sgd_solver.cpp:106] Iteration 89400, lr = 0.000125043
    I1227 20:26:24.668643  5629 solver.cpp:237] Iteration 89500, loss = 0.635959
    I1227 20:26:24.668687  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:26:24.668702  5629 solver.cpp:253]     Train net output #1: loss = 0.635959 (* 1 = 0.635959 loss)
    I1227 20:26:24.668715  5629 sgd_solver.cpp:106] Iteration 89500, lr = 0.000124948
    I1227 20:26:31.740746  5629 solver.cpp:237] Iteration 89600, loss = 0.707127
    I1227 20:26:31.740787  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:26:31.740803  5629 solver.cpp:253]     Train net output #1: loss = 0.707127 (* 1 = 0.707127 loss)
    I1227 20:26:31.740814  5629 sgd_solver.cpp:106] Iteration 89600, lr = 0.000124854
    I1227 20:26:38.778105  5629 solver.cpp:237] Iteration 89700, loss = 0.602276
    I1227 20:26:38.778147  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:26:38.778162  5629 solver.cpp:253]     Train net output #1: loss = 0.602276 (* 1 = 0.602276 loss)
    I1227 20:26:38.778172  5629 sgd_solver.cpp:106] Iteration 89700, lr = 0.00012476
    I1227 20:26:45.827289  5629 solver.cpp:237] Iteration 89800, loss = 0.544569
    I1227 20:26:45.827332  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:26:45.827347  5629 solver.cpp:253]     Train net output #1: loss = 0.544569 (* 1 = 0.544569 loss)
    I1227 20:26:45.827358  5629 sgd_solver.cpp:106] Iteration 89800, lr = 0.000124667
    I1227 20:26:52.846451  5629 solver.cpp:237] Iteration 89900, loss = 0.704871
    I1227 20:26:52.846555  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:26:52.846572  5629 solver.cpp:253]     Train net output #1: loss = 0.704871 (* 1 = 0.704871 loss)
    I1227 20:26:52.846583  5629 sgd_solver.cpp:106] Iteration 89900, lr = 0.000124573
    I1227 20:26:59.979568  5629 solver.cpp:341] Iteration 90000, Testing net (#0)
    I1227 20:27:02.817407  5629 solver.cpp:409]     Test net output #0: accuracy = 0.745833
    I1227 20:27:02.817456  5629 solver.cpp:409]     Test net output #1: loss = 0.721202 (* 1 = 0.721202 loss)
    I1227 20:27:02.845674  5629 solver.cpp:237] Iteration 90000, loss = 0.562254
    I1227 20:27:02.845701  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:27:02.845711  5629 solver.cpp:253]     Train net output #1: loss = 0.562254 (* 1 = 0.562254 loss)
    I1227 20:27:02.845721  5629 sgd_solver.cpp:106] Iteration 90000, lr = 0.00012448
    I1227 20:27:09.901664  5629 solver.cpp:237] Iteration 90100, loss = 0.736222
    I1227 20:27:09.901707  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:27:09.901718  5629 solver.cpp:253]     Train net output #1: loss = 0.736222 (* 1 = 0.736222 loss)
    I1227 20:27:09.901728  5629 sgd_solver.cpp:106] Iteration 90100, lr = 0.000124386
    I1227 20:27:16.934690  5629 solver.cpp:237] Iteration 90200, loss = 0.48065
    I1227 20:27:16.934727  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:27:16.934739  5629 solver.cpp:253]     Train net output #1: loss = 0.48065 (* 1 = 0.48065 loss)
    I1227 20:27:16.934748  5629 sgd_solver.cpp:106] Iteration 90200, lr = 0.000124293
    I1227 20:27:24.154608  5629 solver.cpp:237] Iteration 90300, loss = 0.575914
    I1227 20:27:24.154783  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:27:24.154799  5629 solver.cpp:253]     Train net output #1: loss = 0.575914 (* 1 = 0.575914 loss)
    I1227 20:27:24.154809  5629 sgd_solver.cpp:106] Iteration 90300, lr = 0.0001242
    I1227 20:27:31.403681  5629 solver.cpp:237] Iteration 90400, loss = 0.721257
    I1227 20:27:31.403723  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:27:31.403738  5629 solver.cpp:253]     Train net output #1: loss = 0.721257 (* 1 = 0.721257 loss)
    I1227 20:27:31.403748  5629 sgd_solver.cpp:106] Iteration 90400, lr = 0.000124107
    I1227 20:27:38.808225  5629 solver.cpp:237] Iteration 90500, loss = 0.505146
    I1227 20:27:38.808264  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:27:38.808275  5629 solver.cpp:253]     Train net output #1: loss = 0.505146 (* 1 = 0.505146 loss)
    I1227 20:27:38.808285  5629 sgd_solver.cpp:106] Iteration 90500, lr = 0.000124015
    I1227 20:27:46.135010  5629 solver.cpp:237] Iteration 90600, loss = 0.634122
    I1227 20:27:46.135051  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:27:46.135066  5629 solver.cpp:253]     Train net output #1: loss = 0.634122 (* 1 = 0.634122 loss)
    I1227 20:27:46.135078  5629 sgd_solver.cpp:106] Iteration 90600, lr = 0.000123922
    I1227 20:27:53.342357  5629 solver.cpp:237] Iteration 90700, loss = 0.492873
    I1227 20:27:53.342394  5629 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1227 20:27:53.342406  5629 solver.cpp:253]     Train net output #1: loss = 0.492873 (* 1 = 0.492873 loss)
    I1227 20:27:53.342414  5629 sgd_solver.cpp:106] Iteration 90700, lr = 0.00012383
    I1227 20:28:00.624688  5629 solver.cpp:237] Iteration 90800, loss = 0.634119
    I1227 20:28:00.624845  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:28:00.624861  5629 solver.cpp:253]     Train net output #1: loss = 0.634119 (* 1 = 0.634119 loss)
    I1227 20:28:00.624868  5629 sgd_solver.cpp:106] Iteration 90800, lr = 0.000123738
    I1227 20:28:07.564128  5629 solver.cpp:237] Iteration 90900, loss = 0.654055
    I1227 20:28:07.564165  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:28:07.564177  5629 solver.cpp:253]     Train net output #1: loss = 0.654055 (* 1 = 0.654055 loss)
    I1227 20:28:07.564185  5629 sgd_solver.cpp:106] Iteration 90900, lr = 0.000123646
    I1227 20:28:14.476286  5629 solver.cpp:341] Iteration 91000, Testing net (#0)
    I1227 20:28:17.477399  5629 solver.cpp:409]     Test net output #0: accuracy = 0.747917
    I1227 20:28:17.477447  5629 solver.cpp:409]     Test net output #1: loss = 0.719645 (* 1 = 0.719645 loss)
    I1227 20:28:17.521703  5629 solver.cpp:237] Iteration 91000, loss = 0.664275
    I1227 20:28:17.521749  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:28:17.521764  5629 solver.cpp:253]     Train net output #1: loss = 0.664275 (* 1 = 0.664275 loss)
    I1227 20:28:17.521776  5629 sgd_solver.cpp:106] Iteration 91000, lr = 0.000123554
    I1227 20:28:24.784601  5629 solver.cpp:237] Iteration 91100, loss = 0.692231
    I1227 20:28:24.784648  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:28:24.784659  5629 solver.cpp:253]     Train net output #1: loss = 0.692231 (* 1 = 0.692231 loss)
    I1227 20:28:24.784668  5629 sgd_solver.cpp:106] Iteration 91100, lr = 0.000123462
    I1227 20:28:32.057834  5629 solver.cpp:237] Iteration 91200, loss = 0.526021
    I1227 20:28:32.058012  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:28:32.058039  5629 solver.cpp:253]     Train net output #1: loss = 0.526021 (* 1 = 0.526021 loss)
    I1227 20:28:32.058055  5629 sgd_solver.cpp:106] Iteration 91200, lr = 0.000123371
    I1227 20:28:39.494942  5629 solver.cpp:237] Iteration 91300, loss = 0.574833
    I1227 20:28:39.494985  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:28:39.495002  5629 solver.cpp:253]     Train net output #1: loss = 0.574833 (* 1 = 0.574833 loss)
    I1227 20:28:39.495013  5629 sgd_solver.cpp:106] Iteration 91300, lr = 0.00012328
    I1227 20:28:47.137694  5629 solver.cpp:237] Iteration 91400, loss = 0.660027
    I1227 20:28:47.137748  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:28:47.137769  5629 solver.cpp:253]     Train net output #1: loss = 0.660027 (* 1 = 0.660027 loss)
    I1227 20:28:47.137786  5629 sgd_solver.cpp:106] Iteration 91400, lr = 0.000123188
    I1227 20:28:54.701467  5629 solver.cpp:237] Iteration 91500, loss = 0.591963
    I1227 20:28:54.701516  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:28:54.701530  5629 solver.cpp:253]     Train net output #1: loss = 0.591963 (* 1 = 0.591963 loss)
    I1227 20:28:54.701540  5629 sgd_solver.cpp:106] Iteration 91500, lr = 0.000123097
    I1227 20:29:01.822254  5629 solver.cpp:237] Iteration 91600, loss = 0.712145
    I1227 20:29:01.822309  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:29:01.822330  5629 solver.cpp:253]     Train net output #1: loss = 0.712145 (* 1 = 0.712145 loss)
    I1227 20:29:01.822345  5629 sgd_solver.cpp:106] Iteration 91600, lr = 0.000123006
    I1227 20:29:08.797729  5629 solver.cpp:237] Iteration 91700, loss = 0.478239
    I1227 20:29:08.797848  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:29:08.797865  5629 solver.cpp:253]     Train net output #1: loss = 0.478239 (* 1 = 0.478239 loss)
    I1227 20:29:08.797873  5629 sgd_solver.cpp:106] Iteration 91700, lr = 0.000122916
    I1227 20:29:15.836422  5629 solver.cpp:237] Iteration 91800, loss = 0.615531
    I1227 20:29:15.836468  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:29:15.836480  5629 solver.cpp:253]     Train net output #1: loss = 0.615531 (* 1 = 0.615531 loss)
    I1227 20:29:15.836489  5629 sgd_solver.cpp:106] Iteration 91800, lr = 0.000122825
    I1227 20:29:22.791450  5629 solver.cpp:237] Iteration 91900, loss = 0.610328
    I1227 20:29:22.791507  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:29:22.791529  5629 solver.cpp:253]     Train net output #1: loss = 0.610328 (* 1 = 0.610328 loss)
    I1227 20:29:22.791544  5629 sgd_solver.cpp:106] Iteration 91900, lr = 0.000122735
    I1227 20:29:29.687660  5629 solver.cpp:341] Iteration 92000, Testing net (#0)
    I1227 20:29:32.486399  5629 solver.cpp:409]     Test net output #0: accuracy = 0.741583
    I1227 20:29:32.486448  5629 solver.cpp:409]     Test net output #1: loss = 0.739138 (* 1 = 0.739138 loss)
    I1227 20:29:32.516690  5629 solver.cpp:237] Iteration 92000, loss = 0.53203
    I1227 20:29:32.516734  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:29:32.516749  5629 solver.cpp:253]     Train net output #1: loss = 0.53203 (* 1 = 0.53203 loss)
    I1227 20:29:32.516762  5629 sgd_solver.cpp:106] Iteration 92000, lr = 0.000122644
    I1227 20:29:39.470762  5629 solver.cpp:237] Iteration 92100, loss = 0.596355
    I1227 20:29:39.470877  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:29:39.470893  5629 solver.cpp:253]     Train net output #1: loss = 0.596355 (* 1 = 0.596355 loss)
    I1227 20:29:39.470901  5629 sgd_solver.cpp:106] Iteration 92100, lr = 0.000122554
    I1227 20:29:46.465685  5629 solver.cpp:237] Iteration 92200, loss = 0.530628
    I1227 20:29:46.465728  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:29:46.465742  5629 solver.cpp:253]     Train net output #1: loss = 0.530628 (* 1 = 0.530628 loss)
    I1227 20:29:46.465752  5629 sgd_solver.cpp:106] Iteration 92200, lr = 0.000122464
    I1227 20:29:53.397088  5629 solver.cpp:237] Iteration 92300, loss = 0.626018
    I1227 20:29:53.397125  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:29:53.397140  5629 solver.cpp:253]     Train net output #1: loss = 0.626018 (* 1 = 0.626018 loss)
    I1227 20:29:53.397151  5629 sgd_solver.cpp:106] Iteration 92300, lr = 0.000122375
    I1227 20:30:00.350266  5629 solver.cpp:237] Iteration 92400, loss = 0.645175
    I1227 20:30:00.350306  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:30:00.350319  5629 solver.cpp:253]     Train net output #1: loss = 0.645175 (* 1 = 0.645175 loss)
    I1227 20:30:00.350329  5629 sgd_solver.cpp:106] Iteration 92400, lr = 0.000122285
    I1227 20:30:07.286907  5629 solver.cpp:237] Iteration 92500, loss = 0.576619
    I1227 20:30:07.286955  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:30:07.286974  5629 solver.cpp:253]     Train net output #1: loss = 0.576619 (* 1 = 0.576619 loss)
    I1227 20:30:07.286988  5629 sgd_solver.cpp:106] Iteration 92500, lr = 0.000122195
    I1227 20:30:14.226053  5629 solver.cpp:237] Iteration 92600, loss = 0.687584
    I1227 20:30:14.226183  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:30:14.226200  5629 solver.cpp:253]     Train net output #1: loss = 0.687584 (* 1 = 0.687584 loss)
    I1227 20:30:14.226208  5629 sgd_solver.cpp:106] Iteration 92600, lr = 0.000122106
    I1227 20:30:24.022390  5629 solver.cpp:237] Iteration 92700, loss = 0.494348
    I1227 20:30:24.022430  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:30:24.022444  5629 solver.cpp:253]     Train net output #1: loss = 0.494348 (* 1 = 0.494348 loss)
    I1227 20:30:24.022451  5629 sgd_solver.cpp:106] Iteration 92700, lr = 0.000122017
    I1227 20:30:31.477195  5629 solver.cpp:237] Iteration 92800, loss = 0.534162
    I1227 20:30:31.477237  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:30:31.477252  5629 solver.cpp:253]     Train net output #1: loss = 0.534162 (* 1 = 0.534162 loss)
    I1227 20:30:31.477262  5629 sgd_solver.cpp:106] Iteration 92800, lr = 0.000121928
    I1227 20:30:38.789435  5629 solver.cpp:237] Iteration 92900, loss = 0.534305
    I1227 20:30:38.789472  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:30:38.789484  5629 solver.cpp:253]     Train net output #1: loss = 0.534305 (* 1 = 0.534305 loss)
    I1227 20:30:38.789494  5629 sgd_solver.cpp:106] Iteration 92900, lr = 0.000121839
    I1227 20:30:46.155716  5629 solver.cpp:341] Iteration 93000, Testing net (#0)
    I1227 20:30:49.199267  5629 solver.cpp:409]     Test net output #0: accuracy = 0.749417
    I1227 20:30:49.199334  5629 solver.cpp:409]     Test net output #1: loss = 0.713871 (* 1 = 0.713871 loss)
    I1227 20:30:49.240506  5629 solver.cpp:237] Iteration 93000, loss = 0.574757
    I1227 20:30:49.240553  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:30:49.240566  5629 solver.cpp:253]     Train net output #1: loss = 0.574757 (* 1 = 0.574757 loss)
    I1227 20:30:49.240581  5629 sgd_solver.cpp:106] Iteration 93000, lr = 0.00012175
    I1227 20:30:56.683928  5629 solver.cpp:237] Iteration 93100, loss = 0.863605
    I1227 20:30:56.683990  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:30:56.684011  5629 solver.cpp:253]     Train net output #1: loss = 0.863605 (* 1 = 0.863605 loss)
    I1227 20:30:56.684028  5629 sgd_solver.cpp:106] Iteration 93100, lr = 0.000121662
    I1227 20:31:04.041908  5629 solver.cpp:237] Iteration 93200, loss = 0.506023
    I1227 20:31:04.041946  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:31:04.041960  5629 solver.cpp:253]     Train net output #1: loss = 0.506023 (* 1 = 0.506023 loss)
    I1227 20:31:04.041968  5629 sgd_solver.cpp:106] Iteration 93200, lr = 0.000121573
    I1227 20:31:10.995313  5629 solver.cpp:237] Iteration 93300, loss = 0.637509
    I1227 20:31:10.995369  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:31:10.995391  5629 solver.cpp:253]     Train net output #1: loss = 0.637509 (* 1 = 0.637509 loss)
    I1227 20:31:10.995406  5629 sgd_solver.cpp:106] Iteration 93300, lr = 0.000121485
    I1227 20:31:18.189795  5629 solver.cpp:237] Iteration 93400, loss = 0.669199
    I1227 20:31:18.189924  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:31:18.189941  5629 solver.cpp:253]     Train net output #1: loss = 0.669199 (* 1 = 0.669199 loss)
    I1227 20:31:18.189949  5629 sgd_solver.cpp:106] Iteration 93400, lr = 0.000121397
    I1227 20:31:25.252257  5629 solver.cpp:237] Iteration 93500, loss = 0.644804
    I1227 20:31:25.252296  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:31:25.252308  5629 solver.cpp:253]     Train net output #1: loss = 0.644804 (* 1 = 0.644804 loss)
    I1227 20:31:25.252317  5629 sgd_solver.cpp:106] Iteration 93500, lr = 0.000121309
    I1227 20:31:32.328739  5629 solver.cpp:237] Iteration 93600, loss = 0.685818
    I1227 20:31:32.328779  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:31:32.328794  5629 solver.cpp:253]     Train net output #1: loss = 0.685817 (* 1 = 0.685817 loss)
    I1227 20:31:32.328804  5629 sgd_solver.cpp:106] Iteration 93600, lr = 0.000121221
    I1227 20:31:39.316133  5629 solver.cpp:237] Iteration 93700, loss = 0.490107
    I1227 20:31:39.316176  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:31:39.316191  5629 solver.cpp:253]     Train net output #1: loss = 0.490107 (* 1 = 0.490107 loss)
    I1227 20:31:39.316201  5629 sgd_solver.cpp:106] Iteration 93700, lr = 0.000121133
    I1227 20:31:46.306689  5629 solver.cpp:237] Iteration 93800, loss = 0.607086
    I1227 20:31:46.306726  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:31:46.306740  5629 solver.cpp:253]     Train net output #1: loss = 0.607086 (* 1 = 0.607086 loss)
    I1227 20:31:46.306748  5629 sgd_solver.cpp:106] Iteration 93800, lr = 0.000121046
    I1227 20:31:53.272600  5629 solver.cpp:237] Iteration 93900, loss = 0.617672
    I1227 20:31:53.272802  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:31:53.272830  5629 solver.cpp:253]     Train net output #1: loss = 0.617672 (* 1 = 0.617672 loss)
    I1227 20:31:53.272845  5629 sgd_solver.cpp:106] Iteration 93900, lr = 0.000120958
    I1227 20:32:00.151852  5629 solver.cpp:341] Iteration 94000, Testing net (#0)
    I1227 20:32:03.041640  5629 solver.cpp:409]     Test net output #0: accuracy = 0.750833
    I1227 20:32:03.041692  5629 solver.cpp:409]     Test net output #1: loss = 0.71487 (* 1 = 0.71487 loss)
    I1227 20:32:03.071110  5629 solver.cpp:237] Iteration 94000, loss = 0.525202
    I1227 20:32:03.071149  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:32:03.071161  5629 solver.cpp:253]     Train net output #1: loss = 0.525202 (* 1 = 0.525202 loss)
    I1227 20:32:03.071171  5629 sgd_solver.cpp:106] Iteration 94000, lr = 0.000120871
    I1227 20:32:10.865382  5629 solver.cpp:237] Iteration 94100, loss = 0.614014
    I1227 20:32:10.865423  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:32:10.865438  5629 solver.cpp:253]     Train net output #1: loss = 0.614014 (* 1 = 0.614014 loss)
    I1227 20:32:10.865448  5629 sgd_solver.cpp:106] Iteration 94100, lr = 0.000120784
    I1227 20:32:17.942646  5629 solver.cpp:237] Iteration 94200, loss = 0.639502
    I1227 20:32:17.942682  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:32:17.942694  5629 solver.cpp:253]     Train net output #1: loss = 0.639502 (* 1 = 0.639502 loss)
    I1227 20:32:17.942703  5629 sgd_solver.cpp:106] Iteration 94200, lr = 0.000120697
    I1227 20:32:25.947934  5629 solver.cpp:237] Iteration 94300, loss = 0.625964
    I1227 20:32:25.948050  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:32:25.948065  5629 solver.cpp:253]     Train net output #1: loss = 0.625964 (* 1 = 0.625964 loss)
    I1227 20:32:25.948074  5629 sgd_solver.cpp:106] Iteration 94300, lr = 0.00012061
    I1227 20:32:35.951175  5629 solver.cpp:237] Iteration 94400, loss = 0.622286
    I1227 20:32:35.951246  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:32:35.951272  5629 solver.cpp:253]     Train net output #1: loss = 0.622286 (* 1 = 0.622286 loss)
    I1227 20:32:35.951289  5629 sgd_solver.cpp:106] Iteration 94400, lr = 0.000120524
    I1227 20:32:45.929355  5629 solver.cpp:237] Iteration 94500, loss = 0.531221
    I1227 20:32:45.929395  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:32:45.929409  5629 solver.cpp:253]     Train net output #1: loss = 0.531221 (* 1 = 0.531221 loss)
    I1227 20:32:45.929417  5629 sgd_solver.cpp:106] Iteration 94500, lr = 0.000120437
    I1227 20:32:54.080317  5629 solver.cpp:237] Iteration 94600, loss = 0.562059
    I1227 20:32:54.080368  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:32:54.080380  5629 solver.cpp:253]     Train net output #1: loss = 0.562058 (* 1 = 0.562058 loss)
    I1227 20:32:54.080390  5629 sgd_solver.cpp:106] Iteration 94600, lr = 0.000120351
    I1227 20:33:02.042143  5629 solver.cpp:237] Iteration 94700, loss = 0.500341
    I1227 20:33:02.042315  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:33:02.042348  5629 solver.cpp:253]     Train net output #1: loss = 0.500341 (* 1 = 0.500341 loss)
    I1227 20:33:02.042361  5629 sgd_solver.cpp:106] Iteration 94700, lr = 0.000120265
    I1227 20:33:10.041076  5629 solver.cpp:237] Iteration 94800, loss = 0.615217
    I1227 20:33:10.041115  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:33:10.041126  5629 solver.cpp:253]     Train net output #1: loss = 0.615217 (* 1 = 0.615217 loss)
    I1227 20:33:10.041136  5629 sgd_solver.cpp:106] Iteration 94800, lr = 0.000120179
    I1227 20:33:18.272347  5629 solver.cpp:237] Iteration 94900, loss = 0.630154
    I1227 20:33:18.272405  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:33:18.272428  5629 solver.cpp:253]     Train net output #1: loss = 0.630154 (* 1 = 0.630154 loss)
    I1227 20:33:18.272444  5629 sgd_solver.cpp:106] Iteration 94900, lr = 0.000120093
    I1227 20:33:26.229249  5629 solver.cpp:341] Iteration 95000, Testing net (#0)
    I1227 20:33:29.394541  5629 solver.cpp:409]     Test net output #0: accuracy = 0.754833
    I1227 20:33:29.394582  5629 solver.cpp:409]     Test net output #1: loss = 0.70961 (* 1 = 0.70961 loss)
    I1227 20:33:29.423553  5629 solver.cpp:237] Iteration 95000, loss = 0.515686
    I1227 20:33:29.423604  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:33:29.423616  5629 solver.cpp:253]     Train net output #1: loss = 0.515686 (* 1 = 0.515686 loss)
    I1227 20:33:29.423627  5629 sgd_solver.cpp:106] Iteration 95000, lr = 0.000120007
    I1227 20:33:37.026134  5629 solver.cpp:237] Iteration 95100, loss = 0.713733
    I1227 20:33:37.026571  5629 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1227 20:33:37.026602  5629 solver.cpp:253]     Train net output #1: loss = 0.713733 (* 1 = 0.713733 loss)
    I1227 20:33:37.026618  5629 sgd_solver.cpp:106] Iteration 95100, lr = 0.000119921
    I1227 20:33:44.377439  5629 solver.cpp:237] Iteration 95200, loss = 0.539436
    I1227 20:33:44.377477  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:33:44.377490  5629 solver.cpp:253]     Train net output #1: loss = 0.539436 (* 1 = 0.539436 loss)
    I1227 20:33:44.377497  5629 sgd_solver.cpp:106] Iteration 95200, lr = 0.000119836
    I1227 20:33:52.132889  5629 solver.cpp:237] Iteration 95300, loss = 0.615343
    I1227 20:33:52.132943  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:33:52.132966  5629 solver.cpp:253]     Train net output #1: loss = 0.615343 (* 1 = 0.615343 loss)
    I1227 20:33:52.132982  5629 sgd_solver.cpp:106] Iteration 95300, lr = 0.00011975
    I1227 20:33:59.498807  5629 solver.cpp:237] Iteration 95400, loss = 0.620004
    I1227 20:33:59.498862  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:33:59.498877  5629 solver.cpp:253]     Train net output #1: loss = 0.620004 (* 1 = 0.620004 loss)
    I1227 20:33:59.498888  5629 sgd_solver.cpp:106] Iteration 95400, lr = 0.000119665
    I1227 20:34:07.806351  5629 solver.cpp:237] Iteration 95500, loss = 0.626013
    I1227 20:34:07.806524  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:34:07.806555  5629 solver.cpp:253]     Train net output #1: loss = 0.626013 (* 1 = 0.626013 loss)
    I1227 20:34:07.806572  5629 sgd_solver.cpp:106] Iteration 95500, lr = 0.00011958
    I1227 20:34:15.482481  5629 solver.cpp:237] Iteration 95600, loss = 0.661914
    I1227 20:34:15.482519  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:34:15.482532  5629 solver.cpp:253]     Train net output #1: loss = 0.661914 (* 1 = 0.661914 loss)
    I1227 20:34:15.482540  5629 sgd_solver.cpp:106] Iteration 95600, lr = 0.000119495
    I1227 20:34:22.751696  5629 solver.cpp:237] Iteration 95700, loss = 0.564544
    I1227 20:34:22.751755  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:34:22.751777  5629 solver.cpp:253]     Train net output #1: loss = 0.564544 (* 1 = 0.564544 loss)
    I1227 20:34:22.751795  5629 sgd_solver.cpp:106] Iteration 95700, lr = 0.00011941
    I1227 20:34:31.133702  5629 solver.cpp:237] Iteration 95800, loss = 0.62666
    I1227 20:34:31.133750  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:34:31.133764  5629 solver.cpp:253]     Train net output #1: loss = 0.62666 (* 1 = 0.62666 loss)
    I1227 20:34:31.133774  5629 sgd_solver.cpp:106] Iteration 95800, lr = 0.000119326
    I1227 20:34:39.393579  5629 solver.cpp:237] Iteration 95900, loss = 0.671204
    I1227 20:34:39.393749  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:34:39.393776  5629 solver.cpp:253]     Train net output #1: loss = 0.671203 (* 1 = 0.671203 loss)
    I1227 20:34:39.393793  5629 sgd_solver.cpp:106] Iteration 95900, lr = 0.000119241
    I1227 20:34:48.228272  5629 solver.cpp:341] Iteration 96000, Testing net (#0)
    I1227 20:34:52.032902  5629 solver.cpp:409]     Test net output #0: accuracy = 0.751083
    I1227 20:34:52.032976  5629 solver.cpp:409]     Test net output #1: loss = 0.718484 (* 1 = 0.718484 loss)
    I1227 20:34:52.070565  5629 solver.cpp:237] Iteration 96000, loss = 0.537601
    I1227 20:34:52.070636  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:34:52.070660  5629 solver.cpp:253]     Train net output #1: loss = 0.537601 (* 1 = 0.537601 loss)
    I1227 20:34:52.070679  5629 sgd_solver.cpp:106] Iteration 96000, lr = 0.000119157
    I1227 20:34:59.810477  5629 solver.cpp:237] Iteration 96100, loss = 0.646005
    I1227 20:34:59.810518  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:34:59.810534  5629 solver.cpp:253]     Train net output #1: loss = 0.646005 (* 1 = 0.646005 loss)
    I1227 20:34:59.810544  5629 sgd_solver.cpp:106] Iteration 96100, lr = 0.000119073
    I1227 20:35:07.287935  5629 solver.cpp:237] Iteration 96200, loss = 0.531893
    I1227 20:35:07.287981  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:35:07.287994  5629 solver.cpp:253]     Train net output #1: loss = 0.531893 (* 1 = 0.531893 loss)
    I1227 20:35:07.288007  5629 sgd_solver.cpp:106] Iteration 96200, lr = 0.000118988
    I1227 20:35:14.628846  5629 solver.cpp:237] Iteration 96300, loss = 0.505778
    I1227 20:35:14.628952  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:35:14.628970  5629 solver.cpp:253]     Train net output #1: loss = 0.505778 (* 1 = 0.505778 loss)
    I1227 20:35:14.628980  5629 sgd_solver.cpp:106] Iteration 96300, lr = 0.000118904
    I1227 20:35:22.920475  5629 solver.cpp:237] Iteration 96400, loss = 0.625665
    I1227 20:35:22.920511  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:35:22.920522  5629 solver.cpp:253]     Train net output #1: loss = 0.625665 (* 1 = 0.625665 loss)
    I1227 20:35:22.920531  5629 sgd_solver.cpp:106] Iteration 96400, lr = 0.000118821
    I1227 20:35:31.354328  5629 solver.cpp:237] Iteration 96500, loss = 0.57691
    I1227 20:35:31.354377  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:35:31.354394  5629 solver.cpp:253]     Train net output #1: loss = 0.57691 (* 1 = 0.57691 loss)
    I1227 20:35:31.354406  5629 sgd_solver.cpp:106] Iteration 96500, lr = 0.000118737
    I1227 20:35:39.486886  5629 solver.cpp:237] Iteration 96600, loss = 0.613615
    I1227 20:35:39.486958  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:35:39.486984  5629 solver.cpp:253]     Train net output #1: loss = 0.613614 (* 1 = 0.613614 loss)
    I1227 20:35:39.487004  5629 sgd_solver.cpp:106] Iteration 96600, lr = 0.000118653
    I1227 20:35:47.492236  5629 solver.cpp:237] Iteration 96700, loss = 0.514375
    I1227 20:35:47.492384  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:35:47.492403  5629 solver.cpp:253]     Train net output #1: loss = 0.514375 (* 1 = 0.514375 loss)
    I1227 20:35:47.492411  5629 sgd_solver.cpp:106] Iteration 96700, lr = 0.00011857
    I1227 20:35:56.198076  5629 solver.cpp:237] Iteration 96800, loss = 0.562845
    I1227 20:35:56.198117  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:35:56.198129  5629 solver.cpp:253]     Train net output #1: loss = 0.562845 (* 1 = 0.562845 loss)
    I1227 20:35:56.198138  5629 sgd_solver.cpp:106] Iteration 96800, lr = 0.000118487
    I1227 20:36:05.713064  5629 solver.cpp:237] Iteration 96900, loss = 0.693792
    I1227 20:36:05.713105  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:36:05.713119  5629 solver.cpp:253]     Train net output #1: loss = 0.693792 (* 1 = 0.693792 loss)
    I1227 20:36:05.713129  5629 sgd_solver.cpp:106] Iteration 96900, lr = 0.000118404
    I1227 20:36:13.490953  5629 solver.cpp:341] Iteration 97000, Testing net (#0)
    I1227 20:36:16.822860  5629 solver.cpp:409]     Test net output #0: accuracy = 0.744083
    I1227 20:36:16.822934  5629 solver.cpp:409]     Test net output #1: loss = 0.725228 (* 1 = 0.725228 loss)
    I1227 20:36:16.878912  5629 solver.cpp:237] Iteration 97000, loss = 0.595033
    I1227 20:36:16.878968  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:36:16.878990  5629 solver.cpp:253]     Train net output #1: loss = 0.595033 (* 1 = 0.595033 loss)
    I1227 20:36:16.879009  5629 sgd_solver.cpp:106] Iteration 97000, lr = 0.000118321
    I1227 20:36:25.401038  5629 solver.cpp:237] Iteration 97100, loss = 0.656176
    I1227 20:36:25.401237  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:36:25.401255  5629 solver.cpp:253]     Train net output #1: loss = 0.656176 (* 1 = 0.656176 loss)
    I1227 20:36:25.401265  5629 sgd_solver.cpp:106] Iteration 97100, lr = 0.000118238
    I1227 20:36:33.926741  5629 solver.cpp:237] Iteration 97200, loss = 0.585665
    I1227 20:36:33.926790  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:36:33.926806  5629 solver.cpp:253]     Train net output #1: loss = 0.585665 (* 1 = 0.585665 loss)
    I1227 20:36:33.926817  5629 sgd_solver.cpp:106] Iteration 97200, lr = 0.000118155
    I1227 20:36:42.385668  5629 solver.cpp:237] Iteration 97300, loss = 0.565643
    I1227 20:36:42.385711  5629 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1227 20:36:42.385725  5629 solver.cpp:253]     Train net output #1: loss = 0.565643 (* 1 = 0.565643 loss)
    I1227 20:36:42.385736  5629 sgd_solver.cpp:106] Iteration 97300, lr = 0.000118072
    I1227 20:36:50.606766  5629 solver.cpp:237] Iteration 97400, loss = 0.585259
    I1227 20:36:50.606811  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:36:50.606824  5629 solver.cpp:253]     Train net output #1: loss = 0.585259 (* 1 = 0.585259 loss)
    I1227 20:36:50.606837  5629 sgd_solver.cpp:106] Iteration 97400, lr = 0.00011799
    I1227 20:36:58.020547  5629 solver.cpp:237] Iteration 97500, loss = 0.469645
    I1227 20:36:58.020709  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:36:58.020726  5629 solver.cpp:253]     Train net output #1: loss = 0.469644 (* 1 = 0.469644 loss)
    I1227 20:36:58.020736  5629 sgd_solver.cpp:106] Iteration 97500, lr = 0.000117908
    I1227 20:37:05.923704  5629 solver.cpp:237] Iteration 97600, loss = 0.611683
    I1227 20:37:05.923748  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:37:05.923761  5629 solver.cpp:253]     Train net output #1: loss = 0.611683 (* 1 = 0.611683 loss)
    I1227 20:37:05.923774  5629 sgd_solver.cpp:106] Iteration 97600, lr = 0.000117825
    I1227 20:37:14.509815  5629 solver.cpp:237] Iteration 97700, loss = 0.532167
    I1227 20:37:14.509856  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:37:14.509870  5629 solver.cpp:253]     Train net output #1: loss = 0.532167 (* 1 = 0.532167 loss)
    I1227 20:37:14.509879  5629 sgd_solver.cpp:106] Iteration 97700, lr = 0.000117743
    I1227 20:37:22.757835  5629 solver.cpp:237] Iteration 97800, loss = 0.550665
    I1227 20:37:22.757877  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:37:22.757891  5629 solver.cpp:253]     Train net output #1: loss = 0.550665 (* 1 = 0.550665 loss)
    I1227 20:37:22.757901  5629 sgd_solver.cpp:106] Iteration 97800, lr = 0.000117661
    I1227 20:37:30.290376  5629 solver.cpp:237] Iteration 97900, loss = 0.641695
    I1227 20:37:30.290542  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:37:30.290556  5629 solver.cpp:253]     Train net output #1: loss = 0.641695 (* 1 = 0.641695 loss)
    I1227 20:37:30.290563  5629 sgd_solver.cpp:106] Iteration 97900, lr = 0.00011758
    I1227 20:37:37.668051  5629 solver.cpp:341] Iteration 98000, Testing net (#0)
    I1227 20:37:40.551532  5629 solver.cpp:409]     Test net output #0: accuracy = 0.74825
    I1227 20:37:40.551575  5629 solver.cpp:409]     Test net output #1: loss = 0.720511 (* 1 = 0.720511 loss)
    I1227 20:37:40.581101  5629 solver.cpp:237] Iteration 98000, loss = 0.506956
    I1227 20:37:40.581145  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:37:40.581157  5629 solver.cpp:253]     Train net output #1: loss = 0.506956 (* 1 = 0.506956 loss)
    I1227 20:37:40.581168  5629 sgd_solver.cpp:106] Iteration 98000, lr = 0.000117498
    I1227 20:37:48.115928  5629 solver.cpp:237] Iteration 98100, loss = 0.662487
    I1227 20:37:48.115975  5629 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 20:37:48.115988  5629 solver.cpp:253]     Train net output #1: loss = 0.662487 (* 1 = 0.662487 loss)
    I1227 20:37:48.115998  5629 sgd_solver.cpp:106] Iteration 98100, lr = 0.000117416
    I1227 20:37:55.538828  5629 solver.cpp:237] Iteration 98200, loss = 0.570958
    I1227 20:37:55.538887  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:37:55.538908  5629 solver.cpp:253]     Train net output #1: loss = 0.570958 (* 1 = 0.570958 loss)
    I1227 20:37:55.538923  5629 sgd_solver.cpp:106] Iteration 98200, lr = 0.000117335
    I1227 20:38:02.789655  5629 solver.cpp:237] Iteration 98300, loss = 0.52752
    I1227 20:38:02.789810  5629 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 20:38:02.789825  5629 solver.cpp:253]     Train net output #1: loss = 0.527519 (* 1 = 0.527519 loss)
    I1227 20:38:02.789834  5629 sgd_solver.cpp:106] Iteration 98300, lr = 0.000117254
    I1227 20:38:09.751081  5629 solver.cpp:237] Iteration 98400, loss = 0.656538
    I1227 20:38:09.751130  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:38:09.751143  5629 solver.cpp:253]     Train net output #1: loss = 0.656538 (* 1 = 0.656538 loss)
    I1227 20:38:09.751153  5629 sgd_solver.cpp:106] Iteration 98400, lr = 0.000117173
    I1227 20:38:16.673975  5629 solver.cpp:237] Iteration 98500, loss = 0.567837
    I1227 20:38:16.674036  5629 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1227 20:38:16.674059  5629 solver.cpp:253]     Train net output #1: loss = 0.567836 (* 1 = 0.567836 loss)
    I1227 20:38:16.674077  5629 sgd_solver.cpp:106] Iteration 98500, lr = 0.000117092
    I1227 20:38:23.635799  5629 solver.cpp:237] Iteration 98600, loss = 0.746434
    I1227 20:38:23.635838  5629 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1227 20:38:23.635849  5629 solver.cpp:253]     Train net output #1: loss = 0.746434 (* 1 = 0.746434 loss)
    I1227 20:38:23.635857  5629 sgd_solver.cpp:106] Iteration 98600, lr = 0.000117011
    I1227 20:38:30.737890  5629 solver.cpp:237] Iteration 98700, loss = 0.472389
    I1227 20:38:30.737948  5629 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 20:38:30.737970  5629 solver.cpp:253]     Train net output #1: loss = 0.472389 (* 1 = 0.472389 loss)
    I1227 20:38:30.737988  5629 sgd_solver.cpp:106] Iteration 98700, lr = 0.00011693
    I1227 20:38:37.919879  5629 solver.cpp:237] Iteration 98800, loss = 0.658665
    I1227 20:38:37.920001  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:38:37.920017  5629 solver.cpp:253]     Train net output #1: loss = 0.658664 (* 1 = 0.658664 loss)
    I1227 20:38:37.920024  5629 sgd_solver.cpp:106] Iteration 98800, lr = 0.000116849
    I1227 20:38:44.866524  5629 solver.cpp:237] Iteration 98900, loss = 0.685292
    I1227 20:38:44.866562  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:38:44.866574  5629 solver.cpp:253]     Train net output #1: loss = 0.685291 (* 1 = 0.685291 loss)
    I1227 20:38:44.866582  5629 sgd_solver.cpp:106] Iteration 98900, lr = 0.000116769
    I1227 20:38:51.898617  5629 solver.cpp:341] Iteration 99000, Testing net (#0)
    I1227 20:38:54.680352  5629 solver.cpp:409]     Test net output #0: accuracy = 0.750583
    I1227 20:38:54.680402  5629 solver.cpp:409]     Test net output #1: loss = 0.71574 (* 1 = 0.71574 loss)
    I1227 20:38:54.720407  5629 solver.cpp:237] Iteration 99000, loss = 0.638861
    I1227 20:38:54.720454  5629 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 20:38:54.720470  5629 solver.cpp:253]     Train net output #1: loss = 0.638861 (* 1 = 0.638861 loss)
    I1227 20:38:54.720482  5629 sgd_solver.cpp:106] Iteration 99000, lr = 0.000116689
    I1227 20:39:01.690875  5629 solver.cpp:237] Iteration 99100, loss = 0.659766
    I1227 20:39:01.690919  5629 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1227 20:39:01.690933  5629 solver.cpp:253]     Train net output #1: loss = 0.659766 (* 1 = 0.659766 loss)
    I1227 20:39:01.690945  5629 sgd_solver.cpp:106] Iteration 99100, lr = 0.000116608
    I1227 20:39:08.628335  5629 solver.cpp:237] Iteration 99200, loss = 0.511542
    I1227 20:39:08.628504  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:39:08.628530  5629 solver.cpp:253]     Train net output #1: loss = 0.511542 (* 1 = 0.511542 loss)
    I1227 20:39:08.628540  5629 sgd_solver.cpp:106] Iteration 99200, lr = 0.000116528
    I1227 20:39:15.593919  5629 solver.cpp:237] Iteration 99300, loss = 0.582279
    I1227 20:39:15.593958  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:39:15.593971  5629 solver.cpp:253]     Train net output #1: loss = 0.582279 (* 1 = 0.582279 loss)
    I1227 20:39:15.593979  5629 sgd_solver.cpp:106] Iteration 99300, lr = 0.000116448
    I1227 20:39:22.613412  5629 solver.cpp:237] Iteration 99400, loss = 0.639401
    I1227 20:39:22.613457  5629 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1227 20:39:22.613469  5629 solver.cpp:253]     Train net output #1: loss = 0.639401 (* 1 = 0.639401 loss)
    I1227 20:39:22.613477  5629 sgd_solver.cpp:106] Iteration 99400, lr = 0.000116368
    I1227 20:39:29.715904  5629 solver.cpp:237] Iteration 99500, loss = 0.54444
    I1227 20:39:29.715967  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:39:29.715982  5629 solver.cpp:253]     Train net output #1: loss = 0.544439 (* 1 = 0.544439 loss)
    I1227 20:39:29.716004  5629 sgd_solver.cpp:106] Iteration 99500, lr = 0.000116289
    I1227 20:39:38.596839  5629 solver.cpp:237] Iteration 99600, loss = 0.703217
    I1227 20:39:38.596878  5629 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1227 20:39:38.596889  5629 solver.cpp:253]     Train net output #1: loss = 0.703217 (* 1 = 0.703217 loss)
    I1227 20:39:38.596900  5629 sgd_solver.cpp:106] Iteration 99600, lr = 0.000116209
    I1227 20:39:46.768013  5629 solver.cpp:237] Iteration 99700, loss = 0.539326
    I1227 20:39:46.768167  5629 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1227 20:39:46.768194  5629 solver.cpp:253]     Train net output #1: loss = 0.539326 (* 1 = 0.539326 loss)
    I1227 20:39:46.768210  5629 sgd_solver.cpp:106] Iteration 99700, lr = 0.00011613
    I1227 20:39:56.347695  5629 solver.cpp:237] Iteration 99800, loss = 0.574326
    I1227 20:39:56.347733  5629 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1227 20:39:56.347744  5629 solver.cpp:253]     Train net output #1: loss = 0.574326 (* 1 = 0.574326 loss)
    I1227 20:39:56.347753  5629 sgd_solver.cpp:106] Iteration 99800, lr = 0.00011605
    I1227 20:40:04.049954  5629 solver.cpp:237] Iteration 99900, loss = 0.643648
    I1227 20:40:04.050011  5629 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1227 20:40:04.050034  5629 solver.cpp:253]     Train net output #1: loss = 0.643648 (* 1 = 0.643648 loss)
    I1227 20:40:04.050050  5629 sgd_solver.cpp:106] Iteration 99900, lr = 0.000115971
    I1227 20:40:11.372995  5629 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_100000.caffemodel
    I1227 20:40:11.412442  5629 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_100000.solverstate
    I1227 20:40:11.437083  5629 solver.cpp:321] Iteration 100000, loss = 0.550242
    I1227 20:40:11.437124  5629 solver.cpp:341] Iteration 100000, Testing net (#0)
    I1227 20:40:14.335026  5629 solver.cpp:409]     Test net output #0: accuracy = 0.741333
    I1227 20:40:14.335078  5629 solver.cpp:409]     Test net output #1: loss = 0.741076 (* 1 = 0.741076 loss)
    I1227 20:40:14.335088  5629 solver.cpp:326] Optimization Done.
    I1227 20:40:14.335095  5629 caffe.cpp:215] Optimization Done.
    CPU times: user 30.4 s, sys: 4.49 s, total: 34.9 s
    Wall time: 2h 13min 46s


Caffe brewed.
## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_100000.caffemodel -iterations 83
```

    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1227 20:40:14.583098  9232 caffe.cpp:234] Use CPU.
    I1227 20:40:14.755923  9232 net.cpp:49] Initializing net from parameters:
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
        num_output: 32
        kernel_size: 4
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
        pool: AVE
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
        pool: MAX
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
    I1227 20:40:14.756417  9232 layer_factory.hpp:77] Creating layer data
    I1227 20:40:14.756434  9232 net.cpp:106] Creating Layer data
    I1227 20:40:14.756443  9232 net.cpp:411] data -> data
    I1227 20:40:14.756461  9232 net.cpp:411] data -> label
    I1227 20:40:14.756474  9232 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1227 20:40:14.756517  9232 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1227 20:40:14.757530  9232 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1227 20:40:15.083159  9232 net.cpp:150] Setting up data
    I1227 20:40:15.083199  9232 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1227 20:40:15.083209  9232 net.cpp:157] Top shape: 120 (120)
    I1227 20:40:15.083214  9232 net.cpp:165] Memory required for data: 1475040
    I1227 20:40:15.083223  9232 layer_factory.hpp:77] Creating layer label_data_1_split
    I1227 20:40:15.083246  9232 net.cpp:106] Creating Layer label_data_1_split
    I1227 20:40:15.083256  9232 net.cpp:454] label_data_1_split <- label
    I1227 20:40:15.083267  9232 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1227 20:40:15.083279  9232 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1227 20:40:15.083294  9232 net.cpp:150] Setting up label_data_1_split
    I1227 20:40:15.083302  9232 net.cpp:157] Top shape: 120 (120)
    I1227 20:40:15.083307  9232 net.cpp:157] Top shape: 120 (120)
    I1227 20:40:15.083312  9232 net.cpp:165] Memory required for data: 1476000
    I1227 20:40:15.083318  9232 layer_factory.hpp:77] Creating layer conv1
    I1227 20:40:15.083330  9232 net.cpp:106] Creating Layer conv1
    I1227 20:40:15.083336  9232 net.cpp:454] conv1 <- data
    I1227 20:40:15.083343  9232 net.cpp:411] conv1 -> conv1
    I1227 20:40:15.083734  9232 net.cpp:150] Setting up conv1
    I1227 20:40:15.083746  9232 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1227 20:40:15.083752  9232 net.cpp:165] Memory required for data: 14393760
    I1227 20:40:15.083801  9232 layer_factory.hpp:77] Creating layer pool1
    I1227 20:40:15.083812  9232 net.cpp:106] Creating Layer pool1
    I1227 20:40:15.083818  9232 net.cpp:454] pool1 <- conv1
    I1227 20:40:15.083827  9232 net.cpp:411] pool1 -> pool1
    I1227 20:40:15.083858  9232 net.cpp:150] Setting up pool1
    I1227 20:40:15.083865  9232 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 20:40:15.083871  9232 net.cpp:165] Memory required for data: 17404320
    I1227 20:40:15.083878  9232 layer_factory.hpp:77] Creating layer drop1
    I1227 20:40:15.083889  9232 net.cpp:106] Creating Layer drop1
    I1227 20:40:15.083894  9232 net.cpp:454] drop1 <- pool1
    I1227 20:40:15.083900  9232 net.cpp:397] drop1 -> pool1 (in-place)
    I1227 20:40:15.083914  9232 net.cpp:150] Setting up drop1
    I1227 20:40:15.083920  9232 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 20:40:15.083925  9232 net.cpp:165] Memory required for data: 20414880
    I1227 20:40:15.083931  9232 layer_factory.hpp:77] Creating layer relu1
    I1227 20:40:15.083938  9232 net.cpp:106] Creating Layer relu1
    I1227 20:40:15.083943  9232 net.cpp:454] relu1 <- pool1
    I1227 20:40:15.083950  9232 net.cpp:397] relu1 -> pool1 (in-place)
    I1227 20:40:15.083957  9232 net.cpp:150] Setting up relu1
    I1227 20:40:15.083963  9232 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1227 20:40:15.083968  9232 net.cpp:165] Memory required for data: 23425440
    I1227 20:40:15.083974  9232 layer_factory.hpp:77] Creating layer conv2
    I1227 20:40:15.083982  9232 net.cpp:106] Creating Layer conv2
    I1227 20:40:15.083987  9232 net.cpp:454] conv2 <- pool1
    I1227 20:40:15.083993  9232 net.cpp:411] conv2 -> conv2
    I1227 20:40:15.084148  9232 net.cpp:150] Setting up conv2
    I1227 20:40:15.084157  9232 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1227 20:40:15.084162  9232 net.cpp:165] Memory required for data: 25864800
    I1227 20:40:15.084172  9232 layer_factory.hpp:77] Creating layer pool2
    I1227 20:40:15.084179  9232 net.cpp:106] Creating Layer pool2
    I1227 20:40:15.084184  9232 net.cpp:454] pool2 <- conv2
    I1227 20:40:15.084192  9232 net.cpp:411] pool2 -> pool2
    I1227 20:40:15.084200  9232 net.cpp:150] Setting up pool2
    I1227 20:40:15.084206  9232 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 20:40:15.084211  9232 net.cpp:165] Memory required for data: 26368800
    I1227 20:40:15.084218  9232 layer_factory.hpp:77] Creating layer drop2
    I1227 20:40:15.084225  9232 net.cpp:106] Creating Layer drop2
    I1227 20:40:15.084230  9232 net.cpp:454] drop2 <- pool2
    I1227 20:40:15.084236  9232 net.cpp:397] drop2 -> pool2 (in-place)
    I1227 20:40:15.084244  9232 net.cpp:150] Setting up drop2
    I1227 20:40:15.084250  9232 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 20:40:15.084255  9232 net.cpp:165] Memory required for data: 26872800
    I1227 20:40:15.084261  9232 layer_factory.hpp:77] Creating layer relu2
    I1227 20:40:15.084269  9232 net.cpp:106] Creating Layer relu2
    I1227 20:40:15.084273  9232 net.cpp:454] relu2 <- pool2
    I1227 20:40:15.084280  9232 net.cpp:397] relu2 -> pool2 (in-place)
    I1227 20:40:15.084286  9232 net.cpp:150] Setting up relu2
    I1227 20:40:15.084292  9232 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1227 20:40:15.084298  9232 net.cpp:165] Memory required for data: 27376800
    I1227 20:40:15.084303  9232 layer_factory.hpp:77] Creating layer conv3
    I1227 20:40:15.084311  9232 net.cpp:106] Creating Layer conv3
    I1227 20:40:15.084316  9232 net.cpp:454] conv3 <- pool2
    I1227 20:40:15.084322  9232 net.cpp:411] conv3 -> conv3
    I1227 20:40:15.084409  9232 net.cpp:150] Setting up conv3
    I1227 20:40:15.084419  9232 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1227 20:40:15.084424  9232 net.cpp:165] Memory required for data: 27868320
    I1227 20:40:15.084432  9232 layer_factory.hpp:77] Creating layer pool3
    I1227 20:40:15.084440  9232 net.cpp:106] Creating Layer pool3
    I1227 20:40:15.084445  9232 net.cpp:454] pool3 <- conv3
    I1227 20:40:15.084452  9232 net.cpp:411] pool3 -> pool3
    I1227 20:40:15.084460  9232 net.cpp:150] Setting up pool3
    I1227 20:40:15.084467  9232 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 20:40:15.084473  9232 net.cpp:165] Memory required for data: 27991200
    I1227 20:40:15.084488  9232 layer_factory.hpp:77] Creating layer relu3
    I1227 20:40:15.084496  9232 net.cpp:106] Creating Layer relu3
    I1227 20:40:15.084501  9232 net.cpp:454] relu3 <- pool3
    I1227 20:40:15.084507  9232 net.cpp:397] relu3 -> pool3 (in-place)
    I1227 20:40:15.084514  9232 net.cpp:150] Setting up relu3
    I1227 20:40:15.084522  9232 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 20:40:15.084527  9232 net.cpp:165] Memory required for data: 28114080
    I1227 20:40:15.084532  9232 layer_factory.hpp:77] Creating layer ip1
    I1227 20:40:15.084542  9232 net.cpp:106] Creating Layer ip1
    I1227 20:40:15.084548  9232 net.cpp:454] ip1 <- pool3
    I1227 20:40:15.084554  9232 net.cpp:411] ip1 -> ip1
    I1227 20:40:15.085425  9232 net.cpp:150] Setting up ip1
    I1227 20:40:15.085436  9232 net.cpp:157] Top shape: 120 512 (61440)
    I1227 20:40:15.085443  9232 net.cpp:165] Memory required for data: 28359840
    I1227 20:40:15.085450  9232 layer_factory.hpp:77] Creating layer sig1
    I1227 20:40:15.085458  9232 net.cpp:106] Creating Layer sig1
    I1227 20:40:15.085464  9232 net.cpp:454] sig1 <- ip1
    I1227 20:40:15.085470  9232 net.cpp:397] sig1 -> ip1 (in-place)
    I1227 20:40:15.085479  9232 net.cpp:150] Setting up sig1
    I1227 20:40:15.085484  9232 net.cpp:157] Top shape: 120 512 (61440)
    I1227 20:40:15.085489  9232 net.cpp:165] Memory required for data: 28605600
    I1227 20:40:15.085494  9232 layer_factory.hpp:77] Creating layer ip2
    I1227 20:40:15.085502  9232 net.cpp:106] Creating Layer ip2
    I1227 20:40:15.085507  9232 net.cpp:454] ip2 <- ip1
    I1227 20:40:15.085513  9232 net.cpp:411] ip2 -> ip2
    I1227 20:40:15.085557  9232 net.cpp:150] Setting up ip2
    I1227 20:40:15.085566  9232 net.cpp:157] Top shape: 120 10 (1200)
    I1227 20:40:15.085571  9232 net.cpp:165] Memory required for data: 28610400
    I1227 20:40:15.085579  9232 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1227 20:40:15.085588  9232 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1227 20:40:15.085593  9232 net.cpp:454] ip2_ip2_0_split <- ip2
    I1227 20:40:15.085600  9232 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1227 20:40:15.085608  9232 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1227 20:40:15.085615  9232 net.cpp:150] Setting up ip2_ip2_0_split
    I1227 20:40:15.085623  9232 net.cpp:157] Top shape: 120 10 (1200)
    I1227 20:40:15.085629  9232 net.cpp:157] Top shape: 120 10 (1200)
    I1227 20:40:15.085634  9232 net.cpp:165] Memory required for data: 28620000
    I1227 20:40:15.085639  9232 layer_factory.hpp:77] Creating layer accuracy
    I1227 20:40:15.085646  9232 net.cpp:106] Creating Layer accuracy
    I1227 20:40:15.085652  9232 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1227 20:40:15.085659  9232 net.cpp:454] accuracy <- label_data_1_split_0
    I1227 20:40:15.085665  9232 net.cpp:411] accuracy -> accuracy
    I1227 20:40:15.085674  9232 net.cpp:150] Setting up accuracy
    I1227 20:40:15.085680  9232 net.cpp:157] Top shape: (1)
    I1227 20:40:15.085685  9232 net.cpp:165] Memory required for data: 28620004
    I1227 20:40:15.085690  9232 layer_factory.hpp:77] Creating layer loss
    I1227 20:40:15.085700  9232 net.cpp:106] Creating Layer loss
    I1227 20:40:15.085706  9232 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1227 20:40:15.085712  9232 net.cpp:454] loss <- label_data_1_split_1
    I1227 20:40:15.085719  9232 net.cpp:411] loss -> loss
    I1227 20:40:15.085732  9232 layer_factory.hpp:77] Creating layer loss
    I1227 20:40:15.085750  9232 net.cpp:150] Setting up loss
    I1227 20:40:15.085757  9232 net.cpp:157] Top shape: (1)
    I1227 20:40:15.085762  9232 net.cpp:160]     with loss weight 1
    I1227 20:40:15.085783  9232 net.cpp:165] Memory required for data: 28620008
    I1227 20:40:15.085789  9232 net.cpp:226] loss needs backward computation.
    I1227 20:40:15.085795  9232 net.cpp:228] accuracy does not need backward computation.
    I1227 20:40:15.085801  9232 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1227 20:40:15.085808  9232 net.cpp:226] ip2 needs backward computation.
    I1227 20:40:15.085813  9232 net.cpp:226] sig1 needs backward computation.
    I1227 20:40:15.085819  9232 net.cpp:226] ip1 needs backward computation.
    I1227 20:40:15.085834  9232 net.cpp:226] relu3 needs backward computation.
    I1227 20:40:15.085839  9232 net.cpp:226] pool3 needs backward computation.
    I1227 20:40:15.085845  9232 net.cpp:226] conv3 needs backward computation.
    I1227 20:40:15.085851  9232 net.cpp:226] relu2 needs backward computation.
    I1227 20:40:15.085856  9232 net.cpp:226] drop2 needs backward computation.
    I1227 20:40:15.085861  9232 net.cpp:226] pool2 needs backward computation.
    I1227 20:40:15.085867  9232 net.cpp:226] conv2 needs backward computation.
    I1227 20:40:15.085872  9232 net.cpp:226] relu1 needs backward computation.
    I1227 20:40:15.085877  9232 net.cpp:226] drop1 needs backward computation.
    I1227 20:40:15.085883  9232 net.cpp:226] pool1 needs backward computation.
    I1227 20:40:15.085888  9232 net.cpp:226] conv1 needs backward computation.
    I1227 20:40:15.085894  9232 net.cpp:228] label_data_1_split does not need backward computation.
    I1227 20:40:15.085901  9232 net.cpp:228] data does not need backward computation.
    I1227 20:40:15.085906  9232 net.cpp:270] This network produces output accuracy
    I1227 20:40:15.085912  9232 net.cpp:270] This network produces output loss
    I1227 20:40:15.085927  9232 net.cpp:283] Network initialization done.
    I1227 20:40:15.086938  9232 caffe.cpp:240] Running for 83 iterations.
    I1227 20:40:16.063180  9232 caffe.cpp:264] Batch 0, accuracy = 0.708333
    I1227 20:40:16.063221  9232 caffe.cpp:264] Batch 0, loss = 0.754358
    I1227 20:40:17.015645  9232 caffe.cpp:264] Batch 1, accuracy = 0.708333
    I1227 20:40:17.015692  9232 caffe.cpp:264] Batch 1, loss = 0.726977
    I1227 20:40:18.028589  9232 caffe.cpp:264] Batch 2, accuracy = 0.766667
    I1227 20:40:18.028640  9232 caffe.cpp:264] Batch 2, loss = 0.774314
    I1227 20:40:19.182984  9232 caffe.cpp:264] Batch 3, accuracy = 0.666667
    I1227 20:40:19.183027  9232 caffe.cpp:264] Batch 3, loss = 0.799929
    I1227 20:40:20.217447  9232 caffe.cpp:264] Batch 4, accuracy = 0.733333
    I1227 20:40:20.217499  9232 caffe.cpp:264] Batch 4, loss = 0.768998
    I1227 20:40:21.289613  9232 caffe.cpp:264] Batch 5, accuracy = 0.741667
    I1227 20:40:21.289682  9232 caffe.cpp:264] Batch 5, loss = 0.879529
    I1227 20:40:22.261850  9232 caffe.cpp:264] Batch 6, accuracy = 0.75
    I1227 20:40:22.261888  9232 caffe.cpp:264] Batch 6, loss = 0.805093
    I1227 20:40:23.258651  9232 caffe.cpp:264] Batch 7, accuracy = 0.733333
    I1227 20:40:23.258695  9232 caffe.cpp:264] Batch 7, loss = 0.672306
    I1227 20:40:24.229828  9232 caffe.cpp:264] Batch 8, accuracy = 0.7
    I1227 20:40:24.229866  9232 caffe.cpp:264] Batch 8, loss = 0.796009
    I1227 20:40:25.226560  9232 caffe.cpp:264] Batch 9, accuracy = 0.708333
    I1227 20:40:25.226665  9232 caffe.cpp:264] Batch 9, loss = 0.788811
    I1227 20:40:26.188984  9232 caffe.cpp:264] Batch 10, accuracy = 0.758333
    I1227 20:40:26.189030  9232 caffe.cpp:264] Batch 10, loss = 0.681093
    I1227 20:40:27.054025  9232 caffe.cpp:264] Batch 11, accuracy = 0.8
    I1227 20:40:27.054075  9232 caffe.cpp:264] Batch 11, loss = 0.661454
    I1227 20:40:27.944124  9232 caffe.cpp:264] Batch 12, accuracy = 0.741667
    I1227 20:40:27.944166  9232 caffe.cpp:264] Batch 12, loss = 0.71575
    I1227 20:40:28.818356  9232 caffe.cpp:264] Batch 13, accuracy = 0.791667
    I1227 20:40:28.818400  9232 caffe.cpp:264] Batch 13, loss = 0.619329
    I1227 20:40:29.674286  9232 caffe.cpp:264] Batch 14, accuracy = 0.758333
    I1227 20:40:29.674331  9232 caffe.cpp:264] Batch 14, loss = 0.744591
    I1227 20:40:30.539984  9232 caffe.cpp:264] Batch 15, accuracy = 0.766667
    I1227 20:40:30.540030  9232 caffe.cpp:264] Batch 15, loss = 0.675875
    I1227 20:40:31.435623  9232 caffe.cpp:264] Batch 16, accuracy = 0.733333
    I1227 20:40:31.435668  9232 caffe.cpp:264] Batch 16, loss = 0.72725
    I1227 20:40:32.352046  9232 caffe.cpp:264] Batch 17, accuracy = 0.733333
    I1227 20:40:32.352087  9232 caffe.cpp:264] Batch 17, loss = 0.710068
    I1227 20:40:33.235224  9232 caffe.cpp:264] Batch 18, accuracy = 0.733333
    I1227 20:40:33.235270  9232 caffe.cpp:264] Batch 18, loss = 0.662349
    I1227 20:40:34.146522  9232 caffe.cpp:264] Batch 19, accuracy = 0.708333
    I1227 20:40:34.146565  9232 caffe.cpp:264] Batch 19, loss = 0.747312
    I1227 20:40:35.030582  9232 caffe.cpp:264] Batch 20, accuracy = 0.683333
    I1227 20:40:35.030627  9232 caffe.cpp:264] Batch 20, loss = 0.873291
    I1227 20:40:35.915493  9232 caffe.cpp:264] Batch 21, accuracy = 0.725
    I1227 20:40:35.915539  9232 caffe.cpp:264] Batch 21, loss = 0.852735
    I1227 20:40:36.806216  9232 caffe.cpp:264] Batch 22, accuracy = 0.766667
    I1227 20:40:36.806259  9232 caffe.cpp:264] Batch 22, loss = 0.685131
    I1227 20:40:37.703965  9232 caffe.cpp:264] Batch 23, accuracy = 0.766667
    I1227 20:40:37.704038  9232 caffe.cpp:264] Batch 23, loss = 0.703953
    I1227 20:40:38.659320  9232 caffe.cpp:264] Batch 24, accuracy = 0.75
    I1227 20:40:38.659366  9232 caffe.cpp:264] Batch 24, loss = 0.701672
    I1227 20:40:39.610344  9232 caffe.cpp:264] Batch 25, accuracy = 0.758333
    I1227 20:40:39.610391  9232 caffe.cpp:264] Batch 25, loss = 0.707204
    I1227 20:40:40.507076  9232 caffe.cpp:264] Batch 26, accuracy = 0.683333
    I1227 20:40:40.507122  9232 caffe.cpp:264] Batch 26, loss = 0.805007
    I1227 20:40:41.394197  9232 caffe.cpp:264] Batch 27, accuracy = 0.791667
    I1227 20:40:41.394243  9232 caffe.cpp:264] Batch 27, loss = 0.649906
    I1227 20:40:42.301686  9232 caffe.cpp:264] Batch 28, accuracy = 0.716667
    I1227 20:40:42.301733  9232 caffe.cpp:264] Batch 28, loss = 0.818462
    I1227 20:40:43.293743  9232 caffe.cpp:264] Batch 29, accuracy = 0.75
    I1227 20:40:43.293795  9232 caffe.cpp:264] Batch 29, loss = 0.847423
    I1227 20:40:44.194972  9232 caffe.cpp:264] Batch 30, accuracy = 0.716667
    I1227 20:40:44.195019  9232 caffe.cpp:264] Batch 30, loss = 0.825866
    I1227 20:40:45.059900  9232 caffe.cpp:264] Batch 31, accuracy = 0.75
    I1227 20:40:45.060144  9232 caffe.cpp:264] Batch 31, loss = 0.732187
    I1227 20:40:45.995919  9232 caffe.cpp:264] Batch 32, accuracy = 0.758333
    I1227 20:40:45.995962  9232 caffe.cpp:264] Batch 32, loss = 0.78053
    I1227 20:40:47.015642  9232 caffe.cpp:264] Batch 33, accuracy = 0.725
    I1227 20:40:47.015697  9232 caffe.cpp:264] Batch 33, loss = 0.848931
    I1227 20:40:47.901129  9232 caffe.cpp:264] Batch 34, accuracy = 0.783333
    I1227 20:40:47.901173  9232 caffe.cpp:264] Batch 34, loss = 0.739903
    I1227 20:40:48.873806  9232 caffe.cpp:264] Batch 35, accuracy = 0.741667
    I1227 20:40:48.873852  9232 caffe.cpp:264] Batch 35, loss = 0.719066
    I1227 20:40:49.766898  9232 caffe.cpp:264] Batch 36, accuracy = 0.758333
    I1227 20:40:49.766944  9232 caffe.cpp:264] Batch 36, loss = 0.685026
    I1227 20:40:50.646106  9232 caffe.cpp:264] Batch 37, accuracy = 0.725
    I1227 20:40:50.646152  9232 caffe.cpp:264] Batch 37, loss = 0.802523
    I1227 20:40:51.499117  9232 caffe.cpp:264] Batch 38, accuracy = 0.725
    I1227 20:40:51.499162  9232 caffe.cpp:264] Batch 38, loss = 0.832365
    I1227 20:40:52.441231  9232 caffe.cpp:264] Batch 39, accuracy = 0.8
    I1227 20:40:52.441277  9232 caffe.cpp:264] Batch 39, loss = 0.626914
    I1227 20:40:53.351686  9232 caffe.cpp:264] Batch 40, accuracy = 0.725
    I1227 20:40:53.351732  9232 caffe.cpp:264] Batch 40, loss = 0.62237
    I1227 20:40:54.241893  9232 caffe.cpp:264] Batch 41, accuracy = 0.666667
    I1227 20:40:54.241937  9232 caffe.cpp:264] Batch 41, loss = 0.981314
    I1227 20:40:55.160522  9232 caffe.cpp:264] Batch 42, accuracy = 0.725
    I1227 20:40:55.160572  9232 caffe.cpp:264] Batch 42, loss = 0.765003
    I1227 20:40:56.042069  9232 caffe.cpp:264] Batch 43, accuracy = 0.8
    I1227 20:40:56.042114  9232 caffe.cpp:264] Batch 43, loss = 0.599424
    I1227 20:40:56.950472  9232 caffe.cpp:264] Batch 44, accuracy = 0.766667
    I1227 20:40:56.950520  9232 caffe.cpp:264] Batch 44, loss = 0.797427
    I1227 20:40:57.805619  9232 caffe.cpp:264] Batch 45, accuracy = 0.783333
    I1227 20:40:57.805665  9232 caffe.cpp:264] Batch 45, loss = 0.590849
    I1227 20:40:58.781040  9232 caffe.cpp:264] Batch 46, accuracy = 0.808333
    I1227 20:40:58.781080  9232 caffe.cpp:264] Batch 46, loss = 0.639637
    I1227 20:40:59.762795  9232 caffe.cpp:264] Batch 47, accuracy = 0.775
    I1227 20:40:59.762832  9232 caffe.cpp:264] Batch 47, loss = 0.640595
    I1227 20:41:00.729555  9232 caffe.cpp:264] Batch 48, accuracy = 0.75
    I1227 20:41:00.729590  9232 caffe.cpp:264] Batch 48, loss = 0.772891
    I1227 20:41:01.693336  9232 caffe.cpp:264] Batch 49, accuracy = 0.8
    I1227 20:41:01.693379  9232 caffe.cpp:264] Batch 49, loss = 0.709891
    I1227 20:41:02.596247  9232 caffe.cpp:264] Batch 50, accuracy = 0.708333
    I1227 20:41:02.596321  9232 caffe.cpp:264] Batch 50, loss = 0.803163
    I1227 20:41:03.461493  9232 caffe.cpp:264] Batch 51, accuracy = 0.666667
    I1227 20:41:03.461539  9232 caffe.cpp:264] Batch 51, loss = 0.934253
    I1227 20:41:04.331547  9232 caffe.cpp:264] Batch 52, accuracy = 0.775
    I1227 20:41:04.331593  9232 caffe.cpp:264] Batch 52, loss = 0.622744
    I1227 20:41:05.196004  9232 caffe.cpp:264] Batch 53, accuracy = 0.7
    I1227 20:41:05.196049  9232 caffe.cpp:264] Batch 53, loss = 0.798313
    I1227 20:41:06.080319  9232 caffe.cpp:264] Batch 54, accuracy = 0.808333
    I1227 20:41:06.080364  9232 caffe.cpp:264] Batch 54, loss = 0.68546
    I1227 20:41:06.969355  9232 caffe.cpp:264] Batch 55, accuracy = 0.725
    I1227 20:41:06.969401  9232 caffe.cpp:264] Batch 55, loss = 0.637003
    I1227 20:41:07.852061  9232 caffe.cpp:264] Batch 56, accuracy = 0.733333
    I1227 20:41:07.852103  9232 caffe.cpp:264] Batch 56, loss = 0.8404
    I1227 20:41:08.727458  9232 caffe.cpp:264] Batch 57, accuracy = 0.683333
    I1227 20:41:08.727515  9232 caffe.cpp:264] Batch 57, loss = 0.798013
    I1227 20:41:09.609346  9232 caffe.cpp:264] Batch 58, accuracy = 0.783333
    I1227 20:41:09.609391  9232 caffe.cpp:264] Batch 58, loss = 0.6508
    I1227 20:41:10.486240  9232 caffe.cpp:264] Batch 59, accuracy = 0.741667
    I1227 20:41:10.486284  9232 caffe.cpp:264] Batch 59, loss = 0.680463
    I1227 20:41:11.373842  9232 caffe.cpp:264] Batch 60, accuracy = 0.675
    I1227 20:41:11.373888  9232 caffe.cpp:264] Batch 60, loss = 0.858766
    I1227 20:41:12.236806  9232 caffe.cpp:264] Batch 61, accuracy = 0.758333
    I1227 20:41:12.236851  9232 caffe.cpp:264] Batch 61, loss = 0.652353
    I1227 20:41:13.121618  9232 caffe.cpp:264] Batch 62, accuracy = 0.741667
    I1227 20:41:13.121664  9232 caffe.cpp:264] Batch 62, loss = 0.696719
    I1227 20:41:13.987992  9232 caffe.cpp:264] Batch 63, accuracy = 0.85
    I1227 20:41:13.988037  9232 caffe.cpp:264] Batch 63, loss = 0.523171
    I1227 20:41:14.861654  9232 caffe.cpp:264] Batch 64, accuracy = 0.708333
    I1227 20:41:14.861698  9232 caffe.cpp:264] Batch 64, loss = 0.74659
    I1227 20:41:15.755142  9232 caffe.cpp:264] Batch 65, accuracy = 0.783333
    I1227 20:41:15.755331  9232 caffe.cpp:264] Batch 65, loss = 0.650345
    I1227 20:41:16.625237  9232 caffe.cpp:264] Batch 66, accuracy = 0.708333
    I1227 20:41:16.625283  9232 caffe.cpp:264] Batch 66, loss = 0.843636
    I1227 20:41:17.507134  9232 caffe.cpp:264] Batch 67, accuracy = 0.666667
    I1227 20:41:17.507180  9232 caffe.cpp:264] Batch 67, loss = 1.02005
    I1227 20:41:18.379364  9232 caffe.cpp:264] Batch 68, accuracy = 0.766667
    I1227 20:41:18.379408  9232 caffe.cpp:264] Batch 68, loss = 0.668076
    I1227 20:41:19.264288  9232 caffe.cpp:264] Batch 69, accuracy = 0.733333
    I1227 20:41:19.264334  9232 caffe.cpp:264] Batch 69, loss = 0.853307
    I1227 20:41:20.143440  9232 caffe.cpp:264] Batch 70, accuracy = 0.775
    I1227 20:41:20.143486  9232 caffe.cpp:264] Batch 70, loss = 0.627918
    I1227 20:41:21.016491  9232 caffe.cpp:264] Batch 71, accuracy = 0.716667
    I1227 20:41:21.016538  9232 caffe.cpp:264] Batch 71, loss = 0.853137
    I1227 20:41:21.917397  9232 caffe.cpp:264] Batch 72, accuracy = 0.716667
    I1227 20:41:21.917441  9232 caffe.cpp:264] Batch 72, loss = 0.810289
    I1227 20:41:22.801612  9232 caffe.cpp:264] Batch 73, accuracy = 0.833333
    I1227 20:41:22.801657  9232 caffe.cpp:264] Batch 73, loss = 0.536913
    I1227 20:41:23.684633  9232 caffe.cpp:264] Batch 74, accuracy = 0.65
    I1227 20:41:23.684679  9232 caffe.cpp:264] Batch 74, loss = 0.93226
    I1227 20:41:24.580243  9232 caffe.cpp:264] Batch 75, accuracy = 0.716667
    I1227 20:41:24.580288  9232 caffe.cpp:264] Batch 75, loss = 0.752806
    I1227 20:41:25.466311  9232 caffe.cpp:264] Batch 76, accuracy = 0.733333
    I1227 20:41:25.466356  9232 caffe.cpp:264] Batch 76, loss = 0.759903
    I1227 20:41:26.367560  9232 caffe.cpp:264] Batch 77, accuracy = 0.783333
    I1227 20:41:26.367605  9232 caffe.cpp:264] Batch 77, loss = 0.670729
    I1227 20:41:27.276728  9232 caffe.cpp:264] Batch 78, accuracy = 0.758333
    I1227 20:41:27.276773  9232 caffe.cpp:264] Batch 78, loss = 0.61588
    I1227 20:41:28.161556  9232 caffe.cpp:264] Batch 79, accuracy = 0.733333
    I1227 20:41:28.161602  9232 caffe.cpp:264] Batch 79, loss = 0.708059
    I1227 20:41:29.024966  9232 caffe.cpp:264] Batch 80, accuracy = 0.741667
    I1227 20:41:29.025009  9232 caffe.cpp:264] Batch 80, loss = 0.738941
    I1227 20:41:29.901319  9232 caffe.cpp:264] Batch 81, accuracy = 0.716667
    I1227 20:41:29.901363  9232 caffe.cpp:264] Batch 81, loss = 0.734711
    I1227 20:41:30.785199  9232 caffe.cpp:264] Batch 82, accuracy = 0.733333
    I1227 20:41:30.785246  9232 caffe.cpp:264] Batch 82, loss = 0.811378
    I1227 20:41:30.785254  9232 caffe.cpp:269] Loss: 0.741993
    I1227 20:41:30.785266  9232 caffe.cpp:281] accuracy = 0.741466
    I1227 20:41:30.785279  9232 caffe.cpp:281] loss = 0.741993 (* 1 = 0.741993 loss)
    CPU times: user 216 ms, sys: 48 ms, total: 264 ms
    Wall time: 1min 16s


## Conclusion: the model achieved 74% accuracy.
This means that upon showing the neural network a picture it had never seen, it will correctly classify it in one of the 10 categories 74% of the time. This is amazing, but the neural network for sure could be fine tuned with better solver parameters.

Let's convert the notebook to github markdown:


```python
!jupyter nbconvert --to markdown custom-cifar-10.ipynb
!mv custom-cifar-10.md README.md
```

    [NbConvertApp] Converting notebook custom-cifar-10.ipynb to markdown
    [NbConvertApp] Writing 404667 bytes to custom-cifar-10.md
