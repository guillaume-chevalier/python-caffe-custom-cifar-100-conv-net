
# Custom cifar-10 conv net with Caffe in Python (Pycaffe)

**Here, I train a custom convnet on the cifar-10 dataset. I did not try to implement any specific known architecture, but to design a new one for learning purposes. It is inspired from the official caffe python ".ipynb" examples available at: https://github.com/BVLC/caffe/tree/master/examples, but not the cifar-10 example itself that was in C++.**

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

    n.conv1 = L.Convolution(n.data, kernel_size=3, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    n.conv2 = L.Convolution(n.relu1, kernel_size=5, num_output=42, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    n.relu2 = L.ReLU(n.pool2, in_place=True)
    n.conv3 = L.Convolution(n.relu2, kernel_size=5, num_output=64, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.conv3, in_place=True)

    n.ip1 = L.InnerProduct(n.sig1, num_output=512, weight_filler=dict(type='xavier'))
    n.sig2 = L.Sigmoid(n.ip1, in_place=True)
    n.ip2 = L.InnerProduct(n.sig2, num_output=10, weight_filler=dict(type='xavier'))

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
     ('conv1', (100, 32, 30, 30)),
     ('pool1', (100, 32, 15, 15)),
     ('conv2', (100, 42, 11, 11)),
     ('pool2', (100, 42, 6, 6)),
     ('conv3', (100, 64, 2, 2)),
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

    [('conv1', (32, 3, 3, 3)),
     ('conv2', (42, 32, 5, 5)),
     ('conv3', (64, 42, 5, 5)),
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
    weight_decay: 0.005

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
    I1226 20:49:03.399009  6129 caffe.cpp:184] Using GPUs 0
    I1226 20:49:03.610534  6129 solver.cpp:48] Initializing solver from parameters:
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
    weight_decay: 0.005
    snapshot: 25000
    snapshot_prefix: "cnn_snapshot"
    solver_mode: GPU
    device_id: 0
    rms_decay: 0.98
    type: "RMSProp"
    I1226 20:49:03.610752  6129 solver.cpp:81] Creating training net from train_net file: cnn_train.prototxt
    I1226 20:49:03.611078  6129 net.cpp:49] Initializing net from parameters:
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
        num_output: 42
        kernel_size: 5
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
      name: "sig1"
      type: "Sigmoid"
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
      name: "sig2"
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
    I1226 20:49:03.611583  6129 layer_factory.hpp:77] Creating layer data
    I1226 20:49:03.611609  6129 net.cpp:106] Creating Layer data
    I1226 20:49:03.611619  6129 net.cpp:411] data -> data
    I1226 20:49:03.611642  6129 net.cpp:411] data -> label
    I1226 20:49:03.611657  6129 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/train.txt
    I1226 20:49:03.611690  6129 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1226 20:49:03.613013  6129 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1226 20:49:05.987900  6129 net.cpp:150] Setting up data
    I1226 20:49:05.987952  6129 net.cpp:157] Top shape: 100 3 32 32 (307200)
    I1226 20:49:05.987962  6129 net.cpp:157] Top shape: 100 (100)
    I1226 20:49:05.987968  6129 net.cpp:165] Memory required for data: 1229200
    I1226 20:49:05.987978  6129 layer_factory.hpp:77] Creating layer label_data_1_split
    I1226 20:49:05.988000  6129 net.cpp:106] Creating Layer label_data_1_split
    I1226 20:49:05.988008  6129 net.cpp:454] label_data_1_split <- label
    I1226 20:49:05.988030  6129 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1226 20:49:05.988041  6129 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1226 20:49:05.988072  6129 net.cpp:150] Setting up label_data_1_split
    I1226 20:49:05.988080  6129 net.cpp:157] Top shape: 100 (100)
    I1226 20:49:05.988085  6129 net.cpp:157] Top shape: 100 (100)
    I1226 20:49:05.988090  6129 net.cpp:165] Memory required for data: 1230000
    I1226 20:49:05.988095  6129 layer_factory.hpp:77] Creating layer conv1
    I1226 20:49:05.988106  6129 net.cpp:106] Creating Layer conv1
    I1226 20:49:05.988112  6129 net.cpp:454] conv1 <- data
    I1226 20:49:05.988152  6129 net.cpp:411] conv1 -> conv1
    I1226 20:49:05.989262  6129 net.cpp:150] Setting up conv1
    I1226 20:49:05.989289  6129 net.cpp:157] Top shape: 100 32 30 30 (2880000)
    I1226 20:49:05.989295  6129 net.cpp:165] Memory required for data: 12750000
    I1226 20:49:05.989310  6129 layer_factory.hpp:77] Creating layer pool1
    I1226 20:49:05.989320  6129 net.cpp:106] Creating Layer pool1
    I1226 20:49:05.989336  6129 net.cpp:454] pool1 <- conv1
    I1226 20:49:05.989342  6129 net.cpp:411] pool1 -> pool1
    I1226 20:49:05.989382  6129 net.cpp:150] Setting up pool1
    I1226 20:49:05.989389  6129 net.cpp:157] Top shape: 100 32 15 15 (720000)
    I1226 20:49:05.989394  6129 net.cpp:165] Memory required for data: 15630000
    I1226 20:49:05.989399  6129 layer_factory.hpp:77] Creating layer relu1
    I1226 20:49:05.989406  6129 net.cpp:106] Creating Layer relu1
    I1226 20:49:05.989411  6129 net.cpp:454] relu1 <- pool1
    I1226 20:49:05.989418  6129 net.cpp:397] relu1 -> pool1 (in-place)
    I1226 20:49:05.989428  6129 net.cpp:150] Setting up relu1
    I1226 20:49:05.989434  6129 net.cpp:157] Top shape: 100 32 15 15 (720000)
    I1226 20:49:05.989439  6129 net.cpp:165] Memory required for data: 18510000
    I1226 20:49:05.989444  6129 layer_factory.hpp:77] Creating layer conv2
    I1226 20:49:05.989452  6129 net.cpp:106] Creating Layer conv2
    I1226 20:49:05.989457  6129 net.cpp:454] conv2 <- pool1
    I1226 20:49:05.989475  6129 net.cpp:411] conv2 -> conv2
    I1226 20:49:05.989850  6129 net.cpp:150] Setting up conv2
    I1226 20:49:05.989871  6129 net.cpp:157] Top shape: 100 42 11 11 (508200)
    I1226 20:49:05.989877  6129 net.cpp:165] Memory required for data: 20542800
    I1226 20:49:05.989886  6129 layer_factory.hpp:77] Creating layer pool2
    I1226 20:49:05.989895  6129 net.cpp:106] Creating Layer pool2
    I1226 20:49:05.989900  6129 net.cpp:454] pool2 <- conv2
    I1226 20:49:05.989918  6129 net.cpp:411] pool2 -> pool2
    I1226 20:49:05.989935  6129 net.cpp:150] Setting up pool2
    I1226 20:49:05.989943  6129 net.cpp:157] Top shape: 100 42 6 6 (151200)
    I1226 20:49:05.989948  6129 net.cpp:165] Memory required for data: 21147600
    I1226 20:49:05.989953  6129 layer_factory.hpp:77] Creating layer relu2
    I1226 20:49:05.989959  6129 net.cpp:106] Creating Layer relu2
    I1226 20:49:05.989964  6129 net.cpp:454] relu2 <- pool2
    I1226 20:49:05.989970  6129 net.cpp:397] relu2 -> pool2 (in-place)
    I1226 20:49:05.989977  6129 net.cpp:150] Setting up relu2
    I1226 20:49:05.989982  6129 net.cpp:157] Top shape: 100 42 6 6 (151200)
    I1226 20:49:05.989987  6129 net.cpp:165] Memory required for data: 21752400
    I1226 20:49:05.989994  6129 layer_factory.hpp:77] Creating layer conv3
    I1226 20:49:05.990001  6129 net.cpp:106] Creating Layer conv3
    I1226 20:49:05.990006  6129 net.cpp:454] conv3 <- pool2
    I1226 20:49:05.990013  6129 net.cpp:411] conv3 -> conv3
    I1226 20:49:05.991032  6129 net.cpp:150] Setting up conv3
    I1226 20:49:05.991061  6129 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1226 20:49:05.991068  6129 net.cpp:165] Memory required for data: 21854800
    I1226 20:49:05.991080  6129 layer_factory.hpp:77] Creating layer sig1
    I1226 20:49:05.991091  6129 net.cpp:106] Creating Layer sig1
    I1226 20:49:05.991097  6129 net.cpp:454] sig1 <- conv3
    I1226 20:49:05.991106  6129 net.cpp:397] sig1 -> conv3 (in-place)
    I1226 20:49:05.991117  6129 net.cpp:150] Setting up sig1
    I1226 20:49:05.991123  6129 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1226 20:49:05.991130  6129 net.cpp:165] Memory required for data: 21957200
    I1226 20:49:05.991137  6129 layer_factory.hpp:77] Creating layer ip1
    I1226 20:49:05.991152  6129 net.cpp:106] Creating Layer ip1
    I1226 20:49:05.991159  6129 net.cpp:454] ip1 <- conv3
    I1226 20:49:05.991168  6129 net.cpp:411] ip1 -> ip1
    I1226 20:49:05.992259  6129 net.cpp:150] Setting up ip1
    I1226 20:49:05.992295  6129 net.cpp:157] Top shape: 100 512 (51200)
    I1226 20:49:05.992303  6129 net.cpp:165] Memory required for data: 22162000
    I1226 20:49:05.992316  6129 layer_factory.hpp:77] Creating layer sig2
    I1226 20:49:05.992328  6129 net.cpp:106] Creating Layer sig2
    I1226 20:49:05.992336  6129 net.cpp:454] sig2 <- ip1
    I1226 20:49:05.992346  6129 net.cpp:397] sig2 -> ip1 (in-place)
    I1226 20:49:05.992383  6129 net.cpp:150] Setting up sig2
    I1226 20:49:05.992391  6129 net.cpp:157] Top shape: 100 512 (51200)
    I1226 20:49:05.992398  6129 net.cpp:165] Memory required for data: 22366800
    I1226 20:49:05.992404  6129 layer_factory.hpp:77] Creating layer ip2
    I1226 20:49:05.992414  6129 net.cpp:106] Creating Layer ip2
    I1226 20:49:05.992421  6129 net.cpp:454] ip2 <- ip1
    I1226 20:49:05.992429  6129 net.cpp:411] ip2 -> ip2
    I1226 20:49:05.993247  6129 net.cpp:150] Setting up ip2
    I1226 20:49:05.993278  6129 net.cpp:157] Top shape: 100 10 (1000)
    I1226 20:49:05.993286  6129 net.cpp:165] Memory required for data: 22370800
    I1226 20:49:05.993304  6129 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1226 20:49:05.993329  6129 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1226 20:49:05.993335  6129 net.cpp:454] ip2_ip2_0_split <- ip2
    I1226 20:49:05.993345  6129 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1226 20:49:05.993366  6129 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1226 20:49:05.993407  6129 net.cpp:150] Setting up ip2_ip2_0_split
    I1226 20:49:05.993418  6129 net.cpp:157] Top shape: 100 10 (1000)
    I1226 20:49:05.993427  6129 net.cpp:157] Top shape: 100 10 (1000)
    I1226 20:49:05.993433  6129 net.cpp:165] Memory required for data: 22378800
    I1226 20:49:05.993440  6129 layer_factory.hpp:77] Creating layer accuracy
    I1226 20:49:05.993450  6129 net.cpp:106] Creating Layer accuracy
    I1226 20:49:05.993458  6129 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1226 20:49:05.993466  6129 net.cpp:454] accuracy <- label_data_1_split_0
    I1226 20:49:05.993486  6129 net.cpp:411] accuracy -> accuracy
    I1226 20:49:05.993508  6129 net.cpp:150] Setting up accuracy
    I1226 20:49:05.993517  6129 net.cpp:157] Top shape: (1)
    I1226 20:49:05.993525  6129 net.cpp:165] Memory required for data: 22378804
    I1226 20:49:05.993531  6129 layer_factory.hpp:77] Creating layer loss
    I1226 20:49:05.993541  6129 net.cpp:106] Creating Layer loss
    I1226 20:49:05.993548  6129 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1226 20:49:05.993567  6129 net.cpp:454] loss <- label_data_1_split_1
    I1226 20:49:05.993577  6129 net.cpp:411] loss -> loss
    I1226 20:49:05.993623  6129 layer_factory.hpp:77] Creating layer loss
    I1226 20:49:05.993729  6129 net.cpp:150] Setting up loss
    I1226 20:49:05.993741  6129 net.cpp:157] Top shape: (1)
    I1226 20:49:05.993749  6129 net.cpp:160]     with loss weight 1
    I1226 20:49:05.993779  6129 net.cpp:165] Memory required for data: 22378808
    I1226 20:49:05.993788  6129 net.cpp:226] loss needs backward computation.
    I1226 20:49:05.993796  6129 net.cpp:228] accuracy does not need backward computation.
    I1226 20:49:05.993805  6129 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1226 20:49:05.993813  6129 net.cpp:226] ip2 needs backward computation.
    I1226 20:49:05.993821  6129 net.cpp:226] sig2 needs backward computation.
    I1226 20:49:05.993829  6129 net.cpp:226] ip1 needs backward computation.
    I1226 20:49:05.993839  6129 net.cpp:226] sig1 needs backward computation.
    I1226 20:49:05.993846  6129 net.cpp:226] conv3 needs backward computation.
    I1226 20:49:05.993854  6129 net.cpp:226] relu2 needs backward computation.
    I1226 20:49:05.993863  6129 net.cpp:226] pool2 needs backward computation.
    I1226 20:49:05.993871  6129 net.cpp:226] conv2 needs backward computation.
    I1226 20:49:05.993880  6129 net.cpp:226] relu1 needs backward computation.
    I1226 20:49:05.993890  6129 net.cpp:226] pool1 needs backward computation.
    I1226 20:49:05.993896  6129 net.cpp:226] conv1 needs backward computation.
    I1226 20:49:05.993906  6129 net.cpp:228] label_data_1_split does not need backward computation.
    I1226 20:49:05.993914  6129 net.cpp:228] data does not need backward computation.
    I1226 20:49:05.993922  6129 net.cpp:270] This network produces output accuracy
    I1226 20:49:05.993930  6129 net.cpp:270] This network produces output loss
    I1226 20:49:05.993952  6129 net.cpp:283] Network initialization done.
    I1226 20:49:05.994276  6129 solver.cpp:181] Creating test net (#0) specified by test_net file: cnn_test.prototxt
    I1226 20:49:05.994410  6129 net.cpp:49] Initializing net from parameters:
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
        num_output: 42
        kernel_size: 5
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
      name: "sig1"
      type: "Sigmoid"
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
      name: "sig2"
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
    I1226 20:49:05.994948  6129 layer_factory.hpp:77] Creating layer data
    I1226 20:49:05.994964  6129 net.cpp:106] Creating Layer data
    I1226 20:49:05.994973  6129 net.cpp:411] data -> data
    I1226 20:49:05.994987  6129 net.cpp:411] data -> label
    I1226 20:49:05.994998  6129 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1226 20:49:05.995026  6129 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1226 20:49:06.368707  6129 net.cpp:150] Setting up data
    I1226 20:49:06.368741  6129 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1226 20:49:06.368749  6129 net.cpp:157] Top shape: 120 (120)
    I1226 20:49:06.368755  6129 net.cpp:165] Memory required for data: 1475040
    I1226 20:49:06.368764  6129 layer_factory.hpp:77] Creating layer label_data_1_split
    I1226 20:49:06.368778  6129 net.cpp:106] Creating Layer label_data_1_split
    I1226 20:49:06.368785  6129 net.cpp:454] label_data_1_split <- label
    I1226 20:49:06.368794  6129 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1226 20:49:06.368806  6129 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1226 20:49:06.368837  6129 net.cpp:150] Setting up label_data_1_split
    I1226 20:49:06.368845  6129 net.cpp:157] Top shape: 120 (120)
    I1226 20:49:06.368852  6129 net.cpp:157] Top shape: 120 (120)
    I1226 20:49:06.368857  6129 net.cpp:165] Memory required for data: 1476000
    I1226 20:49:06.368862  6129 layer_factory.hpp:77] Creating layer conv1
    I1226 20:49:06.368875  6129 net.cpp:106] Creating Layer conv1
    I1226 20:49:06.368880  6129 net.cpp:454] conv1 <- data
    I1226 20:49:06.368887  6129 net.cpp:411] conv1 -> conv1
    I1226 20:49:06.369060  6129 net.cpp:150] Setting up conv1
    I1226 20:49:06.369071  6129 net.cpp:157] Top shape: 120 32 30 30 (3456000)
    I1226 20:49:06.369076  6129 net.cpp:165] Memory required for data: 15300000
    I1226 20:49:06.369087  6129 layer_factory.hpp:77] Creating layer pool1
    I1226 20:49:06.369097  6129 net.cpp:106] Creating Layer pool1
    I1226 20:49:06.369103  6129 net.cpp:454] pool1 <- conv1
    I1226 20:49:06.369109  6129 net.cpp:411] pool1 -> pool1
    I1226 20:49:06.369139  6129 net.cpp:150] Setting up pool1
    I1226 20:49:06.369148  6129 net.cpp:157] Top shape: 120 32 15 15 (864000)
    I1226 20:49:06.369191  6129 net.cpp:165] Memory required for data: 18756000
    I1226 20:49:06.369209  6129 layer_factory.hpp:77] Creating layer relu1
    I1226 20:49:06.369216  6129 net.cpp:106] Creating Layer relu1
    I1226 20:49:06.369222  6129 net.cpp:454] relu1 <- pool1
    I1226 20:49:06.369228  6129 net.cpp:397] relu1 -> pool1 (in-place)
    I1226 20:49:06.369236  6129 net.cpp:150] Setting up relu1
    I1226 20:49:06.369243  6129 net.cpp:157] Top shape: 120 32 15 15 (864000)
    I1226 20:49:06.369249  6129 net.cpp:165] Memory required for data: 22212000
    I1226 20:49:06.369254  6129 layer_factory.hpp:77] Creating layer conv2
    I1226 20:49:06.369262  6129 net.cpp:106] Creating Layer conv2
    I1226 20:49:06.369267  6129 net.cpp:454] conv2 <- pool1
    I1226 20:49:06.369274  6129 net.cpp:411] conv2 -> conv2
    I1226 20:49:06.369642  6129 net.cpp:150] Setting up conv2
    I1226 20:49:06.369653  6129 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1226 20:49:06.369678  6129 net.cpp:165] Memory required for data: 24651360
    I1226 20:49:06.369699  6129 layer_factory.hpp:77] Creating layer pool2
    I1226 20:49:06.369706  6129 net.cpp:106] Creating Layer pool2
    I1226 20:49:06.369711  6129 net.cpp:454] pool2 <- conv2
    I1226 20:49:06.369719  6129 net.cpp:411] pool2 -> pool2
    I1226 20:49:06.369740  6129 net.cpp:150] Setting up pool2
    I1226 20:49:06.369746  6129 net.cpp:157] Top shape: 120 42 6 6 (181440)
    I1226 20:49:06.369751  6129 net.cpp:165] Memory required for data: 25377120
    I1226 20:49:06.369756  6129 layer_factory.hpp:77] Creating layer relu2
    I1226 20:49:06.369763  6129 net.cpp:106] Creating Layer relu2
    I1226 20:49:06.369770  6129 net.cpp:454] relu2 <- pool2
    I1226 20:49:06.369776  6129 net.cpp:397] relu2 -> pool2 (in-place)
    I1226 20:49:06.369782  6129 net.cpp:150] Setting up relu2
    I1226 20:49:06.369788  6129 net.cpp:157] Top shape: 120 42 6 6 (181440)
    I1226 20:49:06.369794  6129 net.cpp:165] Memory required for data: 26102880
    I1226 20:49:06.369799  6129 layer_factory.hpp:77] Creating layer conv3
    I1226 20:49:06.369807  6129 net.cpp:106] Creating Layer conv3
    I1226 20:49:06.369812  6129 net.cpp:454] conv3 <- pool2
    I1226 20:49:06.369819  6129 net.cpp:411] conv3 -> conv3
    I1226 20:49:06.370846  6129 net.cpp:150] Setting up conv3
    I1226 20:49:06.370862  6129 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1226 20:49:06.370868  6129 net.cpp:165] Memory required for data: 26225760
    I1226 20:49:06.370878  6129 layer_factory.hpp:77] Creating layer sig1
    I1226 20:49:06.370887  6129 net.cpp:106] Creating Layer sig1
    I1226 20:49:06.370893  6129 net.cpp:454] sig1 <- conv3
    I1226 20:49:06.370900  6129 net.cpp:397] sig1 -> conv3 (in-place)
    I1226 20:49:06.370909  6129 net.cpp:150] Setting up sig1
    I1226 20:49:06.370915  6129 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1226 20:49:06.370920  6129 net.cpp:165] Memory required for data: 26348640
    I1226 20:49:06.370925  6129 layer_factory.hpp:77] Creating layer ip1
    I1226 20:49:06.370934  6129 net.cpp:106] Creating Layer ip1
    I1226 20:49:06.370940  6129 net.cpp:454] ip1 <- conv3
    I1226 20:49:06.370947  6129 net.cpp:411] ip1 -> ip1
    I1226 20:49:06.371783  6129 net.cpp:150] Setting up ip1
    I1226 20:49:06.371803  6129 net.cpp:157] Top shape: 120 512 (61440)
    I1226 20:49:06.371809  6129 net.cpp:165] Memory required for data: 26594400
    I1226 20:49:06.371816  6129 layer_factory.hpp:77] Creating layer sig2
    I1226 20:49:06.371834  6129 net.cpp:106] Creating Layer sig2
    I1226 20:49:06.371840  6129 net.cpp:454] sig2 <- ip1
    I1226 20:49:06.371845  6129 net.cpp:397] sig2 -> ip1 (in-place)
    I1226 20:49:06.371851  6129 net.cpp:150] Setting up sig2
    I1226 20:49:06.371857  6129 net.cpp:157] Top shape: 120 512 (61440)
    I1226 20:49:06.371862  6129 net.cpp:165] Memory required for data: 26840160
    I1226 20:49:06.371866  6129 layer_factory.hpp:77] Creating layer ip2
    I1226 20:49:06.371873  6129 net.cpp:106] Creating Layer ip2
    I1226 20:49:06.371878  6129 net.cpp:454] ip2 <- ip1
    I1226 20:49:06.371884  6129 net.cpp:411] ip2 -> ip2
    I1226 20:49:06.371995  6129 net.cpp:150] Setting up ip2
    I1226 20:49:06.372014  6129 net.cpp:157] Top shape: 120 10 (1200)
    I1226 20:49:06.372030  6129 net.cpp:165] Memory required for data: 26844960
    I1226 20:49:06.372038  6129 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1226 20:49:06.372045  6129 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1226 20:49:06.372051  6129 net.cpp:454] ip2_ip2_0_split <- ip2
    I1226 20:49:06.372056  6129 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1226 20:49:06.372064  6129 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1226 20:49:06.372090  6129 net.cpp:150] Setting up ip2_ip2_0_split
    I1226 20:49:06.372097  6129 net.cpp:157] Top shape: 120 10 (1200)
    I1226 20:49:06.372103  6129 net.cpp:157] Top shape: 120 10 (1200)
    I1226 20:49:06.372107  6129 net.cpp:165] Memory required for data: 26854560
    I1226 20:49:06.372112  6129 layer_factory.hpp:77] Creating layer accuracy
    I1226 20:49:06.372119  6129 net.cpp:106] Creating Layer accuracy
    I1226 20:49:06.372124  6129 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1226 20:49:06.372130  6129 net.cpp:454] accuracy <- label_data_1_split_0
    I1226 20:49:06.372136  6129 net.cpp:411] accuracy -> accuracy
    I1226 20:49:06.372145  6129 net.cpp:150] Setting up accuracy
    I1226 20:49:06.372161  6129 net.cpp:157] Top shape: (1)
    I1226 20:49:06.372167  6129 net.cpp:165] Memory required for data: 26854564
    I1226 20:49:06.372172  6129 layer_factory.hpp:77] Creating layer loss
    I1226 20:49:06.372180  6129 net.cpp:106] Creating Layer loss
    I1226 20:49:06.372185  6129 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1226 20:49:06.372192  6129 net.cpp:454] loss <- label_data_1_split_1
    I1226 20:49:06.372198  6129 net.cpp:411] loss -> loss
    I1226 20:49:06.372208  6129 layer_factory.hpp:77] Creating layer loss
    I1226 20:49:06.372681  6129 net.cpp:150] Setting up loss
    I1226 20:49:06.372706  6129 net.cpp:157] Top shape: (1)
    I1226 20:49:06.372712  6129 net.cpp:160]     with loss weight 1
    I1226 20:49:06.372735  6129 net.cpp:165] Memory required for data: 26854568
    I1226 20:49:06.372740  6129 net.cpp:226] loss needs backward computation.
    I1226 20:49:06.372746  6129 net.cpp:228] accuracy does not need backward computation.
    I1226 20:49:06.372751  6129 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1226 20:49:06.372757  6129 net.cpp:226] ip2 needs backward computation.
    I1226 20:49:06.372762  6129 net.cpp:226] sig2 needs backward computation.
    I1226 20:49:06.372766  6129 net.cpp:226] ip1 needs backward computation.
    I1226 20:49:06.372772  6129 net.cpp:226] sig1 needs backward computation.
    I1226 20:49:06.372777  6129 net.cpp:226] conv3 needs backward computation.
    I1226 20:49:06.372782  6129 net.cpp:226] relu2 needs backward computation.
    I1226 20:49:06.372787  6129 net.cpp:226] pool2 needs backward computation.
    I1226 20:49:06.372792  6129 net.cpp:226] conv2 needs backward computation.
    I1226 20:49:06.372797  6129 net.cpp:226] relu1 needs backward computation.
    I1226 20:49:06.372802  6129 net.cpp:226] pool1 needs backward computation.
    I1226 20:49:06.372807  6129 net.cpp:226] conv1 needs backward computation.
    I1226 20:49:06.372812  6129 net.cpp:228] label_data_1_split does not need backward computation.
    I1226 20:49:06.372818  6129 net.cpp:228] data does not need backward computation.
    I1226 20:49:06.372823  6129 net.cpp:270] This network produces output accuracy
    I1226 20:49:06.372828  6129 net.cpp:270] This network produces output loss
    I1226 20:49:06.372841  6129 net.cpp:283] Network initialization done.
    I1226 20:49:06.372898  6129 solver.cpp:60] Solver scaffolding done.
    I1226 20:49:06.373181  6129 caffe.cpp:212] Starting Optimization
    I1226 20:49:06.373191  6129 solver.cpp:288] Solving
    I1226 20:49:06.373196  6129 solver.cpp:289] Learning Rate Policy: inv
    I1226 20:49:06.373792  6129 solver.cpp:341] Iteration 0, Testing net (#0)
    I1226 20:49:12.967999  6129 solver.cpp:409]     Test net output #0: accuracy = 0.0941667
    I1226 20:49:12.968077  6129 solver.cpp:409]     Test net output #1: loss = 2.37358 (* 1 = 2.37358 loss)
    I1226 20:49:13.055029  6129 solver.cpp:237] Iteration 0, loss = 2.33023
    I1226 20:49:13.055079  6129 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1226 20:49:13.055094  6129 solver.cpp:253]     Train net output #1: loss = 2.33023 (* 1 = 2.33023 loss)
    I1226 20:49:13.055151  6129 sgd_solver.cpp:106] Iteration 0, lr = 0.0007
    I1226 20:49:28.804519  6129 solver.cpp:237] Iteration 100, loss = 2.29429
    I1226 20:49:28.804568  6129 solver.cpp:253]     Train net output #0: accuracy = 0.13
    I1226 20:49:28.804582  6129 solver.cpp:253]     Train net output #1: loss = 2.29429 (* 1 = 2.29429 loss)
    I1226 20:49:28.804594  6129 sgd_solver.cpp:106] Iteration 100, lr = 0.000694796
    I1226 20:49:44.934286  6129 solver.cpp:237] Iteration 200, loss = 2.34597
    I1226 20:49:44.934377  6129 solver.cpp:253]     Train net output #0: accuracy = 0.09
    I1226 20:49:44.934396  6129 solver.cpp:253]     Train net output #1: loss = 2.34597 (* 1 = 2.34597 loss)
    I1226 20:49:44.934407  6129 sgd_solver.cpp:106] Iteration 200, lr = 0.00068968
    I1226 20:50:00.849244  6129 solver.cpp:237] Iteration 300, loss = 2.32553
    I1226 20:50:00.849304  6129 solver.cpp:253]     Train net output #0: accuracy = 0.1
    I1226 20:50:00.849328  6129 solver.cpp:253]     Train net output #1: loss = 2.32553 (* 1 = 2.32553 loss)
    I1226 20:50:00.849345  6129 sgd_solver.cpp:106] Iteration 300, lr = 0.000684652
    I1226 20:50:16.246145  6129 solver.cpp:237] Iteration 400, loss = 1.94783
    I1226 20:50:16.246350  6129 solver.cpp:253]     Train net output #0: accuracy = 0.24
    I1226 20:50:16.246399  6129 solver.cpp:253]     Train net output #1: loss = 1.94783 (* 1 = 1.94783 loss)
    I1226 20:50:16.246425  6129 sgd_solver.cpp:106] Iteration 400, lr = 0.000679709
    I1226 20:50:27.600265  6129 solver.cpp:237] Iteration 500, loss = 1.68079
    I1226 20:50:27.600306  6129 solver.cpp:253]     Train net output #0: accuracy = 0.4
    I1226 20:50:27.600317  6129 solver.cpp:253]     Train net output #1: loss = 1.68079 (* 1 = 1.68079 loss)
    I1226 20:50:27.600327  6129 sgd_solver.cpp:106] Iteration 500, lr = 0.000674848
    I1226 20:50:44.527806  6129 solver.cpp:237] Iteration 600, loss = 1.82447
    I1226 20:50:44.527858  6129 solver.cpp:253]     Train net output #0: accuracy = 0.37
    I1226 20:50:44.527874  6129 solver.cpp:253]     Train net output #1: loss = 1.82447 (* 1 = 1.82447 loss)
    I1226 20:50:44.527884  6129 sgd_solver.cpp:106] Iteration 600, lr = 0.000670068
    I1226 20:51:00.771272  6129 solver.cpp:237] Iteration 700, loss = 1.52572
    I1226 20:51:00.771435  6129 solver.cpp:253]     Train net output #0: accuracy = 0.38
    I1226 20:51:00.771464  6129 solver.cpp:253]     Train net output #1: loss = 1.52572 (* 1 = 1.52572 loss)
    I1226 20:51:00.771477  6129 sgd_solver.cpp:106] Iteration 700, lr = 0.000665365
    I1226 20:51:16.964146  6129 solver.cpp:237] Iteration 800, loss = 1.64916
    I1226 20:51:16.964184  6129 solver.cpp:253]     Train net output #0: accuracy = 0.36
    I1226 20:51:16.964198  6129 solver.cpp:253]     Train net output #1: loss = 1.64916 (* 1 = 1.64916 loss)
    I1226 20:51:16.964210  6129 sgd_solver.cpp:106] Iteration 800, lr = 0.000660739
    I1226 20:51:33.051338  6129 solver.cpp:237] Iteration 900, loss = 1.40114
    I1226 20:51:33.051465  6129 solver.cpp:253]     Train net output #0: accuracy = 0.47
    I1226 20:51:33.051483  6129 solver.cpp:253]     Train net output #1: loss = 1.40114 (* 1 = 1.40114 loss)
    I1226 20:51:33.051496  6129 sgd_solver.cpp:106] Iteration 900, lr = 0.000656188
    I1226 20:51:43.699990  6129 solver.cpp:341] Iteration 1000, Testing net (#0)
    I1226 20:51:49.789225  6129 solver.cpp:409]     Test net output #0: accuracy = 0.445083
    I1226 20:51:49.789297  6129 solver.cpp:409]     Test net output #1: loss = 1.53439 (* 1 = 1.53439 loss)
    I1226 20:51:49.864742  6129 solver.cpp:237] Iteration 1000, loss = 1.47693
    I1226 20:51:49.864792  6129 solver.cpp:253]     Train net output #0: accuracy = 0.42
    I1226 20:51:49.864805  6129 solver.cpp:253]     Train net output #1: loss = 1.47693 (* 1 = 1.47693 loss)
    I1226 20:51:49.864816  6129 sgd_solver.cpp:106] Iteration 1000, lr = 0.000651709
    I1226 20:52:04.338650  6129 solver.cpp:237] Iteration 1100, loss = 1.4977
    I1226 20:52:04.338789  6129 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1226 20:52:04.338809  6129 solver.cpp:253]     Train net output #1: loss = 1.4977 (* 1 = 1.4977 loss)
    I1226 20:52:04.338819  6129 sgd_solver.cpp:106] Iteration 1100, lr = 0.0006473
    I1226 20:52:16.627610  6129 solver.cpp:237] Iteration 1200, loss = 1.36028
    I1226 20:52:16.627653  6129 solver.cpp:253]     Train net output #0: accuracy = 0.45
    I1226 20:52:16.627667  6129 solver.cpp:253]     Train net output #1: loss = 1.36028 (* 1 = 1.36028 loss)
    I1226 20:52:16.627678  6129 sgd_solver.cpp:106] Iteration 1200, lr = 0.000642961
    I1226 20:52:28.178668  6129 solver.cpp:237] Iteration 1300, loss = 1.43542
    I1226 20:52:28.178727  6129 solver.cpp:253]     Train net output #0: accuracy = 0.46
    I1226 20:52:28.178750  6129 solver.cpp:253]     Train net output #1: loss = 1.43542 (* 1 = 1.43542 loss)
    I1226 20:52:28.178766  6129 sgd_solver.cpp:106] Iteration 1300, lr = 0.000638689
    I1226 20:52:39.822965  6129 solver.cpp:237] Iteration 1400, loss = 1.26321
    I1226 20:52:39.823194  6129 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1226 20:52:39.823228  6129 solver.cpp:253]     Train net output #1: loss = 1.26321 (* 1 = 1.26321 loss)
    I1226 20:52:39.823251  6129 sgd_solver.cpp:106] Iteration 1400, lr = 0.000634482
    I1226 20:52:51.269434  6129 solver.cpp:237] Iteration 1500, loss = 1.40885
    I1226 20:52:51.269484  6129 solver.cpp:253]     Train net output #0: accuracy = 0.45
    I1226 20:52:51.269500  6129 solver.cpp:253]     Train net output #1: loss = 1.40885 (* 1 = 1.40885 loss)
    I1226 20:52:51.269511  6129 sgd_solver.cpp:106] Iteration 1500, lr = 0.00063034
    I1226 20:53:02.752974  6129 solver.cpp:237] Iteration 1600, loss = 1.39655
    I1226 20:53:02.753033  6129 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1226 20:53:02.753057  6129 solver.cpp:253]     Train net output #1: loss = 1.39655 (* 1 = 1.39655 loss)
    I1226 20:53:02.753073  6129 sgd_solver.cpp:106] Iteration 1600, lr = 0.00062626
    I1226 20:53:14.415395  6129 solver.cpp:237] Iteration 1700, loss = 1.21483
    I1226 20:53:14.415559  6129 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1226 20:53:14.415591  6129 solver.cpp:253]     Train net output #1: loss = 1.21483 (* 1 = 1.21483 loss)
    I1226 20:53:14.415611  6129 sgd_solver.cpp:106] Iteration 1700, lr = 0.000622241
    I1226 20:53:25.988653  6129 solver.cpp:237] Iteration 1800, loss = 1.3041
    I1226 20:53:25.988701  6129 solver.cpp:253]     Train net output #0: accuracy = 0.52
    I1226 20:53:25.988716  6129 solver.cpp:253]     Train net output #1: loss = 1.3041 (* 1 = 1.3041 loss)
    I1226 20:53:25.988729  6129 sgd_solver.cpp:106] Iteration 1800, lr = 0.000618282
    I1226 20:53:37.405911  6129 solver.cpp:237] Iteration 1900, loss = 1.18563
    I1226 20:53:37.405969  6129 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1226 20:53:37.405985  6129 solver.cpp:253]     Train net output #1: loss = 1.18563 (* 1 = 1.18563 loss)
    I1226 20:53:37.405997  6129 sgd_solver.cpp:106] Iteration 1900, lr = 0.000614381
    I1226 20:53:48.900472  6129 solver.cpp:341] Iteration 2000, Testing net (#0)
    I1226 20:53:54.006171  6129 solver.cpp:409]     Test net output #0: accuracy = 0.512583
    I1226 20:53:54.006239  6129 solver.cpp:409]     Test net output #1: loss = 1.35604 (* 1 = 1.35604 loss)
    I1226 20:53:54.060031  6129 solver.cpp:237] Iteration 2000, loss = 1.35527
    I1226 20:53:54.060078  6129 solver.cpp:253]     Train net output #0: accuracy = 0.49
    I1226 20:53:54.060093  6129 solver.cpp:253]     Train net output #1: loss = 1.35527 (* 1 = 1.35527 loss)
    I1226 20:53:54.060104  6129 sgd_solver.cpp:106] Iteration 2000, lr = 0.000610537
    I1226 20:54:09.036943  6129 solver.cpp:237] Iteration 2100, loss = 1.34925
    I1226 20:54:09.036984  6129 solver.cpp:253]     Train net output #0: accuracy = 0.55
    I1226 20:54:09.036998  6129 solver.cpp:253]     Train net output #1: loss = 1.34925 (* 1 = 1.34925 loss)
    I1226 20:54:09.037010  6129 sgd_solver.cpp:106] Iteration 2100, lr = 0.000606749
    I1226 20:54:25.154124  6129 solver.cpp:237] Iteration 2200, loss = 1.15764
    I1226 20:54:25.154278  6129 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1226 20:54:25.154296  6129 solver.cpp:253]     Train net output #1: loss = 1.15764 (* 1 = 1.15764 loss)
    I1226 20:54:25.154309  6129 sgd_solver.cpp:106] Iteration 2200, lr = 0.000603015
    I1226 20:54:41.216261  6129 solver.cpp:237] Iteration 2300, loss = 1.28191
    I1226 20:54:41.216308  6129 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1226 20:54:41.216336  6129 solver.cpp:253]     Train net output #1: loss = 1.28191 (* 1 = 1.28191 loss)
    I1226 20:54:41.216349  6129 sgd_solver.cpp:106] Iteration 2300, lr = 0.000599334
    I1226 20:54:57.182061  6129 solver.cpp:237] Iteration 2400, loss = 1.1472
    I1226 20:54:57.182225  6129 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1226 20:54:57.182252  6129 solver.cpp:253]     Train net output #1: loss = 1.1472 (* 1 = 1.1472 loss)
    I1226 20:54:57.182271  6129 sgd_solver.cpp:106] Iteration 2400, lr = 0.000595706
    I1226 20:55:13.350539  6129 solver.cpp:237] Iteration 2500, loss = 1.28022
    I1226 20:55:13.350586  6129 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1226 20:55:13.350601  6129 solver.cpp:253]     Train net output #1: loss = 1.28022 (* 1 = 1.28022 loss)
    I1226 20:55:13.350612  6129 sgd_solver.cpp:106] Iteration 2500, lr = 0.000592128
    I1226 20:55:29.478302  6129 solver.cpp:237] Iteration 2600, loss = 1.26874
    I1226 20:55:29.478468  6129 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1226 20:55:29.478489  6129 solver.cpp:253]     Train net output #1: loss = 1.26874 (* 1 = 1.26874 loss)
    I1226 20:55:29.478503  6129 sgd_solver.cpp:106] Iteration 2600, lr = 0.0005886
    I1226 20:55:45.503538  6129 solver.cpp:237] Iteration 2700, loss = 1.15414
    I1226 20:55:45.503578  6129 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1226 20:55:45.503592  6129 solver.cpp:253]     Train net output #1: loss = 1.15414 (* 1 = 1.15414 loss)
    I1226 20:55:45.503603  6129 sgd_solver.cpp:106] Iteration 2700, lr = 0.00058512
    I1226 20:56:01.534715  6129 solver.cpp:237] Iteration 2800, loss = 1.27066
    I1226 20:56:01.534862  6129 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1226 20:56:01.534878  6129 solver.cpp:253]     Train net output #1: loss = 1.27066 (* 1 = 1.27066 loss)
    I1226 20:56:01.534889  6129 sgd_solver.cpp:106] Iteration 2800, lr = 0.000581689
    I1226 20:56:17.705806  6129 solver.cpp:237] Iteration 2900, loss = 1.12119
    I1226 20:56:17.705847  6129 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1226 20:56:17.705862  6129 solver.cpp:253]     Train net output #1: loss = 1.12119 (* 1 = 1.12119 loss)
    I1226 20:56:17.705873  6129 sgd_solver.cpp:106] Iteration 2900, lr = 0.000578303
    I1226 20:56:33.693312  6129 solver.cpp:341] Iteration 3000, Testing net (#0)
    I1226 20:56:40.380368  6129 solver.cpp:409]     Test net output #0: accuracy = 0.5375
    I1226 20:56:40.380504  6129 solver.cpp:409]     Test net output #1: loss = 1.30424 (* 1 = 1.30424 loss)
    I1226 20:56:40.450760  6129 solver.cpp:237] Iteration 3000, loss = 1.27811
    I1226 20:56:40.450829  6129 solver.cpp:253]     Train net output #0: accuracy = 0.49
    I1226 20:56:40.450853  6129 solver.cpp:253]     Train net output #1: loss = 1.27811 (* 1 = 1.27811 loss)
    I1226 20:56:40.450875  6129 sgd_solver.cpp:106] Iteration 3000, lr = 0.000574964
    I1226 20:56:56.661046  6129 solver.cpp:237] Iteration 3100, loss = 1.22134
    I1226 20:56:56.661098  6129 solver.cpp:253]     Train net output #0: accuracy = 0.57
    I1226 20:56:56.661114  6129 solver.cpp:253]     Train net output #1: loss = 1.22134 (* 1 = 1.22134 loss)
    I1226 20:56:56.661128  6129 sgd_solver.cpp:106] Iteration 3100, lr = 0.000571669
    I1226 20:57:12.685245  6129 solver.cpp:237] Iteration 3200, loss = 1.08796
    I1226 20:57:12.685407  6129 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1226 20:57:12.685441  6129 solver.cpp:253]     Train net output #1: loss = 1.08796 (* 1 = 1.08796 loss)
    I1226 20:57:12.685459  6129 sgd_solver.cpp:106] Iteration 3200, lr = 0.000568418
    I1226 20:57:28.831531  6129 solver.cpp:237] Iteration 3300, loss = 1.1771
    I1226 20:57:28.831601  6129 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1226 20:57:28.831627  6129 solver.cpp:253]     Train net output #1: loss = 1.1771 (* 1 = 1.1771 loss)
    I1226 20:57:28.831647  6129 sgd_solver.cpp:106] Iteration 3300, lr = 0.000565209
    I1226 20:57:44.996680  6129 solver.cpp:237] Iteration 3400, loss = 1.15284
    I1226 20:57:44.996901  6129 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1226 20:57:44.996932  6129 solver.cpp:253]     Train net output #1: loss = 1.15284 (* 1 = 1.15284 loss)
    I1226 20:57:44.996951  6129 sgd_solver.cpp:106] Iteration 3400, lr = 0.000562043
    I1226 20:58:01.298144  6129 solver.cpp:237] Iteration 3500, loss = 1.25821
    I1226 20:58:01.298214  6129 solver.cpp:253]     Train net output #0: accuracy = 0.51
    I1226 20:58:01.298240  6129 solver.cpp:253]     Train net output #1: loss = 1.25821 (* 1 = 1.25821 loss)
    I1226 20:58:01.298260  6129 sgd_solver.cpp:106] Iteration 3500, lr = 0.000558917
    I1226 20:58:15.071629  6129 solver.cpp:237] Iteration 3600, loss = 1.18489
    I1226 20:58:15.071763  6129 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1226 20:58:15.071779  6129 solver.cpp:253]     Train net output #1: loss = 1.18489 (* 1 = 1.18489 loss)
    I1226 20:58:15.071789  6129 sgd_solver.cpp:106] Iteration 3600, lr = 0.000555832
    I1226 20:58:25.851501  6129 solver.cpp:237] Iteration 3700, loss = 1.03209
    I1226 20:58:25.851544  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 20:58:25.851559  6129 solver.cpp:253]     Train net output #1: loss = 1.03209 (* 1 = 1.03209 loss)
    I1226 20:58:25.851570  6129 sgd_solver.cpp:106] Iteration 3700, lr = 0.000552787
    I1226 20:58:37.044618  6129 solver.cpp:237] Iteration 3800, loss = 1.17086
    I1226 20:58:37.044656  6129 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1226 20:58:37.044668  6129 solver.cpp:253]     Train net output #1: loss = 1.17086 (* 1 = 1.17086 loss)
    I1226 20:58:37.044678  6129 sgd_solver.cpp:106] Iteration 3800, lr = 0.00054978
    I1226 20:58:50.646102  6129 solver.cpp:237] Iteration 3900, loss = 1.13614
    I1226 20:58:50.646225  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 20:58:50.646241  6129 solver.cpp:253]     Train net output #1: loss = 1.13614 (* 1 = 1.13614 loss)
    I1226 20:58:50.646251  6129 sgd_solver.cpp:106] Iteration 3900, lr = 0.000546811
    I1226 20:59:02.000016  6129 solver.cpp:341] Iteration 4000, Testing net (#0)
    I1226 20:59:06.413503  6129 solver.cpp:409]     Test net output #0: accuracy = 0.56375
    I1226 20:59:06.413555  6129 solver.cpp:409]     Test net output #1: loss = 1.23643 (* 1 = 1.23643 loss)
    I1226 20:59:06.458045  6129 solver.cpp:237] Iteration 4000, loss = 1.17729
    I1226 20:59:06.458084  6129 solver.cpp:253]     Train net output #0: accuracy = 0.53
    I1226 20:59:06.458099  6129 solver.cpp:253]     Train net output #1: loss = 1.17729 (* 1 = 1.17729 loss)
    I1226 20:59:06.458112  6129 sgd_solver.cpp:106] Iteration 4000, lr = 0.000543879
    I1226 20:59:18.336098  6129 solver.cpp:237] Iteration 4100, loss = 1.15974
    I1226 20:59:18.336155  6129 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1226 20:59:18.336177  6129 solver.cpp:253]     Train net output #1: loss = 1.15974 (* 1 = 1.15974 loss)
    I1226 20:59:18.336194  6129 sgd_solver.cpp:106] Iteration 4100, lr = 0.000540983
    I1226 20:59:29.528046  6129 solver.cpp:237] Iteration 4200, loss = 1.0011
    I1226 20:59:29.528194  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 20:59:29.528213  6129 solver.cpp:253]     Train net output #1: loss = 1.0011 (* 1 = 1.0011 loss)
    I1226 20:59:29.528224  6129 sgd_solver.cpp:106] Iteration 4200, lr = 0.000538123
    I1226 20:59:40.235769  6129 solver.cpp:237] Iteration 4300, loss = 1.1191
    I1226 20:59:40.235811  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 20:59:40.235826  6129 solver.cpp:253]     Train net output #1: loss = 1.1191 (* 1 = 1.1191 loss)
    I1226 20:59:40.235839  6129 sgd_solver.cpp:106] Iteration 4300, lr = 0.000535298
    I1226 20:59:50.889926  6129 solver.cpp:237] Iteration 4400, loss = 1.10152
    I1226 20:59:50.889972  6129 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1226 20:59:50.889986  6129 solver.cpp:253]     Train net output #1: loss = 1.10152 (* 1 = 1.10152 loss)
    I1226 20:59:50.889994  6129 sgd_solver.cpp:106] Iteration 4400, lr = 0.000532508
    I1226 21:00:01.401993  6129 solver.cpp:237] Iteration 4500, loss = 1.14086
    I1226 21:00:01.402168  6129 solver.cpp:253]     Train net output #0: accuracy = 0.55
    I1226 21:00:01.402195  6129 solver.cpp:253]     Train net output #1: loss = 1.14086 (* 1 = 1.14086 loss)
    I1226 21:00:01.402211  6129 sgd_solver.cpp:106] Iteration 4500, lr = 0.000529751
    I1226 21:00:11.984277  6129 solver.cpp:237] Iteration 4600, loss = 1.1461
    I1226 21:00:11.984319  6129 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1226 21:00:11.984334  6129 solver.cpp:253]     Train net output #1: loss = 1.1461 (* 1 = 1.1461 loss)
    I1226 21:00:11.984346  6129 sgd_solver.cpp:106] Iteration 4600, lr = 0.000527028
    I1226 21:00:22.498891  6129 solver.cpp:237] Iteration 4700, loss = 0.940118
    I1226 21:00:22.498936  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:00:22.498949  6129 solver.cpp:253]     Train net output #1: loss = 0.940118 (* 1 = 0.940118 loss)
    I1226 21:00:22.498957  6129 sgd_solver.cpp:106] Iteration 4700, lr = 0.000524336
    I1226 21:00:32.980849  6129 solver.cpp:237] Iteration 4800, loss = 1.14882
    I1226 21:00:32.981019  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 21:00:32.981046  6129 solver.cpp:253]     Train net output #1: loss = 1.14882 (* 1 = 1.14882 loss)
    I1226 21:00:32.981065  6129 sgd_solver.cpp:106] Iteration 4800, lr = 0.000521677
    I1226 21:00:43.523448  6129 solver.cpp:237] Iteration 4900, loss = 1.08399
    I1226 21:00:43.523494  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:00:43.523509  6129 solver.cpp:253]     Train net output #1: loss = 1.08399 (* 1 = 1.08399 loss)
    I1226 21:00:43.523520  6129 sgd_solver.cpp:106] Iteration 4900, lr = 0.000519049
    I1226 21:00:53.934823  6129 solver.cpp:341] Iteration 5000, Testing net (#0)
    I1226 21:00:58.229168  6129 solver.cpp:409]     Test net output #0: accuracy = 0.581417
    I1226 21:00:58.229209  6129 solver.cpp:409]     Test net output #1: loss = 1.19384 (* 1 = 1.19384 loss)
    I1226 21:00:58.284265  6129 solver.cpp:237] Iteration 5000, loss = 1.12264
    I1226 21:00:58.284302  6129 solver.cpp:253]     Train net output #0: accuracy = 0.56
    I1226 21:00:58.284317  6129 solver.cpp:253]     Train net output #1: loss = 1.12264 (* 1 = 1.12264 loss)
    I1226 21:00:58.284327  6129 sgd_solver.cpp:106] Iteration 5000, lr = 0.000516452
    I1226 21:01:08.827504  6129 solver.cpp:237] Iteration 5100, loss = 1.15813
    I1226 21:01:08.827612  6129 solver.cpp:253]     Train net output #0: accuracy = 0.58
    I1226 21:01:08.827631  6129 solver.cpp:253]     Train net output #1: loss = 1.15813 (* 1 = 1.15813 loss)
    I1226 21:01:08.827642  6129 sgd_solver.cpp:106] Iteration 5100, lr = 0.000513884
    I1226 21:01:19.322942  6129 solver.cpp:237] Iteration 5200, loss = 0.918832
    I1226 21:01:19.322988  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:01:19.322999  6129 solver.cpp:253]     Train net output #1: loss = 0.918832 (* 1 = 0.918832 loss)
    I1226 21:01:19.323009  6129 sgd_solver.cpp:106] Iteration 5200, lr = 0.000511347
    I1226 21:01:29.830476  6129 solver.cpp:237] Iteration 5300, loss = 1.0835
    I1226 21:01:29.830518  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 21:01:29.830534  6129 solver.cpp:253]     Train net output #1: loss = 1.0835 (* 1 = 1.0835 loss)
    I1226 21:01:29.830546  6129 sgd_solver.cpp:106] Iteration 5300, lr = 0.000508838
    I1226 21:01:40.370647  6129 solver.cpp:237] Iteration 5400, loss = 1.02248
    I1226 21:01:40.370843  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:01:40.370873  6129 solver.cpp:253]     Train net output #1: loss = 1.02248 (* 1 = 1.02248 loss)
    I1226 21:01:40.370889  6129 sgd_solver.cpp:106] Iteration 5400, lr = 0.000506358
    I1226 21:01:50.903198  6129 solver.cpp:237] Iteration 5500, loss = 1.09857
    I1226 21:01:50.903235  6129 solver.cpp:253]     Train net output #0: accuracy = 0.59
    I1226 21:01:50.903247  6129 solver.cpp:253]     Train net output #1: loss = 1.09857 (* 1 = 1.09857 loss)
    I1226 21:01:50.903256  6129 sgd_solver.cpp:106] Iteration 5500, lr = 0.000503906
    I1226 21:02:01.453426  6129 solver.cpp:237] Iteration 5600, loss = 1.10649
    I1226 21:02:01.453469  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 21:02:01.453485  6129 solver.cpp:253]     Train net output #1: loss = 1.10649 (* 1 = 1.10649 loss)
    I1226 21:02:01.453495  6129 sgd_solver.cpp:106] Iteration 5600, lr = 0.000501481
    I1226 21:02:11.949615  6129 solver.cpp:237] Iteration 5700, loss = 0.895661
    I1226 21:02:11.950537  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:02:11.950564  6129 solver.cpp:253]     Train net output #1: loss = 0.895661 (* 1 = 0.895661 loss)
    I1226 21:02:11.950573  6129 sgd_solver.cpp:106] Iteration 5700, lr = 0.000499084
    I1226 21:02:22.475729  6129 solver.cpp:237] Iteration 5800, loss = 1.05216
    I1226 21:02:22.475785  6129 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1226 21:02:22.475807  6129 solver.cpp:253]     Train net output #1: loss = 1.05216 (* 1 = 1.05216 loss)
    I1226 21:02:22.475822  6129 sgd_solver.cpp:106] Iteration 5800, lr = 0.000496713
    I1226 21:02:32.998970  6129 solver.cpp:237] Iteration 5900, loss = 1.0021
    I1226 21:02:32.999014  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:02:32.999030  6129 solver.cpp:253]     Train net output #1: loss = 1.0021 (* 1 = 1.0021 loss)
    I1226 21:02:32.999042  6129 sgd_solver.cpp:106] Iteration 5900, lr = 0.000494368
    I1226 21:02:43.361788  6129 solver.cpp:341] Iteration 6000, Testing net (#0)
    I1226 21:02:47.709872  6129 solver.cpp:409]     Test net output #0: accuracy = 0.59025
    I1226 21:02:47.709933  6129 solver.cpp:409]     Test net output #1: loss = 1.16906 (* 1 = 1.16906 loss)
    I1226 21:02:47.755013  6129 solver.cpp:237] Iteration 6000, loss = 1.08319
    I1226 21:02:47.755071  6129 solver.cpp:253]     Train net output #0: accuracy = 0.6
    I1226 21:02:47.755095  6129 solver.cpp:253]     Train net output #1: loss = 1.08319 (* 1 = 1.08319 loss)
    I1226 21:02:47.755110  6129 sgd_solver.cpp:106] Iteration 6000, lr = 0.000492049
    I1226 21:02:58.326727  6129 solver.cpp:237] Iteration 6100, loss = 1.10086
    I1226 21:02:58.326764  6129 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1226 21:02:58.326776  6129 solver.cpp:253]     Train net output #1: loss = 1.10086 (* 1 = 1.10086 loss)
    I1226 21:02:58.326786  6129 sgd_solver.cpp:106] Iteration 6100, lr = 0.000489755
    I1226 21:03:08.866021  6129 solver.cpp:237] Iteration 6200, loss = 0.868991
    I1226 21:03:08.866070  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:03:08.866085  6129 solver.cpp:253]     Train net output #1: loss = 0.868991 (* 1 = 0.868991 loss)
    I1226 21:03:08.866093  6129 sgd_solver.cpp:106] Iteration 6200, lr = 0.000487486
    I1226 21:03:19.389869  6129 solver.cpp:237] Iteration 6300, loss = 1.07867
    I1226 21:03:19.390004  6129 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1226 21:03:19.390020  6129 solver.cpp:253]     Train net output #1: loss = 1.07867 (* 1 = 1.07867 loss)
    I1226 21:03:19.390030  6129 sgd_solver.cpp:106] Iteration 6300, lr = 0.000485241
    I1226 21:03:29.917409  6129 solver.cpp:237] Iteration 6400, loss = 0.988683
    I1226 21:03:29.917464  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:03:29.917485  6129 solver.cpp:253]     Train net output #1: loss = 0.988683 (* 1 = 0.988683 loss)
    I1226 21:03:29.917501  6129 sgd_solver.cpp:106] Iteration 6400, lr = 0.00048302
    I1226 21:03:40.421913  6129 solver.cpp:237] Iteration 6500, loss = 1.04476
    I1226 21:03:40.421953  6129 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1226 21:03:40.421967  6129 solver.cpp:253]     Train net output #1: loss = 1.04476 (* 1 = 1.04476 loss)
    I1226 21:03:40.421978  6129 sgd_solver.cpp:106] Iteration 6500, lr = 0.000480823
    I1226 21:03:50.915287  6129 solver.cpp:237] Iteration 6600, loss = 1.07227
    I1226 21:03:50.915472  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:03:50.915488  6129 solver.cpp:253]     Train net output #1: loss = 1.07227 (* 1 = 1.07227 loss)
    I1226 21:03:50.915498  6129 sgd_solver.cpp:106] Iteration 6600, lr = 0.000478649
    I1226 21:04:01.429602  6129 solver.cpp:237] Iteration 6700, loss = 0.886793
    I1226 21:04:01.429643  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:04:01.429658  6129 solver.cpp:253]     Train net output #1: loss = 0.886793 (* 1 = 0.886793 loss)
    I1226 21:04:01.429669  6129 sgd_solver.cpp:106] Iteration 6700, lr = 0.000476498
    I1226 21:04:11.936102  6129 solver.cpp:237] Iteration 6800, loss = 1.02812
    I1226 21:04:11.936156  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:04:11.936178  6129 solver.cpp:253]     Train net output #1: loss = 1.02812 (* 1 = 1.02812 loss)
    I1226 21:04:11.936193  6129 sgd_solver.cpp:106] Iteration 6800, lr = 0.000474369
    I1226 21:04:22.478634  6129 solver.cpp:237] Iteration 6900, loss = 0.988804
    I1226 21:04:22.478812  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:04:22.478827  6129 solver.cpp:253]     Train net output #1: loss = 0.988804 (* 1 = 0.988804 loss)
    I1226 21:04:22.478835  6129 sgd_solver.cpp:106] Iteration 6900, lr = 0.000472262
    I1226 21:04:32.879061  6129 solver.cpp:341] Iteration 7000, Testing net (#0)
    I1226 21:04:37.179563  6129 solver.cpp:409]     Test net output #0: accuracy = 0.596917
    I1226 21:04:37.179608  6129 solver.cpp:409]     Test net output #1: loss = 1.15562 (* 1 = 1.15562 loss)
    I1226 21:04:37.224262  6129 solver.cpp:237] Iteration 7000, loss = 1.04525
    I1226 21:04:37.224285  6129 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1226 21:04:37.224298  6129 solver.cpp:253]     Train net output #1: loss = 1.04525 (* 1 = 1.04525 loss)
    I1226 21:04:37.224309  6129 sgd_solver.cpp:106] Iteration 7000, lr = 0.000470177
    I1226 21:04:47.732547  6129 solver.cpp:237] Iteration 7100, loss = 1.05863
    I1226 21:04:47.732592  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:04:47.732605  6129 solver.cpp:253]     Train net output #1: loss = 1.05863 (* 1 = 1.05863 loss)
    I1226 21:04:47.732614  6129 sgd_solver.cpp:106] Iteration 7100, lr = 0.000468113
    I1226 21:05:05.655910  6129 solver.cpp:237] Iteration 7200, loss = 0.851039
    I1226 21:05:05.656067  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:05:05.656082  6129 solver.cpp:253]     Train net output #1: loss = 0.851039 (* 1 = 0.851039 loss)
    I1226 21:05:05.656092  6129 sgd_solver.cpp:106] Iteration 7200, lr = 0.000466071
    I1226 21:05:16.661943  6129 solver.cpp:237] Iteration 7300, loss = 1.01962
    I1226 21:05:16.661977  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:05:16.661988  6129 solver.cpp:253]     Train net output #1: loss = 1.01962 (* 1 = 1.01962 loss)
    I1226 21:05:16.661998  6129 sgd_solver.cpp:106] Iteration 7300, lr = 0.000464049
    I1226 21:05:27.653597  6129 solver.cpp:237] Iteration 7400, loss = 0.960675
    I1226 21:05:27.653635  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:05:27.653650  6129 solver.cpp:253]     Train net output #1: loss = 0.960675 (* 1 = 0.960675 loss)
    I1226 21:05:27.653661  6129 sgd_solver.cpp:106] Iteration 7400, lr = 0.000462047
    I1226 21:05:38.618870  6129 solver.cpp:237] Iteration 7500, loss = 0.997356
    I1226 21:05:38.619022  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:05:38.619040  6129 solver.cpp:253]     Train net output #1: loss = 0.997356 (* 1 = 0.997356 loss)
    I1226 21:05:38.619052  6129 sgd_solver.cpp:106] Iteration 7500, lr = 0.000460065
    I1226 21:05:49.554896  6129 solver.cpp:237] Iteration 7600, loss = 1.0535
    I1226 21:05:49.554936  6129 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1226 21:05:49.554951  6129 solver.cpp:253]     Train net output #1: loss = 1.0535 (* 1 = 1.0535 loss)
    I1226 21:05:49.554963  6129 sgd_solver.cpp:106] Iteration 7600, lr = 0.000458103
    I1226 21:06:00.507771  6129 solver.cpp:237] Iteration 7700, loss = 0.784325
    I1226 21:06:00.507817  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:06:00.507829  6129 solver.cpp:253]     Train net output #1: loss = 0.784325 (* 1 = 0.784325 loss)
    I1226 21:06:00.507840  6129 sgd_solver.cpp:106] Iteration 7700, lr = 0.000456161
    I1226 21:06:11.552194  6129 solver.cpp:237] Iteration 7800, loss = 1.02386
    I1226 21:06:11.552319  6129 solver.cpp:253]     Train net output #0: accuracy = 0.64
    I1226 21:06:11.552337  6129 solver.cpp:253]     Train net output #1: loss = 1.02386 (* 1 = 1.02386 loss)
    I1226 21:06:11.552348  6129 sgd_solver.cpp:106] Iteration 7800, lr = 0.000454238
    I1226 21:06:22.541985  6129 solver.cpp:237] Iteration 7900, loss = 0.950034
    I1226 21:06:22.542037  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:06:22.542059  6129 solver.cpp:253]     Train net output #1: loss = 0.950034 (* 1 = 0.950034 loss)
    I1226 21:06:22.542075  6129 sgd_solver.cpp:106] Iteration 7900, lr = 0.000452333
    I1226 21:06:33.402127  6129 solver.cpp:341] Iteration 8000, Testing net (#0)
    I1226 21:06:37.898149  6129 solver.cpp:409]     Test net output #0: accuracy = 0.613333
    I1226 21:06:37.898185  6129 solver.cpp:409]     Test net output #1: loss = 1.09038 (* 1 = 1.09038 loss)
    I1226 21:06:37.942915  6129 solver.cpp:237] Iteration 8000, loss = 0.975661
    I1226 21:06:37.942960  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:06:37.942973  6129 solver.cpp:253]     Train net output #1: loss = 0.975661 (* 1 = 0.975661 loss)
    I1226 21:06:37.942986  6129 sgd_solver.cpp:106] Iteration 8000, lr = 0.000450447
    I1226 21:06:49.016008  6129 solver.cpp:237] Iteration 8100, loss = 1.03668
    I1226 21:06:49.016116  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:06:49.016134  6129 solver.cpp:253]     Train net output #1: loss = 1.03668 (* 1 = 1.03668 loss)
    I1226 21:06:49.016144  6129 sgd_solver.cpp:106] Iteration 8100, lr = 0.000448579
    I1226 21:07:00.065723  6129 solver.cpp:237] Iteration 8200, loss = 0.801996
    I1226 21:07:00.065774  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:07:00.065795  6129 solver.cpp:253]     Train net output #1: loss = 0.801996 (* 1 = 0.801996 loss)
    I1226 21:07:00.065814  6129 sgd_solver.cpp:106] Iteration 8200, lr = 0.000446729
    I1226 21:07:11.174825  6129 solver.cpp:237] Iteration 8300, loss = 1.02557
    I1226 21:07:11.174860  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:07:11.174872  6129 solver.cpp:253]     Train net output #1: loss = 1.02557 (* 1 = 1.02557 loss)
    I1226 21:07:11.174880  6129 sgd_solver.cpp:106] Iteration 8300, lr = 0.000444897
    I1226 21:07:22.196245  6129 solver.cpp:237] Iteration 8400, loss = 0.931184
    I1226 21:07:22.196357  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:07:22.196372  6129 solver.cpp:253]     Train net output #1: loss = 0.931184 (* 1 = 0.931184 loss)
    I1226 21:07:22.196382  6129 sgd_solver.cpp:106] Iteration 8400, lr = 0.000443083
    I1226 21:07:33.181789  6129 solver.cpp:237] Iteration 8500, loss = 1.07034
    I1226 21:07:33.181841  6129 solver.cpp:253]     Train net output #0: accuracy = 0.61
    I1226 21:07:33.181864  6129 solver.cpp:253]     Train net output #1: loss = 1.07034 (* 1 = 1.07034 loss)
    I1226 21:07:33.181879  6129 sgd_solver.cpp:106] Iteration 8500, lr = 0.000441285
    I1226 21:07:44.244998  6129 solver.cpp:237] Iteration 8600, loss = 1.03415
    I1226 21:07:44.245041  6129 solver.cpp:253]     Train net output #0: accuracy = 0.63
    I1226 21:07:44.245055  6129 solver.cpp:253]     Train net output #1: loss = 1.03415 (* 1 = 1.03415 loss)
    I1226 21:07:44.245065  6129 sgd_solver.cpp:106] Iteration 8600, lr = 0.000439505
    I1226 21:07:55.228271  6129 solver.cpp:237] Iteration 8700, loss = 0.784538
    I1226 21:07:55.228400  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:07:55.228415  6129 solver.cpp:253]     Train net output #1: loss = 0.784538 (* 1 = 0.784538 loss)
    I1226 21:07:55.228425  6129 sgd_solver.cpp:106] Iteration 8700, lr = 0.000437741
    I1226 21:08:06.207958  6129 solver.cpp:237] Iteration 8800, loss = 1.03228
    I1226 21:08:06.208011  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:08:06.208034  6129 solver.cpp:253]     Train net output #1: loss = 1.03228 (* 1 = 1.03228 loss)
    I1226 21:08:06.208050  6129 sgd_solver.cpp:106] Iteration 8800, lr = 0.000435993
    I1226 21:08:17.273957  6129 solver.cpp:237] Iteration 8900, loss = 0.902677
    I1226 21:08:17.273995  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:08:17.274010  6129 solver.cpp:253]     Train net output #1: loss = 0.902677 (* 1 = 0.902677 loss)
    I1226 21:08:17.274021  6129 sgd_solver.cpp:106] Iteration 8900, lr = 0.000434262
    I1226 21:08:28.226266  6129 solver.cpp:341] Iteration 9000, Testing net (#0)
    I1226 21:08:32.751677  6129 solver.cpp:409]     Test net output #0: accuracy = 0.597167
    I1226 21:08:32.751716  6129 solver.cpp:409]     Test net output #1: loss = 1.15735 (* 1 = 1.15735 loss)
    I1226 21:08:32.796164  6129 solver.cpp:237] Iteration 9000, loss = 1.02802
    I1226 21:08:32.796205  6129 solver.cpp:253]     Train net output #0: accuracy = 0.62
    I1226 21:08:32.796217  6129 solver.cpp:253]     Train net output #1: loss = 1.02802 (* 1 = 1.02802 loss)
    I1226 21:08:32.796227  6129 sgd_solver.cpp:106] Iteration 9000, lr = 0.000432547
    I1226 21:08:43.916416  6129 solver.cpp:237] Iteration 9100, loss = 0.998299
    I1226 21:08:43.916458  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:08:43.916472  6129 solver.cpp:253]     Train net output #1: loss = 0.998299 (* 1 = 0.998299 loss)
    I1226 21:08:43.916483  6129 sgd_solver.cpp:106] Iteration 9100, lr = 0.000430847
    I1226 21:08:54.893193  6129 solver.cpp:237] Iteration 9200, loss = 0.76632
    I1226 21:08:54.893237  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:08:54.893256  6129 solver.cpp:253]     Train net output #1: loss = 0.76632 (* 1 = 0.76632 loss)
    I1226 21:08:54.893270  6129 sgd_solver.cpp:106] Iteration 9200, lr = 0.000429163
    I1226 21:09:05.926028  6129 solver.cpp:237] Iteration 9300, loss = 0.982026
    I1226 21:09:05.926152  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:09:05.926174  6129 solver.cpp:253]     Train net output #1: loss = 0.982026 (* 1 = 0.982026 loss)
    I1226 21:09:05.926189  6129 sgd_solver.cpp:106] Iteration 9300, lr = 0.000427494
    I1226 21:09:16.965981  6129 solver.cpp:237] Iteration 9400, loss = 0.924036
    I1226 21:09:16.966022  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:09:16.966037  6129 solver.cpp:253]     Train net output #1: loss = 0.924036 (* 1 = 0.924036 loss)
    I1226 21:09:16.966048  6129 sgd_solver.cpp:106] Iteration 9400, lr = 0.00042584
    I1226 21:09:27.981384  6129 solver.cpp:237] Iteration 9500, loss = 0.940432
    I1226 21:09:27.981425  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:09:27.981438  6129 solver.cpp:253]     Train net output #1: loss = 0.940432 (* 1 = 0.940432 loss)
    I1226 21:09:27.981451  6129 sgd_solver.cpp:106] Iteration 9500, lr = 0.000424201
    I1226 21:09:38.924898  6129 solver.cpp:237] Iteration 9600, loss = 0.98112
    I1226 21:09:38.925017  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:09:38.925040  6129 solver.cpp:253]     Train net output #1: loss = 0.98112 (* 1 = 0.98112 loss)
    I1226 21:09:38.925053  6129 sgd_solver.cpp:106] Iteration 9600, lr = 0.000422577
    I1226 21:09:49.905697  6129 solver.cpp:237] Iteration 9700, loss = 0.7489
    I1226 21:09:49.905735  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:09:49.905748  6129 solver.cpp:253]     Train net output #1: loss = 0.7489 (* 1 = 0.7489 loss)
    I1226 21:09:49.905760  6129 sgd_solver.cpp:106] Iteration 9700, lr = 0.000420967
    I1226 21:10:00.897140  6129 solver.cpp:237] Iteration 9800, loss = 0.972466
    I1226 21:10:00.897176  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:10:00.897191  6129 solver.cpp:253]     Train net output #1: loss = 0.972466 (* 1 = 0.972466 loss)
    I1226 21:10:00.897200  6129 sgd_solver.cpp:106] Iteration 9800, lr = 0.000419372
    I1226 21:10:11.863248  6129 solver.cpp:237] Iteration 9900, loss = 0.884251
    I1226 21:10:11.863365  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:10:11.863384  6129 solver.cpp:253]     Train net output #1: loss = 0.884251 (* 1 = 0.884251 loss)
    I1226 21:10:11.863394  6129 sgd_solver.cpp:106] Iteration 9900, lr = 0.00041779
    I1226 21:10:22.806237  6129 solver.cpp:341] Iteration 10000, Testing net (#0)
    I1226 21:10:27.290216  6129 solver.cpp:409]     Test net output #0: accuracy = 0.632083
    I1226 21:10:27.290258  6129 solver.cpp:409]     Test net output #1: loss = 1.04692 (* 1 = 1.04692 loss)
    I1226 21:10:27.334764  6129 solver.cpp:237] Iteration 10000, loss = 0.931082
    I1226 21:10:27.334811  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:10:27.334825  6129 solver.cpp:253]     Train net output #1: loss = 0.931082 (* 1 = 0.931082 loss)
    I1226 21:10:27.334839  6129 sgd_solver.cpp:106] Iteration 10000, lr = 0.000416222
    I1226 21:10:38.466285  6129 solver.cpp:237] Iteration 10100, loss = 0.9831
    I1226 21:10:38.466322  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:10:38.466336  6129 solver.cpp:253]     Train net output #1: loss = 0.9831 (* 1 = 0.9831 loss)
    I1226 21:10:38.466347  6129 sgd_solver.cpp:106] Iteration 10100, lr = 0.000414668
    I1226 21:10:49.489186  6129 solver.cpp:237] Iteration 10200, loss = 0.748263
    I1226 21:10:49.489331  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:10:49.489353  6129 solver.cpp:253]     Train net output #1: loss = 0.748263 (* 1 = 0.748263 loss)
    I1226 21:10:49.489367  6129 sgd_solver.cpp:106] Iteration 10200, lr = 0.000413128
    I1226 21:11:00.573035  6129 solver.cpp:237] Iteration 10300, loss = 0.982309
    I1226 21:11:00.573073  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:11:00.573087  6129 solver.cpp:253]     Train net output #1: loss = 0.982309 (* 1 = 0.982309 loss)
    I1226 21:11:00.573099  6129 sgd_solver.cpp:106] Iteration 10300, lr = 0.000411601
    I1226 21:11:11.508268  6129 solver.cpp:237] Iteration 10400, loss = 0.885086
    I1226 21:11:11.508306  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:11:11.508321  6129 solver.cpp:253]     Train net output #1: loss = 0.885086 (* 1 = 0.885086 loss)
    I1226 21:11:11.508329  6129 sgd_solver.cpp:106] Iteration 10400, lr = 0.000410086
    I1226 21:11:23.008344  6129 solver.cpp:237] Iteration 10500, loss = 0.914752
    I1226 21:11:23.008460  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:11:23.008478  6129 solver.cpp:253]     Train net output #1: loss = 0.914752 (* 1 = 0.914752 loss)
    I1226 21:11:23.008491  6129 sgd_solver.cpp:106] Iteration 10500, lr = 0.000408585
    I1226 21:11:33.973565  6129 solver.cpp:237] Iteration 10600, loss = 0.976193
    I1226 21:11:33.973608  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:11:33.973621  6129 solver.cpp:253]     Train net output #1: loss = 0.976193 (* 1 = 0.976193 loss)
    I1226 21:11:33.973634  6129 sgd_solver.cpp:106] Iteration 10600, lr = 0.000407097
    I1226 21:11:44.889214  6129 solver.cpp:237] Iteration 10700, loss = 0.722384
    I1226 21:11:44.889251  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:11:44.889266  6129 solver.cpp:253]     Train net output #1: loss = 0.722384 (* 1 = 0.722384 loss)
    I1226 21:11:44.889276  6129 sgd_solver.cpp:106] Iteration 10700, lr = 0.000405621
    I1226 21:11:55.962420  6129 solver.cpp:237] Iteration 10800, loss = 0.921905
    I1226 21:11:55.962535  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:11:55.962551  6129 solver.cpp:253]     Train net output #1: loss = 0.921905 (* 1 = 0.921905 loss)
    I1226 21:11:55.962564  6129 sgd_solver.cpp:106] Iteration 10800, lr = 0.000404157
    I1226 21:12:06.880040  6129 solver.cpp:237] Iteration 10900, loss = 0.86868
    I1226 21:12:06.880084  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:12:06.880095  6129 solver.cpp:253]     Train net output #1: loss = 0.86868 (* 1 = 0.86868 loss)
    I1226 21:12:06.880103  6129 sgd_solver.cpp:106] Iteration 10900, lr = 0.000402706
    I1226 21:12:17.785290  6129 solver.cpp:341] Iteration 11000, Testing net (#0)
    I1226 21:12:22.321548  6129 solver.cpp:409]     Test net output #0: accuracy = 0.634
    I1226 21:12:22.321604  6129 solver.cpp:409]     Test net output #1: loss = 1.0346 (* 1 = 1.0346 loss)
    I1226 21:12:22.366338  6129 solver.cpp:237] Iteration 11000, loss = 0.909143
    I1226 21:12:22.366371  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:12:22.366391  6129 solver.cpp:253]     Train net output #1: loss = 0.909143 (* 1 = 0.909143 loss)
    I1226 21:12:22.366408  6129 sgd_solver.cpp:106] Iteration 11000, lr = 0.000401267
    I1226 21:12:33.298256  6129 solver.cpp:237] Iteration 11100, loss = 0.972628
    I1226 21:12:33.298431  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:12:33.298450  6129 solver.cpp:253]     Train net output #1: loss = 0.972628 (* 1 = 0.972628 loss)
    I1226 21:12:33.298461  6129 sgd_solver.cpp:106] Iteration 11100, lr = 0.00039984
    I1226 21:12:44.217905  6129 solver.cpp:237] Iteration 11200, loss = 0.741822
    I1226 21:12:44.217937  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:12:44.217949  6129 solver.cpp:253]     Train net output #1: loss = 0.741822 (* 1 = 0.741822 loss)
    I1226 21:12:44.217957  6129 sgd_solver.cpp:106] Iteration 11200, lr = 0.000398425
    I1226 21:12:55.167465  6129 solver.cpp:237] Iteration 11300, loss = 0.941473
    I1226 21:12:55.167500  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:12:55.167511  6129 solver.cpp:253]     Train net output #1: loss = 0.941473 (* 1 = 0.941473 loss)
    I1226 21:12:55.167520  6129 sgd_solver.cpp:106] Iteration 11300, lr = 0.000397021
    I1226 21:13:06.163977  6129 solver.cpp:237] Iteration 11400, loss = 0.856937
    I1226 21:13:06.164126  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:13:06.164151  6129 solver.cpp:253]     Train net output #1: loss = 0.856937 (* 1 = 0.856937 loss)
    I1226 21:13:06.164167  6129 sgd_solver.cpp:106] Iteration 11400, lr = 0.000395629
    I1226 21:13:17.108952  6129 solver.cpp:237] Iteration 11500, loss = 0.894247
    I1226 21:13:17.109000  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:13:17.109015  6129 solver.cpp:253]     Train net output #1: loss = 0.894247 (* 1 = 0.894247 loss)
    I1226 21:13:17.109028  6129 sgd_solver.cpp:106] Iteration 11500, lr = 0.000394248
    I1226 21:13:28.102768  6129 solver.cpp:237] Iteration 11600, loss = 0.96387
    I1226 21:13:28.102818  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:13:28.102838  6129 solver.cpp:253]     Train net output #1: loss = 0.96387 (* 1 = 0.96387 loss)
    I1226 21:13:28.102854  6129 sgd_solver.cpp:106] Iteration 11600, lr = 0.000392878
    I1226 21:13:39.125912  6129 solver.cpp:237] Iteration 11700, loss = 0.701087
    I1226 21:13:39.126066  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:13:39.126087  6129 solver.cpp:253]     Train net output #1: loss = 0.701087 (* 1 = 0.701087 loss)
    I1226 21:13:39.126099  6129 sgd_solver.cpp:106] Iteration 11700, lr = 0.000391519
    I1226 21:13:50.063604  6129 solver.cpp:237] Iteration 11800, loss = 0.91355
    I1226 21:13:50.063644  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:13:50.063659  6129 solver.cpp:253]     Train net output #1: loss = 0.91355 (* 1 = 0.91355 loss)
    I1226 21:13:50.063671  6129 sgd_solver.cpp:106] Iteration 11800, lr = 0.000390172
    I1226 21:14:01.128732  6129 solver.cpp:237] Iteration 11900, loss = 0.849595
    I1226 21:14:01.128769  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:14:01.128783  6129 solver.cpp:253]     Train net output #1: loss = 0.849595 (* 1 = 0.849595 loss)
    I1226 21:14:01.128794  6129 sgd_solver.cpp:106] Iteration 11900, lr = 0.000388835
    I1226 21:14:11.952973  6129 solver.cpp:341] Iteration 12000, Testing net (#0)
    I1226 21:14:16.559238  6129 solver.cpp:409]     Test net output #0: accuracy = 0.642167
    I1226 21:14:16.559284  6129 solver.cpp:409]     Test net output #1: loss = 1.01173 (* 1 = 1.01173 loss)
    I1226 21:14:16.603924  6129 solver.cpp:237] Iteration 12000, loss = 0.883077
    I1226 21:14:16.603963  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:14:16.603977  6129 solver.cpp:253]     Train net output #1: loss = 0.883077 (* 1 = 0.883077 loss)
    I1226 21:14:16.603988  6129 sgd_solver.cpp:106] Iteration 12000, lr = 0.000387508
    I1226 21:14:27.624681  6129 solver.cpp:237] Iteration 12100, loss = 0.94701
    I1226 21:14:27.624727  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:14:27.624739  6129 solver.cpp:253]     Train net output #1: loss = 0.94701 (* 1 = 0.94701 loss)
    I1226 21:14:27.624749  6129 sgd_solver.cpp:106] Iteration 12100, lr = 0.000386192
    I1226 21:14:38.592070  6129 solver.cpp:237] Iteration 12200, loss = 0.730384
    I1226 21:14:38.592116  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:14:38.592135  6129 solver.cpp:253]     Train net output #1: loss = 0.730384 (* 1 = 0.730384 loss)
    I1226 21:14:38.592151  6129 sgd_solver.cpp:106] Iteration 12200, lr = 0.000384887
    I1226 21:14:49.492880  6129 solver.cpp:237] Iteration 12300, loss = 0.903743
    I1226 21:14:49.493046  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:14:49.493060  6129 solver.cpp:253]     Train net output #1: loss = 0.903743 (* 1 = 0.903743 loss)
    I1226 21:14:49.493069  6129 sgd_solver.cpp:106] Iteration 12300, lr = 0.000383592
    I1226 21:15:00.606415  6129 solver.cpp:237] Iteration 12400, loss = 0.837378
    I1226 21:15:00.606453  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:15:00.606467  6129 solver.cpp:253]     Train net output #1: loss = 0.837378 (* 1 = 0.837378 loss)
    I1226 21:15:00.606479  6129 sgd_solver.cpp:106] Iteration 12400, lr = 0.000382307
    I1226 21:15:11.610438  6129 solver.cpp:237] Iteration 12500, loss = 0.864022
    I1226 21:15:11.610473  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:15:11.610484  6129 solver.cpp:253]     Train net output #1: loss = 0.864022 (* 1 = 0.864022 loss)
    I1226 21:15:11.610493  6129 sgd_solver.cpp:106] Iteration 12500, lr = 0.000381032
    I1226 21:15:22.628337  6129 solver.cpp:237] Iteration 12600, loss = 0.934401
    I1226 21:15:22.628468  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:15:22.628491  6129 solver.cpp:253]     Train net output #1: loss = 0.934401 (* 1 = 0.934401 loss)
    I1226 21:15:22.628501  6129 sgd_solver.cpp:106] Iteration 12600, lr = 0.000379767
    I1226 21:15:33.636432  6129 solver.cpp:237] Iteration 12700, loss = 0.689688
    I1226 21:15:33.636476  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:15:33.636490  6129 solver.cpp:253]     Train net output #1: loss = 0.689688 (* 1 = 0.689688 loss)
    I1226 21:15:33.636502  6129 sgd_solver.cpp:106] Iteration 12700, lr = 0.000378511
    I1226 21:15:44.508950  6129 solver.cpp:237] Iteration 12800, loss = 0.904487
    I1226 21:15:44.508982  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:15:44.508994  6129 solver.cpp:253]     Train net output #1: loss = 0.904487 (* 1 = 0.904487 loss)
    I1226 21:15:44.509004  6129 sgd_solver.cpp:106] Iteration 12800, lr = 0.000377265
    I1226 21:15:55.513583  6129 solver.cpp:237] Iteration 12900, loss = 0.825195
    I1226 21:15:55.513731  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:15:55.513757  6129 solver.cpp:253]     Train net output #1: loss = 0.825195 (* 1 = 0.825195 loss)
    I1226 21:15:55.513767  6129 sgd_solver.cpp:106] Iteration 12900, lr = 0.000376029
    I1226 21:16:06.661746  6129 solver.cpp:341] Iteration 13000, Testing net (#0)
    I1226 21:16:11.178076  6129 solver.cpp:409]     Test net output #0: accuracy = 0.65025
    I1226 21:16:11.178123  6129 solver.cpp:409]     Test net output #1: loss = 0.995486 (* 1 = 0.995486 loss)
    I1226 21:16:11.222568  6129 solver.cpp:237] Iteration 13000, loss = 0.856003
    I1226 21:16:11.222627  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:16:11.222642  6129 solver.cpp:253]     Train net output #1: loss = 0.856003 (* 1 = 0.856003 loss)
    I1226 21:16:11.222654  6129 sgd_solver.cpp:106] Iteration 13000, lr = 0.000374802
    I1226 21:16:22.303505  6129 solver.cpp:237] Iteration 13100, loss = 0.917757
    I1226 21:16:22.303541  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:16:22.303553  6129 solver.cpp:253]     Train net output #1: loss = 0.917757 (* 1 = 0.917757 loss)
    I1226 21:16:22.303562  6129 sgd_solver.cpp:106] Iteration 13100, lr = 0.000373585
    I1226 21:16:33.325016  6129 solver.cpp:237] Iteration 13200, loss = 0.691707
    I1226 21:16:33.325189  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:16:33.325214  6129 solver.cpp:253]     Train net output #1: loss = 0.691707 (* 1 = 0.691707 loss)
    I1226 21:16:33.325224  6129 sgd_solver.cpp:106] Iteration 13200, lr = 0.000372376
    I1226 21:16:44.330808  6129 solver.cpp:237] Iteration 13300, loss = 0.871964
    I1226 21:16:44.330850  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:16:44.330864  6129 solver.cpp:253]     Train net output #1: loss = 0.871964 (* 1 = 0.871964 loss)
    I1226 21:16:44.330876  6129 sgd_solver.cpp:106] Iteration 13300, lr = 0.000371177
    I1226 21:16:55.296227  6129 solver.cpp:237] Iteration 13400, loss = 0.832162
    I1226 21:16:55.296272  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:16:55.296283  6129 solver.cpp:253]     Train net output #1: loss = 0.832162 (* 1 = 0.832162 loss)
    I1226 21:16:55.296293  6129 sgd_solver.cpp:106] Iteration 13400, lr = 0.000369987
    I1226 21:17:06.308148  6129 solver.cpp:237] Iteration 13500, loss = 0.841876
    I1226 21:17:06.308341  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:17:06.308357  6129 solver.cpp:253]     Train net output #1: loss = 0.841876 (* 1 = 0.841876 loss)
    I1226 21:17:06.308367  6129 sgd_solver.cpp:106] Iteration 13500, lr = 0.000368805
    I1226 21:17:17.244760  6129 solver.cpp:237] Iteration 13600, loss = 0.903397
    I1226 21:17:17.244803  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:17:17.244815  6129 solver.cpp:253]     Train net output #1: loss = 0.903397 (* 1 = 0.903397 loss)
    I1226 21:17:17.244823  6129 sgd_solver.cpp:106] Iteration 13600, lr = 0.000367633
    I1226 21:17:28.249732  6129 solver.cpp:237] Iteration 13700, loss = 0.721655
    I1226 21:17:28.249774  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:17:28.249788  6129 solver.cpp:253]     Train net output #1: loss = 0.721655 (* 1 = 0.721655 loss)
    I1226 21:17:28.249800  6129 sgd_solver.cpp:106] Iteration 13700, lr = 0.000366469
    I1226 21:17:39.230890  6129 solver.cpp:237] Iteration 13800, loss = 0.84816
    I1226 21:17:39.231035  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:17:39.231060  6129 solver.cpp:253]     Train net output #1: loss = 0.84816 (* 1 = 0.84816 loss)
    I1226 21:17:39.231068  6129 sgd_solver.cpp:106] Iteration 13800, lr = 0.000365313
    I1226 21:17:50.204887  6129 solver.cpp:237] Iteration 13900, loss = 0.809601
    I1226 21:17:50.204923  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:17:50.204936  6129 solver.cpp:253]     Train net output #1: loss = 0.809601 (* 1 = 0.809601 loss)
    I1226 21:17:50.204943  6129 sgd_solver.cpp:106] Iteration 13900, lr = 0.000364166
    I1226 21:18:01.184556  6129 solver.cpp:341] Iteration 14000, Testing net (#0)
    I1226 21:18:05.720937  6129 solver.cpp:409]     Test net output #0: accuracy = 0.657417
    I1226 21:18:05.720978  6129 solver.cpp:409]     Test net output #1: loss = 0.977707 (* 1 = 0.977707 loss)
    I1226 21:18:05.765450  6129 solver.cpp:237] Iteration 14000, loss = 0.829052
    I1226 21:18:05.765503  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:18:05.765525  6129 solver.cpp:253]     Train net output #1: loss = 0.829052 (* 1 = 0.829052 loss)
    I1226 21:18:05.765543  6129 sgd_solver.cpp:106] Iteration 14000, lr = 0.000363028
    I1226 21:18:16.850920  6129 solver.cpp:237] Iteration 14100, loss = 0.907661
    I1226 21:18:16.851012  6129 solver.cpp:253]     Train net output #0: accuracy = 0.65
    I1226 21:18:16.851030  6129 solver.cpp:253]     Train net output #1: loss = 0.907661 (* 1 = 0.907661 loss)
    I1226 21:18:16.851042  6129 sgd_solver.cpp:106] Iteration 14100, lr = 0.000361897
    I1226 21:18:27.798041  6129 solver.cpp:237] Iteration 14200, loss = 0.67552
    I1226 21:18:27.798077  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 21:18:27.798089  6129 solver.cpp:253]     Train net output #1: loss = 0.67552 (* 1 = 0.67552 loss)
    I1226 21:18:27.798099  6129 sgd_solver.cpp:106] Iteration 14200, lr = 0.000360775
    I1226 21:18:38.710253  6129 solver.cpp:237] Iteration 14300, loss = 0.831352
    I1226 21:18:38.710297  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:18:38.710309  6129 solver.cpp:253]     Train net output #1: loss = 0.831352 (* 1 = 0.831352 loss)
    I1226 21:18:38.710317  6129 sgd_solver.cpp:106] Iteration 14300, lr = 0.000359661
    I1226 21:18:49.705095  6129 solver.cpp:237] Iteration 14400, loss = 0.80052
    I1226 21:18:49.705235  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:18:49.705253  6129 solver.cpp:253]     Train net output #1: loss = 0.80052 (* 1 = 0.80052 loss)
    I1226 21:18:49.705265  6129 sgd_solver.cpp:106] Iteration 14400, lr = 0.000358555
    I1226 21:19:00.679250  6129 solver.cpp:237] Iteration 14500, loss = 0.82136
    I1226 21:19:00.679298  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:19:00.679311  6129 solver.cpp:253]     Train net output #1: loss = 0.82136 (* 1 = 0.82136 loss)
    I1226 21:19:00.679322  6129 sgd_solver.cpp:106] Iteration 14500, lr = 0.000357457
    I1226 21:19:11.644748  6129 solver.cpp:237] Iteration 14600, loss = 0.908905
    I1226 21:19:11.644784  6129 solver.cpp:253]     Train net output #0: accuracy = 0.66
    I1226 21:19:11.644798  6129 solver.cpp:253]     Train net output #1: loss = 0.908905 (* 1 = 0.908905 loss)
    I1226 21:19:11.644807  6129 sgd_solver.cpp:106] Iteration 14600, lr = 0.000356366
    I1226 21:19:22.642206  6129 solver.cpp:237] Iteration 14700, loss = 0.690218
    I1226 21:19:22.642345  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:19:22.642369  6129 solver.cpp:253]     Train net output #1: loss = 0.690218 (* 1 = 0.690218 loss)
    I1226 21:19:22.642379  6129 sgd_solver.cpp:106] Iteration 14700, lr = 0.000355284
    I1226 21:19:33.592031  6129 solver.cpp:237] Iteration 14800, loss = 0.814543
    I1226 21:19:33.592073  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:19:33.592088  6129 solver.cpp:253]     Train net output #1: loss = 0.814543 (* 1 = 0.814543 loss)
    I1226 21:19:33.592100  6129 sgd_solver.cpp:106] Iteration 14800, lr = 0.000354209
    I1226 21:19:44.558639  6129 solver.cpp:237] Iteration 14900, loss = 0.800524
    I1226 21:19:44.558681  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:19:44.558696  6129 solver.cpp:253]     Train net output #1: loss = 0.800524 (* 1 = 0.800524 loss)
    I1226 21:19:44.558708  6129 sgd_solver.cpp:106] Iteration 14900, lr = 0.000353141
    I1226 21:19:55.418452  6129 solver.cpp:341] Iteration 15000, Testing net (#0)
    I1226 21:19:59.887343  6129 solver.cpp:409]     Test net output #0: accuracy = 0.663667
    I1226 21:19:59.887390  6129 solver.cpp:409]     Test net output #1: loss = 0.96036 (* 1 = 0.96036 loss)
    I1226 21:19:59.957293  6129 solver.cpp:237] Iteration 15000, loss = 0.810295
    I1226 21:19:59.957341  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:19:59.957355  6129 solver.cpp:253]     Train net output #1: loss = 0.810295 (* 1 = 0.810295 loss)
    I1226 21:19:59.957365  6129 sgd_solver.cpp:106] Iteration 15000, lr = 0.000352081
    I1226 21:20:10.997361  6129 solver.cpp:237] Iteration 15100, loss = 0.891971
    I1226 21:20:10.997406  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:20:10.997422  6129 solver.cpp:253]     Train net output #1: loss = 0.891971 (* 1 = 0.891971 loss)
    I1226 21:20:10.997434  6129 sgd_solver.cpp:106] Iteration 15100, lr = 0.000351029
    I1226 21:20:24.544417  6129 solver.cpp:237] Iteration 15200, loss = 0.673745
    I1226 21:20:24.544472  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:20:24.544487  6129 solver.cpp:253]     Train net output #1: loss = 0.673745 (* 1 = 0.673745 loss)
    I1226 21:20:24.544497  6129 sgd_solver.cpp:106] Iteration 15200, lr = 0.000349984
    I1226 21:20:35.272665  6129 solver.cpp:237] Iteration 15300, loss = 0.810382
    I1226 21:20:35.272786  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:20:35.272805  6129 solver.cpp:253]     Train net output #1: loss = 0.810382 (* 1 = 0.810382 loss)
    I1226 21:20:35.272816  6129 sgd_solver.cpp:106] Iteration 15300, lr = 0.000348946
    I1226 21:20:46.023255  6129 solver.cpp:237] Iteration 15400, loss = 0.811863
    I1226 21:20:46.023295  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:20:46.023309  6129 solver.cpp:253]     Train net output #1: loss = 0.811863 (* 1 = 0.811863 loss)
    I1226 21:20:46.023321  6129 sgd_solver.cpp:106] Iteration 15400, lr = 0.000347915
    I1226 21:20:56.517411  6129 solver.cpp:237] Iteration 15500, loss = 0.804951
    I1226 21:20:56.517460  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:20:56.517475  6129 solver.cpp:253]     Train net output #1: loss = 0.804951 (* 1 = 0.804951 loss)
    I1226 21:20:56.517487  6129 sgd_solver.cpp:106] Iteration 15500, lr = 0.000346891
    I1226 21:21:07.029353  6129 solver.cpp:237] Iteration 15600, loss = 0.888472
    I1226 21:21:07.029543  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:21:07.029558  6129 solver.cpp:253]     Train net output #1: loss = 0.888472 (* 1 = 0.888472 loss)
    I1226 21:21:07.029567  6129 sgd_solver.cpp:106] Iteration 15600, lr = 0.000345874
    I1226 21:21:17.523074  6129 solver.cpp:237] Iteration 15700, loss = 0.683455
    I1226 21:21:17.523118  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:21:17.523130  6129 solver.cpp:253]     Train net output #1: loss = 0.683455 (* 1 = 0.683455 loss)
    I1226 21:21:17.523141  6129 sgd_solver.cpp:106] Iteration 15700, lr = 0.000344864
    I1226 21:21:27.992673  6129 solver.cpp:237] Iteration 15800, loss = 0.790901
    I1226 21:21:27.992712  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:21:27.992723  6129 solver.cpp:253]     Train net output #1: loss = 0.790901 (* 1 = 0.790901 loss)
    I1226 21:21:27.992733  6129 sgd_solver.cpp:106] Iteration 15800, lr = 0.000343861
    I1226 21:21:38.484661  6129 solver.cpp:237] Iteration 15900, loss = 0.79026
    I1226 21:21:38.484819  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:21:38.484844  6129 solver.cpp:253]     Train net output #1: loss = 0.79026 (* 1 = 0.79026 loss)
    I1226 21:21:38.484853  6129 sgd_solver.cpp:106] Iteration 15900, lr = 0.000342865
    I1226 21:21:48.844409  6129 solver.cpp:341] Iteration 16000, Testing net (#0)
    I1226 21:21:53.121953  6129 solver.cpp:409]     Test net output #0: accuracy = 0.663
    I1226 21:21:53.122002  6129 solver.cpp:409]     Test net output #1: loss = 0.959659 (* 1 = 0.959659 loss)
    I1226 21:21:53.182926  6129 solver.cpp:237] Iteration 16000, loss = 0.797464
    I1226 21:21:53.182978  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:21:53.182992  6129 solver.cpp:253]     Train net output #1: loss = 0.797464 (* 1 = 0.797464 loss)
    I1226 21:21:53.183003  6129 sgd_solver.cpp:106] Iteration 16000, lr = 0.000341876
    I1226 21:22:03.744602  6129 solver.cpp:237] Iteration 16100, loss = 0.881778
    I1226 21:22:03.744639  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:22:03.744652  6129 solver.cpp:253]     Train net output #1: loss = 0.881778 (* 1 = 0.881778 loss)
    I1226 21:22:03.744659  6129 sgd_solver.cpp:106] Iteration 16100, lr = 0.000340893
    I1226 21:22:14.194248  6129 solver.cpp:237] Iteration 16200, loss = 0.675328
    I1226 21:22:14.194402  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:22:14.194427  6129 solver.cpp:253]     Train net output #1: loss = 0.675328 (* 1 = 0.675328 loss)
    I1226 21:22:14.194434  6129 sgd_solver.cpp:106] Iteration 16200, lr = 0.000339916
    I1226 21:22:24.682219  6129 solver.cpp:237] Iteration 16300, loss = 0.781637
    I1226 21:22:24.682276  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:22:24.682296  6129 solver.cpp:253]     Train net output #1: loss = 0.781637 (* 1 = 0.781637 loss)
    I1226 21:22:24.682312  6129 sgd_solver.cpp:106] Iteration 16300, lr = 0.000338947
    I1226 21:22:35.162775  6129 solver.cpp:237] Iteration 16400, loss = 0.780941
    I1226 21:22:35.162817  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:22:35.162830  6129 solver.cpp:253]     Train net output #1: loss = 0.780941 (* 1 = 0.780941 loss)
    I1226 21:22:35.162840  6129 sgd_solver.cpp:106] Iteration 16400, lr = 0.000337983
    I1226 21:22:45.643823  6129 solver.cpp:237] Iteration 16500, loss = 0.796348
    I1226 21:22:45.643995  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:22:45.644019  6129 solver.cpp:253]     Train net output #1: loss = 0.796348 (* 1 = 0.796348 loss)
    I1226 21:22:45.644027  6129 sgd_solver.cpp:106] Iteration 16500, lr = 0.000337026
    I1226 21:22:56.130159  6129 solver.cpp:237] Iteration 16600, loss = 0.872012
    I1226 21:22:56.130197  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:22:56.130208  6129 solver.cpp:253]     Train net output #1: loss = 0.872012 (* 1 = 0.872012 loss)
    I1226 21:22:56.130216  6129 sgd_solver.cpp:106] Iteration 16600, lr = 0.000336075
    I1226 21:23:06.656844  6129 solver.cpp:237] Iteration 16700, loss = 0.667306
    I1226 21:23:06.656886  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:23:06.656901  6129 solver.cpp:253]     Train net output #1: loss = 0.667306 (* 1 = 0.667306 loss)
    I1226 21:23:06.656911  6129 sgd_solver.cpp:106] Iteration 16700, lr = 0.000335131
    I1226 21:23:17.150861  6129 solver.cpp:237] Iteration 16800, loss = 0.775927
    I1226 21:23:17.150974  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:23:17.150991  6129 solver.cpp:253]     Train net output #1: loss = 0.775927 (* 1 = 0.775927 loss)
    I1226 21:23:17.151002  6129 sgd_solver.cpp:106] Iteration 16800, lr = 0.000334193
    I1226 21:23:27.636080  6129 solver.cpp:237] Iteration 16900, loss = 0.805485
    I1226 21:23:27.636124  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:23:27.636135  6129 solver.cpp:253]     Train net output #1: loss = 0.805485 (* 1 = 0.805485 loss)
    I1226 21:23:27.636145  6129 sgd_solver.cpp:106] Iteration 16900, lr = 0.00033326
    I1226 21:23:38.001014  6129 solver.cpp:341] Iteration 17000, Testing net (#0)
    I1226 21:23:42.264189  6129 solver.cpp:409]     Test net output #0: accuracy = 0.666
    I1226 21:23:42.264236  6129 solver.cpp:409]     Test net output #1: loss = 0.95026 (* 1 = 0.95026 loss)
    I1226 21:23:42.308768  6129 solver.cpp:237] Iteration 17000, loss = 0.795679
    I1226 21:23:42.308818  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:23:42.308832  6129 solver.cpp:253]     Train net output #1: loss = 0.795679 (* 1 = 0.795679 loss)
    I1226 21:23:42.308843  6129 sgd_solver.cpp:106] Iteration 17000, lr = 0.000332334
    I1226 21:23:52.799666  6129 solver.cpp:237] Iteration 17100, loss = 0.87014
    I1226 21:23:52.799805  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:23:52.799818  6129 solver.cpp:253]     Train net output #1: loss = 0.87014 (* 1 = 0.87014 loss)
    I1226 21:23:52.799828  6129 sgd_solver.cpp:106] Iteration 17100, lr = 0.000331414
    I1226 21:24:03.347997  6129 solver.cpp:237] Iteration 17200, loss = 0.685858
    I1226 21:24:03.348036  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:24:03.348047  6129 solver.cpp:253]     Train net output #1: loss = 0.685858 (* 1 = 0.685858 loss)
    I1226 21:24:03.348055  6129 sgd_solver.cpp:106] Iteration 17200, lr = 0.0003305
    I1226 21:24:13.828887  6129 solver.cpp:237] Iteration 17300, loss = 0.776166
    I1226 21:24:13.828923  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:24:13.828935  6129 solver.cpp:253]     Train net output #1: loss = 0.776166 (* 1 = 0.776166 loss)
    I1226 21:24:13.828943  6129 sgd_solver.cpp:106] Iteration 17300, lr = 0.000329592
    I1226 21:24:24.322463  6129 solver.cpp:237] Iteration 17400, loss = 0.797402
    I1226 21:24:24.322609  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:24:24.322623  6129 solver.cpp:253]     Train net output #1: loss = 0.797402 (* 1 = 0.797402 loss)
    I1226 21:24:24.322633  6129 sgd_solver.cpp:106] Iteration 17400, lr = 0.000328689
    I1226 21:24:34.792454  6129 solver.cpp:237] Iteration 17500, loss = 0.791896
    I1226 21:24:34.792500  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:24:34.792511  6129 solver.cpp:253]     Train net output #1: loss = 0.791896 (* 1 = 0.791896 loss)
    I1226 21:24:34.792520  6129 sgd_solver.cpp:106] Iteration 17500, lr = 0.000327792
    I1226 21:24:45.737306  6129 solver.cpp:237] Iteration 17600, loss = 0.879224
    I1226 21:24:45.737354  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:24:45.737365  6129 solver.cpp:253]     Train net output #1: loss = 0.879224 (* 1 = 0.879224 loss)
    I1226 21:24:45.737377  6129 sgd_solver.cpp:106] Iteration 17600, lr = 0.000326901
    I1226 21:24:57.896863  6129 solver.cpp:237] Iteration 17700, loss = 0.684026
    I1226 21:24:57.897006  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:24:57.897024  6129 solver.cpp:253]     Train net output #1: loss = 0.684026 (* 1 = 0.684026 loss)
    I1226 21:24:57.897035  6129 sgd_solver.cpp:106] Iteration 17700, lr = 0.000326015
    I1226 21:25:09.690311  6129 solver.cpp:237] Iteration 17800, loss = 0.763027
    I1226 21:25:09.690366  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:25:09.690379  6129 solver.cpp:253]     Train net output #1: loss = 0.763027 (* 1 = 0.763027 loss)
    I1226 21:25:09.690392  6129 sgd_solver.cpp:106] Iteration 17800, lr = 0.000325136
    I1226 21:25:22.272939  6129 solver.cpp:237] Iteration 17900, loss = 0.758556
    I1226 21:25:22.273001  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:25:22.273025  6129 solver.cpp:253]     Train net output #1: loss = 0.758556 (* 1 = 0.758556 loss)
    I1226 21:25:22.273041  6129 sgd_solver.cpp:106] Iteration 17900, lr = 0.000324261
    I1226 21:25:33.097745  6129 solver.cpp:341] Iteration 18000, Testing net (#0)
    I1226 21:25:37.377290  6129 solver.cpp:409]     Test net output #0: accuracy = 0.6705
    I1226 21:25:37.377341  6129 solver.cpp:409]     Test net output #1: loss = 0.940732 (* 1 = 0.940732 loss)
    I1226 21:25:37.421736  6129 solver.cpp:237] Iteration 18000, loss = 0.779485
    I1226 21:25:37.421774  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:25:37.421787  6129 solver.cpp:253]     Train net output #1: loss = 0.779485 (* 1 = 0.779485 loss)
    I1226 21:25:37.421797  6129 sgd_solver.cpp:106] Iteration 18000, lr = 0.000323392
    I1226 21:25:48.341068  6129 solver.cpp:237] Iteration 18100, loss = 0.867395
    I1226 21:25:48.341107  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:25:48.341120  6129 solver.cpp:253]     Train net output #1: loss = 0.867395 (* 1 = 0.867395 loss)
    I1226 21:25:48.341130  6129 sgd_solver.cpp:106] Iteration 18100, lr = 0.000322529
    I1226 21:26:00.553625  6129 solver.cpp:237] Iteration 18200, loss = 0.711998
    I1226 21:26:00.553665  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:26:00.553678  6129 solver.cpp:253]     Train net output #1: loss = 0.711998 (* 1 = 0.711998 loss)
    I1226 21:26:00.553688  6129 sgd_solver.cpp:106] Iteration 18200, lr = 0.00032167
    I1226 21:26:11.307704  6129 solver.cpp:237] Iteration 18300, loss = 0.764716
    I1226 21:26:11.307857  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:26:11.307871  6129 solver.cpp:253]     Train net output #1: loss = 0.764716 (* 1 = 0.764716 loss)
    I1226 21:26:11.307880  6129 sgd_solver.cpp:106] Iteration 18300, lr = 0.000320818
    I1226 21:26:21.783859  6129 solver.cpp:237] Iteration 18400, loss = 0.76299
    I1226 21:26:21.783895  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:26:21.783906  6129 solver.cpp:253]     Train net output #1: loss = 0.76299 (* 1 = 0.76299 loss)
    I1226 21:26:21.783915  6129 sgd_solver.cpp:106] Iteration 18400, lr = 0.00031997
    I1226 21:26:32.452302  6129 solver.cpp:237] Iteration 18500, loss = 0.779927
    I1226 21:26:32.452338  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:26:32.452349  6129 solver.cpp:253]     Train net output #1: loss = 0.779927 (* 1 = 0.779927 loss)
    I1226 21:26:32.452358  6129 sgd_solver.cpp:106] Iteration 18500, lr = 0.000319128
    I1226 21:26:44.908669  6129 solver.cpp:237] Iteration 18600, loss = 0.85643
    I1226 21:26:44.908849  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:26:44.908870  6129 solver.cpp:253]     Train net output #1: loss = 0.85643 (* 1 = 0.85643 loss)
    I1226 21:26:44.908885  6129 sgd_solver.cpp:106] Iteration 18600, lr = 0.00031829
    I1226 21:26:57.940567  6129 solver.cpp:237] Iteration 18700, loss = 0.672024
    I1226 21:26:57.940603  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:26:57.940615  6129 solver.cpp:253]     Train net output #1: loss = 0.672024 (* 1 = 0.672024 loss)
    I1226 21:26:57.940625  6129 sgd_solver.cpp:106] Iteration 18700, lr = 0.000317458
    I1226 21:27:10.344529  6129 solver.cpp:237] Iteration 18800, loss = 0.762614
    I1226 21:27:10.344571  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:27:10.344585  6129 solver.cpp:253]     Train net output #1: loss = 0.762614 (* 1 = 0.762614 loss)
    I1226 21:27:10.344596  6129 sgd_solver.cpp:106] Iteration 18800, lr = 0.000316631
    I1226 21:27:25.356773  6129 solver.cpp:237] Iteration 18900, loss = 0.755971
    I1226 21:27:25.356922  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:27:25.356937  6129 solver.cpp:253]     Train net output #1: loss = 0.755971 (* 1 = 0.755971 loss)
    I1226 21:27:25.356946  6129 sgd_solver.cpp:106] Iteration 18900, lr = 0.000315809
    I1226 21:27:35.853704  6129 solver.cpp:341] Iteration 19000, Testing net (#0)
    I1226 21:27:41.663375  6129 solver.cpp:409]     Test net output #0: accuracy = 0.669167
    I1226 21:27:41.663442  6129 solver.cpp:409]     Test net output #1: loss = 0.944496 (* 1 = 0.944496 loss)
    I1226 21:27:41.743607  6129 solver.cpp:237] Iteration 19000, loss = 0.776356
    I1226 21:27:41.743643  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:27:41.743655  6129 solver.cpp:253]     Train net output #1: loss = 0.776356 (* 1 = 0.776356 loss)
    I1226 21:27:41.743666  6129 sgd_solver.cpp:106] Iteration 19000, lr = 0.000314992
    I1226 21:27:54.488598  6129 solver.cpp:237] Iteration 19100, loss = 0.861283
    I1226 21:27:54.488642  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:27:54.488654  6129 solver.cpp:253]     Train net output #1: loss = 0.861283 (* 1 = 0.861283 loss)
    I1226 21:27:54.488662  6129 sgd_solver.cpp:106] Iteration 19100, lr = 0.00031418
    I1226 21:28:05.142873  6129 solver.cpp:237] Iteration 19200, loss = 0.681308
    I1226 21:28:05.143013  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:28:05.143039  6129 solver.cpp:253]     Train net output #1: loss = 0.681308 (* 1 = 0.681308 loss)
    I1226 21:28:05.143054  6129 sgd_solver.cpp:106] Iteration 19200, lr = 0.000313372
    I1226 21:28:17.076589  6129 solver.cpp:237] Iteration 19300, loss = 0.761243
    I1226 21:28:17.076647  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:28:17.076663  6129 solver.cpp:253]     Train net output #1: loss = 0.761243 (* 1 = 0.761243 loss)
    I1226 21:28:17.076675  6129 sgd_solver.cpp:106] Iteration 19300, lr = 0.00031257
    I1226 21:28:34.717196  6129 solver.cpp:237] Iteration 19400, loss = 0.745417
    I1226 21:28:34.717249  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:28:34.717263  6129 solver.cpp:253]     Train net output #1: loss = 0.745417 (* 1 = 0.745417 loss)
    I1226 21:28:34.717277  6129 sgd_solver.cpp:106] Iteration 19400, lr = 0.000311772
    I1226 21:28:47.941190  6129 solver.cpp:237] Iteration 19500, loss = 0.772812
    I1226 21:28:47.941386  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:28:47.941402  6129 solver.cpp:253]     Train net output #1: loss = 0.772812 (* 1 = 0.772812 loss)
    I1226 21:28:47.941411  6129 sgd_solver.cpp:106] Iteration 19500, lr = 0.000310979
    I1226 21:28:58.533803  6129 solver.cpp:237] Iteration 19600, loss = 0.856524
    I1226 21:28:58.533848  6129 solver.cpp:253]     Train net output #0: accuracy = 0.68
    I1226 21:28:58.533864  6129 solver.cpp:253]     Train net output #1: loss = 0.856524 (* 1 = 0.856524 loss)
    I1226 21:28:58.533875  6129 sgd_solver.cpp:106] Iteration 19600, lr = 0.000310191
    I1226 21:29:09.399830  6129 solver.cpp:237] Iteration 19700, loss = 0.659197
    I1226 21:29:09.399876  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:29:09.399888  6129 solver.cpp:253]     Train net output #1: loss = 0.659197 (* 1 = 0.659197 loss)
    I1226 21:29:09.399899  6129 sgd_solver.cpp:106] Iteration 19700, lr = 0.000309407
    I1226 21:29:19.910868  6129 solver.cpp:237] Iteration 19800, loss = 0.759439
    I1226 21:29:19.911001  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:29:19.911020  6129 solver.cpp:253]     Train net output #1: loss = 0.759439 (* 1 = 0.759439 loss)
    I1226 21:29:19.911031  6129 sgd_solver.cpp:106] Iteration 19800, lr = 0.000308628
    I1226 21:29:30.421373  6129 solver.cpp:237] Iteration 19900, loss = 0.763147
    I1226 21:29:30.421419  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:29:30.421432  6129 solver.cpp:253]     Train net output #1: loss = 0.763147 (* 1 = 0.763147 loss)
    I1226 21:29:30.421439  6129 sgd_solver.cpp:106] Iteration 19900, lr = 0.000307854
    I1226 21:29:40.823917  6129 solver.cpp:341] Iteration 20000, Testing net (#0)
    I1226 21:29:45.090294  6129 solver.cpp:409]     Test net output #0: accuracy = 0.676417
    I1226 21:29:45.090335  6129 solver.cpp:409]     Test net output #1: loss = 0.935948 (* 1 = 0.935948 loss)
    I1226 21:29:45.134790  6129 solver.cpp:237] Iteration 20000, loss = 0.772501
    I1226 21:29:45.134841  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:29:45.134855  6129 solver.cpp:253]     Train net output #1: loss = 0.772501 (* 1 = 0.772501 loss)
    I1226 21:29:45.134865  6129 sgd_solver.cpp:106] Iteration 20000, lr = 0.000307084
    I1226 21:29:55.645251  6129 solver.cpp:237] Iteration 20100, loss = 0.849763
    I1226 21:29:55.645423  6129 solver.cpp:253]     Train net output #0: accuracy = 0.67
    I1226 21:29:55.645437  6129 solver.cpp:253]     Train net output #1: loss = 0.849763 (* 1 = 0.849763 loss)
    I1226 21:29:55.645447  6129 sgd_solver.cpp:106] Iteration 20100, lr = 0.000306318
    I1226 21:30:06.535924  6129 solver.cpp:237] Iteration 20200, loss = 0.662019
    I1226 21:30:06.535980  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:30:06.536002  6129 solver.cpp:253]     Train net output #1: loss = 0.662019 (* 1 = 0.662019 loss)
    I1226 21:30:06.536020  6129 sgd_solver.cpp:106] Iteration 20200, lr = 0.000305557
    I1226 21:30:17.014475  6129 solver.cpp:237] Iteration 20300, loss = 0.755685
    I1226 21:30:17.014523  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:30:17.014534  6129 solver.cpp:253]     Train net output #1: loss = 0.755685 (* 1 = 0.755685 loss)
    I1226 21:30:17.014544  6129 sgd_solver.cpp:106] Iteration 20300, lr = 0.000304801
    I1226 21:30:27.547433  6129 solver.cpp:237] Iteration 20400, loss = 0.747445
    I1226 21:30:27.547626  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:30:27.547643  6129 solver.cpp:253]     Train net output #1: loss = 0.747445 (* 1 = 0.747445 loss)
    I1226 21:30:27.547654  6129 sgd_solver.cpp:106] Iteration 20400, lr = 0.000304048
    I1226 21:30:38.635495  6129 solver.cpp:237] Iteration 20500, loss = 0.776243
    I1226 21:30:38.635532  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:30:38.635545  6129 solver.cpp:253]     Train net output #1: loss = 0.776243 (* 1 = 0.776243 loss)
    I1226 21:30:38.635553  6129 sgd_solver.cpp:106] Iteration 20500, lr = 0.000303301
    I1226 21:30:49.163055  6129 solver.cpp:237] Iteration 20600, loss = 0.845364
    I1226 21:30:49.163092  6129 solver.cpp:253]     Train net output #0: accuracy = 0.7
    I1226 21:30:49.163105  6129 solver.cpp:253]     Train net output #1: loss = 0.845364 (* 1 = 0.845364 loss)
    I1226 21:30:49.163115  6129 sgd_solver.cpp:106] Iteration 20600, lr = 0.000302557
    I1226 21:31:02.415242  6129 solver.cpp:237] Iteration 20700, loss = 0.726473
    I1226 21:31:02.415403  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:31:02.415429  6129 solver.cpp:253]     Train net output #1: loss = 0.726473 (* 1 = 0.726473 loss)
    I1226 21:31:02.415446  6129 sgd_solver.cpp:106] Iteration 20700, lr = 0.000301817
    I1226 21:31:12.932284  6129 solver.cpp:237] Iteration 20800, loss = 0.750781
    I1226 21:31:12.932328  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:31:12.932343  6129 solver.cpp:253]     Train net output #1: loss = 0.750781 (* 1 = 0.750781 loss)
    I1226 21:31:12.932354  6129 sgd_solver.cpp:106] Iteration 20800, lr = 0.000301082
    I1226 21:31:24.989122  6129 solver.cpp:237] Iteration 20900, loss = 0.738815
    I1226 21:31:24.989181  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:31:24.989202  6129 solver.cpp:253]     Train net output #1: loss = 0.738815 (* 1 = 0.738815 loss)
    I1226 21:31:24.989219  6129 sgd_solver.cpp:106] Iteration 20900, lr = 0.000300351
    I1226 21:31:35.493852  6129 solver.cpp:341] Iteration 21000, Testing net (#0)
    I1226 21:31:39.817219  6129 solver.cpp:409]     Test net output #0: accuracy = 0.670667
    I1226 21:31:39.817260  6129 solver.cpp:409]     Test net output #1: loss = 0.943914 (* 1 = 0.943914 loss)
    I1226 21:31:39.861696  6129 solver.cpp:237] Iteration 21000, loss = 0.777211
    I1226 21:31:39.861740  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:31:39.861753  6129 solver.cpp:253]     Train net output #1: loss = 0.777211 (* 1 = 0.777211 loss)
    I1226 21:31:39.861764  6129 sgd_solver.cpp:106] Iteration 21000, lr = 0.000299624
    I1226 21:31:52.998560  6129 solver.cpp:237] Iteration 21100, loss = 0.839513
    I1226 21:31:52.998611  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:31:52.998625  6129 solver.cpp:253]     Train net output #1: loss = 0.839513 (* 1 = 0.839513 loss)
    I1226 21:31:52.998636  6129 sgd_solver.cpp:106] Iteration 21100, lr = 0.000298901
    I1226 21:32:05.610591  6129 solver.cpp:237] Iteration 21200, loss = 0.663247
    I1226 21:32:05.610693  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:32:05.610709  6129 solver.cpp:253]     Train net output #1: loss = 0.663247 (* 1 = 0.663247 loss)
    I1226 21:32:05.610721  6129 sgd_solver.cpp:106] Iteration 21200, lr = 0.000298182
    I1226 21:32:17.372678  6129 solver.cpp:237] Iteration 21300, loss = 0.748441
    I1226 21:32:17.372721  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:32:17.372735  6129 solver.cpp:253]     Train net output #1: loss = 0.748441 (* 1 = 0.748441 loss)
    I1226 21:32:17.372746  6129 sgd_solver.cpp:106] Iteration 21300, lr = 0.000297468
    I1226 21:32:30.756845  6129 solver.cpp:237] Iteration 21400, loss = 0.742113
    I1226 21:32:30.756901  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:32:30.756916  6129 solver.cpp:253]     Train net output #1: loss = 0.742113 (* 1 = 0.742113 loss)
    I1226 21:32:30.756927  6129 sgd_solver.cpp:106] Iteration 21400, lr = 0.000296757
    I1226 21:32:44.862617  6129 solver.cpp:237] Iteration 21500, loss = 0.748637
    I1226 21:32:44.862771  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:32:44.862797  6129 solver.cpp:253]     Train net output #1: loss = 0.748637 (* 1 = 0.748637 loss)
    I1226 21:32:44.862813  6129 sgd_solver.cpp:106] Iteration 21500, lr = 0.00029605
    I1226 21:32:58.548921  6129 solver.cpp:237] Iteration 21600, loss = 0.835084
    I1226 21:32:58.548995  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:32:58.549018  6129 solver.cpp:253]     Train net output #1: loss = 0.835084 (* 1 = 0.835084 loss)
    I1226 21:32:58.549034  6129 sgd_solver.cpp:106] Iteration 21600, lr = 0.000295347
    I1226 21:33:09.749017  6129 solver.cpp:237] Iteration 21700, loss = 0.699498
    I1226 21:33:09.749059  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:33:09.749074  6129 solver.cpp:253]     Train net output #1: loss = 0.699498 (* 1 = 0.699498 loss)
    I1226 21:33:09.749088  6129 sgd_solver.cpp:106] Iteration 21700, lr = 0.000294648
    I1226 21:33:22.707226  6129 solver.cpp:237] Iteration 21800, loss = 0.744289
    I1226 21:33:22.707375  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:33:22.707401  6129 solver.cpp:253]     Train net output #1: loss = 0.744289 (* 1 = 0.744289 loss)
    I1226 21:33:22.707419  6129 sgd_solver.cpp:106] Iteration 21800, lr = 0.000293953
    I1226 21:33:34.875195  6129 solver.cpp:237] Iteration 21900, loss = 0.739244
    I1226 21:33:34.875233  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:33:34.875248  6129 solver.cpp:253]     Train net output #1: loss = 0.739244 (* 1 = 0.739244 loss)
    I1226 21:33:34.875258  6129 sgd_solver.cpp:106] Iteration 21900, lr = 0.000293261
    I1226 21:33:45.367619  6129 solver.cpp:341] Iteration 22000, Testing net (#0)
    I1226 21:33:49.670652  6129 solver.cpp:409]     Test net output #0: accuracy = 0.676333
    I1226 21:33:49.670697  6129 solver.cpp:409]     Test net output #1: loss = 0.931646 (* 1 = 0.931646 loss)
    I1226 21:33:49.715217  6129 solver.cpp:237] Iteration 22000, loss = 0.760497
    I1226 21:33:49.715240  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:33:49.715253  6129 solver.cpp:253]     Train net output #1: loss = 0.760497 (* 1 = 0.760497 loss)
    I1226 21:33:49.715265  6129 sgd_solver.cpp:106] Iteration 22000, lr = 0.000292574
    I1226 21:34:00.218307  6129 solver.cpp:237] Iteration 22100, loss = 0.831554
    I1226 21:34:00.218444  6129 solver.cpp:253]     Train net output #0: accuracy = 0.69
    I1226 21:34:00.218462  6129 solver.cpp:253]     Train net output #1: loss = 0.831554 (* 1 = 0.831554 loss)
    I1226 21:34:00.218474  6129 sgd_solver.cpp:106] Iteration 22100, lr = 0.00029189
    I1226 21:34:10.855471  6129 solver.cpp:237] Iteration 22200, loss = 0.680117
    I1226 21:34:10.855521  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:34:10.855542  6129 solver.cpp:253]     Train net output #1: loss = 0.680117 (* 1 = 0.680117 loss)
    I1226 21:34:10.855558  6129 sgd_solver.cpp:106] Iteration 22200, lr = 0.00029121
    I1226 21:34:21.392156  6129 solver.cpp:237] Iteration 22300, loss = 0.745507
    I1226 21:34:21.392196  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:34:21.392211  6129 solver.cpp:253]     Train net output #1: loss = 0.745507 (* 1 = 0.745507 loss)
    I1226 21:34:21.392221  6129 sgd_solver.cpp:106] Iteration 22300, lr = 0.000290533
    I1226 21:34:32.340364  6129 solver.cpp:237] Iteration 22400, loss = 0.72914
    I1226 21:34:32.340507  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:34:32.340524  6129 solver.cpp:253]     Train net output #1: loss = 0.72914 (* 1 = 0.72914 loss)
    I1226 21:34:32.340534  6129 sgd_solver.cpp:106] Iteration 22400, lr = 0.000289861
    I1226 21:34:42.822618  6129 solver.cpp:237] Iteration 22500, loss = 0.752427
    I1226 21:34:42.822655  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:34:42.822669  6129 solver.cpp:253]     Train net output #1: loss = 0.752427 (* 1 = 0.752427 loss)
    I1226 21:34:42.822679  6129 sgd_solver.cpp:106] Iteration 22500, lr = 0.000289191
    I1226 21:34:55.289194  6129 solver.cpp:237] Iteration 22600, loss = 0.820226
    I1226 21:34:55.289234  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:34:55.289248  6129 solver.cpp:253]     Train net output #1: loss = 0.820226 (* 1 = 0.820226 loss)
    I1226 21:34:55.289258  6129 sgd_solver.cpp:106] Iteration 22600, lr = 0.000288526
    I1226 21:35:07.380908  6129 solver.cpp:237] Iteration 22700, loss = 0.642496
    I1226 21:35:07.381062  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:35:07.381077  6129 solver.cpp:253]     Train net output #1: loss = 0.642496 (* 1 = 0.642496 loss)
    I1226 21:35:07.381084  6129 sgd_solver.cpp:106] Iteration 22700, lr = 0.000287864
    I1226 21:35:18.555248  6129 solver.cpp:237] Iteration 22800, loss = 0.742388
    I1226 21:35:18.555305  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:35:18.555326  6129 solver.cpp:253]     Train net output #1: loss = 0.742388 (* 1 = 0.742388 loss)
    I1226 21:35:18.555343  6129 sgd_solver.cpp:106] Iteration 22800, lr = 0.000287205
    I1226 21:35:29.041431  6129 solver.cpp:237] Iteration 22900, loss = 0.7261
    I1226 21:35:29.041466  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:35:29.041477  6129 solver.cpp:253]     Train net output #1: loss = 0.7261 (* 1 = 0.7261 loss)
    I1226 21:35:29.041486  6129 sgd_solver.cpp:106] Iteration 22900, lr = 0.00028655
    I1226 21:35:41.176548  6129 solver.cpp:341] Iteration 23000, Testing net (#0)
    I1226 21:35:46.052469  6129 solver.cpp:409]     Test net output #0: accuracy = 0.682417
    I1226 21:35:46.052505  6129 solver.cpp:409]     Test net output #1: loss = 0.916428 (* 1 = 0.916428 loss)
    I1226 21:35:46.096987  6129 solver.cpp:237] Iteration 23000, loss = 0.74458
    I1226 21:35:46.097034  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:35:46.097048  6129 solver.cpp:253]     Train net output #1: loss = 0.74458 (* 1 = 0.74458 loss)
    I1226 21:35:46.097059  6129 sgd_solver.cpp:106] Iteration 23000, lr = 0.000285899
    I1226 21:36:00.017096  6129 solver.cpp:237] Iteration 23100, loss = 0.818305
    I1226 21:36:00.017148  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:36:00.017160  6129 solver.cpp:253]     Train net output #1: loss = 0.818305 (* 1 = 0.818305 loss)
    I1226 21:36:00.017171  6129 sgd_solver.cpp:106] Iteration 23100, lr = 0.000285251
    I1226 21:36:13.188923  6129 solver.cpp:237] Iteration 23200, loss = 0.680362
    I1226 21:36:13.189067  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:36:13.189095  6129 solver.cpp:253]     Train net output #1: loss = 0.680362 (* 1 = 0.680362 loss)
    I1226 21:36:13.189105  6129 sgd_solver.cpp:106] Iteration 23200, lr = 0.000284606
    I1226 21:36:25.736469  6129 solver.cpp:237] Iteration 23300, loss = 0.749362
    I1226 21:36:25.736510  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:36:25.736524  6129 solver.cpp:253]     Train net output #1: loss = 0.749362 (* 1 = 0.749362 loss)
    I1226 21:36:25.736534  6129 sgd_solver.cpp:106] Iteration 23300, lr = 0.000283965
    I1226 21:36:36.990331  6129 solver.cpp:237] Iteration 23400, loss = 0.723998
    I1226 21:36:36.990388  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:36:36.990411  6129 solver.cpp:253]     Train net output #1: loss = 0.723998 (* 1 = 0.723998 loss)
    I1226 21:36:36.990427  6129 sgd_solver.cpp:106] Iteration 23400, lr = 0.000283327
    I1226 21:36:48.016216  6129 solver.cpp:237] Iteration 23500, loss = 0.738337
    I1226 21:36:48.016331  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:36:48.016350  6129 solver.cpp:253]     Train net output #1: loss = 0.738337 (* 1 = 0.738337 loss)
    I1226 21:36:48.016361  6129 sgd_solver.cpp:106] Iteration 23500, lr = 0.000282693
    I1226 21:37:01.159009  6129 solver.cpp:237] Iteration 23600, loss = 0.812356
    I1226 21:37:01.159065  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:37:01.159080  6129 solver.cpp:253]     Train net output #1: loss = 0.812356 (* 1 = 0.812356 loss)
    I1226 21:37:01.159095  6129 sgd_solver.cpp:106] Iteration 23600, lr = 0.000282061
    I1226 21:37:13.890600  6129 solver.cpp:237] Iteration 23700, loss = 0.690482
    I1226 21:37:13.890660  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:37:13.890681  6129 solver.cpp:253]     Train net output #1: loss = 0.690482 (* 1 = 0.690482 loss)
    I1226 21:37:13.890697  6129 sgd_solver.cpp:106] Iteration 23700, lr = 0.000281433
    I1226 21:37:27.025089  6129 solver.cpp:237] Iteration 23800, loss = 0.741166
    I1226 21:37:27.025249  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:37:27.025274  6129 solver.cpp:253]     Train net output #1: loss = 0.741166 (* 1 = 0.741166 loss)
    I1226 21:37:27.025292  6129 sgd_solver.cpp:106] Iteration 23800, lr = 0.000280809
    I1226 21:37:39.723389  6129 solver.cpp:237] Iteration 23900, loss = 0.726111
    I1226 21:37:39.723425  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:37:39.723436  6129 solver.cpp:253]     Train net output #1: loss = 0.726111 (* 1 = 0.726111 loss)
    I1226 21:37:39.723445  6129 sgd_solver.cpp:106] Iteration 23900, lr = 0.000280187
    I1226 21:37:52.275923  6129 solver.cpp:341] Iteration 24000, Testing net (#0)
    I1226 21:37:58.101598  6129 solver.cpp:409]     Test net output #0: accuracy = 0.681917
    I1226 21:37:58.101807  6129 solver.cpp:409]     Test net output #1: loss = 0.921076 (* 1 = 0.921076 loss)
    I1226 21:37:58.146625  6129 solver.cpp:237] Iteration 24000, loss = 0.741994
    I1226 21:37:58.146677  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:37:58.146697  6129 solver.cpp:253]     Train net output #1: loss = 0.741994 (* 1 = 0.741994 loss)
    I1226 21:37:58.146713  6129 sgd_solver.cpp:106] Iteration 24000, lr = 0.000279569
    I1226 21:38:09.502691  6129 solver.cpp:237] Iteration 24100, loss = 0.813045
    I1226 21:38:09.502748  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:38:09.502770  6129 solver.cpp:253]     Train net output #1: loss = 0.813045 (* 1 = 0.813045 loss)
    I1226 21:38:09.502786  6129 sgd_solver.cpp:106] Iteration 24100, lr = 0.000278954
    I1226 21:38:20.814682  6129 solver.cpp:237] Iteration 24200, loss = 0.630032
    I1226 21:38:20.814731  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:38:20.814752  6129 solver.cpp:253]     Train net output #1: loss = 0.630032 (* 1 = 0.630032 loss)
    I1226 21:38:20.814770  6129 sgd_solver.cpp:106] Iteration 24200, lr = 0.000278342
    I1226 21:38:32.388826  6129 solver.cpp:237] Iteration 24300, loss = 0.748503
    I1226 21:38:32.389010  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:38:32.389039  6129 solver.cpp:253]     Train net output #1: loss = 0.748503 (* 1 = 0.748503 loss)
    I1226 21:38:32.389055  6129 sgd_solver.cpp:106] Iteration 24300, lr = 0.000277733
    I1226 21:38:43.883831  6129 solver.cpp:237] Iteration 24400, loss = 0.717431
    I1226 21:38:43.883883  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:38:43.883905  6129 solver.cpp:253]     Train net output #1: loss = 0.717431 (* 1 = 0.717431 loss)
    I1226 21:38:43.883922  6129 sgd_solver.cpp:106] Iteration 24400, lr = 0.000277127
    I1226 21:38:54.885051  6129 solver.cpp:237] Iteration 24500, loss = 0.727719
    I1226 21:38:54.885102  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:38:54.885123  6129 solver.cpp:253]     Train net output #1: loss = 0.727719 (* 1 = 0.727719 loss)
    I1226 21:38:54.885138  6129 sgd_solver.cpp:106] Iteration 24500, lr = 0.000276525
    I1226 21:39:05.426445  6129 solver.cpp:237] Iteration 24600, loss = 0.7928
    I1226 21:39:05.426587  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:39:05.426614  6129 solver.cpp:253]     Train net output #1: loss = 0.7928 (* 1 = 0.7928 loss)
    I1226 21:39:05.426630  6129 sgd_solver.cpp:106] Iteration 24600, lr = 0.000275925
    I1226 21:39:24.701246  6129 solver.cpp:237] Iteration 24700, loss = 0.702699
    I1226 21:39:24.701285  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:39:24.701300  6129 solver.cpp:253]     Train net output #1: loss = 0.702699 (* 1 = 0.702699 loss)
    I1226 21:39:24.701310  6129 sgd_solver.cpp:106] Iteration 24700, lr = 0.000275328
    I1226 21:39:35.277415  6129 solver.cpp:237] Iteration 24800, loss = 0.746142
    I1226 21:39:35.277452  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:39:35.277465  6129 solver.cpp:253]     Train net output #1: loss = 0.746142 (* 1 = 0.746142 loss)
    I1226 21:39:35.277477  6129 sgd_solver.cpp:106] Iteration 24800, lr = 0.000274735
    I1226 21:39:45.862989  6129 solver.cpp:237] Iteration 24900, loss = 0.715297
    I1226 21:39:45.863137  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:39:45.863157  6129 solver.cpp:253]     Train net output #1: loss = 0.715297 (* 1 = 0.715297 loss)
    I1226 21:39:45.863168  6129 sgd_solver.cpp:106] Iteration 24900, lr = 0.000274144
    I1226 21:39:56.360914  6129 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_25000.caffemodel
    I1226 21:39:56.417382  6129 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_25000.solverstate
    I1226 21:39:56.419203  6129 solver.cpp:341] Iteration 25000, Testing net (#0)
    I1226 21:40:00.795881  6129 solver.cpp:409]     Test net output #0: accuracy = 0.688167
    I1226 21:40:00.795923  6129 solver.cpp:409]     Test net output #1: loss = 0.90642 (* 1 = 0.90642 loss)
    I1226 21:40:00.840428  6129 solver.cpp:237] Iteration 25000, loss = 0.724157
    I1226 21:40:00.840456  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:40:00.840469  6129 solver.cpp:253]     Train net output #1: loss = 0.724157 (* 1 = 0.724157 loss)
    I1226 21:40:00.840481  6129 sgd_solver.cpp:106] Iteration 25000, lr = 0.000273556
    I1226 21:40:11.430106  6129 solver.cpp:237] Iteration 25100, loss = 0.803662
    I1226 21:40:11.430145  6129 solver.cpp:253]     Train net output #0: accuracy = 0.71
    I1226 21:40:11.430158  6129 solver.cpp:253]     Train net output #1: loss = 0.803662 (* 1 = 0.803662 loss)
    I1226 21:40:11.430168  6129 sgd_solver.cpp:106] Iteration 25100, lr = 0.000272972
    I1226 21:40:22.072558  6129 solver.cpp:237] Iteration 25200, loss = 0.617988
    I1226 21:40:22.072685  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:40:22.072703  6129 solver.cpp:253]     Train net output #1: loss = 0.617988 (* 1 = 0.617988 loss)
    I1226 21:40:22.072713  6129 sgd_solver.cpp:106] Iteration 25200, lr = 0.00027239
    I1226 21:40:32.676796  6129 solver.cpp:237] Iteration 25300, loss = 0.743786
    I1226 21:40:32.676833  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:40:32.676847  6129 solver.cpp:253]     Train net output #1: loss = 0.743786 (* 1 = 0.743786 loss)
    I1226 21:40:32.676858  6129 sgd_solver.cpp:106] Iteration 25300, lr = 0.000271811
    I1226 21:40:43.248375  6129 solver.cpp:237] Iteration 25400, loss = 0.712879
    I1226 21:40:43.248414  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:40:43.248427  6129 solver.cpp:253]     Train net output #1: loss = 0.712879 (* 1 = 0.712879 loss)
    I1226 21:40:43.248437  6129 sgd_solver.cpp:106] Iteration 25400, lr = 0.000271235
    I1226 21:40:53.837102  6129 solver.cpp:237] Iteration 25500, loss = 0.729128
    I1226 21:40:53.837213  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:40:53.837230  6129 solver.cpp:253]     Train net output #1: loss = 0.729128 (* 1 = 0.729128 loss)
    I1226 21:40:53.837240  6129 sgd_solver.cpp:106] Iteration 25500, lr = 0.000270662
    I1226 21:41:04.411149  6129 solver.cpp:237] Iteration 25600, loss = 0.790902
    I1226 21:41:04.411190  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:41:04.411203  6129 solver.cpp:253]     Train net output #1: loss = 0.790902 (* 1 = 0.790902 loss)
    I1226 21:41:04.411216  6129 sgd_solver.cpp:106] Iteration 25600, lr = 0.000270091
    I1226 21:41:15.049202  6129 solver.cpp:237] Iteration 25700, loss = 0.715845
    I1226 21:41:15.049242  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:41:15.049257  6129 solver.cpp:253]     Train net output #1: loss = 0.715845 (* 1 = 0.715845 loss)
    I1226 21:41:15.049268  6129 sgd_solver.cpp:106] Iteration 25700, lr = 0.000269524
    I1226 21:41:25.639605  6129 solver.cpp:237] Iteration 25800, loss = 0.765817
    I1226 21:41:25.639698  6129 solver.cpp:253]     Train net output #0: accuracy = 0.72
    I1226 21:41:25.639715  6129 solver.cpp:253]     Train net output #1: loss = 0.765817 (* 1 = 0.765817 loss)
    I1226 21:41:25.639727  6129 sgd_solver.cpp:106] Iteration 25800, lr = 0.000268959
    I1226 21:41:36.200155  6129 solver.cpp:237] Iteration 25900, loss = 0.712884
    I1226 21:41:36.200193  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:41:36.200208  6129 solver.cpp:253]     Train net output #1: loss = 0.712884 (* 1 = 0.712884 loss)
    I1226 21:41:36.200218  6129 sgd_solver.cpp:106] Iteration 25900, lr = 0.000268397
    I1226 21:41:46.725795  6129 solver.cpp:341] Iteration 26000, Testing net (#0)
    I1226 21:41:51.048101  6129 solver.cpp:409]     Test net output #0: accuracy = 0.683
    I1226 21:41:51.048142  6129 solver.cpp:409]     Test net output #1: loss = 0.912655 (* 1 = 0.912655 loss)
    I1226 21:41:51.092641  6129 solver.cpp:237] Iteration 26000, loss = 0.730109
    I1226 21:41:51.092679  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:41:51.092694  6129 solver.cpp:253]     Train net output #1: loss = 0.730109 (* 1 = 0.730109 loss)
    I1226 21:41:51.092705  6129 sgd_solver.cpp:106] Iteration 26000, lr = 0.000267837
    I1226 21:42:01.808223  6129 solver.cpp:237] Iteration 26100, loss = 0.789325
    I1226 21:42:01.808346  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:42:01.808364  6129 solver.cpp:253]     Train net output #1: loss = 0.789325 (* 1 = 0.789325 loss)
    I1226 21:42:01.808375  6129 sgd_solver.cpp:106] Iteration 26100, lr = 0.000267281
    I1226 21:42:12.371312  6129 solver.cpp:237] Iteration 26200, loss = 0.706835
    I1226 21:42:12.371350  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:42:12.371363  6129 solver.cpp:253]     Train net output #1: loss = 0.706835 (* 1 = 0.706835 loss)
    I1226 21:42:12.371373  6129 sgd_solver.cpp:106] Iteration 26200, lr = 0.000266727
    I1226 21:42:23.003125  6129 solver.cpp:237] Iteration 26300, loss = 0.730798
    I1226 21:42:23.003165  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:42:23.003180  6129 solver.cpp:253]     Train net output #1: loss = 0.730798 (* 1 = 0.730798 loss)
    I1226 21:42:23.003191  6129 sgd_solver.cpp:106] Iteration 26300, lr = 0.000266175
    I1226 21:42:33.597996  6129 solver.cpp:237] Iteration 26400, loss = 0.715687
    I1226 21:42:33.598103  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:42:33.598119  6129 solver.cpp:253]     Train net output #1: loss = 0.715687 (* 1 = 0.715687 loss)
    I1226 21:42:33.598130  6129 sgd_solver.cpp:106] Iteration 26400, lr = 0.000265627
    I1226 21:42:44.164479  6129 solver.cpp:237] Iteration 26500, loss = 0.714078
    I1226 21:42:44.164516  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:42:44.164531  6129 solver.cpp:253]     Train net output #1: loss = 0.714078 (* 1 = 0.714078 loss)
    I1226 21:42:44.164541  6129 sgd_solver.cpp:106] Iteration 26500, lr = 0.000265081
    I1226 21:42:54.742403  6129 solver.cpp:237] Iteration 26600, loss = 0.788624
    I1226 21:42:54.742441  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:42:54.742455  6129 solver.cpp:253]     Train net output #1: loss = 0.788624 (* 1 = 0.788624 loss)
    I1226 21:42:54.742466  6129 sgd_solver.cpp:106] Iteration 26600, lr = 0.000264537
    I1226 21:43:05.362488  6129 solver.cpp:237] Iteration 26700, loss = 0.608105
    I1226 21:43:05.362593  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:43:05.362610  6129 solver.cpp:253]     Train net output #1: loss = 0.608105 (* 1 = 0.608105 loss)
    I1226 21:43:05.362620  6129 sgd_solver.cpp:106] Iteration 26700, lr = 0.000263997
    I1226 21:43:15.935062  6129 solver.cpp:237] Iteration 26800, loss = 0.737645
    I1226 21:43:15.935101  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:43:15.935114  6129 solver.cpp:253]     Train net output #1: loss = 0.737645 (* 1 = 0.737645 loss)
    I1226 21:43:15.935125  6129 sgd_solver.cpp:106] Iteration 26800, lr = 0.000263458
    I1226 21:43:26.516935  6129 solver.cpp:237] Iteration 26900, loss = 0.711692
    I1226 21:43:26.516975  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:43:26.516989  6129 solver.cpp:253]     Train net output #1: loss = 0.711692 (* 1 = 0.711692 loss)
    I1226 21:43:26.517001  6129 sgd_solver.cpp:106] Iteration 26900, lr = 0.000262923
    I1226 21:43:36.985780  6129 solver.cpp:341] Iteration 27000, Testing net (#0)
    I1226 21:43:41.301661  6129 solver.cpp:409]     Test net output #0: accuracy = 0.68675
    I1226 21:43:41.301702  6129 solver.cpp:409]     Test net output #1: loss = 0.901768 (* 1 = 0.901768 loss)
    I1226 21:43:41.368847  6129 solver.cpp:237] Iteration 27000, loss = 0.718264
    I1226 21:43:41.368892  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:43:41.368907  6129 solver.cpp:253]     Train net output #1: loss = 0.718264 (* 1 = 0.718264 loss)
    I1226 21:43:41.368919  6129 sgd_solver.cpp:106] Iteration 27000, lr = 0.00026239
    I1226 21:43:52.053402  6129 solver.cpp:237] Iteration 27100, loss = 0.779948
    I1226 21:43:52.053442  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:43:52.053457  6129 solver.cpp:253]     Train net output #1: loss = 0.779948 (* 1 = 0.779948 loss)
    I1226 21:43:52.053467  6129 sgd_solver.cpp:106] Iteration 27100, lr = 0.000261859
    I1226 21:44:02.609509  6129 solver.cpp:237] Iteration 27200, loss = 0.697607
    I1226 21:44:02.609565  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:44:02.609586  6129 solver.cpp:253]     Train net output #1: loss = 0.697607 (* 1 = 0.697607 loss)
    I1226 21:44:02.609596  6129 sgd_solver.cpp:106] Iteration 27200, lr = 0.000261331
    I1226 21:44:13.159788  6129 solver.cpp:237] Iteration 27300, loss = 0.742272
    I1226 21:44:13.159971  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:44:13.159986  6129 solver.cpp:253]     Train net output #1: loss = 0.742272 (* 1 = 0.742272 loss)
    I1226 21:44:13.159994  6129 sgd_solver.cpp:106] Iteration 27300, lr = 0.000260805
    I1226 21:44:23.791577  6129 solver.cpp:237] Iteration 27400, loss = 0.705794
    I1226 21:44:23.791612  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:44:23.791625  6129 solver.cpp:253]     Train net output #1: loss = 0.705794 (* 1 = 0.705794 loss)
    I1226 21:44:23.791633  6129 sgd_solver.cpp:106] Iteration 27400, lr = 0.000260282
    I1226 21:44:34.368410  6129 solver.cpp:237] Iteration 27500, loss = 0.710346
    I1226 21:44:34.368477  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:44:34.368506  6129 solver.cpp:253]     Train net output #1: loss = 0.710346 (* 1 = 0.710346 loss)
    I1226 21:44:34.368526  6129 sgd_solver.cpp:106] Iteration 27500, lr = 0.000259761
    I1226 21:44:44.992171  6129 solver.cpp:237] Iteration 27600, loss = 0.777039
    I1226 21:44:44.992314  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:44:44.992332  6129 solver.cpp:253]     Train net output #1: loss = 0.777039 (* 1 = 0.777039 loss)
    I1226 21:44:44.992343  6129 sgd_solver.cpp:106] Iteration 27600, lr = 0.000259243
    I1226 21:44:55.633256  6129 solver.cpp:237] Iteration 27700, loss = 0.612187
    I1226 21:44:55.633290  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:44:55.633301  6129 solver.cpp:253]     Train net output #1: loss = 0.612187 (* 1 = 0.612187 loss)
    I1226 21:44:55.633311  6129 sgd_solver.cpp:106] Iteration 27700, lr = 0.000258727
    I1226 21:45:06.316723  6129 solver.cpp:237] Iteration 27800, loss = 0.739663
    I1226 21:45:06.316763  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:45:06.316778  6129 solver.cpp:253]     Train net output #1: loss = 0.739663 (* 1 = 0.739663 loss)
    I1226 21:45:06.316789  6129 sgd_solver.cpp:106] Iteration 27800, lr = 0.000258214
    I1226 21:45:16.918584  6129 solver.cpp:237] Iteration 27900, loss = 0.703752
    I1226 21:45:16.918707  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:45:16.918721  6129 solver.cpp:253]     Train net output #1: loss = 0.703752 (* 1 = 0.703752 loss)
    I1226 21:45:16.918730  6129 sgd_solver.cpp:106] Iteration 27900, lr = 0.000257702
    I1226 21:45:27.412021  6129 solver.cpp:341] Iteration 28000, Testing net (#0)
    I1226 21:45:31.760399  6129 solver.cpp:409]     Test net output #0: accuracy = 0.686583
    I1226 21:45:31.760442  6129 solver.cpp:409]     Test net output #1: loss = 0.899926 (* 1 = 0.899926 loss)
    I1226 21:45:31.805014  6129 solver.cpp:237] Iteration 28000, loss = 0.714817
    I1226 21:45:31.805057  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:45:31.805070  6129 solver.cpp:253]     Train net output #1: loss = 0.714817 (* 1 = 0.714817 loss)
    I1226 21:45:31.805083  6129 sgd_solver.cpp:106] Iteration 28000, lr = 0.000257194
    I1226 21:45:42.454628  6129 solver.cpp:237] Iteration 28100, loss = 0.769756
    I1226 21:45:42.454663  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:45:42.454675  6129 solver.cpp:253]     Train net output #1: loss = 0.769756 (* 1 = 0.769756 loss)
    I1226 21:45:42.454684  6129 sgd_solver.cpp:106] Iteration 28100, lr = 0.000256687
    I1226 21:45:53.039844  6129 solver.cpp:237] Iteration 28200, loss = 0.676739
    I1226 21:45:53.040035  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:45:53.040050  6129 solver.cpp:253]     Train net output #1: loss = 0.676739 (* 1 = 0.676739 loss)
    I1226 21:45:53.040060  6129 sgd_solver.cpp:106] Iteration 28200, lr = 0.000256183
    I1226 21:46:03.720996  6129 solver.cpp:237] Iteration 28300, loss = 0.740124
    I1226 21:46:03.721035  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:46:03.721047  6129 solver.cpp:253]     Train net output #1: loss = 0.740124 (* 1 = 0.740124 loss)
    I1226 21:46:03.721057  6129 sgd_solver.cpp:106] Iteration 28300, lr = 0.000255681
    I1226 21:46:14.355039  6129 solver.cpp:237] Iteration 28400, loss = 0.69714
    I1226 21:46:14.355098  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:46:14.355124  6129 solver.cpp:253]     Train net output #1: loss = 0.69714 (* 1 = 0.69714 loss)
    I1226 21:46:14.355144  6129 sgd_solver.cpp:106] Iteration 28400, lr = 0.000255182
    I1226 21:46:25.025493  6129 solver.cpp:237] Iteration 28500, loss = 0.704737
    I1226 21:46:25.025660  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:46:25.025681  6129 solver.cpp:253]     Train net output #1: loss = 0.704737 (* 1 = 0.704737 loss)
    I1226 21:46:25.025693  6129 sgd_solver.cpp:106] Iteration 28500, lr = 0.000254684
    I1226 21:46:35.612567  6129 solver.cpp:237] Iteration 28600, loss = 0.762128
    I1226 21:46:35.612601  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:46:35.612612  6129 solver.cpp:253]     Train net output #1: loss = 0.762128 (* 1 = 0.762128 loss)
    I1226 21:46:35.612622  6129 sgd_solver.cpp:106] Iteration 28600, lr = 0.000254189
    I1226 21:46:46.226616  6129 solver.cpp:237] Iteration 28700, loss = 0.600624
    I1226 21:46:46.226655  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:46:46.226670  6129 solver.cpp:253]     Train net output #1: loss = 0.600624 (* 1 = 0.600624 loss)
    I1226 21:46:46.226680  6129 sgd_solver.cpp:106] Iteration 28700, lr = 0.000253697
    I1226 21:46:56.865325  6129 solver.cpp:237] Iteration 28800, loss = 0.732448
    I1226 21:46:56.865458  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:46:56.865483  6129 solver.cpp:253]     Train net output #1: loss = 0.732448 (* 1 = 0.732448 loss)
    I1226 21:46:56.865491  6129 sgd_solver.cpp:106] Iteration 28800, lr = 0.000253206
    I1226 21:47:07.512547  6129 solver.cpp:237] Iteration 28900, loss = 0.699989
    I1226 21:47:07.512585  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:47:07.512599  6129 solver.cpp:253]     Train net output #1: loss = 0.699989 (* 1 = 0.699989 loss)
    I1226 21:47:07.512610  6129 sgd_solver.cpp:106] Iteration 28900, lr = 0.000252718
    I1226 21:47:18.027290  6129 solver.cpp:341] Iteration 29000, Testing net (#0)
    I1226 21:47:22.372212  6129 solver.cpp:409]     Test net output #0: accuracy = 0.687917
    I1226 21:47:22.372273  6129 solver.cpp:409]     Test net output #1: loss = 0.896611 (* 1 = 0.896611 loss)
    I1226 21:47:22.416993  6129 solver.cpp:237] Iteration 29000, loss = 0.699412
    I1226 21:47:22.417034  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:47:22.417054  6129 solver.cpp:253]     Train net output #1: loss = 0.699412 (* 1 = 0.699412 loss)
    I1226 21:47:22.417070  6129 sgd_solver.cpp:106] Iteration 29000, lr = 0.000252232
    I1226 21:47:33.115990  6129 solver.cpp:237] Iteration 29100, loss = 0.763536
    I1226 21:47:33.116099  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:47:33.116117  6129 solver.cpp:253]     Train net output #1: loss = 0.763536 (* 1 = 0.763536 loss)
    I1226 21:47:33.116128  6129 sgd_solver.cpp:106] Iteration 29100, lr = 0.000251748
    I1226 21:47:43.785867  6129 solver.cpp:237] Iteration 29200, loss = 0.674954
    I1226 21:47:43.785907  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:47:43.785920  6129 solver.cpp:253]     Train net output #1: loss = 0.674954 (* 1 = 0.674954 loss)
    I1226 21:47:43.785931  6129 sgd_solver.cpp:106] Iteration 29200, lr = 0.000251266
    I1226 21:47:54.390333  6129 solver.cpp:237] Iteration 29300, loss = 0.73015
    I1226 21:47:54.390375  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:47:54.390389  6129 solver.cpp:253]     Train net output #1: loss = 0.73015 (* 1 = 0.73015 loss)
    I1226 21:47:54.390401  6129 sgd_solver.cpp:106] Iteration 29300, lr = 0.000250786
    I1226 21:48:04.976624  6129 solver.cpp:237] Iteration 29400, loss = 0.695632
    I1226 21:48:04.976788  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:48:04.976801  6129 solver.cpp:253]     Train net output #1: loss = 0.695632 (* 1 = 0.695632 loss)
    I1226 21:48:04.976811  6129 sgd_solver.cpp:106] Iteration 29400, lr = 0.000250309
    I1226 21:48:15.518100  6129 solver.cpp:237] Iteration 29500, loss = 0.698499
    I1226 21:48:15.518153  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:48:15.518169  6129 solver.cpp:253]     Train net output #1: loss = 0.698499 (* 1 = 0.698499 loss)
    I1226 21:48:15.518182  6129 sgd_solver.cpp:106] Iteration 29500, lr = 0.000249833
    I1226 21:48:26.149166  6129 solver.cpp:237] Iteration 29600, loss = 0.764831
    I1226 21:48:26.149206  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:48:26.149220  6129 solver.cpp:253]     Train net output #1: loss = 0.764831 (* 1 = 0.764831 loss)
    I1226 21:48:26.149232  6129 sgd_solver.cpp:106] Iteration 29600, lr = 0.00024936
    I1226 21:48:36.773802  6129 solver.cpp:237] Iteration 29700, loss = 0.613431
    I1226 21:48:36.773947  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:48:36.773973  6129 solver.cpp:253]     Train net output #1: loss = 0.613431 (* 1 = 0.613431 loss)
    I1226 21:48:36.773990  6129 sgd_solver.cpp:106] Iteration 29700, lr = 0.000248889
    I1226 21:48:47.382824  6129 solver.cpp:237] Iteration 29800, loss = 0.73178
    I1226 21:48:47.382858  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:48:47.382869  6129 solver.cpp:253]     Train net output #1: loss = 0.73178 (* 1 = 0.73178 loss)
    I1226 21:48:47.382876  6129 sgd_solver.cpp:106] Iteration 29800, lr = 0.00024842
    I1226 21:48:57.981035  6129 solver.cpp:237] Iteration 29900, loss = 0.695781
    I1226 21:48:57.981084  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:48:57.981102  6129 solver.cpp:253]     Train net output #1: loss = 0.695781 (* 1 = 0.695781 loss)
    I1226 21:48:57.981118  6129 sgd_solver.cpp:106] Iteration 29900, lr = 0.000247952
    I1226 21:49:08.474342  6129 solver.cpp:341] Iteration 30000, Testing net (#0)
    I1226 21:49:12.878821  6129 solver.cpp:409]     Test net output #0: accuracy = 0.6885
    I1226 21:49:12.878876  6129 solver.cpp:409]     Test net output #1: loss = 0.89269 (* 1 = 0.89269 loss)
    I1226 21:49:12.923686  6129 solver.cpp:237] Iteration 30000, loss = 0.693579
    I1226 21:49:12.923735  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:49:12.923755  6129 solver.cpp:253]     Train net output #1: loss = 0.693579 (* 1 = 0.693579 loss)
    I1226 21:49:12.923773  6129 sgd_solver.cpp:106] Iteration 30000, lr = 0.000247487
    I1226 21:49:23.605023  6129 solver.cpp:237] Iteration 30100, loss = 0.760314
    I1226 21:49:23.605067  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:49:23.605078  6129 solver.cpp:253]     Train net output #1: loss = 0.760314 (* 1 = 0.760314 loss)
    I1226 21:49:23.605087  6129 sgd_solver.cpp:106] Iteration 30100, lr = 0.000247024
    I1226 21:49:34.165457  6129 solver.cpp:237] Iteration 30200, loss = 0.621305
    I1226 21:49:34.165508  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:49:34.165529  6129 solver.cpp:253]     Train net output #1: loss = 0.621305 (* 1 = 0.621305 loss)
    I1226 21:49:34.165544  6129 sgd_solver.cpp:106] Iteration 30200, lr = 0.000246563
    I1226 21:49:44.807504  6129 solver.cpp:237] Iteration 30300, loss = 0.720262
    I1226 21:49:44.807651  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:49:44.807664  6129 solver.cpp:253]     Train net output #1: loss = 0.720262 (* 1 = 0.720262 loss)
    I1226 21:49:44.807673  6129 sgd_solver.cpp:106] Iteration 30300, lr = 0.000246104
    I1226 21:49:55.407730  6129 solver.cpp:237] Iteration 30400, loss = 0.695474
    I1226 21:49:55.407768  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:49:55.407783  6129 solver.cpp:253]     Train net output #1: loss = 0.695474 (* 1 = 0.695474 loss)
    I1226 21:49:55.407793  6129 sgd_solver.cpp:106] Iteration 30400, lr = 0.000245647
    I1226 21:50:06.025395  6129 solver.cpp:237] Iteration 30500, loss = 0.707144
    I1226 21:50:06.025452  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:50:06.025480  6129 solver.cpp:253]     Train net output #1: loss = 0.707144 (* 1 = 0.707144 loss)
    I1226 21:50:06.025501  6129 sgd_solver.cpp:106] Iteration 30500, lr = 0.000245192
    I1226 21:50:16.664113  6129 solver.cpp:237] Iteration 30600, loss = 0.754573
    I1226 21:50:16.664305  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:50:16.664324  6129 solver.cpp:253]     Train net output #1: loss = 0.754573 (* 1 = 0.754573 loss)
    I1226 21:50:16.664333  6129 sgd_solver.cpp:106] Iteration 30600, lr = 0.000244739
    I1226 21:50:27.259917  6129 solver.cpp:237] Iteration 30700, loss = 0.600844
    I1226 21:50:27.259953  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:50:27.259968  6129 solver.cpp:253]     Train net output #1: loss = 0.600844 (* 1 = 0.600844 loss)
    I1226 21:50:27.259979  6129 sgd_solver.cpp:106] Iteration 30700, lr = 0.000244288
    I1226 21:50:37.886261  6129 solver.cpp:237] Iteration 30800, loss = 0.720075
    I1226 21:50:37.886318  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:50:37.886344  6129 solver.cpp:253]     Train net output #1: loss = 0.720075 (* 1 = 0.720075 loss)
    I1226 21:50:37.886363  6129 sgd_solver.cpp:106] Iteration 30800, lr = 0.000243839
    I1226 21:50:48.541535  6129 solver.cpp:237] Iteration 30900, loss = 0.697472
    I1226 21:50:48.541687  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:50:48.541699  6129 solver.cpp:253]     Train net output #1: loss = 0.697472 (* 1 = 0.697472 loss)
    I1226 21:50:48.541709  6129 sgd_solver.cpp:106] Iteration 30900, lr = 0.000243392
    I1226 21:50:59.036573  6129 solver.cpp:341] Iteration 31000, Testing net (#0)
    I1226 21:51:03.391187  6129 solver.cpp:409]     Test net output #0: accuracy = 0.686667
    I1226 21:51:03.391224  6129 solver.cpp:409]     Test net output #1: loss = 0.900517 (* 1 = 0.900517 loss)
    I1226 21:51:03.435781  6129 solver.cpp:237] Iteration 31000, loss = 0.697997
    I1226 21:51:03.435822  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:51:03.435835  6129 solver.cpp:253]     Train net output #1: loss = 0.697997 (* 1 = 0.697997 loss)
    I1226 21:51:03.435847  6129 sgd_solver.cpp:106] Iteration 31000, lr = 0.000242946
    I1226 21:51:14.103935  6129 solver.cpp:237] Iteration 31100, loss = 0.749971
    I1226 21:51:14.103967  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:51:14.103978  6129 solver.cpp:253]     Train net output #1: loss = 0.749971 (* 1 = 0.749971 loss)
    I1226 21:51:14.103987  6129 sgd_solver.cpp:106] Iteration 31100, lr = 0.000242503
    I1226 21:51:25.063393  6129 solver.cpp:237] Iteration 31200, loss = 0.677621
    I1226 21:51:25.063527  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 21:51:25.063550  6129 solver.cpp:253]     Train net output #1: loss = 0.677621 (* 1 = 0.677621 loss)
    I1226 21:51:25.063560  6129 sgd_solver.cpp:106] Iteration 31200, lr = 0.000242061
    I1226 21:51:35.667407  6129 solver.cpp:237] Iteration 31300, loss = 0.713303
    I1226 21:51:35.667459  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:51:35.667481  6129 solver.cpp:253]     Train net output #1: loss = 0.713303 (* 1 = 0.713303 loss)
    I1226 21:51:35.667498  6129 sgd_solver.cpp:106] Iteration 31300, lr = 0.000241621
    I1226 21:51:46.289907  6129 solver.cpp:237] Iteration 31400, loss = 0.699218
    I1226 21:51:46.289947  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:51:46.289963  6129 solver.cpp:253]     Train net output #1: loss = 0.699218 (* 1 = 0.699218 loss)
    I1226 21:51:46.289973  6129 sgd_solver.cpp:106] Iteration 31400, lr = 0.000241184
    I1226 21:51:56.897369  6129 solver.cpp:237] Iteration 31500, loss = 0.68873
    I1226 21:51:56.897537  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:51:56.897554  6129 solver.cpp:253]     Train net output #1: loss = 0.68873 (* 1 = 0.68873 loss)
    I1226 21:51:56.897564  6129 sgd_solver.cpp:106] Iteration 31500, lr = 0.000240748
    I1226 21:52:07.578177  6129 solver.cpp:237] Iteration 31600, loss = 0.748469
    I1226 21:52:07.578228  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:52:07.578249  6129 solver.cpp:253]     Train net output #1: loss = 0.748469 (* 1 = 0.748469 loss)
    I1226 21:52:07.578265  6129 sgd_solver.cpp:106] Iteration 31600, lr = 0.000240313
    I1226 21:52:18.207011  6129 solver.cpp:237] Iteration 31700, loss = 0.590957
    I1226 21:52:18.207044  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:52:18.207056  6129 solver.cpp:253]     Train net output #1: loss = 0.590957 (* 1 = 0.590957 loss)
    I1226 21:52:18.207063  6129 sgd_solver.cpp:106] Iteration 31700, lr = 0.000239881
    I1226 21:52:28.824211  6129 solver.cpp:237] Iteration 31800, loss = 0.708554
    I1226 21:52:28.824334  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:52:28.824352  6129 solver.cpp:253]     Train net output #1: loss = 0.708554 (* 1 = 0.708554 loss)
    I1226 21:52:28.824362  6129 sgd_solver.cpp:106] Iteration 31800, lr = 0.000239451
    I1226 21:52:39.431892  6129 solver.cpp:237] Iteration 31900, loss = 0.699024
    I1226 21:52:39.431942  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:52:39.431962  6129 solver.cpp:253]     Train net output #1: loss = 0.699024 (* 1 = 0.699024 loss)
    I1226 21:52:39.431978  6129 sgd_solver.cpp:106] Iteration 31900, lr = 0.000239022
    I1226 21:52:49.996235  6129 solver.cpp:341] Iteration 32000, Testing net (#0)
    I1226 21:52:54.330446  6129 solver.cpp:409]     Test net output #0: accuracy = 0.692
    I1226 21:52:54.330489  6129 solver.cpp:409]     Test net output #1: loss = 0.887487 (* 1 = 0.887487 loss)
    I1226 21:52:54.374933  6129 solver.cpp:237] Iteration 32000, loss = 0.68428
    I1226 21:52:54.374976  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:52:54.374989  6129 solver.cpp:253]     Train net output #1: loss = 0.68428 (* 1 = 0.68428 loss)
    I1226 21:52:54.375000  6129 sgd_solver.cpp:106] Iteration 32000, lr = 0.000238595
    I1226 21:53:05.091907  6129 solver.cpp:237] Iteration 32100, loss = 0.753969
    I1226 21:53:05.092008  6129 solver.cpp:253]     Train net output #0: accuracy = 0.74
    I1226 21:53:05.092025  6129 solver.cpp:253]     Train net output #1: loss = 0.753969 (* 1 = 0.753969 loss)
    I1226 21:53:05.092034  6129 sgd_solver.cpp:106] Iteration 32100, lr = 0.00023817
    I1226 21:53:15.710412  6129 solver.cpp:237] Iteration 32200, loss = 0.657145
    I1226 21:53:15.710453  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:53:15.710464  6129 solver.cpp:253]     Train net output #1: loss = 0.657145 (* 1 = 0.657145 loss)
    I1226 21:53:15.710471  6129 sgd_solver.cpp:106] Iteration 32200, lr = 0.000237746
    I1226 21:53:26.318241  6129 solver.cpp:237] Iteration 32300, loss = 0.706294
    I1226 21:53:26.318277  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:53:26.318291  6129 solver.cpp:253]     Train net output #1: loss = 0.706294 (* 1 = 0.706294 loss)
    I1226 21:53:26.318301  6129 sgd_solver.cpp:106] Iteration 32300, lr = 0.000237325
    I1226 21:53:36.896673  6129 solver.cpp:237] Iteration 32400, loss = 0.697332
    I1226 21:53:36.896821  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:53:36.896842  6129 solver.cpp:253]     Train net output #1: loss = 0.697332 (* 1 = 0.697332 loss)
    I1226 21:53:36.896852  6129 sgd_solver.cpp:106] Iteration 32400, lr = 0.000236905
    I1226 21:53:47.492552  6129 solver.cpp:237] Iteration 32500, loss = 0.69106
    I1226 21:53:47.492589  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:53:47.492602  6129 solver.cpp:253]     Train net output #1: loss = 0.69106 (* 1 = 0.69106 loss)
    I1226 21:53:47.492612  6129 sgd_solver.cpp:106] Iteration 32500, lr = 0.000236486
    I1226 21:53:58.084884  6129 solver.cpp:237] Iteration 32600, loss = 0.752491
    I1226 21:53:58.084923  6129 solver.cpp:253]     Train net output #0: accuracy = 0.73
    I1226 21:53:58.084938  6129 solver.cpp:253]     Train net output #1: loss = 0.752491 (* 1 = 0.752491 loss)
    I1226 21:53:58.084949  6129 sgd_solver.cpp:106] Iteration 32600, lr = 0.00023607
    I1226 21:54:08.750864  6129 solver.cpp:237] Iteration 32700, loss = 0.601713
    I1226 21:54:08.751029  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:54:08.751052  6129 solver.cpp:253]     Train net output #1: loss = 0.601713 (* 1 = 0.601713 loss)
    I1226 21:54:08.751060  6129 sgd_solver.cpp:106] Iteration 32700, lr = 0.000235655
    I1226 21:54:19.362057  6129 solver.cpp:237] Iteration 32800, loss = 0.698967
    I1226 21:54:19.362105  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:54:19.362126  6129 solver.cpp:253]     Train net output #1: loss = 0.698967 (* 1 = 0.698967 loss)
    I1226 21:54:19.362141  6129 sgd_solver.cpp:106] Iteration 32800, lr = 0.000235242
    I1226 21:54:29.986874  6129 solver.cpp:237] Iteration 32900, loss = 0.696207
    I1226 21:54:29.986912  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:54:29.986922  6129 solver.cpp:253]     Train net output #1: loss = 0.696207 (* 1 = 0.696207 loss)
    I1226 21:54:29.986932  6129 sgd_solver.cpp:106] Iteration 32900, lr = 0.000234831
    I1226 21:54:40.624495  6129 solver.cpp:341] Iteration 33000, Testing net (#0)
    I1226 21:54:44.965992  6129 solver.cpp:409]     Test net output #0: accuracy = 0.692
    I1226 21:54:44.966040  6129 solver.cpp:409]     Test net output #1: loss = 0.887381 (* 1 = 0.887381 loss)
    I1226 21:54:45.010591  6129 solver.cpp:237] Iteration 33000, loss = 0.679324
    I1226 21:54:45.010630  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:54:45.010644  6129 solver.cpp:253]     Train net output #1: loss = 0.679324 (* 1 = 0.679324 loss)
    I1226 21:54:45.010656  6129 sgd_solver.cpp:106] Iteration 33000, lr = 0.000234421
    I1226 21:54:55.672574  6129 solver.cpp:237] Iteration 33100, loss = 0.737135
    I1226 21:54:55.672608  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:54:55.672621  6129 solver.cpp:253]     Train net output #1: loss = 0.737135 (* 1 = 0.737135 loss)
    I1226 21:54:55.672628  6129 sgd_solver.cpp:106] Iteration 33100, lr = 0.000234013
    I1226 21:55:06.318714  6129 solver.cpp:237] Iteration 33200, loss = 0.575055
    I1226 21:55:06.318758  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 21:55:06.318776  6129 solver.cpp:253]     Train net output #1: loss = 0.575055 (* 1 = 0.575055 loss)
    I1226 21:55:06.318790  6129 sgd_solver.cpp:106] Iteration 33200, lr = 0.000233607
    I1226 21:55:16.908128  6129 solver.cpp:237] Iteration 33300, loss = 0.704708
    I1226 21:55:16.908323  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:55:16.908339  6129 solver.cpp:253]     Train net output #1: loss = 0.704708 (* 1 = 0.704708 loss)
    I1226 21:55:16.908346  6129 sgd_solver.cpp:106] Iteration 33300, lr = 0.000233202
    I1226 21:55:27.552592  6129 solver.cpp:237] Iteration 33400, loss = 0.693714
    I1226 21:55:27.552636  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:55:27.552649  6129 solver.cpp:253]     Train net output #1: loss = 0.693714 (* 1 = 0.693714 loss)
    I1226 21:55:27.552659  6129 sgd_solver.cpp:106] Iteration 33400, lr = 0.000232799
    I1226 21:55:38.143640  6129 solver.cpp:237] Iteration 33500, loss = 0.67217
    I1226 21:55:38.143673  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:55:38.143687  6129 solver.cpp:253]     Train net output #1: loss = 0.67217 (* 1 = 0.67217 loss)
    I1226 21:55:38.143697  6129 sgd_solver.cpp:106] Iteration 33500, lr = 0.000232397
    I1226 21:55:48.776680  6129 solver.cpp:237] Iteration 33600, loss = 0.747964
    I1226 21:55:48.776862  6129 solver.cpp:253]     Train net output #0: accuracy = 0.75
    I1226 21:55:48.776880  6129 solver.cpp:253]     Train net output #1: loss = 0.747964 (* 1 = 0.747964 loss)
    I1226 21:55:48.776887  6129 sgd_solver.cpp:106] Iteration 33600, lr = 0.000231997
    I1226 21:55:59.377143  6129 solver.cpp:237] Iteration 33700, loss = 0.697261
    I1226 21:55:59.377182  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:55:59.377195  6129 solver.cpp:253]     Train net output #1: loss = 0.697261 (* 1 = 0.697261 loss)
    I1226 21:55:59.377205  6129 sgd_solver.cpp:106] Iteration 33700, lr = 0.000231599
    I1226 21:56:10.017685  6129 solver.cpp:237] Iteration 33800, loss = 0.694948
    I1226 21:56:10.017724  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:56:10.017738  6129 solver.cpp:253]     Train net output #1: loss = 0.694948 (* 1 = 0.694948 loss)
    I1226 21:56:10.017750  6129 sgd_solver.cpp:106] Iteration 33800, lr = 0.000231202
    I1226 21:56:20.689851  6129 solver.cpp:237] Iteration 33900, loss = 0.698686
    I1226 21:56:20.690023  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:56:20.690052  6129 solver.cpp:253]     Train net output #1: loss = 0.698686 (* 1 = 0.698686 loss)
    I1226 21:56:20.690067  6129 sgd_solver.cpp:106] Iteration 33900, lr = 0.000230807
    I1226 21:56:31.228620  6129 solver.cpp:341] Iteration 34000, Testing net (#0)
    I1226 21:56:35.589407  6129 solver.cpp:409]     Test net output #0: accuracy = 0.690417
    I1226 21:56:35.589462  6129 solver.cpp:409]     Test net output #1: loss = 0.894755 (* 1 = 0.894755 loss)
    I1226 21:56:35.634141  6129 solver.cpp:237] Iteration 34000, loss = 0.682448
    I1226 21:56:35.634176  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:56:35.634196  6129 solver.cpp:253]     Train net output #1: loss = 0.682448 (* 1 = 0.682448 loss)
    I1226 21:56:35.634212  6129 sgd_solver.cpp:106] Iteration 34000, lr = 0.000230414
    I1226 21:56:46.284260  6129 solver.cpp:237] Iteration 34100, loss = 0.742592
    I1226 21:56:46.284304  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:56:46.284322  6129 solver.cpp:253]     Train net output #1: loss = 0.742592 (* 1 = 0.742592 loss)
    I1226 21:56:46.284337  6129 sgd_solver.cpp:106] Iteration 34100, lr = 0.000230022
    I1226 21:56:56.894879  6129 solver.cpp:237] Iteration 34200, loss = 0.680911
    I1226 21:56:56.895043  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:56:56.895061  6129 solver.cpp:253]     Train net output #1: loss = 0.680911 (* 1 = 0.680911 loss)
    I1226 21:56:56.895069  6129 sgd_solver.cpp:106] Iteration 34200, lr = 0.000229631
    I1226 21:57:07.577698  6129 solver.cpp:237] Iteration 34300, loss = 0.691833
    I1226 21:57:07.577735  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:57:07.577749  6129 solver.cpp:253]     Train net output #1: loss = 0.691833 (* 1 = 0.691833 loss)
    I1226 21:57:07.577761  6129 sgd_solver.cpp:106] Iteration 34300, lr = 0.000229243
    I1226 21:57:18.185322  6129 solver.cpp:237] Iteration 34400, loss = 0.690638
    I1226 21:57:18.185361  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:57:18.185376  6129 solver.cpp:253]     Train net output #1: loss = 0.690638 (* 1 = 0.690638 loss)
    I1226 21:57:18.185389  6129 sgd_solver.cpp:106] Iteration 34400, lr = 0.000228855
    I1226 21:57:28.752599  6129 solver.cpp:237] Iteration 34500, loss = 0.664885
    I1226 21:57:28.752789  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:57:28.752806  6129 solver.cpp:253]     Train net output #1: loss = 0.664885 (* 1 = 0.664885 loss)
    I1226 21:57:28.752817  6129 sgd_solver.cpp:106] Iteration 34500, lr = 0.000228469
    I1226 21:57:39.354006  6129 solver.cpp:237] Iteration 34600, loss = 0.736657
    I1226 21:57:39.354043  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:57:39.354055  6129 solver.cpp:253]     Train net output #1: loss = 0.736657 (* 1 = 0.736657 loss)
    I1226 21:57:39.354065  6129 sgd_solver.cpp:106] Iteration 34600, lr = 0.000228085
    I1226 21:57:49.893333  6129 solver.cpp:237] Iteration 34700, loss = 0.683365
    I1226 21:57:49.893367  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:57:49.893378  6129 solver.cpp:253]     Train net output #1: loss = 0.683365 (* 1 = 0.683365 loss)
    I1226 21:57:49.893385  6129 sgd_solver.cpp:106] Iteration 34700, lr = 0.000227702
    I1226 21:58:00.517758  6129 solver.cpp:237] Iteration 34800, loss = 0.689862
    I1226 21:58:00.517879  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:58:00.517900  6129 solver.cpp:253]     Train net output #1: loss = 0.689862 (* 1 = 0.689862 loss)
    I1226 21:58:00.517913  6129 sgd_solver.cpp:106] Iteration 34800, lr = 0.000227321
    I1226 21:58:11.082552  6129 solver.cpp:237] Iteration 34900, loss = 0.688807
    I1226 21:58:11.082586  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:58:11.082597  6129 solver.cpp:253]     Train net output #1: loss = 0.688807 (* 1 = 0.688807 loss)
    I1226 21:58:11.082604  6129 sgd_solver.cpp:106] Iteration 34900, lr = 0.000226941
    I1226 21:58:21.634201  6129 solver.cpp:341] Iteration 35000, Testing net (#0)
    I1226 21:58:26.011425  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69775
    I1226 21:58:26.011474  6129 solver.cpp:409]     Test net output #1: loss = 0.876345 (* 1 = 0.876345 loss)
    I1226 21:58:26.056058  6129 solver.cpp:237] Iteration 35000, loss = 0.659806
    I1226 21:58:26.056103  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:58:26.056121  6129 solver.cpp:253]     Train net output #1: loss = 0.659806 (* 1 = 0.659806 loss)
    I1226 21:58:26.056136  6129 sgd_solver.cpp:106] Iteration 35000, lr = 0.000226563
    I1226 21:58:36.700644  6129 solver.cpp:237] Iteration 35100, loss = 0.736257
    I1226 21:58:36.700772  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:58:36.700788  6129 solver.cpp:253]     Train net output #1: loss = 0.736257 (* 1 = 0.736257 loss)
    I1226 21:58:36.700799  6129 sgd_solver.cpp:106] Iteration 35100, lr = 0.000226186
    I1226 21:58:47.285107  6129 solver.cpp:237] Iteration 35200, loss = 0.58534
    I1226 21:58:47.285143  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 21:58:47.285157  6129 solver.cpp:253]     Train net output #1: loss = 0.58534 (* 1 = 0.58534 loss)
    I1226 21:58:47.285167  6129 sgd_solver.cpp:106] Iteration 35200, lr = 0.000225811
    I1226 21:58:57.880046  6129 solver.cpp:237] Iteration 35300, loss = 0.687496
    I1226 21:58:57.880081  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:58:57.880092  6129 solver.cpp:253]     Train net output #1: loss = 0.687496 (* 1 = 0.687496 loss)
    I1226 21:58:57.880100  6129 sgd_solver.cpp:106] Iteration 35300, lr = 0.000225437
    I1226 21:59:08.452533  6129 solver.cpp:237] Iteration 35400, loss = 0.691017
    I1226 21:59:08.452726  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 21:59:08.452744  6129 solver.cpp:253]     Train net output #1: loss = 0.691017 (* 1 = 0.691017 loss)
    I1226 21:59:08.452752  6129 sgd_solver.cpp:106] Iteration 35400, lr = 0.000225064
    I1226 21:59:19.090646  6129 solver.cpp:237] Iteration 35500, loss = 0.664303
    I1226 21:59:19.090679  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 21:59:19.090690  6129 solver.cpp:253]     Train net output #1: loss = 0.664303 (* 1 = 0.664303 loss)
    I1226 21:59:19.090699  6129 sgd_solver.cpp:106] Iteration 35500, lr = 0.000224693
    I1226 21:59:29.713044  6129 solver.cpp:237] Iteration 35600, loss = 0.730025
    I1226 21:59:29.713099  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 21:59:29.713119  6129 solver.cpp:253]     Train net output #1: loss = 0.730025 (* 1 = 0.730025 loss)
    I1226 21:59:29.713135  6129 sgd_solver.cpp:106] Iteration 35600, lr = 0.000224323
    I1226 21:59:40.306042  6129 solver.cpp:237] Iteration 35700, loss = 0.639803
    I1226 21:59:40.306227  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 21:59:40.306243  6129 solver.cpp:253]     Train net output #1: loss = 0.639803 (* 1 = 0.639803 loss)
    I1226 21:59:40.306252  6129 sgd_solver.cpp:106] Iteration 35700, lr = 0.000223955
    I1226 21:59:50.925688  6129 solver.cpp:237] Iteration 35800, loss = 0.682126
    I1226 21:59:50.925729  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 21:59:50.925743  6129 solver.cpp:253]     Train net output #1: loss = 0.682126 (* 1 = 0.682126 loss)
    I1226 21:59:50.925753  6129 sgd_solver.cpp:106] Iteration 35800, lr = 0.000223588
    I1226 22:00:01.541529  6129 solver.cpp:237] Iteration 35900, loss = 0.687702
    I1226 22:00:01.541568  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:00:01.541579  6129 solver.cpp:253]     Train net output #1: loss = 0.687702 (* 1 = 0.687702 loss)
    I1226 22:00:01.541589  6129 sgd_solver.cpp:106] Iteration 35900, lr = 0.000223223
    I1226 22:00:12.045356  6129 solver.cpp:341] Iteration 36000, Testing net (#0)
    I1226 22:00:16.373991  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69325
    I1226 22:00:16.374040  6129 solver.cpp:409]     Test net output #1: loss = 0.884197 (* 1 = 0.884197 loss)
    I1226 22:00:16.437140  6129 solver.cpp:237] Iteration 36000, loss = 0.661855
    I1226 22:00:16.437186  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:00:16.437199  6129 solver.cpp:253]     Train net output #1: loss = 0.661855 (* 1 = 0.661855 loss)
    I1226 22:00:16.437209  6129 sgd_solver.cpp:106] Iteration 36000, lr = 0.000222859
    I1226 22:00:27.031898  6129 solver.cpp:237] Iteration 36100, loss = 0.730095
    I1226 22:00:27.031941  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:00:27.031955  6129 solver.cpp:253]     Train net output #1: loss = 0.730095 (* 1 = 0.730095 loss)
    I1226 22:00:27.031965  6129 sgd_solver.cpp:106] Iteration 36100, lr = 0.000222496
    I1226 22:00:37.593133  6129 solver.cpp:237] Iteration 36200, loss = 0.579268
    I1226 22:00:37.593186  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:00:37.593204  6129 solver.cpp:253]     Train net output #1: loss = 0.579268 (* 1 = 0.579268 loss)
    I1226 22:00:37.593219  6129 sgd_solver.cpp:106] Iteration 36200, lr = 0.000222135
    I1226 22:00:48.186614  6129 solver.cpp:237] Iteration 36300, loss = 0.671987
    I1226 22:00:48.186754  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:00:48.186769  6129 solver.cpp:253]     Train net output #1: loss = 0.671987 (* 1 = 0.671987 loss)
    I1226 22:00:48.186779  6129 sgd_solver.cpp:106] Iteration 36300, lr = 0.000221775
    I1226 22:00:58.795053  6129 solver.cpp:237] Iteration 36400, loss = 0.686658
    I1226 22:00:58.795090  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:00:58.795104  6129 solver.cpp:253]     Train net output #1: loss = 0.686658 (* 1 = 0.686658 loss)
    I1226 22:00:58.795114  6129 sgd_solver.cpp:106] Iteration 36400, lr = 0.000221416
    I1226 22:01:09.400929  6129 solver.cpp:237] Iteration 36500, loss = 0.659181
    I1226 22:01:09.400967  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:01:09.400982  6129 solver.cpp:253]     Train net output #1: loss = 0.659181 (* 1 = 0.659181 loss)
    I1226 22:01:09.400993  6129 sgd_solver.cpp:106] Iteration 36500, lr = 0.000221059
    I1226 22:01:20.019958  6129 solver.cpp:237] Iteration 36600, loss = 0.722214
    I1226 22:01:20.020114  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:01:20.020130  6129 solver.cpp:253]     Train net output #1: loss = 0.722214 (* 1 = 0.722214 loss)
    I1226 22:01:20.020140  6129 sgd_solver.cpp:106] Iteration 36600, lr = 0.000220703
    I1226 22:01:30.616739  6129 solver.cpp:237] Iteration 36700, loss = 0.65427
    I1226 22:01:30.616786  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:01:30.616807  6129 solver.cpp:253]     Train net output #1: loss = 0.65427 (* 1 = 0.65427 loss)
    I1226 22:01:30.616824  6129 sgd_solver.cpp:106] Iteration 36700, lr = 0.000220349
    I1226 22:01:41.257132  6129 solver.cpp:237] Iteration 36800, loss = 0.675723
    I1226 22:01:41.257164  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:01:41.257175  6129 solver.cpp:253]     Train net output #1: loss = 0.675723 (* 1 = 0.675723 loss)
    I1226 22:01:41.257184  6129 sgd_solver.cpp:106] Iteration 36800, lr = 0.000219995
    I1226 22:01:51.839126  6129 solver.cpp:237] Iteration 36900, loss = 0.685328
    I1226 22:01:51.839272  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:01:51.839287  6129 solver.cpp:253]     Train net output #1: loss = 0.685328 (* 1 = 0.685328 loss)
    I1226 22:01:51.839294  6129 sgd_solver.cpp:106] Iteration 36900, lr = 0.000219644
    I1226 22:02:02.359292  6129 solver.cpp:341] Iteration 37000, Testing net (#0)
    I1226 22:02:06.720219  6129 solver.cpp:409]     Test net output #0: accuracy = 0.695833
    I1226 22:02:06.720264  6129 solver.cpp:409]     Test net output #1: loss = 0.872904 (* 1 = 0.872904 loss)
    I1226 22:02:06.764827  6129 solver.cpp:237] Iteration 37000, loss = 0.651774
    I1226 22:02:06.764878  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:02:06.764894  6129 solver.cpp:253]     Train net output #1: loss = 0.651774 (* 1 = 0.651774 loss)
    I1226 22:02:06.764907  6129 sgd_solver.cpp:106] Iteration 37000, lr = 0.000219293
    I1226 22:02:17.482897  6129 solver.cpp:237] Iteration 37100, loss = 0.729545
    I1226 22:02:17.482939  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:02:17.482956  6129 solver.cpp:253]     Train net output #1: loss = 0.729545 (* 1 = 0.729545 loss)
    I1226 22:02:17.482969  6129 sgd_solver.cpp:106] Iteration 37100, lr = 0.000218944
    I1226 22:02:28.055871  6129 solver.cpp:237] Iteration 37200, loss = 0.571823
    I1226 22:02:28.056042  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:02:28.056068  6129 solver.cpp:253]     Train net output #1: loss = 0.571823 (* 1 = 0.571823 loss)
    I1226 22:02:28.056085  6129 sgd_solver.cpp:106] Iteration 37200, lr = 0.000218596
    I1226 22:02:38.707787  6129 solver.cpp:237] Iteration 37300, loss = 0.672052
    I1226 22:02:38.707826  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:02:38.707841  6129 solver.cpp:253]     Train net output #1: loss = 0.672052 (* 1 = 0.672052 loss)
    I1226 22:02:38.707852  6129 sgd_solver.cpp:106] Iteration 37300, lr = 0.000218249
    I1226 22:02:49.306602  6129 solver.cpp:237] Iteration 37400, loss = 0.674948
    I1226 22:02:49.306646  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:02:49.306665  6129 solver.cpp:253]     Train net output #1: loss = 0.674948 (* 1 = 0.674948 loss)
    I1226 22:02:49.306679  6129 sgd_solver.cpp:106] Iteration 37400, lr = 0.000217904
    I1226 22:02:59.875296  6129 solver.cpp:237] Iteration 37500, loss = 0.663332
    I1226 22:02:59.875494  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:02:59.875509  6129 solver.cpp:253]     Train net output #1: loss = 0.663332 (* 1 = 0.663332 loss)
    I1226 22:02:59.875517  6129 sgd_solver.cpp:106] Iteration 37500, lr = 0.000217559
    I1226 22:03:10.486454  6129 solver.cpp:237] Iteration 37600, loss = 0.735928
    I1226 22:03:10.486503  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:03:10.486524  6129 solver.cpp:253]     Train net output #1: loss = 0.735928 (* 1 = 0.735928 loss)
    I1226 22:03:10.486539  6129 sgd_solver.cpp:106] Iteration 37600, lr = 0.000217216
    I1226 22:03:21.096226  6129 solver.cpp:237] Iteration 37700, loss = 0.639087
    I1226 22:03:21.096259  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:03:21.096271  6129 solver.cpp:253]     Train net output #1: loss = 0.639087 (* 1 = 0.639087 loss)
    I1226 22:03:21.096278  6129 sgd_solver.cpp:106] Iteration 37700, lr = 0.000216875
    I1226 22:03:31.663578  6129 solver.cpp:237] Iteration 37800, loss = 0.669041
    I1226 22:03:31.663736  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:03:31.663754  6129 solver.cpp:253]     Train net output #1: loss = 0.669041 (* 1 = 0.669041 loss)
    I1226 22:03:31.663764  6129 sgd_solver.cpp:106] Iteration 37800, lr = 0.000216535
    I1226 22:03:42.292551  6129 solver.cpp:237] Iteration 37900, loss = 0.686276
    I1226 22:03:42.292595  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:03:42.292614  6129 solver.cpp:253]     Train net output #1: loss = 0.686276 (* 1 = 0.686276 loss)
    I1226 22:03:42.292629  6129 sgd_solver.cpp:106] Iteration 37900, lr = 0.000216195
    I1226 22:03:52.730784  6129 solver.cpp:341] Iteration 38000, Testing net (#0)
    I1226 22:03:57.090283  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69675
    I1226 22:03:57.090338  6129 solver.cpp:409]     Test net output #1: loss = 0.874821 (* 1 = 0.874821 loss)
    I1226 22:03:57.135099  6129 solver.cpp:237] Iteration 38000, loss = 0.652709
    I1226 22:03:57.135151  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:03:57.135171  6129 solver.cpp:253]     Train net output #1: loss = 0.652709 (* 1 = 0.652709 loss)
    I1226 22:03:57.135188  6129 sgd_solver.cpp:106] Iteration 38000, lr = 0.000215857
    I1226 22:04:07.873244  6129 solver.cpp:237] Iteration 38100, loss = 0.719733
    I1226 22:04:07.873365  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:04:07.873383  6129 solver.cpp:253]     Train net output #1: loss = 0.719733 (* 1 = 0.719733 loss)
    I1226 22:04:07.873392  6129 sgd_solver.cpp:106] Iteration 38100, lr = 0.000215521
    I1226 22:04:18.497149  6129 solver.cpp:237] Iteration 38200, loss = 0.565948
    I1226 22:04:18.497200  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:04:18.497220  6129 solver.cpp:253]     Train net output #1: loss = 0.565948 (* 1 = 0.565948 loss)
    I1226 22:04:18.497236  6129 sgd_solver.cpp:106] Iteration 38200, lr = 0.000215185
    I1226 22:04:29.062429  6129 solver.cpp:237] Iteration 38300, loss = 0.66684
    I1226 22:04:29.062470  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:04:29.062480  6129 solver.cpp:253]     Train net output #1: loss = 0.66684 (* 1 = 0.66684 loss)
    I1226 22:04:29.062489  6129 sgd_solver.cpp:106] Iteration 38300, lr = 0.000214851
    I1226 22:04:39.641203  6129 solver.cpp:237] Iteration 38400, loss = 0.682724
    I1226 22:04:39.641299  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:04:39.641314  6129 solver.cpp:253]     Train net output #1: loss = 0.682724 (* 1 = 0.682724 loss)
    I1226 22:04:39.641326  6129 sgd_solver.cpp:106] Iteration 38400, lr = 0.000214518
    I1226 22:04:50.208233  6129 solver.cpp:237] Iteration 38500, loss = 0.652613
    I1226 22:04:50.208272  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:04:50.208287  6129 solver.cpp:253]     Train net output #1: loss = 0.652613 (* 1 = 0.652613 loss)
    I1226 22:04:50.208298  6129 sgd_solver.cpp:106] Iteration 38500, lr = 0.000214186
    I1226 22:05:00.873849  6129 solver.cpp:237] Iteration 38600, loss = 0.717777
    I1226 22:05:00.873889  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:05:00.873903  6129 solver.cpp:253]     Train net output #1: loss = 0.717777 (* 1 = 0.717777 loss)
    I1226 22:05:00.873914  6129 sgd_solver.cpp:106] Iteration 38600, lr = 0.000213856
    I1226 22:05:11.449638  6129 solver.cpp:237] Iteration 38700, loss = 0.609607
    I1226 22:05:11.449753  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:05:11.449769  6129 solver.cpp:253]     Train net output #1: loss = 0.609607 (* 1 = 0.609607 loss)
    I1226 22:05:11.449780  6129 sgd_solver.cpp:106] Iteration 38700, lr = 0.000213526
    I1226 22:05:22.093338  6129 solver.cpp:237] Iteration 38800, loss = 0.664515
    I1226 22:05:22.093382  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:05:22.093392  6129 solver.cpp:253]     Train net output #1: loss = 0.664515 (* 1 = 0.664515 loss)
    I1226 22:05:22.093401  6129 sgd_solver.cpp:106] Iteration 38800, lr = 0.000213198
    I1226 22:05:32.686842  6129 solver.cpp:237] Iteration 38900, loss = 0.68516
    I1226 22:05:32.686882  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:05:32.686897  6129 solver.cpp:253]     Train net output #1: loss = 0.68516 (* 1 = 0.68516 loss)
    I1226 22:05:32.686908  6129 sgd_solver.cpp:106] Iteration 38900, lr = 0.000212871
    I1226 22:05:43.135594  6129 solver.cpp:341] Iteration 39000, Testing net (#0)
    I1226 22:05:47.536145  6129 solver.cpp:409]     Test net output #0: accuracy = 0.696
    I1226 22:05:47.536192  6129 solver.cpp:409]     Test net output #1: loss = 0.874367 (* 1 = 0.874367 loss)
    I1226 22:05:47.586186  6129 solver.cpp:237] Iteration 39000, loss = 0.643166
    I1226 22:05:47.586223  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:05:47.586236  6129 solver.cpp:253]     Train net output #1: loss = 0.643166 (* 1 = 0.643166 loss)
    I1226 22:05:47.586248  6129 sgd_solver.cpp:106] Iteration 39000, lr = 0.000212545
    I1226 22:05:58.198251  6129 solver.cpp:237] Iteration 39100, loss = 0.724732
    I1226 22:05:58.198289  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:05:58.198304  6129 solver.cpp:253]     Train net output #1: loss = 0.724732 (* 1 = 0.724732 loss)
    I1226 22:05:58.198315  6129 sgd_solver.cpp:106] Iteration 39100, lr = 0.00021222
    I1226 22:06:08.848418  6129 solver.cpp:237] Iteration 39200, loss = 0.576061
    I1226 22:06:08.848458  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:06:08.848474  6129 solver.cpp:253]     Train net output #1: loss = 0.576061 (* 1 = 0.576061 loss)
    I1226 22:06:08.848484  6129 sgd_solver.cpp:106] Iteration 39200, lr = 0.000211897
    I1226 22:06:19.451279  6129 solver.cpp:237] Iteration 39300, loss = 0.662768
    I1226 22:06:19.451407  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:06:19.451426  6129 solver.cpp:253]     Train net output #1: loss = 0.662768 (* 1 = 0.662768 loss)
    I1226 22:06:19.451436  6129 sgd_solver.cpp:106] Iteration 39300, lr = 0.000211574
    I1226 22:06:30.013562  6129 solver.cpp:237] Iteration 39400, loss = 0.682164
    I1226 22:06:30.013610  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:06:30.013625  6129 solver.cpp:253]     Train net output #1: loss = 0.682164 (* 1 = 0.682164 loss)
    I1226 22:06:30.013638  6129 sgd_solver.cpp:106] Iteration 39400, lr = 0.000211253
    I1226 22:06:40.631649  6129 solver.cpp:237] Iteration 39500, loss = 0.653587
    I1226 22:06:40.631685  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:06:40.631700  6129 solver.cpp:253]     Train net output #1: loss = 0.653587 (* 1 = 0.653587 loss)
    I1226 22:06:40.631711  6129 sgd_solver.cpp:106] Iteration 39500, lr = 0.000210933
    I1226 22:06:51.194093  6129 solver.cpp:237] Iteration 39600, loss = 0.721287
    I1226 22:06:51.194291  6129 solver.cpp:253]     Train net output #0: accuracy = 0.76
    I1226 22:06:51.194308  6129 solver.cpp:253]     Train net output #1: loss = 0.721287 (* 1 = 0.721287 loss)
    I1226 22:06:51.194316  6129 sgd_solver.cpp:106] Iteration 39600, lr = 0.000210614
    I1226 22:07:01.816485  6129 solver.cpp:237] Iteration 39700, loss = 0.564947
    I1226 22:07:01.816530  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:07:01.816550  6129 solver.cpp:253]     Train net output #1: loss = 0.564947 (* 1 = 0.564947 loss)
    I1226 22:07:01.816565  6129 sgd_solver.cpp:106] Iteration 39700, lr = 0.000210296
    I1226 22:07:12.406491  6129 solver.cpp:237] Iteration 39800, loss = 0.663255
    I1226 22:07:12.406534  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:07:12.406546  6129 solver.cpp:253]     Train net output #1: loss = 0.663255 (* 1 = 0.663255 loss)
    I1226 22:07:12.406558  6129 sgd_solver.cpp:106] Iteration 39800, lr = 0.000209979
    I1226 22:07:23.021296  6129 solver.cpp:237] Iteration 39900, loss = 0.67802
    I1226 22:07:23.022370  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:07:23.022392  6129 solver.cpp:253]     Train net output #1: loss = 0.67802 (* 1 = 0.67802 loss)
    I1226 22:07:23.022403  6129 sgd_solver.cpp:106] Iteration 39900, lr = 0.000209663
    I1226 22:07:33.536777  6129 solver.cpp:341] Iteration 40000, Testing net (#0)
    I1226 22:07:37.882372  6129 solver.cpp:409]     Test net output #0: accuracy = 0.699
    I1226 22:07:37.882424  6129 solver.cpp:409]     Test net output #1: loss = 0.871118 (* 1 = 0.871118 loss)
    I1226 22:07:37.926960  6129 solver.cpp:237] Iteration 40000, loss = 0.642938
    I1226 22:07:37.926990  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:07:37.927003  6129 solver.cpp:253]     Train net output #1: loss = 0.642938 (* 1 = 0.642938 loss)
    I1226 22:07:37.927016  6129 sgd_solver.cpp:106] Iteration 40000, lr = 0.000209349
    I1226 22:07:48.485370  6129 solver.cpp:237] Iteration 40100, loss = 0.712777
    I1226 22:07:48.485419  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:07:48.485440  6129 solver.cpp:253]     Train net output #1: loss = 0.712777 (* 1 = 0.712777 loss)
    I1226 22:07:48.485455  6129 sgd_solver.cpp:106] Iteration 40100, lr = 0.000209035
    I1226 22:07:59.126113  6129 solver.cpp:237] Iteration 40200, loss = 0.679773
    I1226 22:07:59.126305  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:07:59.126322  6129 solver.cpp:253]     Train net output #1: loss = 0.679773 (* 1 = 0.679773 loss)
    I1226 22:07:59.126330  6129 sgd_solver.cpp:106] Iteration 40200, lr = 0.000208723
    I1226 22:08:09.777789  6129 solver.cpp:237] Iteration 40300, loss = 0.65859
    I1226 22:08:09.777834  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:08:09.777853  6129 solver.cpp:253]     Train net output #1: loss = 0.65859 (* 1 = 0.65859 loss)
    I1226 22:08:09.777866  6129 sgd_solver.cpp:106] Iteration 40300, lr = 0.000208412
    I1226 22:08:20.394695  6129 solver.cpp:237] Iteration 40400, loss = 0.677254
    I1226 22:08:20.394744  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:08:20.394765  6129 solver.cpp:253]     Train net output #1: loss = 0.677254 (* 1 = 0.677254 loss)
    I1226 22:08:20.394781  6129 sgd_solver.cpp:106] Iteration 40400, lr = 0.000208101
    I1226 22:08:31.030231  6129 solver.cpp:237] Iteration 40500, loss = 0.646438
    I1226 22:08:31.030349  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:08:31.030370  6129 solver.cpp:253]     Train net output #1: loss = 0.646438 (* 1 = 0.646438 loss)
    I1226 22:08:31.030385  6129 sgd_solver.cpp:106] Iteration 40500, lr = 0.000207792
    I1226 22:08:41.628403  6129 solver.cpp:237] Iteration 40600, loss = 0.723733
    I1226 22:08:41.628443  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:08:41.628473  6129 solver.cpp:253]     Train net output #1: loss = 0.723733 (* 1 = 0.723733 loss)
    I1226 22:08:41.628489  6129 sgd_solver.cpp:106] Iteration 40600, lr = 0.000207484
    I1226 22:08:52.231925  6129 solver.cpp:237] Iteration 40700, loss = 0.581686
    I1226 22:08:52.231958  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:08:52.231969  6129 solver.cpp:253]     Train net output #1: loss = 0.581686 (* 1 = 0.581686 loss)
    I1226 22:08:52.231977  6129 sgd_solver.cpp:106] Iteration 40700, lr = 0.000207177
    I1226 22:09:02.856609  6129 solver.cpp:237] Iteration 40800, loss = 0.657864
    I1226 22:09:02.856734  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:09:02.856755  6129 solver.cpp:253]     Train net output #1: loss = 0.657864 (* 1 = 0.657864 loss)
    I1226 22:09:02.856770  6129 sgd_solver.cpp:106] Iteration 40800, lr = 0.000206871
    I1226 22:09:13.424865  6129 solver.cpp:237] Iteration 40900, loss = 0.67684
    I1226 22:09:13.424906  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:09:13.424916  6129 solver.cpp:253]     Train net output #1: loss = 0.67684 (* 1 = 0.67684 loss)
    I1226 22:09:13.424924  6129 sgd_solver.cpp:106] Iteration 40900, lr = 0.000206566
    I1226 22:09:23.955245  6129 solver.cpp:341] Iteration 41000, Testing net (#0)
    I1226 22:09:28.309020  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69425
    I1226 22:09:28.309062  6129 solver.cpp:409]     Test net output #1: loss = 0.879653 (* 1 = 0.879653 loss)
    I1226 22:09:28.353621  6129 solver.cpp:237] Iteration 41000, loss = 0.644626
    I1226 22:09:28.353660  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:09:28.353674  6129 solver.cpp:253]     Train net output #1: loss = 0.644626 (* 1 = 0.644626 loss)
    I1226 22:09:28.353685  6129 sgd_solver.cpp:106] Iteration 41000, lr = 0.000206263
    I1226 22:09:38.960420  6129 solver.cpp:237] Iteration 41100, loss = 0.709952
    I1226 22:09:38.960592  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:09:38.960623  6129 solver.cpp:253]     Train net output #1: loss = 0.709952 (* 1 = 0.709952 loss)
    I1226 22:09:38.960644  6129 sgd_solver.cpp:106] Iteration 41100, lr = 0.00020596
    I1226 22:09:49.587806  6129 solver.cpp:237] Iteration 41200, loss = 0.608463
    I1226 22:09:49.587852  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:09:49.587872  6129 solver.cpp:253]     Train net output #1: loss = 0.608463 (* 1 = 0.608463 loss)
    I1226 22:09:49.587887  6129 sgd_solver.cpp:106] Iteration 41200, lr = 0.000205658
    I1226 22:10:00.149528  6129 solver.cpp:237] Iteration 41300, loss = 0.656793
    I1226 22:10:00.149561  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:10:00.149572  6129 solver.cpp:253]     Train net output #1: loss = 0.656793 (* 1 = 0.656793 loss)
    I1226 22:10:00.149581  6129 sgd_solver.cpp:106] Iteration 41300, lr = 0.000205357
    I1226 22:10:10.789791  6129 solver.cpp:237] Iteration 41400, loss = 0.677819
    I1226 22:10:10.789935  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:10:10.789957  6129 solver.cpp:253]     Train net output #1: loss = 0.677819 (* 1 = 0.677819 loss)
    I1226 22:10:10.789971  6129 sgd_solver.cpp:106] Iteration 41400, lr = 0.000205058
    I1226 22:10:21.403821  6129 solver.cpp:237] Iteration 41500, loss = 0.63875
    I1226 22:10:21.403873  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:10:21.403893  6129 solver.cpp:253]     Train net output #1: loss = 0.63875 (* 1 = 0.63875 loss)
    I1226 22:10:21.403910  6129 sgd_solver.cpp:106] Iteration 41500, lr = 0.000204759
    I1226 22:10:32.013829  6129 solver.cpp:237] Iteration 41600, loss = 0.708286
    I1226 22:10:32.013866  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:10:32.013881  6129 solver.cpp:253]     Train net output #1: loss = 0.708286 (* 1 = 0.708286 loss)
    I1226 22:10:32.013892  6129 sgd_solver.cpp:106] Iteration 41600, lr = 0.000204461
    I1226 22:10:42.615342  6129 solver.cpp:237] Iteration 41700, loss = 0.645151
    I1226 22:10:42.615483  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:10:42.615507  6129 solver.cpp:253]     Train net output #1: loss = 0.645151 (* 1 = 0.645151 loss)
    I1226 22:10:42.615525  6129 sgd_solver.cpp:106] Iteration 41700, lr = 0.000204164
    I1226 22:10:53.194550  6129 solver.cpp:237] Iteration 41800, loss = 0.655378
    I1226 22:10:53.194594  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:10:53.194612  6129 solver.cpp:253]     Train net output #1: loss = 0.655378 (* 1 = 0.655378 loss)
    I1226 22:10:53.194627  6129 sgd_solver.cpp:106] Iteration 41800, lr = 0.000203869
    I1226 22:11:03.807950  6129 solver.cpp:237] Iteration 41900, loss = 0.673942
    I1226 22:11:03.807981  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:11:03.807992  6129 solver.cpp:253]     Train net output #1: loss = 0.673942 (* 1 = 0.673942 loss)
    I1226 22:11:03.808001  6129 sgd_solver.cpp:106] Iteration 41900, lr = 0.000203574
    I1226 22:11:14.288266  6129 solver.cpp:341] Iteration 42000, Testing net (#0)
    I1226 22:11:18.649792  6129 solver.cpp:409]     Test net output #0: accuracy = 0.695
    I1226 22:11:18.649835  6129 solver.cpp:409]     Test net output #1: loss = 0.870047 (* 1 = 0.870047 loss)
    I1226 22:11:18.694293  6129 solver.cpp:237] Iteration 42000, loss = 0.634336
    I1226 22:11:18.694344  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:11:18.694360  6129 solver.cpp:253]     Train net output #1: loss = 0.634336 (* 1 = 0.634336 loss)
    I1226 22:11:18.694373  6129 sgd_solver.cpp:106] Iteration 42000, lr = 0.00020328
    I1226 22:11:29.745401  6129 solver.cpp:237] Iteration 42100, loss = 0.718213
    I1226 22:11:29.745443  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:11:29.745460  6129 solver.cpp:253]     Train net output #1: loss = 0.718213 (* 1 = 0.718213 loss)
    I1226 22:11:29.745472  6129 sgd_solver.cpp:106] Iteration 42100, lr = 0.000202988
    I1226 22:11:40.312793  6129 solver.cpp:237] Iteration 42200, loss = 0.573332
    I1226 22:11:40.312842  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:11:40.312863  6129 solver.cpp:253]     Train net output #1: loss = 0.573332 (* 1 = 0.573332 loss)
    I1226 22:11:40.312878  6129 sgd_solver.cpp:106] Iteration 42200, lr = 0.000202696
    I1226 22:11:50.942014  6129 solver.cpp:237] Iteration 42300, loss = 0.652487
    I1226 22:11:50.942163  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:11:50.942179  6129 solver.cpp:253]     Train net output #1: loss = 0.652487 (* 1 = 0.652487 loss)
    I1226 22:11:50.942191  6129 sgd_solver.cpp:106] Iteration 42300, lr = 0.000202405
    I1226 22:12:01.568205  6129 solver.cpp:237] Iteration 42400, loss = 0.682068
    I1226 22:12:01.568258  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:12:01.568279  6129 solver.cpp:253]     Train net output #1: loss = 0.682068 (* 1 = 0.682068 loss)
    I1226 22:12:01.568295  6129 sgd_solver.cpp:106] Iteration 42400, lr = 0.000202115
    I1226 22:12:12.172478  6129 solver.cpp:237] Iteration 42500, loss = 0.650779
    I1226 22:12:12.172518  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:12:12.172529  6129 solver.cpp:253]     Train net output #1: loss = 0.650779 (* 1 = 0.650779 loss)
    I1226 22:12:12.172538  6129 sgd_solver.cpp:106] Iteration 42500, lr = 0.000201827
    I1226 22:12:22.763188  6129 solver.cpp:237] Iteration 42600, loss = 0.70276
    I1226 22:12:22.763355  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:12:22.763373  6129 solver.cpp:253]     Train net output #1: loss = 0.70276 (* 1 = 0.70276 loss)
    I1226 22:12:22.763384  6129 sgd_solver.cpp:106] Iteration 42600, lr = 0.000201539
    I1226 22:12:33.353027  6129 solver.cpp:237] Iteration 42700, loss = 0.625513
    I1226 22:12:33.353072  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:12:33.353091  6129 solver.cpp:253]     Train net output #1: loss = 0.625513 (* 1 = 0.625513 loss)
    I1226 22:12:33.353104  6129 sgd_solver.cpp:106] Iteration 42700, lr = 0.000201252
    I1226 22:12:43.925776  6129 solver.cpp:237] Iteration 42800, loss = 0.654304
    I1226 22:12:43.925809  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:12:43.925820  6129 solver.cpp:253]     Train net output #1: loss = 0.654304 (* 1 = 0.654304 loss)
    I1226 22:12:43.925828  6129 sgd_solver.cpp:106] Iteration 42800, lr = 0.000200966
    I1226 22:12:54.496372  6129 solver.cpp:237] Iteration 42900, loss = 0.677151
    I1226 22:12:54.496537  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:12:54.496556  6129 solver.cpp:253]     Train net output #1: loss = 0.677151 (* 1 = 0.677151 loss)
    I1226 22:12:54.496567  6129 sgd_solver.cpp:106] Iteration 42900, lr = 0.000200681
    I1226 22:13:05.002540  6129 solver.cpp:341] Iteration 43000, Testing net (#0)
    I1226 22:13:09.337263  6129 solver.cpp:409]     Test net output #0: accuracy = 0.696416
    I1226 22:13:09.337308  6129 solver.cpp:409]     Test net output #1: loss = 0.869688 (* 1 = 0.869688 loss)
    I1226 22:13:09.381841  6129 solver.cpp:237] Iteration 43000, loss = 0.631716
    I1226 22:13:09.381893  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:13:09.381909  6129 solver.cpp:253]     Train net output #1: loss = 0.631716 (* 1 = 0.631716 loss)
    I1226 22:13:09.381923  6129 sgd_solver.cpp:106] Iteration 43000, lr = 0.000200397
    I1226 22:13:20.059288  6129 solver.cpp:237] Iteration 43100, loss = 0.704461
    I1226 22:13:20.059325  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:13:20.059340  6129 solver.cpp:253]     Train net output #1: loss = 0.704461 (* 1 = 0.704461 loss)
    I1226 22:13:20.059350  6129 sgd_solver.cpp:106] Iteration 43100, lr = 0.000200114
    I1226 22:13:30.635241  6129 solver.cpp:237] Iteration 43200, loss = 0.645124
    I1226 22:13:30.635395  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:13:30.635411  6129 solver.cpp:253]     Train net output #1: loss = 0.645124 (* 1 = 0.645124 loss)
    I1226 22:13:30.635423  6129 sgd_solver.cpp:106] Iteration 43200, lr = 0.000199832
    I1226 22:13:41.223526  6129 solver.cpp:237] Iteration 43300, loss = 0.656302
    I1226 22:13:41.223575  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:13:41.223595  6129 solver.cpp:253]     Train net output #1: loss = 0.656302 (* 1 = 0.656302 loss)
    I1226 22:13:41.223611  6129 sgd_solver.cpp:106] Iteration 43300, lr = 0.00019955
    I1226 22:13:51.860304  6129 solver.cpp:237] Iteration 43400, loss = 0.677731
    I1226 22:13:51.860337  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:13:51.860347  6129 solver.cpp:253]     Train net output #1: loss = 0.677731 (* 1 = 0.677731 loss)
    I1226 22:13:51.860355  6129 sgd_solver.cpp:106] Iteration 43400, lr = 0.00019927
    I1226 22:14:02.471009  6129 solver.cpp:237] Iteration 43500, loss = 0.625957
    I1226 22:14:02.471166  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:14:02.471185  6129 solver.cpp:253]     Train net output #1: loss = 0.625957 (* 1 = 0.625957 loss)
    I1226 22:14:02.471199  6129 sgd_solver.cpp:106] Iteration 43500, lr = 0.000198991
    I1226 22:14:13.053262  6129 solver.cpp:237] Iteration 43600, loss = 0.70766
    I1226 22:14:13.053297  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:14:13.053308  6129 solver.cpp:253]     Train net output #1: loss = 0.70766 (* 1 = 0.70766 loss)
    I1226 22:14:13.053316  6129 sgd_solver.cpp:106] Iteration 43600, lr = 0.000198712
    I1226 22:14:23.692064  6129 solver.cpp:237] Iteration 43700, loss = 0.571385
    I1226 22:14:23.692101  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:14:23.692113  6129 solver.cpp:253]     Train net output #1: loss = 0.571385 (* 1 = 0.571385 loss)
    I1226 22:14:23.692126  6129 sgd_solver.cpp:106] Iteration 43700, lr = 0.000198435
    I1226 22:14:34.229133  6129 solver.cpp:237] Iteration 43800, loss = 0.654076
    I1226 22:14:34.230468  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:14:34.230496  6129 solver.cpp:253]     Train net output #1: loss = 0.654076 (* 1 = 0.654076 loss)
    I1226 22:14:34.230509  6129 sgd_solver.cpp:106] Iteration 43800, lr = 0.000198158
    I1226 22:14:44.814504  6129 solver.cpp:237] Iteration 43900, loss = 0.67625
    I1226 22:14:44.814546  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:14:44.814559  6129 solver.cpp:253]     Train net output #1: loss = 0.67625 (* 1 = 0.67625 loss)
    I1226 22:14:44.814568  6129 sgd_solver.cpp:106] Iteration 43900, lr = 0.000197882
    I1226 22:14:55.329061  6129 solver.cpp:341] Iteration 44000, Testing net (#0)
    I1226 22:14:59.668920  6129 solver.cpp:409]     Test net output #0: accuracy = 0.696167
    I1226 22:14:59.668964  6129 solver.cpp:409]     Test net output #1: loss = 0.874651 (* 1 = 0.874651 loss)
    I1226 22:14:59.713418  6129 solver.cpp:237] Iteration 44000, loss = 0.627935
    I1226 22:14:59.713464  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:14:59.713476  6129 solver.cpp:253]     Train net output #1: loss = 0.627935 (* 1 = 0.627935 loss)
    I1226 22:14:59.713487  6129 sgd_solver.cpp:106] Iteration 44000, lr = 0.000197607
    I1226 22:15:10.353359  6129 solver.cpp:237] Iteration 44100, loss = 0.693979
    I1226 22:15:10.353513  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:15:10.353530  6129 solver.cpp:253]     Train net output #1: loss = 0.693979 (* 1 = 0.693979 loss)
    I1226 22:15:10.353543  6129 sgd_solver.cpp:106] Iteration 44100, lr = 0.000197333
    I1226 22:15:20.977710  6129 solver.cpp:237] Iteration 44200, loss = 0.623141
    I1226 22:15:20.977746  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:15:20.977757  6129 solver.cpp:253]     Train net output #1: loss = 0.623141 (* 1 = 0.623141 loss)
    I1226 22:15:20.977766  6129 sgd_solver.cpp:106] Iteration 44200, lr = 0.00019706
    I1226 22:15:31.598834  6129 solver.cpp:237] Iteration 44300, loss = 0.64835
    I1226 22:15:31.598873  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:15:31.598887  6129 solver.cpp:253]     Train net output #1: loss = 0.64835 (* 1 = 0.64835 loss)
    I1226 22:15:31.598899  6129 sgd_solver.cpp:106] Iteration 44300, lr = 0.000196788
    I1226 22:15:42.156553  6129 solver.cpp:237] Iteration 44400, loss = 0.676937
    I1226 22:15:42.156671  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:15:42.156692  6129 solver.cpp:253]     Train net output #1: loss = 0.676937 (* 1 = 0.676937 loss)
    I1226 22:15:42.156705  6129 sgd_solver.cpp:106] Iteration 44400, lr = 0.000196516
    I1226 22:15:52.768010  6129 solver.cpp:237] Iteration 44500, loss = 0.625389
    I1226 22:15:52.768043  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:15:52.768054  6129 solver.cpp:253]     Train net output #1: loss = 0.625389 (* 1 = 0.625389 loss)
    I1226 22:15:52.768061  6129 sgd_solver.cpp:106] Iteration 44500, lr = 0.000196246
    I1226 22:16:03.372700  6129 solver.cpp:237] Iteration 44600, loss = 0.693851
    I1226 22:16:03.372735  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:16:03.372748  6129 solver.cpp:253]     Train net output #1: loss = 0.693851 (* 1 = 0.693851 loss)
    I1226 22:16:03.372758  6129 sgd_solver.cpp:106] Iteration 44600, lr = 0.000195976
    I1226 22:16:13.964646  6129 solver.cpp:237] Iteration 44700, loss = 0.624327
    I1226 22:16:13.964815  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:16:13.964840  6129 solver.cpp:253]     Train net output #1: loss = 0.624327 (* 1 = 0.624327 loss)
    I1226 22:16:13.964849  6129 sgd_solver.cpp:106] Iteration 44700, lr = 0.000195708
    I1226 22:16:24.564610  6129 solver.cpp:237] Iteration 44800, loss = 0.650001
    I1226 22:16:24.564653  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:16:24.564666  6129 solver.cpp:253]     Train net output #1: loss = 0.650001 (* 1 = 0.650001 loss)
    I1226 22:16:24.564677  6129 sgd_solver.cpp:106] Iteration 44800, lr = 0.00019544
    I1226 22:16:35.141242  6129 solver.cpp:237] Iteration 44900, loss = 0.670929
    I1226 22:16:35.141293  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:16:35.141314  6129 solver.cpp:253]     Train net output #1: loss = 0.670929 (* 1 = 0.670929 loss)
    I1226 22:16:35.141330  6129 sgd_solver.cpp:106] Iteration 44900, lr = 0.000195173
    I1226 22:16:45.681483  6129 solver.cpp:341] Iteration 45000, Testing net (#0)
    I1226 22:16:50.037137  6129 solver.cpp:409]     Test net output #0: accuracy = 0.698833
    I1226 22:16:50.037195  6129 solver.cpp:409]     Test net output #1: loss = 0.872398 (* 1 = 0.872398 loss)
    I1226 22:16:50.081881  6129 solver.cpp:237] Iteration 45000, loss = 0.627897
    I1226 22:16:50.081934  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:16:50.081954  6129 solver.cpp:253]     Train net output #1: loss = 0.627897 (* 1 = 0.627897 loss)
    I1226 22:16:50.081971  6129 sgd_solver.cpp:106] Iteration 45000, lr = 0.000194906
    I1226 22:17:00.779055  6129 solver.cpp:237] Iteration 45100, loss = 0.691758
    I1226 22:17:00.779088  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:17:00.779100  6129 solver.cpp:253]     Train net output #1: loss = 0.691758 (* 1 = 0.691758 loss)
    I1226 22:17:00.779109  6129 sgd_solver.cpp:106] Iteration 45100, lr = 0.000194641
    I1226 22:17:11.374651  6129 solver.cpp:237] Iteration 45200, loss = 0.563185
    I1226 22:17:11.374688  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:17:11.374702  6129 solver.cpp:253]     Train net output #1: loss = 0.563185 (* 1 = 0.563185 loss)
    I1226 22:17:11.374713  6129 sgd_solver.cpp:106] Iteration 45200, lr = 0.000194376
    I1226 22:17:22.017448  6129 solver.cpp:237] Iteration 45300, loss = 0.64829
    I1226 22:17:22.017596  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:17:22.017611  6129 solver.cpp:253]     Train net output #1: loss = 0.64829 (* 1 = 0.64829 loss)
    I1226 22:17:22.017621  6129 sgd_solver.cpp:106] Iteration 45300, lr = 0.000194113
    I1226 22:17:32.616227  6129 solver.cpp:237] Iteration 45400, loss = 0.671137
    I1226 22:17:32.616273  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:17:32.616286  6129 solver.cpp:253]     Train net output #1: loss = 0.671137 (* 1 = 0.671137 loss)
    I1226 22:17:32.616297  6129 sgd_solver.cpp:106] Iteration 45400, lr = 0.00019385
    I1226 22:17:43.221276  6129 solver.cpp:237] Iteration 45500, loss = 0.626924
    I1226 22:17:43.221334  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:17:43.221355  6129 solver.cpp:253]     Train net output #1: loss = 0.626924 (* 1 = 0.626924 loss)
    I1226 22:17:43.221372  6129 sgd_solver.cpp:106] Iteration 45500, lr = 0.000193588
    I1226 22:17:53.770848  6129 solver.cpp:237] Iteration 45600, loss = 0.689238
    I1226 22:17:53.770973  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:17:53.770989  6129 solver.cpp:253]     Train net output #1: loss = 0.689238 (* 1 = 0.689238 loss)
    I1226 22:17:53.770999  6129 sgd_solver.cpp:106] Iteration 45600, lr = 0.000193327
    I1226 22:18:04.402109  6129 solver.cpp:237] Iteration 45700, loss = 0.619349
    I1226 22:18:04.402148  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:18:04.402163  6129 solver.cpp:253]     Train net output #1: loss = 0.619349 (* 1 = 0.619349 loss)
    I1226 22:18:04.402173  6129 sgd_solver.cpp:106] Iteration 45700, lr = 0.000193066
    I1226 22:18:15.069525  6129 solver.cpp:237] Iteration 45800, loss = 0.64693
    I1226 22:18:15.069574  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:18:15.069596  6129 solver.cpp:253]     Train net output #1: loss = 0.64693 (* 1 = 0.64693 loss)
    I1226 22:18:15.069610  6129 sgd_solver.cpp:106] Iteration 45800, lr = 0.000192807
    I1226 22:18:25.666751  6129 solver.cpp:237] Iteration 45900, loss = 0.671115
    I1226 22:18:25.666901  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:18:25.666924  6129 solver.cpp:253]     Train net output #1: loss = 0.671115 (* 1 = 0.671115 loss)
    I1226 22:18:25.666939  6129 sgd_solver.cpp:106] Iteration 45900, lr = 0.000192548
    I1226 22:18:36.159787  6129 solver.cpp:341] Iteration 46000, Testing net (#0)
    I1226 22:18:40.461663  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69525
    I1226 22:18:40.461699  6129 solver.cpp:409]     Test net output #1: loss = 0.873225 (* 1 = 0.873225 loss)
    I1226 22:18:40.506120  6129 solver.cpp:237] Iteration 46000, loss = 0.6172
    I1226 22:18:40.506165  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:18:40.506178  6129 solver.cpp:253]     Train net output #1: loss = 0.6172 (* 1 = 0.6172 loss)
    I1226 22:18:40.506189  6129 sgd_solver.cpp:106] Iteration 46000, lr = 0.00019229
    I1226 22:18:51.159704  6129 solver.cpp:237] Iteration 46100, loss = 0.689973
    I1226 22:18:51.159736  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:18:51.159747  6129 solver.cpp:253]     Train net output #1: loss = 0.689973 (* 1 = 0.689973 loss)
    I1226 22:18:51.159755  6129 sgd_solver.cpp:106] Iteration 46100, lr = 0.000192033
    I1226 22:19:01.838909  6129 solver.cpp:237] Iteration 46200, loss = 0.622127
    I1226 22:19:01.839011  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:19:01.839028  6129 solver.cpp:253]     Train net output #1: loss = 0.622127 (* 1 = 0.622127 loss)
    I1226 22:19:01.839038  6129 sgd_solver.cpp:106] Iteration 46200, lr = 0.000191777
    I1226 22:19:12.442394  6129 solver.cpp:237] Iteration 46300, loss = 0.649946
    I1226 22:19:12.442443  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:19:12.442464  6129 solver.cpp:253]     Train net output #1: loss = 0.649946 (* 1 = 0.649946 loss)
    I1226 22:19:12.442478  6129 sgd_solver.cpp:106] Iteration 46300, lr = 0.000191521
    I1226 22:19:23.101729  6129 solver.cpp:237] Iteration 46400, loss = 0.666918
    I1226 22:19:23.101763  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:19:23.101773  6129 solver.cpp:253]     Train net output #1: loss = 0.666918 (* 1 = 0.666918 loss)
    I1226 22:19:23.101781  6129 sgd_solver.cpp:106] Iteration 46400, lr = 0.000191266
    I1226 22:19:34.576876  6129 solver.cpp:237] Iteration 46500, loss = 0.623388
    I1226 22:19:34.577036  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:19:34.577054  6129 solver.cpp:253]     Train net output #1: loss = 0.623388 (* 1 = 0.623388 loss)
    I1226 22:19:34.577067  6129 sgd_solver.cpp:106] Iteration 46500, lr = 0.000191012
    I1226 22:19:45.952571  6129 solver.cpp:237] Iteration 46600, loss = 0.687239
    I1226 22:19:45.952631  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:19:45.952648  6129 solver.cpp:253]     Train net output #1: loss = 0.687239 (* 1 = 0.687239 loss)
    I1226 22:19:45.952662  6129 sgd_solver.cpp:106] Iteration 46600, lr = 0.000190759
    I1226 22:19:57.490296  6129 solver.cpp:237] Iteration 46700, loss = 0.562873
    I1226 22:19:57.490345  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:19:57.490362  6129 solver.cpp:253]     Train net output #1: loss = 0.562873 (* 1 = 0.562873 loss)
    I1226 22:19:57.490387  6129 sgd_solver.cpp:106] Iteration 46700, lr = 0.000190507
    I1226 22:20:09.979928  6129 solver.cpp:237] Iteration 46800, loss = 0.648408
    I1226 22:20:09.980304  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:20:09.980368  6129 solver.cpp:253]     Train net output #1: loss = 0.648408 (* 1 = 0.648408 loss)
    I1226 22:20:09.980393  6129 sgd_solver.cpp:106] Iteration 46800, lr = 0.000190255
    I1226 22:20:21.165956  6129 solver.cpp:237] Iteration 46900, loss = 0.666104
    I1226 22:20:21.166002  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:20:21.166021  6129 solver.cpp:253]     Train net output #1: loss = 0.666104 (* 1 = 0.666104 loss)
    I1226 22:20:21.166036  6129 sgd_solver.cpp:106] Iteration 46900, lr = 0.000190004
    I1226 22:20:31.855469  6129 solver.cpp:341] Iteration 47000, Testing net (#0)
    I1226 22:20:36.159817  6129 solver.cpp:409]     Test net output #0: accuracy = 0.698333
    I1226 22:20:36.159859  6129 solver.cpp:409]     Test net output #1: loss = 0.867188 (* 1 = 0.867188 loss)
    I1226 22:20:36.204382  6129 solver.cpp:237] Iteration 47000, loss = 0.612492
    I1226 22:20:36.204419  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:20:36.204433  6129 solver.cpp:253]     Train net output #1: loss = 0.612492 (* 1 = 0.612492 loss)
    I1226 22:20:36.204445  6129 sgd_solver.cpp:106] Iteration 47000, lr = 0.000189754
    I1226 22:20:46.699295  6129 solver.cpp:237] Iteration 47100, loss = 0.682062
    I1226 22:20:46.699405  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:20:46.699422  6129 solver.cpp:253]     Train net output #1: loss = 0.682062 (* 1 = 0.682062 loss)
    I1226 22:20:46.699432  6129 sgd_solver.cpp:106] Iteration 47100, lr = 0.000189505
    I1226 22:20:57.481035  6129 solver.cpp:237] Iteration 47200, loss = 0.603556
    I1226 22:20:57.481075  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:20:57.481089  6129 solver.cpp:253]     Train net output #1: loss = 0.603556 (* 1 = 0.603556 loss)
    I1226 22:20:57.481101  6129 sgd_solver.cpp:106] Iteration 47200, lr = 0.000189257
    I1226 22:21:09.851887  6129 solver.cpp:237] Iteration 47300, loss = 0.645146
    I1226 22:21:09.851927  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:21:09.851941  6129 solver.cpp:253]     Train net output #1: loss = 0.645146 (* 1 = 0.645146 loss)
    I1226 22:21:09.851953  6129 sgd_solver.cpp:106] Iteration 47300, lr = 0.000189009
    I1226 22:21:21.582082  6129 solver.cpp:237] Iteration 47400, loss = 0.665333
    I1226 22:21:21.582214  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:21:21.582234  6129 solver.cpp:253]     Train net output #1: loss = 0.665333 (* 1 = 0.665333 loss)
    I1226 22:21:21.582247  6129 sgd_solver.cpp:106] Iteration 47400, lr = 0.000188762
    I1226 22:21:32.582972  6129 solver.cpp:237] Iteration 47500, loss = 0.614447
    I1226 22:21:32.583009  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:21:32.583024  6129 solver.cpp:253]     Train net output #1: loss = 0.614447 (* 1 = 0.614447 loss)
    I1226 22:21:32.583035  6129 sgd_solver.cpp:106] Iteration 47500, lr = 0.000188516
    I1226 22:21:43.406792  6129 solver.cpp:237] Iteration 47600, loss = 0.685528
    I1226 22:21:43.406831  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:21:43.406844  6129 solver.cpp:253]     Train net output #1: loss = 0.685528 (* 1 = 0.685528 loss)
    I1226 22:21:43.406854  6129 sgd_solver.cpp:106] Iteration 47600, lr = 0.00018827
    I1226 22:21:54.962741  6129 solver.cpp:237] Iteration 47700, loss = 0.612284
    I1226 22:21:54.962893  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:21:54.962919  6129 solver.cpp:253]     Train net output #1: loss = 0.612284 (* 1 = 0.612284 loss)
    I1226 22:21:54.962939  6129 sgd_solver.cpp:106] Iteration 47700, lr = 0.000188025
    I1226 22:22:07.276290  6129 solver.cpp:237] Iteration 47800, loss = 0.645809
    I1226 22:22:07.276329  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:22:07.276340  6129 solver.cpp:253]     Train net output #1: loss = 0.645809 (* 1 = 0.645809 loss)
    I1226 22:22:07.276350  6129 sgd_solver.cpp:106] Iteration 47800, lr = 0.000187781
    I1226 22:22:18.485275  6129 solver.cpp:237] Iteration 47900, loss = 0.663218
    I1226 22:22:18.485321  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:22:18.485337  6129 solver.cpp:253]     Train net output #1: loss = 0.663218 (* 1 = 0.663218 loss)
    I1226 22:22:18.485348  6129 sgd_solver.cpp:106] Iteration 47900, lr = 0.000187538
    I1226 22:22:29.652601  6129 solver.cpp:341] Iteration 48000, Testing net (#0)
    I1226 22:22:33.949888  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69925
    I1226 22:22:33.949956  6129 solver.cpp:409]     Test net output #1: loss = 0.866184 (* 1 = 0.866184 loss)
    I1226 22:22:33.994786  6129 solver.cpp:237] Iteration 48000, loss = 0.60868
    I1226 22:22:33.994834  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:22:33.994854  6129 solver.cpp:253]     Train net output #1: loss = 0.60868 (* 1 = 0.60868 loss)
    I1226 22:22:33.994871  6129 sgd_solver.cpp:106] Iteration 48000, lr = 0.000187295
    I1226 22:22:44.760009  6129 solver.cpp:237] Iteration 48100, loss = 0.678122
    I1226 22:22:44.760042  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:22:44.760053  6129 solver.cpp:253]     Train net output #1: loss = 0.678122 (* 1 = 0.678122 loss)
    I1226 22:22:44.760062  6129 sgd_solver.cpp:106] Iteration 48100, lr = 0.000187054
    I1226 22:22:55.515090  6129 solver.cpp:237] Iteration 48200, loss = 0.580057
    I1226 22:22:55.515126  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:22:55.515139  6129 solver.cpp:253]     Train net output #1: loss = 0.580057 (* 1 = 0.580057 loss)
    I1226 22:22:55.515148  6129 sgd_solver.cpp:106] Iteration 48200, lr = 0.000186812
    I1226 22:23:07.500114  6129 solver.cpp:237] Iteration 48300, loss = 0.643526
    I1226 22:23:07.500314  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:23:07.500344  6129 solver.cpp:253]     Train net output #1: loss = 0.643526 (* 1 = 0.643526 loss)
    I1226 22:23:07.500360  6129 sgd_solver.cpp:106] Iteration 48300, lr = 0.000186572
    I1226 22:23:18.052242  6129 solver.cpp:237] Iteration 48400, loss = 0.662883
    I1226 22:23:18.052291  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:23:18.052309  6129 solver.cpp:253]     Train net output #1: loss = 0.662883 (* 1 = 0.662883 loss)
    I1226 22:23:18.052325  6129 sgd_solver.cpp:106] Iteration 48400, lr = 0.000186332
    I1226 22:23:28.524384  6129 solver.cpp:237] Iteration 48500, loss = 0.60886
    I1226 22:23:28.524440  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:23:28.524467  6129 solver.cpp:253]     Train net output #1: loss = 0.60886 (* 1 = 0.60886 loss)
    I1226 22:23:28.524484  6129 sgd_solver.cpp:106] Iteration 48500, lr = 0.000186093
    I1226 22:23:39.168795  6129 solver.cpp:237] Iteration 48600, loss = 0.676195
    I1226 22:23:39.168943  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:23:39.168956  6129 solver.cpp:253]     Train net output #1: loss = 0.676195 (* 1 = 0.676195 loss)
    I1226 22:23:39.168965  6129 sgd_solver.cpp:106] Iteration 48600, lr = 0.000185855
    I1226 22:23:49.671308  6129 solver.cpp:237] Iteration 48700, loss = 0.598596
    I1226 22:23:49.671353  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:23:49.671372  6129 solver.cpp:253]     Train net output #1: loss = 0.598596 (* 1 = 0.598596 loss)
    I1226 22:23:49.671386  6129 sgd_solver.cpp:106] Iteration 48700, lr = 0.000185618
    I1226 22:24:00.133172  6129 solver.cpp:237] Iteration 48800, loss = 0.641366
    I1226 22:24:00.133213  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:24:00.133224  6129 solver.cpp:253]     Train net output #1: loss = 0.641366 (* 1 = 0.641366 loss)
    I1226 22:24:00.133232  6129 sgd_solver.cpp:106] Iteration 48800, lr = 0.000185381
    I1226 22:24:10.614032  6129 solver.cpp:237] Iteration 48900, loss = 0.66269
    I1226 22:24:10.615486  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:24:10.615507  6129 solver.cpp:253]     Train net output #1: loss = 0.66269 (* 1 = 0.66269 loss)
    I1226 22:24:10.615523  6129 sgd_solver.cpp:106] Iteration 48900, lr = 0.000185145
    I1226 22:24:20.989202  6129 solver.cpp:341] Iteration 49000, Testing net (#0)
    I1226 22:24:25.263814  6129 solver.cpp:409]     Test net output #0: accuracy = 0.698667
    I1226 22:24:25.263871  6129 solver.cpp:409]     Test net output #1: loss = 0.870104 (* 1 = 0.870104 loss)
    I1226 22:24:25.308579  6129 solver.cpp:237] Iteration 49000, loss = 0.606335
    I1226 22:24:25.308620  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:24:25.308640  6129 solver.cpp:253]     Train net output #1: loss = 0.606335 (* 1 = 0.606335 loss)
    I1226 22:24:25.308655  6129 sgd_solver.cpp:106] Iteration 49000, lr = 0.000184909
    I1226 22:24:36.023829  6129 solver.cpp:237] Iteration 49100, loss = 0.686588
    I1226 22:24:36.023862  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:24:36.023874  6129 solver.cpp:253]     Train net output #1: loss = 0.686588 (* 1 = 0.686588 loss)
    I1226 22:24:36.023883  6129 sgd_solver.cpp:106] Iteration 49100, lr = 0.000184675
    I1226 22:24:46.503959  6129 solver.cpp:237] Iteration 49200, loss = 0.5735
    I1226 22:24:46.504108  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:24:46.504130  6129 solver.cpp:253]     Train net output #1: loss = 0.5735 (* 1 = 0.5735 loss)
    I1226 22:24:46.504143  6129 sgd_solver.cpp:106] Iteration 49200, lr = 0.000184441
    I1226 22:24:58.146579  6129 solver.cpp:237] Iteration 49300, loss = 0.640505
    I1226 22:24:58.146615  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:24:58.146625  6129 solver.cpp:253]     Train net output #1: loss = 0.640505 (* 1 = 0.640505 loss)
    I1226 22:24:58.146636  6129 sgd_solver.cpp:106] Iteration 49300, lr = 0.000184207
    I1226 22:25:10.817720  6129 solver.cpp:237] Iteration 49400, loss = 0.658391
    I1226 22:25:10.817760  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:25:10.817775  6129 solver.cpp:253]     Train net output #1: loss = 0.658391 (* 1 = 0.658391 loss)
    I1226 22:25:10.817786  6129 sgd_solver.cpp:106] Iteration 49400, lr = 0.000183975
    I1226 22:25:21.764853  6129 solver.cpp:237] Iteration 49500, loss = 0.607486
    I1226 22:25:21.764972  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:25:21.764992  6129 solver.cpp:253]     Train net output #1: loss = 0.607486 (* 1 = 0.607486 loss)
    I1226 22:25:21.765007  6129 sgd_solver.cpp:106] Iteration 49500, lr = 0.000183743
    I1226 22:25:32.711915  6129 solver.cpp:237] Iteration 49600, loss = 0.679041
    I1226 22:25:32.711966  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:25:32.711987  6129 solver.cpp:253]     Train net output #1: loss = 0.679041 (* 1 = 0.679041 loss)
    I1226 22:25:32.712003  6129 sgd_solver.cpp:106] Iteration 49600, lr = 0.000183512
    I1226 22:25:43.538229  6129 solver.cpp:237] Iteration 49700, loss = 0.584202
    I1226 22:25:43.538272  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:25:43.538288  6129 solver.cpp:253]     Train net output #1: loss = 0.584202 (* 1 = 0.584202 loss)
    I1226 22:25:43.538300  6129 sgd_solver.cpp:106] Iteration 49700, lr = 0.000183281
    I1226 22:25:54.235972  6129 solver.cpp:237] Iteration 49800, loss = 0.640959
    I1226 22:25:54.236095  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:25:54.236109  6129 solver.cpp:253]     Train net output #1: loss = 0.640959 (* 1 = 0.640959 loss)
    I1226 22:25:54.236119  6129 sgd_solver.cpp:106] Iteration 49800, lr = 0.000183051
    I1226 22:26:04.909116  6129 solver.cpp:237] Iteration 49900, loss = 0.659593
    I1226 22:26:04.909163  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:26:04.909184  6129 solver.cpp:253]     Train net output #1: loss = 0.659593 (* 1 = 0.659593 loss)
    I1226 22:26:04.909198  6129 sgd_solver.cpp:106] Iteration 49900, lr = 0.000182822
    I1226 22:26:15.576726  6129 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_50000.caffemodel
    I1226 22:26:15.672888  6129 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_50000.solverstate
    I1226 22:26:15.674625  6129 solver.cpp:341] Iteration 50000, Testing net (#0)
    I1226 22:26:20.876086  6129 solver.cpp:409]     Test net output #0: accuracy = 0.702417
    I1226 22:26:20.876122  6129 solver.cpp:409]     Test net output #1: loss = 0.863403 (* 1 = 0.863403 loss)
    I1226 22:26:20.921905  6129 solver.cpp:237] Iteration 50000, loss = 0.603513
    I1226 22:26:20.921953  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:26:20.921967  6129 solver.cpp:253]     Train net output #1: loss = 0.603513 (* 1 = 0.603513 loss)
    I1226 22:26:20.921978  6129 sgd_solver.cpp:106] Iteration 50000, lr = 0.000182593
    I1226 22:26:33.507192  6129 solver.cpp:237] Iteration 50100, loss = 0.671972
    I1226 22:26:33.507346  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:26:33.507365  6129 solver.cpp:253]     Train net output #1: loss = 0.671972 (* 1 = 0.671972 loss)
    I1226 22:26:33.507376  6129 sgd_solver.cpp:106] Iteration 50100, lr = 0.000182365
    I1226 22:26:45.559264  6129 solver.cpp:237] Iteration 50200, loss = 0.60202
    I1226 22:26:45.559319  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:26:45.559340  6129 solver.cpp:253]     Train net output #1: loss = 0.60202 (* 1 = 0.60202 loss)
    I1226 22:26:45.559356  6129 sgd_solver.cpp:106] Iteration 50200, lr = 0.000182138
    I1226 22:26:56.943961  6129 solver.cpp:237] Iteration 50300, loss = 0.642042
    I1226 22:26:56.944008  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:26:56.944021  6129 solver.cpp:253]     Train net output #1: loss = 0.642042 (* 1 = 0.642042 loss)
    I1226 22:26:56.944032  6129 sgd_solver.cpp:106] Iteration 50300, lr = 0.000181911
    I1226 22:27:07.556545  6129 solver.cpp:237] Iteration 50400, loss = 0.659201
    I1226 22:27:07.556691  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:27:07.556712  6129 solver.cpp:253]     Train net output #1: loss = 0.659201 (* 1 = 0.659201 loss)
    I1226 22:27:07.556725  6129 sgd_solver.cpp:106] Iteration 50400, lr = 0.000181686
    I1226 22:27:18.367341  6129 solver.cpp:237] Iteration 50500, loss = 0.602553
    I1226 22:27:18.367393  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:27:18.367414  6129 solver.cpp:253]     Train net output #1: loss = 0.602553 (* 1 = 0.602553 loss)
    I1226 22:27:18.367431  6129 sgd_solver.cpp:106] Iteration 50500, lr = 0.00018146
    I1226 22:27:30.699939  6129 solver.cpp:237] Iteration 50600, loss = 0.683202
    I1226 22:27:30.699981  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:27:30.699996  6129 solver.cpp:253]     Train net output #1: loss = 0.683202 (* 1 = 0.683202 loss)
    I1226 22:27:30.700007  6129 sgd_solver.cpp:106] Iteration 50600, lr = 0.000181236
    I1226 22:27:42.434259  6129 solver.cpp:237] Iteration 50700, loss = 0.579642
    I1226 22:27:42.434403  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:27:42.434423  6129 solver.cpp:253]     Train net output #1: loss = 0.579642 (* 1 = 0.579642 loss)
    I1226 22:27:42.434435  6129 sgd_solver.cpp:106] Iteration 50700, lr = 0.000181012
    I1226 22:27:55.283326  6129 solver.cpp:237] Iteration 50800, loss = 0.639196
    I1226 22:27:55.283380  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:27:55.283401  6129 solver.cpp:253]     Train net output #1: loss = 0.639196 (* 1 = 0.639196 loss)
    I1226 22:27:55.283418  6129 sgd_solver.cpp:106] Iteration 50800, lr = 0.000180788
    I1226 22:28:07.755427  6129 solver.cpp:237] Iteration 50900, loss = 0.656616
    I1226 22:28:07.755463  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:28:07.755475  6129 solver.cpp:253]     Train net output #1: loss = 0.656616 (* 1 = 0.656616 loss)
    I1226 22:28:07.755483  6129 sgd_solver.cpp:106] Iteration 50900, lr = 0.000180566
    I1226 22:28:20.062479  6129 solver.cpp:341] Iteration 51000, Testing net (#0)
    I1226 22:28:24.352152  6129 solver.cpp:409]     Test net output #0: accuracy = 0.696667
    I1226 22:28:24.352192  6129 solver.cpp:409]     Test net output #1: loss = 0.869163 (* 1 = 0.869163 loss)
    I1226 22:28:24.396600  6129 solver.cpp:237] Iteration 51000, loss = 0.602475
    I1226 22:28:24.396618  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:28:24.396628  6129 solver.cpp:253]     Train net output #1: loss = 0.602475 (* 1 = 0.602475 loss)
    I1226 22:28:24.396637  6129 sgd_solver.cpp:106] Iteration 51000, lr = 0.000180344
    I1226 22:28:35.068578  6129 solver.cpp:237] Iteration 51100, loss = 0.68282
    I1226 22:28:35.068614  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:28:35.068625  6129 solver.cpp:253]     Train net output #1: loss = 0.68282 (* 1 = 0.68282 loss)
    I1226 22:28:35.068635  6129 sgd_solver.cpp:106] Iteration 51100, lr = 0.000180122
    I1226 22:28:46.299361  6129 solver.cpp:237] Iteration 51200, loss = 0.569114
    I1226 22:28:46.299407  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:28:46.299425  6129 solver.cpp:253]     Train net output #1: loss = 0.569114 (* 1 = 0.569114 loss)
    I1226 22:28:46.299439  6129 sgd_solver.cpp:106] Iteration 51200, lr = 0.000179901
    I1226 22:28:59.626296  6129 solver.cpp:237] Iteration 51300, loss = 0.638053
    I1226 22:28:59.626516  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:28:59.626548  6129 solver.cpp:253]     Train net output #1: loss = 0.638053 (* 1 = 0.638053 loss)
    I1226 22:28:59.626564  6129 sgd_solver.cpp:106] Iteration 51300, lr = 0.000179681
    I1226 22:29:15.349264  6129 solver.cpp:237] Iteration 51400, loss = 0.653649
    I1226 22:29:15.349320  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:29:15.349341  6129 solver.cpp:253]     Train net output #1: loss = 0.653649 (* 1 = 0.653649 loss)
    I1226 22:29:15.349359  6129 sgd_solver.cpp:106] Iteration 51400, lr = 0.000179462
    I1226 22:29:27.288348  6129 solver.cpp:237] Iteration 51500, loss = 0.600614
    I1226 22:29:27.288384  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:29:27.288396  6129 solver.cpp:253]     Train net output #1: loss = 0.600614 (* 1 = 0.600614 loss)
    I1226 22:29:27.288406  6129 sgd_solver.cpp:106] Iteration 51500, lr = 0.000179243
    I1226 22:29:39.027325  6129 solver.cpp:237] Iteration 51600, loss = 0.673264
    I1226 22:29:39.027482  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:29:39.027503  6129 solver.cpp:253]     Train net output #1: loss = 0.673264 (* 1 = 0.673264 loss)
    I1226 22:29:39.027515  6129 sgd_solver.cpp:106] Iteration 51600, lr = 0.000179025
    I1226 22:29:52.560624  6129 solver.cpp:237] Iteration 51700, loss = 0.585903
    I1226 22:29:52.560660  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:29:52.560673  6129 solver.cpp:253]     Train net output #1: loss = 0.585903 (* 1 = 0.585903 loss)
    I1226 22:29:52.560684  6129 sgd_solver.cpp:106] Iteration 51700, lr = 0.000178807
    I1226 22:30:05.670588  6129 solver.cpp:237] Iteration 51800, loss = 0.639545
    I1226 22:30:05.670627  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:30:05.670640  6129 solver.cpp:253]     Train net output #1: loss = 0.639545 (* 1 = 0.639545 loss)
    I1226 22:30:05.670651  6129 sgd_solver.cpp:106] Iteration 51800, lr = 0.00017859
    I1226 22:30:16.818745  6129 solver.cpp:237] Iteration 51900, loss = 0.653117
    I1226 22:30:16.818891  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:30:16.818903  6129 solver.cpp:253]     Train net output #1: loss = 0.653117 (* 1 = 0.653117 loss)
    I1226 22:30:16.818912  6129 sgd_solver.cpp:106] Iteration 51900, lr = 0.000178373
    I1226 22:30:29.225162  6129 solver.cpp:341] Iteration 52000, Testing net (#0)
    I1226 22:30:34.449988  6129 solver.cpp:409]     Test net output #0: accuracy = 0.699667
    I1226 22:30:34.450044  6129 solver.cpp:409]     Test net output #1: loss = 0.862357 (* 1 = 0.862357 loss)
    I1226 22:30:34.510581  6129 solver.cpp:237] Iteration 52000, loss = 0.599879
    I1226 22:30:34.510634  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:30:34.510655  6129 solver.cpp:253]     Train net output #1: loss = 0.599879 (* 1 = 0.599879 loss)
    I1226 22:30:34.510671  6129 sgd_solver.cpp:106] Iteration 52000, lr = 0.000178158
    I1226 22:30:46.881777  6129 solver.cpp:237] Iteration 52100, loss = 0.671719
    I1226 22:30:46.881922  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:30:46.881937  6129 solver.cpp:253]     Train net output #1: loss = 0.671719 (* 1 = 0.671719 loss)
    I1226 22:30:46.881947  6129 sgd_solver.cpp:106] Iteration 52100, lr = 0.000177942
    I1226 22:30:58.485553  6129 solver.cpp:237] Iteration 52200, loss = 0.579899
    I1226 22:30:58.485589  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:30:58.485599  6129 solver.cpp:253]     Train net output #1: loss = 0.579899 (* 1 = 0.579899 loss)
    I1226 22:30:58.485607  6129 sgd_solver.cpp:106] Iteration 52200, lr = 0.000177728
    I1226 22:31:10.737795  6129 solver.cpp:237] Iteration 52300, loss = 0.639581
    I1226 22:31:10.737833  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:31:10.737845  6129 solver.cpp:253]     Train net output #1: loss = 0.639581 (* 1 = 0.639581 loss)
    I1226 22:31:10.737854  6129 sgd_solver.cpp:106] Iteration 52300, lr = 0.000177514
    I1226 22:31:26.571568  6129 solver.cpp:237] Iteration 52400, loss = 0.653522
    I1226 22:31:26.571707  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:31:26.571732  6129 solver.cpp:253]     Train net output #1: loss = 0.653522 (* 1 = 0.653522 loss)
    I1226 22:31:26.571744  6129 sgd_solver.cpp:106] Iteration 52400, lr = 0.0001773
    I1226 22:31:38.142192  6129 solver.cpp:237] Iteration 52500, loss = 0.600255
    I1226 22:31:38.142256  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:31:38.142283  6129 solver.cpp:253]     Train net output #1: loss = 0.600255 (* 1 = 0.600255 loss)
    I1226 22:31:38.142305  6129 sgd_solver.cpp:106] Iteration 52500, lr = 0.000177088
    I1226 22:31:50.579557  6129 solver.cpp:237] Iteration 52600, loss = 0.670778
    I1226 22:31:50.579598  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:31:50.579614  6129 solver.cpp:253]     Train net output #1: loss = 0.670778 (* 1 = 0.670778 loss)
    I1226 22:31:50.579627  6129 sgd_solver.cpp:106] Iteration 52600, lr = 0.000176875
    I1226 22:32:02.073405  6129 solver.cpp:237] Iteration 52700, loss = 0.578259
    I1226 22:32:02.073549  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:32:02.073564  6129 solver.cpp:253]     Train net output #1: loss = 0.578259 (* 1 = 0.578259 loss)
    I1226 22:32:02.073571  6129 sgd_solver.cpp:106] Iteration 52700, lr = 0.000176664
    I1226 22:32:14.219310  6129 solver.cpp:237] Iteration 52800, loss = 0.637827
    I1226 22:32:14.219357  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:32:14.219377  6129 solver.cpp:253]     Train net output #1: loss = 0.637827 (* 1 = 0.637827 loss)
    I1226 22:32:14.219390  6129 sgd_solver.cpp:106] Iteration 52800, lr = 0.000176453
    I1226 22:32:24.744017  6129 solver.cpp:237] Iteration 52900, loss = 0.649021
    I1226 22:32:24.744051  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:32:24.744063  6129 solver.cpp:253]     Train net output #1: loss = 0.649021 (* 1 = 0.649021 loss)
    I1226 22:32:24.744072  6129 sgd_solver.cpp:106] Iteration 52900, lr = 0.000176242
    I1226 22:32:35.624383  6129 solver.cpp:341] Iteration 53000, Testing net (#0)
    I1226 22:32:40.102162  6129 solver.cpp:409]     Test net output #0: accuracy = 0.700417
    I1226 22:32:40.102200  6129 solver.cpp:409]     Test net output #1: loss = 0.859691 (* 1 = 0.859691 loss)
    I1226 22:32:40.146628  6129 solver.cpp:237] Iteration 53000, loss = 0.594994
    I1226 22:32:40.146677  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:32:40.146690  6129 solver.cpp:253]     Train net output #1: loss = 0.594994 (* 1 = 0.594994 loss)
    I1226 22:32:40.146700  6129 sgd_solver.cpp:106] Iteration 53000, lr = 0.000176032
    I1226 22:32:50.700603  6129 solver.cpp:237] Iteration 53100, loss = 0.668953
    I1226 22:32:50.700642  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:32:50.700659  6129 solver.cpp:253]     Train net output #1: loss = 0.668953 (* 1 = 0.668953 loss)
    I1226 22:32:50.700670  6129 sgd_solver.cpp:106] Iteration 53100, lr = 0.000175823
    I1226 22:33:01.966820  6129 solver.cpp:237] Iteration 53200, loss = 0.565159
    I1226 22:33:01.966859  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:33:01.966871  6129 solver.cpp:253]     Train net output #1: loss = 0.565159 (* 1 = 0.565159 loss)
    I1226 22:33:01.966881  6129 sgd_solver.cpp:106] Iteration 53200, lr = 0.000175614
    I1226 22:33:12.963239  6129 solver.cpp:237] Iteration 53300, loss = 0.63912
    I1226 22:33:12.963399  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:33:12.963424  6129 solver.cpp:253]     Train net output #1: loss = 0.63912 (* 1 = 0.63912 loss)
    I1226 22:33:12.963441  6129 sgd_solver.cpp:106] Iteration 53300, lr = 0.000175406
    I1226 22:33:24.207543  6129 solver.cpp:237] Iteration 53400, loss = 0.650274
    I1226 22:33:24.207576  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:33:24.207587  6129 solver.cpp:253]     Train net output #1: loss = 0.650274 (* 1 = 0.650274 loss)
    I1226 22:33:24.207597  6129 sgd_solver.cpp:106] Iteration 53400, lr = 0.000175199
    I1226 22:33:35.910930  6129 solver.cpp:237] Iteration 53500, loss = 0.594242
    I1226 22:33:35.910969  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:33:35.910980  6129 solver.cpp:253]     Train net output #1: loss = 0.594242 (* 1 = 0.594242 loss)
    I1226 22:33:35.910991  6129 sgd_solver.cpp:106] Iteration 53500, lr = 0.000174992
    I1226 22:33:49.609544  6129 solver.cpp:237] Iteration 53600, loss = 0.665495
    I1226 22:33:49.609694  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:33:49.609706  6129 solver.cpp:253]     Train net output #1: loss = 0.665495 (* 1 = 0.665495 loss)
    I1226 22:33:49.609716  6129 sgd_solver.cpp:106] Iteration 53600, lr = 0.000174785
    I1226 22:34:00.172186  6129 solver.cpp:237] Iteration 53700, loss = 0.561978
    I1226 22:34:00.172240  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:34:00.172260  6129 solver.cpp:253]     Train net output #1: loss = 0.561978 (* 1 = 0.561978 loss)
    I1226 22:34:00.172277  6129 sgd_solver.cpp:106] Iteration 53700, lr = 0.00017458
    I1226 22:34:10.733784  6129 solver.cpp:237] Iteration 53800, loss = 0.639372
    I1226 22:34:10.733819  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:34:10.733830  6129 solver.cpp:253]     Train net output #1: loss = 0.639372 (* 1 = 0.639372 loss)
    I1226 22:34:10.733839  6129 sgd_solver.cpp:106] Iteration 53800, lr = 0.000174374
    I1226 22:34:21.229171  6129 solver.cpp:237] Iteration 53900, loss = 0.650254
    I1226 22:34:21.229269  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:34:21.229285  6129 solver.cpp:253]     Train net output #1: loss = 0.650254 (* 1 = 0.650254 loss)
    I1226 22:34:21.229295  6129 sgd_solver.cpp:106] Iteration 53900, lr = 0.00017417
    I1226 22:34:32.998778  6129 solver.cpp:341] Iteration 54000, Testing net (#0)
    I1226 22:34:37.856745  6129 solver.cpp:409]     Test net output #0: accuracy = 0.699083
    I1226 22:34:37.856801  6129 solver.cpp:409]     Test net output #1: loss = 0.863368 (* 1 = 0.863368 loss)
    I1226 22:34:37.901409  6129 solver.cpp:237] Iteration 54000, loss = 0.592496
    I1226 22:34:37.901463  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:34:37.901479  6129 solver.cpp:253]     Train net output #1: loss = 0.592496 (* 1 = 0.592496 loss)
    I1226 22:34:37.901492  6129 sgd_solver.cpp:106] Iteration 54000, lr = 0.000173965
    I1226 22:34:49.764955  6129 solver.cpp:237] Iteration 54100, loss = 0.664327
    I1226 22:34:49.764993  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:34:49.765007  6129 solver.cpp:253]     Train net output #1: loss = 0.664327 (* 1 = 0.664327 loss)
    I1226 22:34:49.765019  6129 sgd_solver.cpp:106] Iteration 54100, lr = 0.000173762
    I1226 22:35:03.011055  6129 solver.cpp:237] Iteration 54200, loss = 0.583751
    I1226 22:35:03.011198  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:35:03.011221  6129 solver.cpp:253]     Train net output #1: loss = 0.583751 (* 1 = 0.583751 loss)
    I1226 22:35:03.011239  6129 sgd_solver.cpp:106] Iteration 54200, lr = 0.000173559
    I1226 22:35:14.033118  6129 solver.cpp:237] Iteration 54300, loss = 0.640112
    I1226 22:35:14.033157  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:35:14.033172  6129 solver.cpp:253]     Train net output #1: loss = 0.640112 (* 1 = 0.640112 loss)
    I1226 22:35:14.033185  6129 sgd_solver.cpp:106] Iteration 54300, lr = 0.000173356
    I1226 22:35:24.861176  6129 solver.cpp:237] Iteration 54400, loss = 0.647689
    I1226 22:35:24.861212  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:35:24.861225  6129 solver.cpp:253]     Train net output #1: loss = 0.647689 (* 1 = 0.647689 loss)
    I1226 22:35:24.861235  6129 sgd_solver.cpp:106] Iteration 54400, lr = 0.000173154
    I1226 22:35:36.582165  6129 solver.cpp:237] Iteration 54500, loss = 0.593148
    I1226 22:35:36.582329  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:35:36.582355  6129 solver.cpp:253]     Train net output #1: loss = 0.593148 (* 1 = 0.593148 loss)
    I1226 22:35:36.582372  6129 sgd_solver.cpp:106] Iteration 54500, lr = 0.000172953
    I1226 22:35:49.152279  6129 solver.cpp:237] Iteration 54600, loss = 0.672466
    I1226 22:35:49.152343  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:35:49.152366  6129 solver.cpp:253]     Train net output #1: loss = 0.672466 (* 1 = 0.672466 loss)
    I1226 22:35:49.152384  6129 sgd_solver.cpp:106] Iteration 54600, lr = 0.000172752
    I1226 22:36:00.995542  6129 solver.cpp:237] Iteration 54700, loss = 0.556771
    I1226 22:36:00.995575  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:36:00.995586  6129 solver.cpp:253]     Train net output #1: loss = 0.556771 (* 1 = 0.556771 loss)
    I1226 22:36:00.995596  6129 sgd_solver.cpp:106] Iteration 54700, lr = 0.000172552
    I1226 22:36:11.484108  6129 solver.cpp:237] Iteration 54800, loss = 0.640872
    I1226 22:36:11.484218  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:36:11.484235  6129 solver.cpp:253]     Train net output #1: loss = 0.640872 (* 1 = 0.640872 loss)
    I1226 22:36:11.484244  6129 sgd_solver.cpp:106] Iteration 54800, lr = 0.000172352
    I1226 22:36:21.983537  6129 solver.cpp:237] Iteration 54900, loss = 0.647272
    I1226 22:36:21.983600  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:36:21.983628  6129 solver.cpp:253]     Train net output #1: loss = 0.647272 (* 1 = 0.647272 loss)
    I1226 22:36:21.983649  6129 sgd_solver.cpp:106] Iteration 54900, lr = 0.000172153
    I1226 22:36:32.358500  6129 solver.cpp:341] Iteration 55000, Testing net (#0)
    I1226 22:36:36.637737  6129 solver.cpp:409]     Test net output #0: accuracy = 0.703333
    I1226 22:36:36.637774  6129 solver.cpp:409]     Test net output #1: loss = 0.857285 (* 1 = 0.857285 loss)
    I1226 22:36:36.682121  6129 solver.cpp:237] Iteration 55000, loss = 0.590064
    I1226 22:36:36.682159  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:36:36.682171  6129 solver.cpp:253]     Train net output #1: loss = 0.590064 (* 1 = 0.590064 loss)
    I1226 22:36:36.682183  6129 sgd_solver.cpp:106] Iteration 55000, lr = 0.000171954
    I1226 22:36:47.200906  6129 solver.cpp:237] Iteration 55100, loss = 0.66108
    I1226 22:36:47.201009  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:36:47.201025  6129 solver.cpp:253]     Train net output #1: loss = 0.66108 (* 1 = 0.66108 loss)
    I1226 22:36:47.201035  6129 sgd_solver.cpp:106] Iteration 55100, lr = 0.000171756
    I1226 22:36:57.695241  6129 solver.cpp:237] Iteration 55200, loss = 0.570186
    I1226 22:36:57.695279  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:36:57.695294  6129 solver.cpp:253]     Train net output #1: loss = 0.570186 (* 1 = 0.570186 loss)
    I1226 22:36:57.695305  6129 sgd_solver.cpp:106] Iteration 55200, lr = 0.000171559
    I1226 22:37:08.148561  6129 solver.cpp:237] Iteration 55300, loss = 0.639018
    I1226 22:37:08.148597  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:37:08.148608  6129 solver.cpp:253]     Train net output #1: loss = 0.639018 (* 1 = 0.639018 loss)
    I1226 22:37:08.148617  6129 sgd_solver.cpp:106] Iteration 55300, lr = 0.000171361
    I1226 22:37:18.643565  6129 solver.cpp:237] Iteration 55400, loss = 0.646376
    I1226 22:37:18.643717  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:37:18.643739  6129 solver.cpp:253]     Train net output #1: loss = 0.646376 (* 1 = 0.646376 loss)
    I1226 22:37:18.643754  6129 sgd_solver.cpp:106] Iteration 55400, lr = 0.000171165
    I1226 22:37:29.260579  6129 solver.cpp:237] Iteration 55500, loss = 0.588726
    I1226 22:37:29.260627  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:37:29.260644  6129 solver.cpp:253]     Train net output #1: loss = 0.588726 (* 1 = 0.588726 loss)
    I1226 22:37:29.260658  6129 sgd_solver.cpp:106] Iteration 55500, lr = 0.000170969
    I1226 22:37:39.733907  6129 solver.cpp:237] Iteration 55600, loss = 0.660546
    I1226 22:37:39.733945  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:37:39.733958  6129 solver.cpp:253]     Train net output #1: loss = 0.660546 (* 1 = 0.660546 loss)
    I1226 22:37:39.733968  6129 sgd_solver.cpp:106] Iteration 55600, lr = 0.000170773
    I1226 22:37:50.182231  6129 solver.cpp:237] Iteration 55700, loss = 0.565305
    I1226 22:37:50.182345  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:37:50.182368  6129 solver.cpp:253]     Train net output #1: loss = 0.565305 (* 1 = 0.565305 loss)
    I1226 22:37:50.182384  6129 sgd_solver.cpp:106] Iteration 55700, lr = 0.000170578
    I1226 22:38:00.681627  6129 solver.cpp:237] Iteration 55800, loss = 0.637909
    I1226 22:38:00.681663  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:38:00.681679  6129 solver.cpp:253]     Train net output #1: loss = 0.637909 (* 1 = 0.637909 loss)
    I1226 22:38:00.681689  6129 sgd_solver.cpp:106] Iteration 55800, lr = 0.000170384
    I1226 22:38:13.316444  6129 solver.cpp:237] Iteration 55900, loss = 0.645359
    I1226 22:38:13.316498  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:38:13.316516  6129 solver.cpp:253]     Train net output #1: loss = 0.645359 (* 1 = 0.645359 loss)
    I1226 22:38:13.316531  6129 sgd_solver.cpp:106] Iteration 55900, lr = 0.00017019
    I1226 22:38:26.614214  6129 solver.cpp:341] Iteration 56000, Testing net (#0)
    I1226 22:38:30.904178  6129 solver.cpp:409]     Test net output #0: accuracy = 0.69925
    I1226 22:38:30.904230  6129 solver.cpp:409]     Test net output #1: loss = 0.862845 (* 1 = 0.862845 loss)
    I1226 22:38:30.948870  6129 solver.cpp:237] Iteration 56000, loss = 0.588421
    I1226 22:38:30.948925  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:38:30.948945  6129 solver.cpp:253]     Train net output #1: loss = 0.588421 (* 1 = 0.588421 loss)
    I1226 22:38:30.948961  6129 sgd_solver.cpp:106] Iteration 56000, lr = 0.000169997
    I1226 22:38:41.438110  6129 solver.cpp:237] Iteration 56100, loss = 0.661481
    I1226 22:38:41.438154  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:38:41.438172  6129 solver.cpp:253]     Train net output #1: loss = 0.661481 (* 1 = 0.661481 loss)
    I1226 22:38:41.438186  6129 sgd_solver.cpp:106] Iteration 56100, lr = 0.000169804
    I1226 22:38:51.914013  6129 solver.cpp:237] Iteration 56200, loss = 0.564787
    I1226 22:38:51.914050  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:38:51.914064  6129 solver.cpp:253]     Train net output #1: loss = 0.564787 (* 1 = 0.564787 loss)
    I1226 22:38:51.914074  6129 sgd_solver.cpp:106] Iteration 56200, lr = 0.000169611
    I1226 22:39:02.419210  6129 solver.cpp:237] Iteration 56300, loss = 0.637371
    I1226 22:39:02.419334  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:39:02.419356  6129 solver.cpp:253]     Train net output #1: loss = 0.637371 (* 1 = 0.637371 loss)
    I1226 22:39:02.419370  6129 sgd_solver.cpp:106] Iteration 56300, lr = 0.000169419
    I1226 22:39:12.886394  6129 solver.cpp:237] Iteration 56400, loss = 0.64556
    I1226 22:39:12.886432  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:39:12.886446  6129 solver.cpp:253]     Train net output #1: loss = 0.64556 (* 1 = 0.64556 loss)
    I1226 22:39:12.886456  6129 sgd_solver.cpp:106] Iteration 56400, lr = 0.000169228
    I1226 22:39:23.337020  6129 solver.cpp:237] Iteration 56500, loss = 0.584807
    I1226 22:39:23.337059  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 22:39:23.337074  6129 solver.cpp:253]     Train net output #1: loss = 0.584807 (* 1 = 0.584807 loss)
    I1226 22:39:23.337083  6129 sgd_solver.cpp:106] Iteration 56500, lr = 0.000169037
    I1226 22:39:33.814355  6129 solver.cpp:237] Iteration 56600, loss = 0.656076
    I1226 22:39:33.814476  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:39:33.814493  6129 solver.cpp:253]     Train net output #1: loss = 0.656076 (* 1 = 0.656076 loss)
    I1226 22:39:33.814504  6129 sgd_solver.cpp:106] Iteration 56600, lr = 0.000168847
    I1226 22:39:44.291385  6129 solver.cpp:237] Iteration 56700, loss = 0.545703
    I1226 22:39:44.291422  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:39:44.291436  6129 solver.cpp:253]     Train net output #1: loss = 0.545703 (* 1 = 0.545703 loss)
    I1226 22:39:44.291446  6129 sgd_solver.cpp:106] Iteration 56700, lr = 0.000168657
    I1226 22:39:54.771469  6129 solver.cpp:237] Iteration 56800, loss = 0.636801
    I1226 22:39:54.771517  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:39:54.771535  6129 solver.cpp:253]     Train net output #1: loss = 0.636801 (* 1 = 0.636801 loss)
    I1226 22:39:54.771548  6129 sgd_solver.cpp:106] Iteration 56800, lr = 0.000168467
    I1226 22:40:05.240617  6129 solver.cpp:237] Iteration 56900, loss = 0.644747
    I1226 22:40:05.240736  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:40:05.240756  6129 solver.cpp:253]     Train net output #1: loss = 0.644747 (* 1 = 0.644747 loss)
    I1226 22:40:05.240769  6129 sgd_solver.cpp:106] Iteration 56900, lr = 0.000168278
    I1226 22:40:15.624739  6129 solver.cpp:341] Iteration 57000, Testing net (#0)
    I1226 22:40:19.939960  6129 solver.cpp:409]     Test net output #0: accuracy = 0.70125
    I1226 22:40:19.940004  6129 solver.cpp:409]     Test net output #1: loss = 0.855579 (* 1 = 0.855579 loss)
    I1226 22:40:19.984557  6129 solver.cpp:237] Iteration 57000, loss = 0.58377
    I1226 22:40:19.984601  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:40:19.984614  6129 solver.cpp:253]     Train net output #1: loss = 0.58377 (* 1 = 0.58377 loss)
    I1226 22:40:19.984627  6129 sgd_solver.cpp:106] Iteration 57000, lr = 0.00016809
    I1226 22:40:30.553458  6129 solver.cpp:237] Iteration 57100, loss = 0.659245
    I1226 22:40:30.553510  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:40:30.553529  6129 solver.cpp:253]     Train net output #1: loss = 0.659245 (* 1 = 0.659245 loss)
    I1226 22:40:30.553544  6129 sgd_solver.cpp:106] Iteration 57100, lr = 0.000167902
    I1226 22:40:41.015050  6129 solver.cpp:237] Iteration 57200, loss = 0.553336
    I1226 22:40:41.015161  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:40:41.015177  6129 solver.cpp:253]     Train net output #1: loss = 0.553336 (* 1 = 0.553336 loss)
    I1226 22:40:41.015188  6129 sgd_solver.cpp:106] Iteration 57200, lr = 0.000167715
    I1226 22:40:51.476477  6129 solver.cpp:237] Iteration 57300, loss = 0.635159
    I1226 22:40:51.476516  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:40:51.476531  6129 solver.cpp:253]     Train net output #1: loss = 0.635159 (* 1 = 0.635159 loss)
    I1226 22:40:51.476541  6129 sgd_solver.cpp:106] Iteration 57300, lr = 0.000167528
    I1226 22:41:01.984714  6129 solver.cpp:237] Iteration 57400, loss = 0.645373
    I1226 22:41:01.984752  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:41:01.984766  6129 solver.cpp:253]     Train net output #1: loss = 0.645373 (* 1 = 0.645373 loss)
    I1226 22:41:01.984777  6129 sgd_solver.cpp:106] Iteration 57400, lr = 0.000167341
    I1226 22:41:12.517141  6129 solver.cpp:237] Iteration 57500, loss = 0.580392
    I1226 22:41:12.517230  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:41:12.517246  6129 solver.cpp:253]     Train net output #1: loss = 0.580392 (* 1 = 0.580392 loss)
    I1226 22:41:12.517256  6129 sgd_solver.cpp:106] Iteration 57500, lr = 0.000167155
    I1226 22:41:22.986479  6129 solver.cpp:237] Iteration 57600, loss = 0.654112
    I1226 22:41:22.986526  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:41:22.986546  6129 solver.cpp:253]     Train net output #1: loss = 0.654112 (* 1 = 0.654112 loss)
    I1226 22:41:22.986560  6129 sgd_solver.cpp:106] Iteration 57600, lr = 0.00016697
    I1226 22:41:34.887972  6129 solver.cpp:237] Iteration 57700, loss = 0.570235
    I1226 22:41:34.888022  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:41:34.888041  6129 solver.cpp:253]     Train net output #1: loss = 0.570235 (* 1 = 0.570235 loss)
    I1226 22:41:34.888056  6129 sgd_solver.cpp:106] Iteration 57700, lr = 0.000166785
    I1226 22:41:47.641012  6129 solver.cpp:237] Iteration 57800, loss = 0.63541
    I1226 22:41:47.641144  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:41:47.641161  6129 solver.cpp:253]     Train net output #1: loss = 0.63541 (* 1 = 0.63541 loss)
    I1226 22:41:47.641173  6129 sgd_solver.cpp:106] Iteration 57800, lr = 0.0001666
    I1226 22:42:00.815574  6129 solver.cpp:237] Iteration 57900, loss = 0.643816
    I1226 22:42:00.815614  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:42:00.815629  6129 solver.cpp:253]     Train net output #1: loss = 0.643816 (* 1 = 0.643816 loss)
    I1226 22:42:00.815640  6129 sgd_solver.cpp:106] Iteration 57900, lr = 0.000166416
    I1226 22:42:11.168509  6129 solver.cpp:341] Iteration 58000, Testing net (#0)
    I1226 22:42:15.459200  6129 solver.cpp:409]     Test net output #0: accuracy = 0.700417
    I1226 22:42:15.459244  6129 solver.cpp:409]     Test net output #1: loss = 0.852711 (* 1 = 0.852711 loss)
    I1226 22:42:15.503705  6129 solver.cpp:237] Iteration 58000, loss = 0.577919
    I1226 22:42:15.503742  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:42:15.503756  6129 solver.cpp:253]     Train net output #1: loss = 0.577919 (* 1 = 0.577919 loss)
    I1226 22:42:15.503767  6129 sgd_solver.cpp:106] Iteration 58000, lr = 0.000166233
    I1226 22:42:26.135685  6129 solver.cpp:237] Iteration 58100, loss = 0.652938
    I1226 22:42:26.135807  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:42:26.135828  6129 solver.cpp:253]     Train net output #1: loss = 0.652938 (* 1 = 0.652938 loss)
    I1226 22:42:26.135841  6129 sgd_solver.cpp:106] Iteration 58100, lr = 0.00016605
    I1226 22:42:37.863090  6129 solver.cpp:237] Iteration 58200, loss = 0.543039
    I1226 22:42:37.863142  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:42:37.863160  6129 solver.cpp:253]     Train net output #1: loss = 0.543039 (* 1 = 0.543039 loss)
    I1226 22:42:37.863175  6129 sgd_solver.cpp:106] Iteration 58200, lr = 0.000165867
    I1226 22:42:48.917529  6129 solver.cpp:237] Iteration 58300, loss = 0.632968
    I1226 22:42:48.917567  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:42:48.917579  6129 solver.cpp:253]     Train net output #1: loss = 0.632968 (* 1 = 0.632968 loss)
    I1226 22:42:48.917592  6129 sgd_solver.cpp:106] Iteration 58300, lr = 0.000165685
    I1226 22:42:59.566125  6129 solver.cpp:237] Iteration 58400, loss = 0.645315
    I1226 22:42:59.566231  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:42:59.566246  6129 solver.cpp:253]     Train net output #1: loss = 0.645315 (* 1 = 0.645315 loss)
    I1226 22:42:59.566257  6129 sgd_solver.cpp:106] Iteration 58400, lr = 0.000165503
    I1226 22:43:10.034451  6129 solver.cpp:237] Iteration 58500, loss = 0.576945
    I1226 22:43:10.034489  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:43:10.034503  6129 solver.cpp:253]     Train net output #1: loss = 0.576945 (* 1 = 0.576945 loss)
    I1226 22:43:10.034514  6129 sgd_solver.cpp:106] Iteration 58500, lr = 0.000165322
    I1226 22:43:20.981209  6129 solver.cpp:237] Iteration 58600, loss = 0.654078
    I1226 22:43:20.981251  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:43:20.981266  6129 solver.cpp:253]     Train net output #1: loss = 0.654078 (* 1 = 0.654078 loss)
    I1226 22:43:20.981276  6129 sgd_solver.cpp:106] Iteration 58600, lr = 0.000165141
    I1226 22:43:31.458576  6129 solver.cpp:237] Iteration 58700, loss = 0.566694
    I1226 22:43:31.459239  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:43:31.459257  6129 solver.cpp:253]     Train net output #1: loss = 0.566694 (* 1 = 0.566694 loss)
    I1226 22:43:31.459266  6129 sgd_solver.cpp:106] Iteration 58700, lr = 0.000164961
    I1226 22:43:43.466688  6129 solver.cpp:237] Iteration 58800, loss = 0.630359
    I1226 22:43:43.466734  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:43:43.466747  6129 solver.cpp:253]     Train net output #1: loss = 0.630359 (* 1 = 0.630359 loss)
    I1226 22:43:43.466754  6129 sgd_solver.cpp:106] Iteration 58800, lr = 0.000164781
    I1226 22:43:54.557943  6129 solver.cpp:237] Iteration 58900, loss = 0.643877
    I1226 22:43:54.557988  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:43:54.558001  6129 solver.cpp:253]     Train net output #1: loss = 0.643877 (* 1 = 0.643877 loss)
    I1226 22:43:54.558012  6129 sgd_solver.cpp:106] Iteration 58900, lr = 0.000164601
    I1226 22:44:04.933336  6129 solver.cpp:341] Iteration 59000, Testing net (#0)
    I1226 22:44:09.195837  6129 solver.cpp:409]     Test net output #0: accuracy = 0.6985
    I1226 22:44:09.195888  6129 solver.cpp:409]     Test net output #1: loss = 0.856037 (* 1 = 0.856037 loss)
    I1226 22:44:09.240350  6129 solver.cpp:237] Iteration 59000, loss = 0.575165
    I1226 22:44:09.240377  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:44:09.240387  6129 solver.cpp:253]     Train net output #1: loss = 0.575165 (* 1 = 0.575165 loss)
    I1226 22:44:09.240396  6129 sgd_solver.cpp:106] Iteration 59000, lr = 0.000164422
    I1226 22:44:19.719137  6129 solver.cpp:237] Iteration 59100, loss = 0.649055
    I1226 22:44:19.719174  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:44:19.719187  6129 solver.cpp:253]     Train net output #1: loss = 0.649055 (* 1 = 0.649055 loss)
    I1226 22:44:19.719197  6129 sgd_solver.cpp:106] Iteration 59100, lr = 0.000164244
    I1226 22:44:30.132853  6129 solver.cpp:237] Iteration 59200, loss = 0.534621
    I1226 22:44:30.132908  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 22:44:30.132930  6129 solver.cpp:253]     Train net output #1: loss = 0.534621 (* 1 = 0.534621 loss)
    I1226 22:44:30.132946  6129 sgd_solver.cpp:106] Iteration 59200, lr = 0.000164066
    I1226 22:44:40.629848  6129 solver.cpp:237] Iteration 59300, loss = 0.629351
    I1226 22:44:40.629956  6129 solver.cpp:253]     Train net output #0: accuracy = 0.77
    I1226 22:44:40.629973  6129 solver.cpp:253]     Train net output #1: loss = 0.629351 (* 1 = 0.629351 loss)
    I1226 22:44:40.629986  6129 sgd_solver.cpp:106] Iteration 59300, lr = 0.000163888
    I1226 22:44:51.110915  6129 solver.cpp:237] Iteration 59400, loss = 0.644537
    I1226 22:44:51.110956  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:44:51.110971  6129 solver.cpp:253]     Train net output #1: loss = 0.644537 (* 1 = 0.644537 loss)
    I1226 22:44:51.110981  6129 sgd_solver.cpp:106] Iteration 59400, lr = 0.000163711
    I1226 22:45:01.604434  6129 solver.cpp:237] Iteration 59500, loss = 0.572682
    I1226 22:45:01.604477  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:45:01.604492  6129 solver.cpp:253]     Train net output #1: loss = 0.572682 (* 1 = 0.572682 loss)
    I1226 22:45:01.604504  6129 sgd_solver.cpp:106] Iteration 59500, lr = 0.000163535
    I1226 22:45:12.055413  6129 solver.cpp:237] Iteration 59600, loss = 0.652382
    I1226 22:45:12.055546  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:45:12.055567  6129 solver.cpp:253]     Train net output #1: loss = 0.652382 (* 1 = 0.652382 loss)
    I1226 22:45:12.055578  6129 sgd_solver.cpp:106] Iteration 59600, lr = 0.000163358
    I1226 22:45:23.331679  6129 solver.cpp:237] Iteration 59700, loss = 0.567921
    I1226 22:45:23.331725  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:45:23.331740  6129 solver.cpp:253]     Train net output #1: loss = 0.567921 (* 1 = 0.567921 loss)
    I1226 22:45:23.331753  6129 sgd_solver.cpp:106] Iteration 59700, lr = 0.000163182
    I1226 22:45:34.625391  6129 solver.cpp:237] Iteration 59800, loss = 0.628186
    I1226 22:45:34.625437  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:45:34.625453  6129 solver.cpp:253]     Train net output #1: loss = 0.628186 (* 1 = 0.628186 loss)
    I1226 22:45:34.625466  6129 sgd_solver.cpp:106] Iteration 59800, lr = 0.000163007
    I1226 22:45:47.737304  6129 solver.cpp:237] Iteration 59900, loss = 0.643533
    I1226 22:45:47.737468  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:45:47.737483  6129 solver.cpp:253]     Train net output #1: loss = 0.643533 (* 1 = 0.643533 loss)
    I1226 22:45:47.737491  6129 sgd_solver.cpp:106] Iteration 59900, lr = 0.000162832
    I1226 22:45:58.333130  6129 solver.cpp:341] Iteration 60000, Testing net (#0)
    I1226 22:46:03.463310  6129 solver.cpp:409]     Test net output #0: accuracy = 0.703417
    I1226 22:46:03.463359  6129 solver.cpp:409]     Test net output #1: loss = 0.849635 (* 1 = 0.849635 loss)
    I1226 22:46:03.507850  6129 solver.cpp:237] Iteration 60000, loss = 0.571041
    I1226 22:46:03.507895  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:46:03.507910  6129 solver.cpp:253]     Train net output #1: loss = 0.571041 (* 1 = 0.571041 loss)
    I1226 22:46:03.507921  6129 sgd_solver.cpp:106] Iteration 60000, lr = 0.000162658
    I1226 22:46:18.852149  6129 solver.cpp:237] Iteration 60100, loss = 0.646842
    I1226 22:46:18.852285  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:46:18.852313  6129 solver.cpp:253]     Train net output #1: loss = 0.646842 (* 1 = 0.646842 loss)
    I1226 22:46:18.852324  6129 sgd_solver.cpp:106] Iteration 60100, lr = 0.000162484
    I1226 22:46:34.703341  6129 solver.cpp:237] Iteration 60200, loss = 0.536164
    I1226 22:46:34.703377  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:46:34.703390  6129 solver.cpp:253]     Train net output #1: loss = 0.536164 (* 1 = 0.536164 loss)
    I1226 22:46:34.703399  6129 sgd_solver.cpp:106] Iteration 60200, lr = 0.00016231
    I1226 22:46:48.077204  6129 solver.cpp:237] Iteration 60300, loss = 0.627031
    I1226 22:46:48.077258  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:46:48.077273  6129 solver.cpp:253]     Train net output #1: loss = 0.627031 (* 1 = 0.627031 loss)
    I1226 22:46:48.077286  6129 sgd_solver.cpp:106] Iteration 60300, lr = 0.000162137
    I1226 22:46:59.729111  6129 solver.cpp:237] Iteration 60400, loss = 0.643573
    I1226 22:46:59.729233  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:46:59.729250  6129 solver.cpp:253]     Train net output #1: loss = 0.643573 (* 1 = 0.643573 loss)
    I1226 22:46:59.729260  6129 sgd_solver.cpp:106] Iteration 60400, lr = 0.000161964
    I1226 22:47:10.193473  6129 solver.cpp:237] Iteration 60500, loss = 0.568863
    I1226 22:47:10.193518  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:47:10.193529  6129 solver.cpp:253]     Train net output #1: loss = 0.568863 (* 1 = 0.568863 loss)
    I1226 22:47:10.193537  6129 sgd_solver.cpp:106] Iteration 60500, lr = 0.000161792
    I1226 22:47:20.840169  6129 solver.cpp:237] Iteration 60600, loss = 0.644495
    I1226 22:47:20.840225  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:47:20.840246  6129 solver.cpp:253]     Train net output #1: loss = 0.644495 (* 1 = 0.644495 loss)
    I1226 22:47:20.840262  6129 sgd_solver.cpp:106] Iteration 60600, lr = 0.00016162
    I1226 22:47:32.665238  6129 solver.cpp:237] Iteration 60700, loss = 0.555186
    I1226 22:47:32.665340  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:47:32.665354  6129 solver.cpp:253]     Train net output #1: loss = 0.555186 (* 1 = 0.555186 loss)
    I1226 22:47:32.665364  6129 sgd_solver.cpp:106] Iteration 60700, lr = 0.000161448
    I1226 22:47:43.271697  6129 solver.cpp:237] Iteration 60800, loss = 0.625044
    I1226 22:47:43.271739  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:47:43.271754  6129 solver.cpp:253]     Train net output #1: loss = 0.625044 (* 1 = 0.625044 loss)
    I1226 22:47:43.271766  6129 sgd_solver.cpp:106] Iteration 60800, lr = 0.000161277
    I1226 22:47:54.272289  6129 solver.cpp:237] Iteration 60900, loss = 0.642109
    I1226 22:47:54.272330  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:47:54.272344  6129 solver.cpp:253]     Train net output #1: loss = 0.642109 (* 1 = 0.642109 loss)
    I1226 22:47:54.272356  6129 sgd_solver.cpp:106] Iteration 60900, lr = 0.000161107
    I1226 22:48:07.384995  6129 solver.cpp:341] Iteration 61000, Testing net (#0)
    I1226 22:48:12.027978  6129 solver.cpp:409]     Test net output #0: accuracy = 0.701
    I1226 22:48:12.028024  6129 solver.cpp:409]     Test net output #1: loss = 0.853295 (* 1 = 0.853295 loss)
    I1226 22:48:12.072600  6129 solver.cpp:237] Iteration 61000, loss = 0.567317
    I1226 22:48:12.072643  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:48:12.072656  6129 solver.cpp:253]     Train net output #1: loss = 0.567317 (* 1 = 0.567317 loss)
    I1226 22:48:12.072669  6129 sgd_solver.cpp:106] Iteration 61000, lr = 0.000160936
    I1226 22:48:22.656390  6129 solver.cpp:237] Iteration 61100, loss = 0.643558
    I1226 22:48:22.656433  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:48:22.656447  6129 solver.cpp:253]     Train net output #1: loss = 0.643558 (* 1 = 0.643558 loss)
    I1226 22:48:22.656464  6129 sgd_solver.cpp:106] Iteration 61100, lr = 0.000160767
    I1226 22:48:33.090580  6129 solver.cpp:237] Iteration 61200, loss = 0.53925
    I1226 22:48:33.090620  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:48:33.090636  6129 solver.cpp:253]     Train net output #1: loss = 0.53925 (* 1 = 0.53925 loss)
    I1226 22:48:33.090648  6129 sgd_solver.cpp:106] Iteration 61200, lr = 0.000160597
    I1226 22:48:43.562721  6129 solver.cpp:237] Iteration 61300, loss = 0.623577
    I1226 22:48:43.562837  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:48:43.562855  6129 solver.cpp:253]     Train net output #1: loss = 0.623577 (* 1 = 0.623577 loss)
    I1226 22:48:43.562867  6129 sgd_solver.cpp:106] Iteration 61300, lr = 0.000160428
    I1226 22:48:54.867110  6129 solver.cpp:237] Iteration 61400, loss = 0.642525
    I1226 22:48:54.867153  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:48:54.867168  6129 solver.cpp:253]     Train net output #1: loss = 0.642525 (* 1 = 0.642525 loss)
    I1226 22:48:54.867179  6129 sgd_solver.cpp:106] Iteration 61400, lr = 0.00016026
    I1226 22:49:06.768260  6129 solver.cpp:237] Iteration 61500, loss = 0.565397
    I1226 22:49:06.768304  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:49:06.768319  6129 solver.cpp:253]     Train net output #1: loss = 0.565397 (* 1 = 0.565397 loss)
    I1226 22:49:06.768331  6129 sgd_solver.cpp:106] Iteration 61500, lr = 0.000160092
    I1226 22:49:18.924041  6129 solver.cpp:237] Iteration 61600, loss = 0.641908
    I1226 22:49:18.924156  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:49:18.924182  6129 solver.cpp:253]     Train net output #1: loss = 0.641908 (* 1 = 0.641908 loss)
    I1226 22:49:18.924193  6129 sgd_solver.cpp:106] Iteration 61600, lr = 0.000159924
    I1226 22:49:30.491173  6129 solver.cpp:237] Iteration 61700, loss = 0.552367
    I1226 22:49:30.491216  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:49:30.491228  6129 solver.cpp:253]     Train net output #1: loss = 0.552367 (* 1 = 0.552367 loss)
    I1226 22:49:30.491236  6129 sgd_solver.cpp:106] Iteration 61700, lr = 0.000159757
    I1226 22:49:43.208626  6129 solver.cpp:237] Iteration 61800, loss = 0.621734
    I1226 22:49:43.208662  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:49:43.208673  6129 solver.cpp:253]     Train net output #1: loss = 0.621734 (* 1 = 0.621734 loss)
    I1226 22:49:43.208680  6129 sgd_solver.cpp:106] Iteration 61800, lr = 0.00015959
    I1226 22:49:55.372432  6129 solver.cpp:237] Iteration 61900, loss = 0.641781
    I1226 22:49:55.372576  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:49:55.372601  6129 solver.cpp:253]     Train net output #1: loss = 0.641781 (* 1 = 0.641781 loss)
    I1226 22:49:55.372611  6129 sgd_solver.cpp:106] Iteration 61900, lr = 0.000159423
    I1226 22:50:07.401335  6129 solver.cpp:341] Iteration 62000, Testing net (#0)
    I1226 22:50:14.733479  6129 solver.cpp:409]     Test net output #0: accuracy = 0.703
    I1226 22:50:14.733538  6129 solver.cpp:409]     Test net output #1: loss = 0.846875 (* 1 = 0.846875 loss)
    I1226 22:50:14.811049  6129 solver.cpp:237] Iteration 62000, loss = 0.563929
    I1226 22:50:14.811097  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:50:14.811111  6129 solver.cpp:253]     Train net output #1: loss = 0.563929 (* 1 = 0.563929 loss)
    I1226 22:50:14.811122  6129 sgd_solver.cpp:106] Iteration 62000, lr = 0.000159257
    I1226 22:50:26.715536  6129 solver.cpp:237] Iteration 62100, loss = 0.641927
    I1226 22:50:26.715677  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:50:26.715689  6129 solver.cpp:253]     Train net output #1: loss = 0.641927 (* 1 = 0.641927 loss)
    I1226 22:50:26.715698  6129 sgd_solver.cpp:106] Iteration 62100, lr = 0.000159091
    I1226 22:50:37.274323  6129 solver.cpp:237] Iteration 62200, loss = 0.54766
    I1226 22:50:37.274359  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:50:37.274371  6129 solver.cpp:253]     Train net output #1: loss = 0.54766 (* 1 = 0.54766 loss)
    I1226 22:50:37.274380  6129 sgd_solver.cpp:106] Iteration 62200, lr = 0.000158926
    I1226 22:50:47.958945  6129 solver.cpp:237] Iteration 62300, loss = 0.621073
    I1226 22:50:47.958983  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:50:47.958995  6129 solver.cpp:253]     Train net output #1: loss = 0.621073 (* 1 = 0.621073 loss)
    I1226 22:50:47.959005  6129 sgd_solver.cpp:106] Iteration 62300, lr = 0.000158761
    I1226 22:50:59.563020  6129 solver.cpp:237] Iteration 62400, loss = 0.641566
    I1226 22:50:59.563155  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:50:59.563180  6129 solver.cpp:253]     Train net output #1: loss = 0.641566 (* 1 = 0.641566 loss)
    I1226 22:50:59.563196  6129 sgd_solver.cpp:106] Iteration 62400, lr = 0.000158597
    I1226 22:51:11.041249  6129 solver.cpp:237] Iteration 62500, loss = 0.562709
    I1226 22:51:11.041295  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:51:11.041306  6129 solver.cpp:253]     Train net output #1: loss = 0.562709 (* 1 = 0.562709 loss)
    I1226 22:51:11.041314  6129 sgd_solver.cpp:106] Iteration 62500, lr = 0.000158433
    I1226 22:51:21.746809  6129 solver.cpp:237] Iteration 62600, loss = 0.638269
    I1226 22:51:21.746865  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:51:21.746886  6129 solver.cpp:253]     Train net output #1: loss = 0.638269 (* 1 = 0.638269 loss)
    I1226 22:51:21.746902  6129 sgd_solver.cpp:106] Iteration 62600, lr = 0.000158269
    I1226 22:51:32.416108  6129 solver.cpp:237] Iteration 62700, loss = 0.543599
    I1226 22:51:32.416278  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:51:32.416295  6129 solver.cpp:253]     Train net output #1: loss = 0.543599 (* 1 = 0.543599 loss)
    I1226 22:51:32.416304  6129 sgd_solver.cpp:106] Iteration 62700, lr = 0.000158106
    I1226 22:51:44.133563  6129 solver.cpp:237] Iteration 62800, loss = 0.61869
    I1226 22:51:44.133606  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:51:44.133620  6129 solver.cpp:253]     Train net output #1: loss = 0.61869 (* 1 = 0.61869 loss)
    I1226 22:51:44.133633  6129 sgd_solver.cpp:106] Iteration 62800, lr = 0.000157943
    I1226 22:51:54.583489  6129 solver.cpp:237] Iteration 62900, loss = 0.641308
    I1226 22:51:54.583531  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:51:54.583546  6129 solver.cpp:253]     Train net output #1: loss = 0.641308 (* 1 = 0.641308 loss)
    I1226 22:51:54.583559  6129 sgd_solver.cpp:106] Iteration 62900, lr = 0.00015778
    I1226 22:52:05.359205  6129 solver.cpp:341] Iteration 63000, Testing net (#0)
    I1226 22:52:09.722213  6129 solver.cpp:409]     Test net output #0: accuracy = 0.704333
    I1226 22:52:09.722275  6129 solver.cpp:409]     Test net output #1: loss = 0.845788 (* 1 = 0.845788 loss)
    I1226 22:52:09.795359  6129 solver.cpp:237] Iteration 63000, loss = 0.561063
    I1226 22:52:09.795408  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:52:09.795423  6129 solver.cpp:253]     Train net output #1: loss = 0.561063 (* 1 = 0.561063 loss)
    I1226 22:52:09.795434  6129 sgd_solver.cpp:106] Iteration 63000, lr = 0.000157618
    I1226 22:52:20.516419  6129 solver.cpp:237] Iteration 63100, loss = 0.637022
    I1226 22:52:20.516465  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:52:20.516481  6129 solver.cpp:253]     Train net output #1: loss = 0.637022 (* 1 = 0.637022 loss)
    I1226 22:52:20.516492  6129 sgd_solver.cpp:106] Iteration 63100, lr = 0.000157456
    I1226 22:52:31.544144  6129 solver.cpp:237] Iteration 63200, loss = 0.550786
    I1226 22:52:31.544194  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:52:31.544209  6129 solver.cpp:253]     Train net output #1: loss = 0.550786 (* 1 = 0.550786 loss)
    I1226 22:52:31.544219  6129 sgd_solver.cpp:106] Iteration 63200, lr = 0.000157295
    I1226 22:52:43.167464  6129 solver.cpp:237] Iteration 63300, loss = 0.616759
    I1226 22:52:43.167647  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:52:43.167666  6129 solver.cpp:253]     Train net output #1: loss = 0.616759 (* 1 = 0.616759 loss)
    I1226 22:52:43.167676  6129 sgd_solver.cpp:106] Iteration 63300, lr = 0.000157134
    I1226 22:52:54.994848  6129 solver.cpp:237] Iteration 63400, loss = 0.639988
    I1226 22:52:54.994882  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:52:54.994894  6129 solver.cpp:253]     Train net output #1: loss = 0.639988 (* 1 = 0.639988 loss)
    I1226 22:52:54.994904  6129 sgd_solver.cpp:106] Iteration 63400, lr = 0.000156973
    I1226 22:53:06.999009  6129 solver.cpp:237] Iteration 63500, loss = 0.560107
    I1226 22:53:06.999050  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:53:06.999065  6129 solver.cpp:253]     Train net output #1: loss = 0.560107 (* 1 = 0.560107 loss)
    I1226 22:53:06.999078  6129 sgd_solver.cpp:106] Iteration 63500, lr = 0.000156813
    I1226 22:53:17.598342  6129 solver.cpp:237] Iteration 63600, loss = 0.63443
    I1226 22:53:17.598495  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:53:17.598508  6129 solver.cpp:253]     Train net output #1: loss = 0.63443 (* 1 = 0.63443 loss)
    I1226 22:53:17.598515  6129 sgd_solver.cpp:106] Iteration 63600, lr = 0.000156653
    I1226 22:53:28.384982  6129 solver.cpp:237] Iteration 63700, loss = 0.535109
    I1226 22:53:28.385025  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 22:53:28.385040  6129 solver.cpp:253]     Train net output #1: loss = 0.535109 (* 1 = 0.535109 loss)
    I1226 22:53:28.385049  6129 sgd_solver.cpp:106] Iteration 63700, lr = 0.000156494
    I1226 22:53:41.244230  6129 solver.cpp:237] Iteration 63800, loss = 0.615031
    I1226 22:53:41.244266  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:53:41.244279  6129 solver.cpp:253]     Train net output #1: loss = 0.615031 (* 1 = 0.615031 loss)
    I1226 22:53:41.244289  6129 sgd_solver.cpp:106] Iteration 63800, lr = 0.000156335
    I1226 22:53:54.296416  6129 solver.cpp:237] Iteration 63900, loss = 0.638945
    I1226 22:53:54.296545  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:53:54.296566  6129 solver.cpp:253]     Train net output #1: loss = 0.638945 (* 1 = 0.638945 loss)
    I1226 22:53:54.296579  6129 sgd_solver.cpp:106] Iteration 63900, lr = 0.000156176
    I1226 22:54:07.527737  6129 solver.cpp:341] Iteration 64000, Testing net (#0)
    I1226 22:54:12.812209  6129 solver.cpp:409]     Test net output #0: accuracy = 0.703167
    I1226 22:54:12.812255  6129 solver.cpp:409]     Test net output #1: loss = 0.849823 (* 1 = 0.849823 loss)
    I1226 22:54:12.856834  6129 solver.cpp:237] Iteration 64000, loss = 0.55908
    I1226 22:54:12.856876  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:54:12.856890  6129 solver.cpp:253]     Train net output #1: loss = 0.55908 (* 1 = 0.55908 loss)
    I1226 22:54:12.856902  6129 sgd_solver.cpp:106] Iteration 64000, lr = 0.000156018
    I1226 22:54:25.770310  6129 solver.cpp:237] Iteration 64100, loss = 0.632777
    I1226 22:54:25.770630  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:54:25.770683  6129 solver.cpp:253]     Train net output #1: loss = 0.632777 (* 1 = 0.632777 loss)
    I1226 22:54:25.770714  6129 sgd_solver.cpp:106] Iteration 64100, lr = 0.00015586
    I1226 22:54:37.978760  6129 solver.cpp:237] Iteration 64200, loss = 0.536386
    I1226 22:54:37.978801  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:54:37.978812  6129 solver.cpp:253]     Train net output #1: loss = 0.536386 (* 1 = 0.536386 loss)
    I1226 22:54:37.978821  6129 sgd_solver.cpp:106] Iteration 64200, lr = 0.000155702
    I1226 22:54:49.458375  6129 solver.cpp:237] Iteration 64300, loss = 0.613207
    I1226 22:54:49.458410  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:54:49.458422  6129 solver.cpp:253]     Train net output #1: loss = 0.613207 (* 1 = 0.613207 loss)
    I1226 22:54:49.458432  6129 sgd_solver.cpp:106] Iteration 64300, lr = 0.000155545
    I1226 22:55:01.728797  6129 solver.cpp:237] Iteration 64400, loss = 0.63805
    I1226 22:55:01.728935  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:55:01.728950  6129 solver.cpp:253]     Train net output #1: loss = 0.63805 (* 1 = 0.63805 loss)
    I1226 22:55:01.728961  6129 sgd_solver.cpp:106] Iteration 64400, lr = 0.000155388
    I1226 22:55:12.856299  6129 solver.cpp:237] Iteration 64500, loss = 0.557957
    I1226 22:55:12.856353  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:55:12.856374  6129 solver.cpp:253]     Train net output #1: loss = 0.557957 (* 1 = 0.557957 loss)
    I1226 22:55:12.856389  6129 sgd_solver.cpp:106] Iteration 64500, lr = 0.000155232
    I1226 22:55:24.280416  6129 solver.cpp:237] Iteration 64600, loss = 0.632656
    I1226 22:55:24.280480  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:55:24.280501  6129 solver.cpp:253]     Train net output #1: loss = 0.632656 (* 1 = 0.632656 loss)
    I1226 22:55:24.280525  6129 sgd_solver.cpp:106] Iteration 64600, lr = 0.000155076
    I1226 22:55:35.713227  6129 solver.cpp:237] Iteration 64700, loss = 0.552314
    I1226 22:55:35.713387  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:55:35.713407  6129 solver.cpp:253]     Train net output #1: loss = 0.552314 (* 1 = 0.552314 loss)
    I1226 22:55:35.713417  6129 sgd_solver.cpp:106] Iteration 64700, lr = 0.00015492
    I1226 22:55:47.675290  6129 solver.cpp:237] Iteration 64800, loss = 0.61078
    I1226 22:55:47.675330  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:55:47.675344  6129 solver.cpp:253]     Train net output #1: loss = 0.61078 (* 1 = 0.61078 loss)
    I1226 22:55:47.675355  6129 sgd_solver.cpp:106] Iteration 64800, lr = 0.000154765
    I1226 22:56:01.085729  6129 solver.cpp:237] Iteration 64900, loss = 0.636729
    I1226 22:56:01.085772  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:56:01.085788  6129 solver.cpp:253]     Train net output #1: loss = 0.636729 (* 1 = 0.636729 loss)
    I1226 22:56:01.085798  6129 sgd_solver.cpp:106] Iteration 64900, lr = 0.00015461
    I1226 22:56:12.997669  6129 solver.cpp:341] Iteration 65000, Testing net (#0)
    I1226 22:56:17.420742  6129 solver.cpp:409]     Test net output #0: accuracy = 0.707583
    I1226 22:56:17.420781  6129 solver.cpp:409]     Test net output #1: loss = 0.84389 (* 1 = 0.84389 loss)
    I1226 22:56:17.465425  6129 solver.cpp:237] Iteration 65000, loss = 0.557131
    I1226 22:56:17.465462  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:56:17.465473  6129 solver.cpp:253]     Train net output #1: loss = 0.557131 (* 1 = 0.557131 loss)
    I1226 22:56:17.465482  6129 sgd_solver.cpp:106] Iteration 65000, lr = 0.000154455
    I1226 22:56:28.466476  6129 solver.cpp:237] Iteration 65100, loss = 0.631787
    I1226 22:56:28.466511  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:56:28.466522  6129 solver.cpp:253]     Train net output #1: loss = 0.631787 (* 1 = 0.631787 loss)
    I1226 22:56:28.466531  6129 sgd_solver.cpp:106] Iteration 65100, lr = 0.000154301
    I1226 22:56:40.918308  6129 solver.cpp:237] Iteration 65200, loss = 0.537672
    I1226 22:56:40.918349  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:56:40.918365  6129 solver.cpp:253]     Train net output #1: loss = 0.537672 (* 1 = 0.537672 loss)
    I1226 22:56:40.918376  6129 sgd_solver.cpp:106] Iteration 65200, lr = 0.000154147
    I1226 22:56:52.502027  6129 solver.cpp:237] Iteration 65300, loss = 0.611058
    I1226 22:56:52.502166  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 22:56:52.502185  6129 solver.cpp:253]     Train net output #1: loss = 0.611058 (* 1 = 0.611058 loss)
    I1226 22:56:52.502195  6129 sgd_solver.cpp:106] Iteration 65300, lr = 0.000153993
    I1226 22:57:03.345137  6129 solver.cpp:237] Iteration 65400, loss = 0.635091
    I1226 22:57:03.345177  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:57:03.345192  6129 solver.cpp:253]     Train net output #1: loss = 0.635091 (* 1 = 0.635091 loss)
    I1226 22:57:03.345202  6129 sgd_solver.cpp:106] Iteration 65400, lr = 0.00015384
    I1226 22:57:14.567668  6129 solver.cpp:237] Iteration 65500, loss = 0.555822
    I1226 22:57:14.567709  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:57:14.567723  6129 solver.cpp:253]     Train net output #1: loss = 0.555822 (* 1 = 0.555822 loss)
    I1226 22:57:14.567734  6129 sgd_solver.cpp:106] Iteration 65500, lr = 0.000153687
    I1226 22:57:26.503644  6129 solver.cpp:237] Iteration 65600, loss = 0.629961
    I1226 22:57:26.503764  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:57:26.503784  6129 solver.cpp:253]     Train net output #1: loss = 0.629961 (* 1 = 0.629961 loss)
    I1226 22:57:26.503795  6129 sgd_solver.cpp:106] Iteration 65600, lr = 0.000153535
    I1226 22:57:38.303148  6129 solver.cpp:237] Iteration 65700, loss = 0.540574
    I1226 22:57:38.303192  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:57:38.303208  6129 solver.cpp:253]     Train net output #1: loss = 0.540574 (* 1 = 0.540574 loss)
    I1226 22:57:38.303220  6129 sgd_solver.cpp:106] Iteration 65700, lr = 0.000153383
    I1226 22:57:49.735901  6129 solver.cpp:237] Iteration 65800, loss = 0.607797
    I1226 22:57:49.735944  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:57:49.735959  6129 solver.cpp:253]     Train net output #1: loss = 0.607797 (* 1 = 0.607797 loss)
    I1226 22:57:49.735970  6129 sgd_solver.cpp:106] Iteration 65800, lr = 0.000153231
    I1226 22:58:00.943245  6129 solver.cpp:237] Iteration 65900, loss = 0.634666
    I1226 22:58:00.943352  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:58:00.943369  6129 solver.cpp:253]     Train net output #1: loss = 0.634666 (* 1 = 0.634666 loss)
    I1226 22:58:00.943377  6129 sgd_solver.cpp:106] Iteration 65900, lr = 0.000153079
    I1226 22:58:11.841617  6129 solver.cpp:341] Iteration 66000, Testing net (#0)
    I1226 22:58:16.148557  6129 solver.cpp:409]     Test net output #0: accuracy = 0.704083
    I1226 22:58:16.148605  6129 solver.cpp:409]     Test net output #1: loss = 0.848538 (* 1 = 0.848538 loss)
    I1226 22:58:16.193032  6129 solver.cpp:237] Iteration 66000, loss = 0.554298
    I1226 22:58:16.193078  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:58:16.193090  6129 solver.cpp:253]     Train net output #1: loss = 0.554298 (* 1 = 0.554298 loss)
    I1226 22:58:16.193102  6129 sgd_solver.cpp:106] Iteration 66000, lr = 0.000152928
    I1226 22:58:26.888334  6129 solver.cpp:237] Iteration 66100, loss = 0.630721
    I1226 22:58:26.888370  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:58:26.888381  6129 solver.cpp:253]     Train net output #1: loss = 0.630721 (* 1 = 0.630721 loss)
    I1226 22:58:26.888389  6129 sgd_solver.cpp:106] Iteration 66100, lr = 0.000152778
    I1226 22:58:37.740407  6129 solver.cpp:237] Iteration 66200, loss = 0.542107
    I1226 22:58:37.740550  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 22:58:37.740567  6129 solver.cpp:253]     Train net output #1: loss = 0.542107 (* 1 = 0.542107 loss)
    I1226 22:58:37.740578  6129 sgd_solver.cpp:106] Iteration 66200, lr = 0.000152627
    I1226 22:58:48.204701  6129 solver.cpp:237] Iteration 66300, loss = 0.607375
    I1226 22:58:48.204741  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 22:58:48.204756  6129 solver.cpp:253]     Train net output #1: loss = 0.607375 (* 1 = 0.607375 loss)
    I1226 22:58:48.204766  6129 sgd_solver.cpp:106] Iteration 66300, lr = 0.000152477
    I1226 22:58:58.985515  6129 solver.cpp:237] Iteration 66400, loss = 0.634484
    I1226 22:58:58.985559  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:58:58.985574  6129 solver.cpp:253]     Train net output #1: loss = 0.634484 (* 1 = 0.634484 loss)
    I1226 22:58:58.985586  6129 sgd_solver.cpp:106] Iteration 66400, lr = 0.000152327
    I1226 22:59:09.628679  6129 solver.cpp:237] Iteration 66500, loss = 0.552585
    I1226 22:59:09.628782  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 22:59:09.628799  6129 solver.cpp:253]     Train net output #1: loss = 0.552585 (* 1 = 0.552585 loss)
    I1226 22:59:09.628808  6129 sgd_solver.cpp:106] Iteration 66500, lr = 0.000152178
    I1226 22:59:20.333562  6129 solver.cpp:237] Iteration 66600, loss = 0.627415
    I1226 22:59:20.333601  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:59:20.333616  6129 solver.cpp:253]     Train net output #1: loss = 0.627415 (* 1 = 0.627415 loss)
    I1226 22:59:20.333626  6129 sgd_solver.cpp:106] Iteration 66600, lr = 0.000152029
    I1226 22:59:31.515928  6129 solver.cpp:237] Iteration 66700, loss = 0.536215
    I1226 22:59:31.515991  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 22:59:31.516013  6129 solver.cpp:253]     Train net output #1: loss = 0.536215 (* 1 = 0.536215 loss)
    I1226 22:59:31.516031  6129 sgd_solver.cpp:106] Iteration 66700, lr = 0.00015188
    I1226 22:59:42.525758  6129 solver.cpp:237] Iteration 66800, loss = 0.605216
    I1226 22:59:42.525871  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 22:59:42.525889  6129 solver.cpp:253]     Train net output #1: loss = 0.605216 (* 1 = 0.605216 loss)
    I1226 22:59:42.525902  6129 sgd_solver.cpp:106] Iteration 66800, lr = 0.000151732
    I1226 22:59:53.173105  6129 solver.cpp:237] Iteration 66900, loss = 0.633138
    I1226 22:59:53.173157  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 22:59:53.173179  6129 solver.cpp:253]     Train net output #1: loss = 0.633138 (* 1 = 0.633138 loss)
    I1226 22:59:53.173194  6129 sgd_solver.cpp:106] Iteration 66900, lr = 0.000151584
    I1226 23:00:04.063874  6129 solver.cpp:341] Iteration 67000, Testing net (#0)
    I1226 23:00:09.071506  6129 solver.cpp:409]     Test net output #0: accuracy = 0.706167
    I1226 23:00:09.071569  6129 solver.cpp:409]     Test net output #1: loss = 0.842611 (* 1 = 0.842611 loss)
    I1226 23:00:09.135854  6129 solver.cpp:237] Iteration 67000, loss = 0.551729
    I1226 23:00:09.135908  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:00:09.135931  6129 solver.cpp:253]     Train net output #1: loss = 0.551729 (* 1 = 0.551729 loss)
    I1226 23:00:09.135946  6129 sgd_solver.cpp:106] Iteration 67000, lr = 0.000151436
    I1226 23:00:21.837452  6129 solver.cpp:237] Iteration 67100, loss = 0.627553
    I1226 23:00:21.837630  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:00:21.837646  6129 solver.cpp:253]     Train net output #1: loss = 0.627553 (* 1 = 0.627553 loss)
    I1226 23:00:21.837656  6129 sgd_solver.cpp:106] Iteration 67100, lr = 0.000151289
    I1226 23:00:34.703666  6129 solver.cpp:237] Iteration 67200, loss = 0.542576
    I1226 23:00:34.703701  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:00:34.703712  6129 solver.cpp:253]     Train net output #1: loss = 0.542576 (* 1 = 0.542576 loss)
    I1226 23:00:34.703722  6129 sgd_solver.cpp:106] Iteration 67200, lr = 0.000151142
    I1226 23:00:45.509171  6129 solver.cpp:237] Iteration 67300, loss = 0.604324
    I1226 23:00:45.509209  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:00:45.509225  6129 solver.cpp:253]     Train net output #1: loss = 0.604324 (* 1 = 0.604324 loss)
    I1226 23:00:45.509237  6129 sgd_solver.cpp:106] Iteration 67300, lr = 0.000150995
    I1226 23:00:56.340628  6129 solver.cpp:237] Iteration 67400, loss = 0.632325
    I1226 23:00:56.340797  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:00:56.340824  6129 solver.cpp:253]     Train net output #1: loss = 0.632325 (* 1 = 0.632325 loss)
    I1226 23:00:56.340839  6129 sgd_solver.cpp:106] Iteration 67400, lr = 0.000150849
    I1226 23:01:07.300813  6129 solver.cpp:237] Iteration 67500, loss = 0.550377
    I1226 23:01:07.300849  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:01:07.300861  6129 solver.cpp:253]     Train net output #1: loss = 0.550377 (* 1 = 0.550377 loss)
    I1226 23:01:07.300868  6129 sgd_solver.cpp:106] Iteration 67500, lr = 0.000150703
    I1226 23:01:19.650650  6129 solver.cpp:237] Iteration 67600, loss = 0.626625
    I1226 23:01:19.650697  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:01:19.650710  6129 solver.cpp:253]     Train net output #1: loss = 0.626625 (* 1 = 0.626625 loss)
    I1226 23:01:19.650722  6129 sgd_solver.cpp:106] Iteration 67600, lr = 0.000150557
    I1226 23:01:31.362512  6129 solver.cpp:237] Iteration 67700, loss = 0.538249
    I1226 23:01:31.362681  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:01:31.362697  6129 solver.cpp:253]     Train net output #1: loss = 0.538249 (* 1 = 0.538249 loss)
    I1226 23:01:31.362704  6129 sgd_solver.cpp:106] Iteration 67700, lr = 0.000150412
    I1226 23:01:42.170939  6129 solver.cpp:237] Iteration 67800, loss = 0.603169
    I1226 23:01:42.170984  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:01:42.170996  6129 solver.cpp:253]     Train net output #1: loss = 0.603169 (* 1 = 0.603169 loss)
    I1226 23:01:42.171005  6129 sgd_solver.cpp:106] Iteration 67800, lr = 0.000150267
    I1226 23:01:53.025168  6129 solver.cpp:237] Iteration 67900, loss = 0.630808
    I1226 23:01:53.025213  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:01:53.025228  6129 solver.cpp:253]     Train net output #1: loss = 0.630808 (* 1 = 0.630808 loss)
    I1226 23:01:53.025238  6129 sgd_solver.cpp:106] Iteration 67900, lr = 0.000150122
    I1226 23:02:03.802536  6129 solver.cpp:341] Iteration 68000, Testing net (#0)
    I1226 23:02:08.173218  6129 solver.cpp:409]     Test net output #0: accuracy = 0.706417
    I1226 23:02:08.173269  6129 solver.cpp:409]     Test net output #1: loss = 0.840987 (* 1 = 0.840987 loss)
    I1226 23:02:08.217787  6129 solver.cpp:237] Iteration 68000, loss = 0.549473
    I1226 23:02:08.217821  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:02:08.217834  6129 solver.cpp:253]     Train net output #1: loss = 0.549473 (* 1 = 0.549473 loss)
    I1226 23:02:08.217847  6129 sgd_solver.cpp:106] Iteration 68000, lr = 0.000149978
    I1226 23:02:18.692360  6129 solver.cpp:237] Iteration 68100, loss = 0.62593
    I1226 23:02:18.692397  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:02:18.692409  6129 solver.cpp:253]     Train net output #1: loss = 0.62593 (* 1 = 0.62593 loss)
    I1226 23:02:18.692419  6129 sgd_solver.cpp:106] Iteration 68100, lr = 0.000149834
    I1226 23:02:29.183748  6129 solver.cpp:237] Iteration 68200, loss = 0.53973
    I1226 23:02:29.183784  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:02:29.183795  6129 solver.cpp:253]     Train net output #1: loss = 0.53973 (* 1 = 0.53973 loss)
    I1226 23:02:29.183804  6129 sgd_solver.cpp:106] Iteration 68200, lr = 0.00014969
    I1226 23:02:39.912536  6129 solver.cpp:237] Iteration 68300, loss = 0.600773
    I1226 23:02:39.912659  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:02:39.912675  6129 solver.cpp:253]     Train net output #1: loss = 0.600773 (* 1 = 0.600773 loss)
    I1226 23:02:39.912685  6129 sgd_solver.cpp:106] Iteration 68300, lr = 0.000149547
    I1226 23:02:51.041662  6129 solver.cpp:237] Iteration 68400, loss = 0.630649
    I1226 23:02:51.041699  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:02:51.041712  6129 solver.cpp:253]     Train net output #1: loss = 0.630649 (* 1 = 0.630649 loss)
    I1226 23:02:51.041721  6129 sgd_solver.cpp:106] Iteration 68400, lr = 0.000149404
    I1226 23:03:01.887790  6129 solver.cpp:237] Iteration 68500, loss = 0.54871
    I1226 23:03:01.887845  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:03:01.887867  6129 solver.cpp:253]     Train net output #1: loss = 0.54871 (* 1 = 0.54871 loss)
    I1226 23:03:01.887883  6129 sgd_solver.cpp:106] Iteration 68500, lr = 0.000149261
    I1226 23:03:12.574872  6129 solver.cpp:237] Iteration 68600, loss = 0.624493
    I1226 23:03:12.575008  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:03:12.575027  6129 solver.cpp:253]     Train net output #1: loss = 0.624493 (* 1 = 0.624493 loss)
    I1226 23:03:12.575039  6129 sgd_solver.cpp:106] Iteration 68600, lr = 0.000149118
    I1226 23:03:23.346020  6129 solver.cpp:237] Iteration 68700, loss = 0.52924
    I1226 23:03:23.346060  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:03:23.346074  6129 solver.cpp:253]     Train net output #1: loss = 0.52924 (* 1 = 0.52924 loss)
    I1226 23:03:23.346086  6129 sgd_solver.cpp:106] Iteration 68700, lr = 0.000148976
    I1226 23:03:34.293501  6129 solver.cpp:237] Iteration 68800, loss = 0.600149
    I1226 23:03:34.293572  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:03:34.293596  6129 solver.cpp:253]     Train net output #1: loss = 0.600149 (* 1 = 0.600149 loss)
    I1226 23:03:34.293612  6129 sgd_solver.cpp:106] Iteration 68800, lr = 0.000148834
    I1226 23:03:45.225028  6129 solver.cpp:237] Iteration 68900, loss = 0.628736
    I1226 23:03:45.225177  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:03:45.225189  6129 solver.cpp:253]     Train net output #1: loss = 0.628736 (* 1 = 0.628736 loss)
    I1226 23:03:45.225195  6129 sgd_solver.cpp:106] Iteration 68900, lr = 0.000148693
    I1226 23:03:56.385934  6129 solver.cpp:341] Iteration 69000, Testing net (#0)
    I1226 23:04:00.692598  6129 solver.cpp:409]     Test net output #0: accuracy = 0.705333
    I1226 23:04:00.692647  6129 solver.cpp:409]     Test net output #1: loss = 0.844956 (* 1 = 0.844956 loss)
    I1226 23:04:00.737133  6129 solver.cpp:237] Iteration 69000, loss = 0.548011
    I1226 23:04:00.737180  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:04:00.737193  6129 solver.cpp:253]     Train net output #1: loss = 0.548011 (* 1 = 0.548011 loss)
    I1226 23:04:00.737205  6129 sgd_solver.cpp:106] Iteration 69000, lr = 0.000148552
    I1226 23:04:11.402065  6129 solver.cpp:237] Iteration 69100, loss = 0.623997
    I1226 23:04:11.402103  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:04:11.402114  6129 solver.cpp:253]     Train net output #1: loss = 0.623997 (* 1 = 0.623997 loss)
    I1226 23:04:11.402123  6129 sgd_solver.cpp:106] Iteration 69100, lr = 0.000148411
    I1226 23:04:22.252362  6129 solver.cpp:237] Iteration 69200, loss = 0.539732
    I1226 23:04:22.252516  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:04:22.252531  6129 solver.cpp:253]     Train net output #1: loss = 0.539732 (* 1 = 0.539732 loss)
    I1226 23:04:22.252539  6129 sgd_solver.cpp:106] Iteration 69200, lr = 0.00014827
    I1226 23:04:33.023679  6129 solver.cpp:237] Iteration 69300, loss = 0.598805
    I1226 23:04:33.023736  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:04:33.023758  6129 solver.cpp:253]     Train net output #1: loss = 0.598805 (* 1 = 0.598805 loss)
    I1226 23:04:33.023774  6129 sgd_solver.cpp:106] Iteration 69300, lr = 0.00014813
    I1226 23:04:43.819401  6129 solver.cpp:237] Iteration 69400, loss = 0.629689
    I1226 23:04:43.819438  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:04:43.819450  6129 solver.cpp:253]     Train net output #1: loss = 0.629689 (* 1 = 0.629689 loss)
    I1226 23:04:43.819460  6129 sgd_solver.cpp:106] Iteration 69400, lr = 0.00014799
    I1226 23:04:55.144819  6129 solver.cpp:237] Iteration 69500, loss = 0.547332
    I1226 23:04:55.144966  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:04:55.144991  6129 solver.cpp:253]     Train net output #1: loss = 0.547332 (* 1 = 0.547332 loss)
    I1226 23:04:55.145000  6129 sgd_solver.cpp:106] Iteration 69500, lr = 0.00014785
    I1226 23:05:06.211436  6129 solver.cpp:237] Iteration 69600, loss = 0.621546
    I1226 23:05:06.211478  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:05:06.211493  6129 solver.cpp:253]     Train net output #1: loss = 0.621546 (* 1 = 0.621546 loss)
    I1226 23:05:06.211503  6129 sgd_solver.cpp:106] Iteration 69600, lr = 0.000147711
    I1226 23:05:17.141185  6129 solver.cpp:237] Iteration 69700, loss = 0.528296
    I1226 23:05:17.141240  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:05:17.141263  6129 solver.cpp:253]     Train net output #1: loss = 0.528296 (* 1 = 0.528296 loss)
    I1226 23:05:17.141278  6129 sgd_solver.cpp:106] Iteration 69700, lr = 0.000147572
    I1226 23:05:28.144541  6129 solver.cpp:237] Iteration 69800, loss = 0.596515
    I1226 23:05:28.144692  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:05:28.144707  6129 solver.cpp:253]     Train net output #1: loss = 0.596515 (* 1 = 0.596515 loss)
    I1226 23:05:28.144718  6129 sgd_solver.cpp:106] Iteration 69800, lr = 0.000147433
    I1226 23:05:39.103273  6129 solver.cpp:237] Iteration 69900, loss = 0.628506
    I1226 23:05:39.103330  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:05:39.103346  6129 solver.cpp:253]     Train net output #1: loss = 0.628506 (* 1 = 0.628506 loss)
    I1226 23:05:39.103359  6129 sgd_solver.cpp:106] Iteration 69900, lr = 0.000147295
    I1226 23:05:50.874467  6129 solver.cpp:341] Iteration 70000, Testing net (#0)
    I1226 23:05:55.398542  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7105
    I1226 23:05:55.398617  6129 solver.cpp:409]     Test net output #1: loss = 0.839918 (* 1 = 0.839918 loss)
    I1226 23:05:55.443519  6129 solver.cpp:237] Iteration 70000, loss = 0.546232
    I1226 23:05:55.443588  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:05:55.443611  6129 solver.cpp:253]     Train net output #1: loss = 0.546232 (* 1 = 0.546232 loss)
    I1226 23:05:55.443630  6129 sgd_solver.cpp:106] Iteration 70000, lr = 0.000147157
    I1226 23:06:07.940289  6129 solver.cpp:237] Iteration 70100, loss = 0.62484
    I1226 23:06:07.940446  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:06:07.940486  6129 solver.cpp:253]     Train net output #1: loss = 0.62484 (* 1 = 0.62484 loss)
    I1226 23:06:07.940493  6129 sgd_solver.cpp:106] Iteration 70100, lr = 0.000147019
    I1226 23:06:22.399929  6129 solver.cpp:237] Iteration 70200, loss = 0.537475
    I1226 23:06:22.399966  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:06:22.399981  6129 solver.cpp:253]     Train net output #1: loss = 0.537475 (* 1 = 0.537475 loss)
    I1226 23:06:22.399991  6129 sgd_solver.cpp:106] Iteration 70200, lr = 0.000146882
    I1226 23:06:33.697460  6129 solver.cpp:237] Iteration 70300, loss = 0.594731
    I1226 23:06:33.697513  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:06:33.697535  6129 solver.cpp:253]     Train net output #1: loss = 0.594731 (* 1 = 0.594731 loss)
    I1226 23:06:33.697552  6129 sgd_solver.cpp:106] Iteration 70300, lr = 0.000146744
    I1226 23:06:44.901002  6129 solver.cpp:237] Iteration 70400, loss = 0.628489
    I1226 23:06:44.901156  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:06:44.901170  6129 solver.cpp:253]     Train net output #1: loss = 0.628489 (* 1 = 0.628489 loss)
    I1226 23:06:44.901176  6129 sgd_solver.cpp:106] Iteration 70400, lr = 0.000146607
    I1226 23:06:56.951357  6129 solver.cpp:237] Iteration 70500, loss = 0.544947
    I1226 23:06:56.951412  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:06:56.951428  6129 solver.cpp:253]     Train net output #1: loss = 0.544947 (* 1 = 0.544947 loss)
    I1226 23:06:56.951442  6129 sgd_solver.cpp:106] Iteration 70500, lr = 0.000146471
    I1226 23:07:07.995139  6129 solver.cpp:237] Iteration 70600, loss = 0.619781
    I1226 23:07:07.995182  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:07:07.995198  6129 solver.cpp:253]     Train net output #1: loss = 0.619781 (* 1 = 0.619781 loss)
    I1226 23:07:07.995210  6129 sgd_solver.cpp:106] Iteration 70600, lr = 0.000146335
    I1226 23:07:18.960908  6129 solver.cpp:237] Iteration 70700, loss = 0.528677
    I1226 23:07:18.961082  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:07:18.961097  6129 solver.cpp:253]     Train net output #1: loss = 0.528677 (* 1 = 0.528677 loss)
    I1226 23:07:18.961104  6129 sgd_solver.cpp:106] Iteration 70700, lr = 0.000146198
    I1226 23:07:29.784855  6129 solver.cpp:237] Iteration 70800, loss = 0.593625
    I1226 23:07:29.784893  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:07:29.784904  6129 solver.cpp:253]     Train net output #1: loss = 0.593625 (* 1 = 0.593625 loss)
    I1226 23:07:29.784914  6129 sgd_solver.cpp:106] Iteration 70800, lr = 0.000146063
    I1226 23:07:40.706122  6129 solver.cpp:237] Iteration 70900, loss = 0.627445
    I1226 23:07:40.706181  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:07:40.706203  6129 solver.cpp:253]     Train net output #1: loss = 0.627445 (* 1 = 0.627445 loss)
    I1226 23:07:40.706219  6129 sgd_solver.cpp:106] Iteration 70900, lr = 0.000145927
    I1226 23:07:52.044837  6129 solver.cpp:341] Iteration 71000, Testing net (#0)
    I1226 23:07:56.579416  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7055
    I1226 23:07:56.579465  6129 solver.cpp:409]     Test net output #1: loss = 0.844642 (* 1 = 0.844642 loss)
    I1226 23:07:56.659073  6129 solver.cpp:237] Iteration 71000, loss = 0.544695
    I1226 23:07:56.659126  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:07:56.659142  6129 solver.cpp:253]     Train net output #1: loss = 0.544695 (* 1 = 0.544695 loss)
    I1226 23:07:56.659165  6129 sgd_solver.cpp:106] Iteration 71000, lr = 0.000145792
    I1226 23:08:07.863251  6129 solver.cpp:237] Iteration 71100, loss = 0.621916
    I1226 23:08:07.863299  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:08:07.863314  6129 solver.cpp:253]     Train net output #1: loss = 0.621916 (* 1 = 0.621916 loss)
    I1226 23:08:07.863327  6129 sgd_solver.cpp:106] Iteration 71100, lr = 0.000145657
    I1226 23:08:20.259660  6129 solver.cpp:237] Iteration 71200, loss = 0.532495
    I1226 23:08:20.259714  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:08:20.259737  6129 solver.cpp:253]     Train net output #1: loss = 0.532495 (* 1 = 0.532495 loss)
    I1226 23:08:20.259752  6129 sgd_solver.cpp:106] Iteration 71200, lr = 0.000145523
    I1226 23:08:31.120283  6129 solver.cpp:237] Iteration 71300, loss = 0.591
    I1226 23:08:31.120436  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:08:31.120470  6129 solver.cpp:253]     Train net output #1: loss = 0.591 (* 1 = 0.591 loss)
    I1226 23:08:31.120489  6129 sgd_solver.cpp:106] Iteration 71300, lr = 0.000145389
    I1226 23:08:41.632432  6129 solver.cpp:237] Iteration 71400, loss = 0.627198
    I1226 23:08:41.632477  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:08:41.632489  6129 solver.cpp:253]     Train net output #1: loss = 0.627198 (* 1 = 0.627198 loss)
    I1226 23:08:41.632500  6129 sgd_solver.cpp:106] Iteration 71400, lr = 0.000145255
    I1226 23:08:52.210021  6129 solver.cpp:237] Iteration 71500, loss = 0.544477
    I1226 23:08:52.210078  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:08:52.210099  6129 solver.cpp:253]     Train net output #1: loss = 0.544477 (* 1 = 0.544477 loss)
    I1226 23:08:52.210116  6129 sgd_solver.cpp:106] Iteration 71500, lr = 0.000145121
    I1226 23:09:05.412986  6129 solver.cpp:237] Iteration 71600, loss = 0.620183
    I1226 23:09:05.413143  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:09:05.413171  6129 solver.cpp:253]     Train net output #1: loss = 0.620183 (* 1 = 0.620183 loss)
    I1226 23:09:05.413183  6129 sgd_solver.cpp:106] Iteration 71600, lr = 0.000144987
    I1226 23:09:16.930428  6129 solver.cpp:237] Iteration 71700, loss = 0.535364
    I1226 23:09:16.930467  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:09:16.930480  6129 solver.cpp:253]     Train net output #1: loss = 0.535364 (* 1 = 0.535364 loss)
    I1226 23:09:16.930487  6129 sgd_solver.cpp:106] Iteration 71700, lr = 0.000144854
    I1226 23:09:28.882577  6129 solver.cpp:237] Iteration 71800, loss = 0.589667
    I1226 23:09:28.882634  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:09:28.882657  6129 solver.cpp:253]     Train net output #1: loss = 0.589667 (* 1 = 0.589667 loss)
    I1226 23:09:28.882671  6129 sgd_solver.cpp:106] Iteration 71800, lr = 0.000144721
    I1226 23:09:40.156716  6129 solver.cpp:237] Iteration 71900, loss = 0.626138
    I1226 23:09:40.156884  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:09:40.156899  6129 solver.cpp:253]     Train net output #1: loss = 0.626138 (* 1 = 0.626138 loss)
    I1226 23:09:40.156908  6129 sgd_solver.cpp:106] Iteration 71900, lr = 0.000144589
    I1226 23:09:51.922672  6129 solver.cpp:341] Iteration 72000, Testing net (#0)
    I1226 23:09:56.350028  6129 solver.cpp:409]     Test net output #0: accuracy = 0.707917
    I1226 23:09:56.350078  6129 solver.cpp:409]     Test net output #1: loss = 0.839636 (* 1 = 0.839636 loss)
    I1226 23:09:56.394512  6129 solver.cpp:237] Iteration 72000, loss = 0.543436
    I1226 23:09:56.394544  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:09:56.394556  6129 solver.cpp:253]     Train net output #1: loss = 0.543436 (* 1 = 0.543436 loss)
    I1226 23:09:56.394567  6129 sgd_solver.cpp:106] Iteration 72000, lr = 0.000144457
    I1226 23:10:07.462060  6129 solver.cpp:237] Iteration 72100, loss = 0.619092
    I1226 23:10:07.462102  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:10:07.462117  6129 solver.cpp:253]     Train net output #1: loss = 0.619092 (* 1 = 0.619092 loss)
    I1226 23:10:07.462128  6129 sgd_solver.cpp:106] Iteration 72100, lr = 0.000144325
    I1226 23:10:18.304570  6129 solver.cpp:237] Iteration 72200, loss = 0.524698
    I1226 23:10:18.304733  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:10:18.304749  6129 solver.cpp:253]     Train net output #1: loss = 0.524698 (* 1 = 0.524698 loss)
    I1226 23:10:18.304759  6129 sgd_solver.cpp:106] Iteration 72200, lr = 0.000144193
    I1226 23:10:29.339668  6129 solver.cpp:237] Iteration 72300, loss = 0.587775
    I1226 23:10:29.339709  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:10:29.339725  6129 solver.cpp:253]     Train net output #1: loss = 0.587775 (* 1 = 0.587775 loss)
    I1226 23:10:29.339735  6129 sgd_solver.cpp:106] Iteration 72300, lr = 0.000144062
    I1226 23:10:40.321305  6129 solver.cpp:237] Iteration 72400, loss = 0.626472
    I1226 23:10:40.321351  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:10:40.321364  6129 solver.cpp:253]     Train net output #1: loss = 0.626472 (* 1 = 0.626472 loss)
    I1226 23:10:40.321377  6129 sgd_solver.cpp:106] Iteration 72400, lr = 0.00014393
    I1226 23:10:51.046349  6129 solver.cpp:237] Iteration 72500, loss = 0.542915
    I1226 23:10:51.046499  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:10:51.046511  6129 solver.cpp:253]     Train net output #1: loss = 0.542915 (* 1 = 0.542915 loss)
    I1226 23:10:51.046517  6129 sgd_solver.cpp:106] Iteration 72500, lr = 0.0001438
    I1226 23:11:03.229866  6129 solver.cpp:237] Iteration 72600, loss = 0.617431
    I1226 23:11:03.229902  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:11:03.229913  6129 solver.cpp:253]     Train net output #1: loss = 0.617431 (* 1 = 0.617431 loss)
    I1226 23:11:03.229923  6129 sgd_solver.cpp:106] Iteration 72600, lr = 0.000143669
    I1226 23:11:14.065058  6129 solver.cpp:237] Iteration 72700, loss = 0.527903
    I1226 23:11:14.065095  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:11:14.065106  6129 solver.cpp:253]     Train net output #1: loss = 0.527903 (* 1 = 0.527903 loss)
    I1226 23:11:14.065116  6129 sgd_solver.cpp:106] Iteration 72700, lr = 0.000143539
    I1226 23:11:25.632580  6129 solver.cpp:237] Iteration 72800, loss = 0.586561
    I1226 23:11:25.632750  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:11:25.632763  6129 solver.cpp:253]     Train net output #1: loss = 0.586561 (* 1 = 0.586561 loss)
    I1226 23:11:25.632769  6129 sgd_solver.cpp:106] Iteration 72800, lr = 0.000143409
    I1226 23:11:36.715662  6129 solver.cpp:237] Iteration 72900, loss = 0.626164
    I1226 23:11:36.715701  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:11:36.715715  6129 solver.cpp:253]     Train net output #1: loss = 0.626164 (* 1 = 0.626164 loss)
    I1226 23:11:36.715728  6129 sgd_solver.cpp:106] Iteration 72900, lr = 0.000143279
    I1226 23:11:47.102387  6129 solver.cpp:341] Iteration 73000, Testing net (#0)
    I1226 23:11:51.406522  6129 solver.cpp:409]     Test net output #0: accuracy = 0.708083
    I1226 23:11:51.406564  6129 solver.cpp:409]     Test net output #1: loss = 0.838196 (* 1 = 0.838196 loss)
    I1226 23:11:51.451009  6129 solver.cpp:237] Iteration 73000, loss = 0.541748
    I1226 23:11:51.451033  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:11:51.451045  6129 solver.cpp:253]     Train net output #1: loss = 0.541748 (* 1 = 0.541748 loss)
    I1226 23:11:51.451057  6129 sgd_solver.cpp:106] Iteration 73000, lr = 0.000143149
    I1226 23:12:01.970007  6129 solver.cpp:237] Iteration 73100, loss = 0.61748
    I1226 23:12:01.970208  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:12:01.970223  6129 solver.cpp:253]     Train net output #1: loss = 0.61748 (* 1 = 0.61748 loss)
    I1226 23:12:01.970233  6129 sgd_solver.cpp:106] Iteration 73100, lr = 0.00014302
    I1226 23:12:12.437824  6129 solver.cpp:237] Iteration 73200, loss = 0.534891
    I1226 23:12:12.437856  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:12:12.437868  6129 solver.cpp:253]     Train net output #1: loss = 0.534891 (* 1 = 0.534891 loss)
    I1226 23:12:12.437876  6129 sgd_solver.cpp:106] Iteration 73200, lr = 0.000142891
    I1226 23:12:22.936241  6129 solver.cpp:237] Iteration 73300, loss = 0.585597
    I1226 23:12:22.936295  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:12:22.936316  6129 solver.cpp:253]     Train net output #1: loss = 0.585597 (* 1 = 0.585597 loss)
    I1226 23:12:22.936331  6129 sgd_solver.cpp:106] Iteration 73300, lr = 0.000142763
    I1226 23:12:33.409662  6129 solver.cpp:237] Iteration 73400, loss = 0.62518
    I1226 23:12:33.409785  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:12:33.409803  6129 solver.cpp:253]     Train net output #1: loss = 0.62518 (* 1 = 0.62518 loss)
    I1226 23:12:33.409814  6129 sgd_solver.cpp:106] Iteration 73400, lr = 0.000142634
    I1226 23:12:43.879884  6129 solver.cpp:237] Iteration 73500, loss = 0.541079
    I1226 23:12:43.879941  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:12:43.879963  6129 solver.cpp:253]     Train net output #1: loss = 0.541079 (* 1 = 0.541079 loss)
    I1226 23:12:43.879979  6129 sgd_solver.cpp:106] Iteration 73500, lr = 0.000142506
    I1226 23:12:54.376373  6129 solver.cpp:237] Iteration 73600, loss = 0.621718
    I1226 23:12:54.376412  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:12:54.376427  6129 solver.cpp:253]     Train net output #1: loss = 0.621718 (* 1 = 0.621718 loss)
    I1226 23:12:54.376437  6129 sgd_solver.cpp:106] Iteration 73600, lr = 0.000142378
    I1226 23:13:04.886934  6129 solver.cpp:237] Iteration 73700, loss = 0.523783
    I1226 23:13:04.887053  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:13:04.887069  6129 solver.cpp:253]     Train net output #1: loss = 0.523783 (* 1 = 0.523783 loss)
    I1226 23:13:04.887076  6129 sgd_solver.cpp:106] Iteration 73700, lr = 0.000142251
    I1226 23:13:15.373555  6129 solver.cpp:237] Iteration 73800, loss = 0.584339
    I1226 23:13:15.373595  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:13:15.373610  6129 solver.cpp:253]     Train net output #1: loss = 0.584339 (* 1 = 0.584339 loss)
    I1226 23:13:15.373620  6129 sgd_solver.cpp:106] Iteration 73800, lr = 0.000142123
    I1226 23:13:25.873381  6129 solver.cpp:237] Iteration 73900, loss = 0.624871
    I1226 23:13:25.873437  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:13:25.873459  6129 solver.cpp:253]     Train net output #1: loss = 0.624871 (* 1 = 0.624871 loss)
    I1226 23:13:25.873476  6129 sgd_solver.cpp:106] Iteration 73900, lr = 0.000141996
    I1226 23:13:36.245053  6129 solver.cpp:341] Iteration 74000, Testing net (#0)
    I1226 23:13:40.519757  6129 solver.cpp:409]     Test net output #0: accuracy = 0.706
    I1226 23:13:40.519821  6129 solver.cpp:409]     Test net output #1: loss = 0.842537 (* 1 = 0.842537 loss)
    I1226 23:13:40.578299  6129 solver.cpp:237] Iteration 74000, loss = 0.540376
    I1226 23:13:40.578348  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:13:40.578361  6129 solver.cpp:253]     Train net output #1: loss = 0.540376 (* 1 = 0.540376 loss)
    I1226 23:13:40.578372  6129 sgd_solver.cpp:106] Iteration 74000, lr = 0.000141869
    I1226 23:13:51.047617  6129 solver.cpp:237] Iteration 74100, loss = 0.616409
    I1226 23:13:51.047667  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:13:51.047682  6129 solver.cpp:253]     Train net output #1: loss = 0.616409 (* 1 = 0.616409 loss)
    I1226 23:13:51.047691  6129 sgd_solver.cpp:106] Iteration 74100, lr = 0.000141743
    I1226 23:14:01.532196  6129 solver.cpp:237] Iteration 74200, loss = 0.529235
    I1226 23:14:01.532246  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:14:01.532260  6129 solver.cpp:253]     Train net output #1: loss = 0.529235 (* 1 = 0.529235 loss)
    I1226 23:14:01.532271  6129 sgd_solver.cpp:106] Iteration 74200, lr = 0.000141617
    I1226 23:14:12.020123  6129 solver.cpp:237] Iteration 74300, loss = 0.583579
    I1226 23:14:12.020241  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:14:12.020259  6129 solver.cpp:253]     Train net output #1: loss = 0.583579 (* 1 = 0.583579 loss)
    I1226 23:14:12.020270  6129 sgd_solver.cpp:106] Iteration 74300, lr = 0.000141491
    I1226 23:14:22.502140  6129 solver.cpp:237] Iteration 74400, loss = 0.624863
    I1226 23:14:22.502187  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:14:22.502197  6129 solver.cpp:253]     Train net output #1: loss = 0.624863 (* 1 = 0.624863 loss)
    I1226 23:14:22.502205  6129 sgd_solver.cpp:106] Iteration 74400, lr = 0.000141365
    I1226 23:14:32.986184  6129 solver.cpp:237] Iteration 74500, loss = 0.539664
    I1226 23:14:32.986240  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:14:32.986261  6129 solver.cpp:253]     Train net output #1: loss = 0.539664 (* 1 = 0.539664 loss)
    I1226 23:14:32.986279  6129 sgd_solver.cpp:106] Iteration 74500, lr = 0.000141239
    I1226 23:14:43.474784  6129 solver.cpp:237] Iteration 74600, loss = 0.620068
    I1226 23:14:43.474916  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:14:43.474941  6129 solver.cpp:253]     Train net output #1: loss = 0.620068 (* 1 = 0.620068 loss)
    I1226 23:14:43.474949  6129 sgd_solver.cpp:106] Iteration 74600, lr = 0.000141114
    I1226 23:14:54.016918  6129 solver.cpp:237] Iteration 74700, loss = 0.525023
    I1226 23:14:54.016963  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:14:54.016978  6129 solver.cpp:253]     Train net output #1: loss = 0.525023 (* 1 = 0.525023 loss)
    I1226 23:14:54.016990  6129 sgd_solver.cpp:106] Iteration 74700, lr = 0.000140989
    I1226 23:15:04.523968  6129 solver.cpp:237] Iteration 74800, loss = 0.581448
    I1226 23:15:04.524013  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:15:04.524027  6129 solver.cpp:253]     Train net output #1: loss = 0.581448 (* 1 = 0.581448 loss)
    I1226 23:15:04.524039  6129 sgd_solver.cpp:106] Iteration 74800, lr = 0.000140864
    I1226 23:15:15.005928  6129 solver.cpp:237] Iteration 74900, loss = 0.624803
    I1226 23:15:15.006068  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:15:15.006088  6129 solver.cpp:253]     Train net output #1: loss = 0.624803 (* 1 = 0.624803 loss)
    I1226 23:15:15.006099  6129 sgd_solver.cpp:106] Iteration 74900, lr = 0.00014074
    I1226 23:15:25.384804  6129 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_75000.caffemodel
    I1226 23:15:25.440449  6129 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_75000.solverstate
    I1226 23:15:25.441988  6129 solver.cpp:341] Iteration 75000, Testing net (#0)
    I1226 23:15:29.730762  6129 solver.cpp:409]     Test net output #0: accuracy = 0.71125
    I1226 23:15:29.730810  6129 solver.cpp:409]     Test net output #1: loss = 0.836843 (* 1 = 0.836843 loss)
    I1226 23:15:29.775353  6129 solver.cpp:237] Iteration 75000, loss = 0.538311
    I1226 23:15:29.775395  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:15:29.775409  6129 solver.cpp:253]     Train net output #1: loss = 0.538311 (* 1 = 0.538311 loss)
    I1226 23:15:29.775421  6129 sgd_solver.cpp:106] Iteration 75000, lr = 0.000140616
    I1226 23:15:40.283421  6129 solver.cpp:237] Iteration 75100, loss = 0.616665
    I1226 23:15:40.283468  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:15:40.283480  6129 solver.cpp:253]     Train net output #1: loss = 0.616665 (* 1 = 0.616665 loss)
    I1226 23:15:40.283488  6129 sgd_solver.cpp:106] Iteration 75100, lr = 0.000140492
    I1226 23:15:50.799875  6129 solver.cpp:237] Iteration 75200, loss = 0.531186
    I1226 23:15:50.800026  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:15:50.800050  6129 solver.cpp:253]     Train net output #1: loss = 0.531186 (* 1 = 0.531186 loss)
    I1226 23:15:50.800060  6129 sgd_solver.cpp:106] Iteration 75200, lr = 0.000140368
    I1226 23:16:01.293568  6129 solver.cpp:237] Iteration 75300, loss = 0.581299
    I1226 23:16:01.293614  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:16:01.293625  6129 solver.cpp:253]     Train net output #1: loss = 0.581299 (* 1 = 0.581299 loss)
    I1226 23:16:01.293633  6129 sgd_solver.cpp:106] Iteration 75300, lr = 0.000140245
    I1226 23:16:11.780184  6129 solver.cpp:237] Iteration 75400, loss = 0.624716
    I1226 23:16:11.780226  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:16:11.780241  6129 solver.cpp:253]     Train net output #1: loss = 0.624716 (* 1 = 0.624716 loss)
    I1226 23:16:11.780251  6129 sgd_solver.cpp:106] Iteration 75400, lr = 0.000140121
    I1226 23:16:22.264968  6129 solver.cpp:237] Iteration 75500, loss = 0.537455
    I1226 23:16:22.265102  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:16:22.265125  6129 solver.cpp:253]     Train net output #1: loss = 0.537455 (* 1 = 0.537455 loss)
    I1226 23:16:22.265130  6129 sgd_solver.cpp:106] Iteration 75500, lr = 0.000139999
    I1226 23:16:39.563308  6129 solver.cpp:237] Iteration 75600, loss = 0.615713
    I1226 23:16:39.563352  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:16:39.563364  6129 solver.cpp:253]     Train net output #1: loss = 0.615713 (* 1 = 0.615713 loss)
    I1226 23:16:39.563374  6129 sgd_solver.cpp:106] Iteration 75600, lr = 0.000139876
    I1226 23:16:50.598340  6129 solver.cpp:237] Iteration 75700, loss = 0.520017
    I1226 23:16:50.598397  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:16:50.598419  6129 solver.cpp:253]     Train net output #1: loss = 0.520017 (* 1 = 0.520017 loss)
    I1226 23:16:50.598435  6129 sgd_solver.cpp:106] Iteration 75700, lr = 0.000139753
    I1226 23:17:01.433825  6129 solver.cpp:237] Iteration 75800, loss = 0.580087
    I1226 23:17:01.433969  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:17:01.433982  6129 solver.cpp:253]     Train net output #1: loss = 0.580087 (* 1 = 0.580087 loss)
    I1226 23:17:01.433990  6129 sgd_solver.cpp:106] Iteration 75800, lr = 0.000139631
    I1226 23:17:12.442060  6129 solver.cpp:237] Iteration 75900, loss = 0.62428
    I1226 23:17:12.442101  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:17:12.442114  6129 solver.cpp:253]     Train net output #1: loss = 0.62428 (* 1 = 0.62428 loss)
    I1226 23:17:12.442126  6129 sgd_solver.cpp:106] Iteration 75900, lr = 0.000139509
    I1226 23:17:23.354341  6129 solver.cpp:341] Iteration 76000, Testing net (#0)
    I1226 23:17:27.864352  6129 solver.cpp:409]     Test net output #0: accuracy = 0.707166
    I1226 23:17:27.864397  6129 solver.cpp:409]     Test net output #1: loss = 0.841806 (* 1 = 0.841806 loss)
    I1226 23:17:27.909024  6129 solver.cpp:237] Iteration 76000, loss = 0.53605
    I1226 23:17:27.909065  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:17:27.909080  6129 solver.cpp:253]     Train net output #1: loss = 0.53605 (* 1 = 0.53605 loss)
    I1226 23:17:27.909090  6129 sgd_solver.cpp:106] Iteration 76000, lr = 0.000139388
    I1226 23:17:38.776054  6129 solver.cpp:237] Iteration 76100, loss = 0.615586
    I1226 23:17:38.776206  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:17:38.776228  6129 solver.cpp:253]     Train net output #1: loss = 0.615586 (* 1 = 0.615586 loss)
    I1226 23:17:38.776234  6129 sgd_solver.cpp:106] Iteration 76100, lr = 0.000139266
    I1226 23:17:49.637208  6129 solver.cpp:237] Iteration 76200, loss = 0.526241
    I1226 23:17:49.637248  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:17:49.637265  6129 solver.cpp:253]     Train net output #1: loss = 0.526241 (* 1 = 0.526241 loss)
    I1226 23:17:49.637280  6129 sgd_solver.cpp:106] Iteration 76200, lr = 0.000139145
    I1226 23:18:00.525559  6129 solver.cpp:237] Iteration 76300, loss = 0.578778
    I1226 23:18:00.525600  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:18:00.525617  6129 solver.cpp:253]     Train net output #1: loss = 0.578778 (* 1 = 0.578778 loss)
    I1226 23:18:00.525631  6129 sgd_solver.cpp:106] Iteration 76300, lr = 0.000139024
    I1226 23:18:11.396293  6129 solver.cpp:237] Iteration 76400, loss = 0.624551
    I1226 23:18:11.396440  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:18:11.396471  6129 solver.cpp:253]     Train net output #1: loss = 0.624551 (* 1 = 0.624551 loss)
    I1226 23:18:11.396482  6129 sgd_solver.cpp:106] Iteration 76400, lr = 0.000138903
    I1226 23:18:22.254001  6129 solver.cpp:237] Iteration 76500, loss = 0.534741
    I1226 23:18:22.254040  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:18:22.254060  6129 solver.cpp:253]     Train net output #1: loss = 0.534741 (* 1 = 0.534741 loss)
    I1226 23:18:22.254070  6129 sgd_solver.cpp:106] Iteration 76500, lr = 0.000138783
    I1226 23:18:33.093933  6129 solver.cpp:237] Iteration 76600, loss = 0.615471
    I1226 23:18:33.093972  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:18:33.093991  6129 solver.cpp:253]     Train net output #1: loss = 0.615471 (* 1 = 0.615471 loss)
    I1226 23:18:33.094002  6129 sgd_solver.cpp:106] Iteration 76600, lr = 0.000138663
    I1226 23:18:43.947728  6129 solver.cpp:237] Iteration 76700, loss = 0.522648
    I1226 23:18:43.947832  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:18:43.947849  6129 solver.cpp:253]     Train net output #1: loss = 0.522648 (* 1 = 0.522648 loss)
    I1226 23:18:43.947854  6129 sgd_solver.cpp:106] Iteration 76700, lr = 0.000138543
    I1226 23:18:54.943673  6129 solver.cpp:237] Iteration 76800, loss = 0.577992
    I1226 23:18:54.943712  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:18:54.943727  6129 solver.cpp:253]     Train net output #1: loss = 0.577992 (* 1 = 0.577992 loss)
    I1226 23:18:54.943737  6129 sgd_solver.cpp:106] Iteration 76800, lr = 0.000138423
    I1226 23:19:05.980439  6129 solver.cpp:237] Iteration 76900, loss = 0.623916
    I1226 23:19:05.980485  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:19:05.980500  6129 solver.cpp:253]     Train net output #1: loss = 0.623916 (* 1 = 0.623916 loss)
    I1226 23:19:05.980511  6129 sgd_solver.cpp:106] Iteration 76900, lr = 0.000138304
    I1226 23:19:16.870429  6129 solver.cpp:341] Iteration 77000, Testing net (#0)
    I1226 23:19:21.276263  6129 solver.cpp:409]     Test net output #0: accuracy = 0.709667
    I1226 23:19:21.276304  6129 solver.cpp:409]     Test net output #1: loss = 0.836121 (* 1 = 0.836121 loss)
    I1226 23:19:21.320730  6129 solver.cpp:237] Iteration 77000, loss = 0.533671
    I1226 23:19:21.320755  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:19:21.320766  6129 solver.cpp:253]     Train net output #1: loss = 0.533671 (* 1 = 0.533671 loss)
    I1226 23:19:21.320775  6129 sgd_solver.cpp:106] Iteration 77000, lr = 0.000138184
    I1226 23:19:32.215473  6129 solver.cpp:237] Iteration 77100, loss = 0.61327
    I1226 23:19:32.215517  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:19:32.215529  6129 solver.cpp:253]     Train net output #1: loss = 0.61327 (* 1 = 0.61327 loss)
    I1226 23:19:32.215538  6129 sgd_solver.cpp:106] Iteration 77100, lr = 0.000138065
    I1226 23:19:43.064913  6129 solver.cpp:237] Iteration 77200, loss = 0.519713
    I1226 23:19:43.064968  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:19:43.064990  6129 solver.cpp:253]     Train net output #1: loss = 0.519713 (* 1 = 0.519713 loss)
    I1226 23:19:43.065006  6129 sgd_solver.cpp:106] Iteration 77200, lr = 0.000137946
    I1226 23:19:53.921303  6129 solver.cpp:237] Iteration 77300, loss = 0.576031
    I1226 23:19:53.921445  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:19:53.921464  6129 solver.cpp:253]     Train net output #1: loss = 0.576031 (* 1 = 0.576031 loss)
    I1226 23:19:53.921475  6129 sgd_solver.cpp:106] Iteration 77300, lr = 0.000137828
    I1226 23:20:04.829603  6129 solver.cpp:237] Iteration 77400, loss = 0.624168
    I1226 23:20:04.829640  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:20:04.829653  6129 solver.cpp:253]     Train net output #1: loss = 0.624168 (* 1 = 0.624168 loss)
    I1226 23:20:04.829661  6129 sgd_solver.cpp:106] Iteration 77400, lr = 0.00013771
    I1226 23:20:15.687899  6129 solver.cpp:237] Iteration 77500, loss = 0.533105
    I1226 23:20:15.687942  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:20:15.687957  6129 solver.cpp:253]     Train net output #1: loss = 0.533105 (* 1 = 0.533105 loss)
    I1226 23:20:15.687966  6129 sgd_solver.cpp:106] Iteration 77500, lr = 0.000137592
    I1226 23:20:26.587388  6129 solver.cpp:237] Iteration 77600, loss = 0.611816
    I1226 23:20:26.587527  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:20:26.587543  6129 solver.cpp:253]     Train net output #1: loss = 0.611816 (* 1 = 0.611816 loss)
    I1226 23:20:26.587553  6129 sgd_solver.cpp:106] Iteration 77600, lr = 0.000137474
    I1226 23:20:37.426385  6129 solver.cpp:237] Iteration 77700, loss = 0.518207
    I1226 23:20:37.426422  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:20:37.426434  6129 solver.cpp:253]     Train net output #1: loss = 0.518207 (* 1 = 0.518207 loss)
    I1226 23:20:37.426443  6129 sgd_solver.cpp:106] Iteration 77700, lr = 0.000137356
    I1226 23:20:48.251968  6129 solver.cpp:237] Iteration 77800, loss = 0.575954
    I1226 23:20:48.252012  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:20:48.252022  6129 solver.cpp:253]     Train net output #1: loss = 0.575954 (* 1 = 0.575954 loss)
    I1226 23:20:48.252032  6129 sgd_solver.cpp:106] Iteration 77800, lr = 0.000137239
    I1226 23:20:59.112591  6129 solver.cpp:237] Iteration 77900, loss = 0.624357
    I1226 23:20:59.112750  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:20:59.112774  6129 solver.cpp:253]     Train net output #1: loss = 0.624357 (* 1 = 0.624357 loss)
    I1226 23:20:59.112782  6129 sgd_solver.cpp:106] Iteration 77900, lr = 0.000137122
    I1226 23:21:09.934959  6129 solver.cpp:341] Iteration 78000, Testing net (#0)
    I1226 23:21:14.388093  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7095
    I1226 23:21:14.388139  6129 solver.cpp:409]     Test net output #1: loss = 0.834984 (* 1 = 0.834984 loss)
    I1226 23:21:14.442164  6129 solver.cpp:237] Iteration 78000, loss = 0.532124
    I1226 23:21:14.442203  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:21:14.442214  6129 solver.cpp:253]     Train net output #1: loss = 0.532124 (* 1 = 0.532124 loss)
    I1226 23:21:14.442224  6129 sgd_solver.cpp:106] Iteration 78000, lr = 0.000137005
    I1226 23:21:25.316436  6129 solver.cpp:237] Iteration 78100, loss = 0.610624
    I1226 23:21:25.316483  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:21:25.316494  6129 solver.cpp:253]     Train net output #1: loss = 0.610624 (* 1 = 0.610624 loss)
    I1226 23:21:25.316504  6129 sgd_solver.cpp:106] Iteration 78100, lr = 0.000136888
    I1226 23:21:36.159155  6129 solver.cpp:237] Iteration 78200, loss = 0.524488
    I1226 23:21:36.159292  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:21:36.159309  6129 solver.cpp:253]     Train net output #1: loss = 0.524488 (* 1 = 0.524488 loss)
    I1226 23:21:36.159317  6129 sgd_solver.cpp:106] Iteration 78200, lr = 0.000136772
    I1226 23:21:47.023983  6129 solver.cpp:237] Iteration 78300, loss = 0.574966
    I1226 23:21:47.024016  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:21:47.024029  6129 solver.cpp:253]     Train net output #1: loss = 0.574966 (* 1 = 0.574966 loss)
    I1226 23:21:47.024037  6129 sgd_solver.cpp:106] Iteration 78300, lr = 0.000136656
    I1226 23:21:57.890769  6129 solver.cpp:237] Iteration 78400, loss = 0.62397
    I1226 23:21:57.890810  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:21:57.890826  6129 solver.cpp:253]     Train net output #1: loss = 0.62397 (* 1 = 0.62397 loss)
    I1226 23:21:57.890837  6129 sgd_solver.cpp:106] Iteration 78400, lr = 0.00013654
    I1226 23:22:08.787101  6129 solver.cpp:237] Iteration 78500, loss = 0.530907
    I1226 23:22:08.787202  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:22:08.787220  6129 solver.cpp:253]     Train net output #1: loss = 0.530907 (* 1 = 0.530907 loss)
    I1226 23:22:08.787230  6129 sgd_solver.cpp:106] Iteration 78500, lr = 0.000136424
    I1226 23:22:19.646013  6129 solver.cpp:237] Iteration 78600, loss = 0.61873
    I1226 23:22:19.646056  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:22:19.646069  6129 solver.cpp:253]     Train net output #1: loss = 0.61873 (* 1 = 0.61873 loss)
    I1226 23:22:19.646077  6129 sgd_solver.cpp:106] Iteration 78600, lr = 0.000136308
    I1226 23:22:30.473356  6129 solver.cpp:237] Iteration 78700, loss = 0.516498
    I1226 23:22:30.473405  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:22:30.473426  6129 solver.cpp:253]     Train net output #1: loss = 0.516498 (* 1 = 0.516498 loss)
    I1226 23:22:30.473440  6129 sgd_solver.cpp:106] Iteration 78700, lr = 0.000136193
    I1226 23:22:41.357861  6129 solver.cpp:237] Iteration 78800, loss = 0.575128
    I1226 23:22:41.357978  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:22:41.357996  6129 solver.cpp:253]     Train net output #1: loss = 0.575128 (* 1 = 0.575128 loss)
    I1226 23:22:41.358006  6129 sgd_solver.cpp:106] Iteration 78800, lr = 0.000136078
    I1226 23:22:52.245501  6129 solver.cpp:237] Iteration 78900, loss = 0.624338
    I1226 23:22:52.245551  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:22:52.245573  6129 solver.cpp:253]     Train net output #1: loss = 0.624338 (* 1 = 0.624338 loss)
    I1226 23:22:52.245589  6129 sgd_solver.cpp:106] Iteration 78900, lr = 0.000135963
    I1226 23:23:03.040516  6129 solver.cpp:341] Iteration 79000, Testing net (#0)
    I1226 23:23:07.485419  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7065
    I1226 23:23:07.485458  6129 solver.cpp:409]     Test net output #1: loss = 0.839239 (* 1 = 0.839239 loss)
    I1226 23:23:07.532676  6129 solver.cpp:237] Iteration 79000, loss = 0.530063
    I1226 23:23:07.532723  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:23:07.532737  6129 solver.cpp:253]     Train net output #1: loss = 0.530063 (* 1 = 0.530063 loss)
    I1226 23:23:07.532747  6129 sgd_solver.cpp:106] Iteration 79000, lr = 0.000135849
    I1226 23:23:18.493291  6129 solver.cpp:237] Iteration 79100, loss = 0.610302
    I1226 23:23:18.493399  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:23:18.493415  6129 solver.cpp:253]     Train net output #1: loss = 0.610302 (* 1 = 0.610302 loss)
    I1226 23:23:18.493422  6129 sgd_solver.cpp:106] Iteration 79100, lr = 0.000135734
    I1226 23:23:29.337148  6129 solver.cpp:237] Iteration 79200, loss = 0.518544
    I1226 23:23:29.337193  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:23:29.337204  6129 solver.cpp:253]     Train net output #1: loss = 0.518544 (* 1 = 0.518544 loss)
    I1226 23:23:29.337213  6129 sgd_solver.cpp:106] Iteration 79200, lr = 0.00013562
    I1226 23:23:40.183467  6129 solver.cpp:237] Iteration 79300, loss = 0.574445
    I1226 23:23:40.183501  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:23:40.183512  6129 solver.cpp:253]     Train net output #1: loss = 0.574445 (* 1 = 0.574445 loss)
    I1226 23:23:40.183521  6129 sgd_solver.cpp:106] Iteration 79300, lr = 0.000135506
    I1226 23:23:51.058982  6129 solver.cpp:237] Iteration 79400, loss = 0.624702
    I1226 23:23:51.059093  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:23:51.059109  6129 solver.cpp:253]     Train net output #1: loss = 0.624702 (* 1 = 0.624702 loss)
    I1226 23:23:51.059116  6129 sgd_solver.cpp:106] Iteration 79400, lr = 0.000135393
    I1226 23:24:01.916219  6129 solver.cpp:237] Iteration 79500, loss = 0.529264
    I1226 23:24:01.916255  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:24:01.916266  6129 solver.cpp:253]     Train net output #1: loss = 0.529264 (* 1 = 0.529264 loss)
    I1226 23:24:01.916275  6129 sgd_solver.cpp:106] Iteration 79500, lr = 0.000135279
    I1226 23:24:12.800118  6129 solver.cpp:237] Iteration 79600, loss = 0.615381
    I1226 23:24:12.800160  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:24:12.800175  6129 solver.cpp:253]     Train net output #1: loss = 0.615381 (* 1 = 0.615381 loss)
    I1226 23:24:12.800186  6129 sgd_solver.cpp:106] Iteration 79600, lr = 0.000135166
    I1226 23:24:23.704905  6129 solver.cpp:237] Iteration 79700, loss = 0.5169
    I1226 23:24:23.705057  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:24:23.705072  6129 solver.cpp:253]     Train net output #1: loss = 0.5169 (* 1 = 0.5169 loss)
    I1226 23:24:23.705082  6129 sgd_solver.cpp:106] Iteration 79700, lr = 0.000135053
    I1226 23:24:34.526602  6129 solver.cpp:237] Iteration 79800, loss = 0.573817
    I1226 23:24:34.526654  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:24:34.526669  6129 solver.cpp:253]     Train net output #1: loss = 0.573817 (* 1 = 0.573817 loss)
    I1226 23:24:34.526680  6129 sgd_solver.cpp:106] Iteration 79800, lr = 0.00013494
    I1226 23:24:45.375910  6129 solver.cpp:237] Iteration 79900, loss = 0.624477
    I1226 23:24:45.375954  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:24:45.375967  6129 solver.cpp:253]     Train net output #1: loss = 0.624477 (* 1 = 0.624477 loss)
    I1226 23:24:45.375974  6129 sgd_solver.cpp:106] Iteration 79900, lr = 0.000134827
    I1226 23:24:56.157637  6129 solver.cpp:341] Iteration 80000, Testing net (#0)
    I1226 23:25:00.657729  6129 solver.cpp:409]     Test net output #0: accuracy = 0.71125
    I1226 23:25:00.657791  6129 solver.cpp:409]     Test net output #1: loss = 0.834311 (* 1 = 0.834311 loss)
    I1226 23:25:00.702584  6129 solver.cpp:237] Iteration 80000, loss = 0.528552
    I1226 23:25:00.702636  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:25:00.702656  6129 solver.cpp:253]     Train net output #1: loss = 0.528552 (* 1 = 0.528552 loss)
    I1226 23:25:00.702672  6129 sgd_solver.cpp:106] Iteration 80000, lr = 0.000134715
    I1226 23:25:11.665843  6129 solver.cpp:237] Iteration 80100, loss = 0.607682
    I1226 23:25:11.665879  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:25:11.665890  6129 solver.cpp:253]     Train net output #1: loss = 0.607682 (* 1 = 0.607682 loss)
    I1226 23:25:11.665899  6129 sgd_solver.cpp:106] Iteration 80100, lr = 0.000134603
    I1226 23:25:22.637492  6129 solver.cpp:237] Iteration 80200, loss = 0.515181
    I1226 23:25:22.637547  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:25:22.637567  6129 solver.cpp:253]     Train net output #1: loss = 0.515181 (* 1 = 0.515181 loss)
    I1226 23:25:22.637584  6129 sgd_solver.cpp:106] Iteration 80200, lr = 0.000134491
    I1226 23:25:33.460961  6129 solver.cpp:237] Iteration 80300, loss = 0.573734
    I1226 23:25:33.461148  6129 solver.cpp:253]     Train net output #0: accuracy = 0.8
    I1226 23:25:33.461164  6129 solver.cpp:253]     Train net output #1: loss = 0.573734 (* 1 = 0.573734 loss)
    I1226 23:25:33.461171  6129 sgd_solver.cpp:106] Iteration 80300, lr = 0.000134379
    I1226 23:25:44.279116  6129 solver.cpp:237] Iteration 80400, loss = 0.625096
    I1226 23:25:44.279153  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:25:44.279167  6129 solver.cpp:253]     Train net output #1: loss = 0.625096 (* 1 = 0.625096 loss)
    I1226 23:25:44.279177  6129 sgd_solver.cpp:106] Iteration 80400, lr = 0.000134268
    I1226 23:25:55.176075  6129 solver.cpp:237] Iteration 80500, loss = 0.527159
    I1226 23:25:55.176115  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:25:55.176129  6129 solver.cpp:253]     Train net output #1: loss = 0.527159 (* 1 = 0.527159 loss)
    I1226 23:25:55.176139  6129 sgd_solver.cpp:106] Iteration 80500, lr = 0.000134156
    I1226 23:26:06.048054  6129 solver.cpp:237] Iteration 80600, loss = 0.610665
    I1226 23:26:06.048163  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:26:06.048179  6129 solver.cpp:253]     Train net output #1: loss = 0.610665 (* 1 = 0.610665 loss)
    I1226 23:26:06.048188  6129 sgd_solver.cpp:106] Iteration 80600, lr = 0.000134045
    I1226 23:26:17.009951  6129 solver.cpp:237] Iteration 80700, loss = 0.511906
    I1226 23:26:17.009986  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:26:17.009999  6129 solver.cpp:253]     Train net output #1: loss = 0.511906 (* 1 = 0.511906 loss)
    I1226 23:26:17.010007  6129 sgd_solver.cpp:106] Iteration 80700, lr = 0.000133935
    I1226 23:26:27.876049  6129 solver.cpp:237] Iteration 80800, loss = 0.573298
    I1226 23:26:27.876085  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:26:27.876097  6129 solver.cpp:253]     Train net output #1: loss = 0.573298 (* 1 = 0.573298 loss)
    I1226 23:26:27.876106  6129 sgd_solver.cpp:106] Iteration 80800, lr = 0.000133824
    I1226 23:26:38.756798  6129 solver.cpp:237] Iteration 80900, loss = 0.62505
    I1226 23:26:38.756950  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:26:38.756965  6129 solver.cpp:253]     Train net output #1: loss = 0.62505 (* 1 = 0.62505 loss)
    I1226 23:26:38.756973  6129 sgd_solver.cpp:106] Iteration 80900, lr = 0.000133713
    I1226 23:26:49.506273  6129 solver.cpp:341] Iteration 81000, Testing net (#0)
    I1226 23:26:54.033546  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7075
    I1226 23:26:54.033588  6129 solver.cpp:409]     Test net output #1: loss = 0.8387 (* 1 = 0.8387 loss)
    I1226 23:26:54.080837  6129 solver.cpp:237] Iteration 81000, loss = 0.526206
    I1226 23:26:54.080885  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:26:54.080899  6129 solver.cpp:253]     Train net output #1: loss = 0.526206 (* 1 = 0.526206 loss)
    I1226 23:26:54.080909  6129 sgd_solver.cpp:106] Iteration 81000, lr = 0.000133603
    I1226 23:27:05.018327  6129 solver.cpp:237] Iteration 81100, loss = 0.61034
    I1226 23:27:05.018368  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:27:05.018383  6129 solver.cpp:253]     Train net output #1: loss = 0.61034 (* 1 = 0.61034 loss)
    I1226 23:27:05.018394  6129 sgd_solver.cpp:106] Iteration 81100, lr = 0.000133493
    I1226 23:27:15.865911  6129 solver.cpp:237] Iteration 81200, loss = 0.514114
    I1226 23:27:15.866025  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:27:15.866044  6129 solver.cpp:253]     Train net output #1: loss = 0.514114 (* 1 = 0.514114 loss)
    I1226 23:27:15.866055  6129 sgd_solver.cpp:106] Iteration 81200, lr = 0.000133383
    I1226 23:27:26.741257  6129 solver.cpp:237] Iteration 81300, loss = 0.57332
    I1226 23:27:26.741292  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:27:26.741304  6129 solver.cpp:253]     Train net output #1: loss = 0.57332 (* 1 = 0.57332 loss)
    I1226 23:27:26.741313  6129 sgd_solver.cpp:106] Iteration 81300, lr = 0.000133274
    I1226 23:27:37.568720  6129 solver.cpp:237] Iteration 81400, loss = 0.624855
    I1226 23:27:37.568779  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:27:37.568801  6129 solver.cpp:253]     Train net output #1: loss = 0.624855 (* 1 = 0.624855 loss)
    I1226 23:27:37.568817  6129 sgd_solver.cpp:106] Iteration 81400, lr = 0.000133164
    I1226 23:27:48.399705  6129 solver.cpp:237] Iteration 81500, loss = 0.525642
    I1226 23:27:48.399834  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:27:48.399850  6129 solver.cpp:253]     Train net output #1: loss = 0.525642 (* 1 = 0.525642 loss)
    I1226 23:27:48.399857  6129 sgd_solver.cpp:106] Iteration 81500, lr = 0.000133055
    I1226 23:27:59.244060  6129 solver.cpp:237] Iteration 81600, loss = 0.605604
    I1226 23:27:59.244105  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:27:59.244117  6129 solver.cpp:253]     Train net output #1: loss = 0.605604 (* 1 = 0.605604 loss)
    I1226 23:27:59.244124  6129 sgd_solver.cpp:106] Iteration 81600, lr = 0.000132946
    I1226 23:28:10.120610  6129 solver.cpp:237] Iteration 81700, loss = 0.509226
    I1226 23:28:10.120652  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:28:10.120667  6129 solver.cpp:253]     Train net output #1: loss = 0.509226 (* 1 = 0.509226 loss)
    I1226 23:28:10.120678  6129 sgd_solver.cpp:106] Iteration 81700, lr = 0.000132838
    I1226 23:28:21.001503  6129 solver.cpp:237] Iteration 81800, loss = 0.572176
    I1226 23:28:21.001653  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:28:21.001668  6129 solver.cpp:253]     Train net output #1: loss = 0.572176 (* 1 = 0.572176 loss)
    I1226 23:28:21.001678  6129 sgd_solver.cpp:106] Iteration 81800, lr = 0.000132729
    I1226 23:28:31.918633  6129 solver.cpp:237] Iteration 81900, loss = 0.625213
    I1226 23:28:31.918675  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:28:31.918689  6129 solver.cpp:253]     Train net output #1: loss = 0.625213 (* 1 = 0.625213 loss)
    I1226 23:28:31.918700  6129 sgd_solver.cpp:106] Iteration 81900, lr = 0.000132621
    I1226 23:28:42.649871  6129 solver.cpp:341] Iteration 82000, Testing net (#0)
    I1226 23:28:47.073364  6129 solver.cpp:409]     Test net output #0: accuracy = 0.711
    I1226 23:28:47.073415  6129 solver.cpp:409]     Test net output #1: loss = 0.833539 (* 1 = 0.833539 loss)
    I1226 23:28:47.140435  6129 solver.cpp:237] Iteration 82000, loss = 0.524702
    I1226 23:28:47.140475  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:28:47.140487  6129 solver.cpp:253]     Train net output #1: loss = 0.524702 (* 1 = 0.524702 loss)
    I1226 23:28:47.140497  6129 sgd_solver.cpp:106] Iteration 82000, lr = 0.000132513
    I1226 23:28:58.046785  6129 solver.cpp:237] Iteration 82100, loss = 0.609171
    I1226 23:28:58.046912  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:28:58.046949  6129 solver.cpp:253]     Train net output #1: loss = 0.609171 (* 1 = 0.609171 loss)
    I1226 23:28:58.046959  6129 sgd_solver.cpp:106] Iteration 82100, lr = 0.000132405
    I1226 23:29:08.950075  6129 solver.cpp:237] Iteration 82200, loss = 0.509371
    I1226 23:29:08.950132  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:29:08.950150  6129 solver.cpp:253]     Train net output #1: loss = 0.509371 (* 1 = 0.509371 loss)
    I1226 23:29:08.950161  6129 sgd_solver.cpp:106] Iteration 82200, lr = 0.000132297
    I1226 23:29:19.814690  6129 solver.cpp:237] Iteration 82300, loss = 0.572845
    I1226 23:29:19.814734  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:29:19.814745  6129 solver.cpp:253]     Train net output #1: loss = 0.572845 (* 1 = 0.572845 loss)
    I1226 23:29:19.814754  6129 sgd_solver.cpp:106] Iteration 82300, lr = 0.000132189
    I1226 23:29:30.644194  6129 solver.cpp:237] Iteration 82400, loss = 0.625347
    I1226 23:29:30.644351  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:29:30.644369  6129 solver.cpp:253]     Train net output #1: loss = 0.625347 (* 1 = 0.625347 loss)
    I1226 23:29:30.644381  6129 sgd_solver.cpp:106] Iteration 82400, lr = 0.000132082
    I1226 23:29:41.503782  6129 solver.cpp:237] Iteration 82500, loss = 0.524083
    I1226 23:29:41.503823  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:29:41.503839  6129 solver.cpp:253]     Train net output #1: loss = 0.524083 (* 1 = 0.524083 loss)
    I1226 23:29:41.503849  6129 sgd_solver.cpp:106] Iteration 82500, lr = 0.000131975
    I1226 23:29:52.369132  6129 solver.cpp:237] Iteration 82600, loss = 0.606674
    I1226 23:29:52.369185  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:29:52.369206  6129 solver.cpp:253]     Train net output #1: loss = 0.606674 (* 1 = 0.606674 loss)
    I1226 23:29:52.369222  6129 sgd_solver.cpp:106] Iteration 82600, lr = 0.000131868
    I1226 23:30:03.232998  6129 solver.cpp:237] Iteration 82700, loss = 0.510978
    I1226 23:30:03.233100  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:30:03.233117  6129 solver.cpp:253]     Train net output #1: loss = 0.510978 (* 1 = 0.510978 loss)
    I1226 23:30:03.233125  6129 sgd_solver.cpp:106] Iteration 82700, lr = 0.000131761
    I1226 23:30:14.091420  6129 solver.cpp:237] Iteration 82800, loss = 0.571938
    I1226 23:30:14.091464  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:30:14.091475  6129 solver.cpp:253]     Train net output #1: loss = 0.571938 (* 1 = 0.571938 loss)
    I1226 23:30:14.091483  6129 sgd_solver.cpp:106] Iteration 82800, lr = 0.000131655
    I1226 23:30:24.913552  6129 solver.cpp:237] Iteration 82900, loss = 0.624571
    I1226 23:30:24.913589  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:30:24.913599  6129 solver.cpp:253]     Train net output #1: loss = 0.624571 (* 1 = 0.624571 loss)
    I1226 23:30:24.913609  6129 sgd_solver.cpp:106] Iteration 82900, lr = 0.000131549
    I1226 23:30:35.675204  6129 solver.cpp:341] Iteration 83000, Testing net (#0)
    I1226 23:30:40.143447  6129 solver.cpp:409]     Test net output #0: accuracy = 0.709917
    I1226 23:30:40.143507  6129 solver.cpp:409]     Test net output #1: loss = 0.832695 (* 1 = 0.832695 loss)
    I1226 23:30:40.188282  6129 solver.cpp:237] Iteration 83000, loss = 0.523521
    I1226 23:30:40.188335  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:30:40.188356  6129 solver.cpp:253]     Train net output #1: loss = 0.523521 (* 1 = 0.523521 loss)
    I1226 23:30:40.188374  6129 sgd_solver.cpp:106] Iteration 83000, lr = 0.000131443
    I1226 23:30:51.180958  6129 solver.cpp:237] Iteration 83100, loss = 0.602737
    I1226 23:30:51.180996  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:30:51.181010  6129 solver.cpp:253]     Train net output #1: loss = 0.602737 (* 1 = 0.602737 loss)
    I1226 23:30:51.181020  6129 sgd_solver.cpp:106] Iteration 83100, lr = 0.000131337
    I1226 23:31:02.037418  6129 solver.cpp:237] Iteration 83200, loss = 0.507229
    I1226 23:31:02.037454  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:31:02.037466  6129 solver.cpp:253]     Train net output #1: loss = 0.507229 (* 1 = 0.507229 loss)
    I1226 23:31:02.037475  6129 sgd_solver.cpp:106] Iteration 83200, lr = 0.000131231
    I1226 23:31:12.913964  6129 solver.cpp:237] Iteration 83300, loss = 0.571073
    I1226 23:31:12.914088  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:31:12.914106  6129 solver.cpp:253]     Train net output #1: loss = 0.571073 (* 1 = 0.571073 loss)
    I1226 23:31:12.914118  6129 sgd_solver.cpp:106] Iteration 83300, lr = 0.000131125
    I1226 23:31:23.770961  6129 solver.cpp:237] Iteration 83400, loss = 0.624418
    I1226 23:31:23.770993  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:31:23.771005  6129 solver.cpp:253]     Train net output #1: loss = 0.624418 (* 1 = 0.624418 loss)
    I1226 23:31:23.771013  6129 sgd_solver.cpp:106] Iteration 83400, lr = 0.00013102
    I1226 23:31:35.038089  6129 solver.cpp:237] Iteration 83500, loss = 0.522592
    I1226 23:31:35.038127  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:31:35.038141  6129 solver.cpp:253]     Train net output #1: loss = 0.522592 (* 1 = 0.522592 loss)
    I1226 23:31:35.038153  6129 sgd_solver.cpp:106] Iteration 83500, lr = 0.000130915
    I1226 23:31:45.875123  6129 solver.cpp:237] Iteration 83600, loss = 0.602639
    I1226 23:31:45.875248  6129 solver.cpp:253]     Train net output #0: accuracy = 0.81
    I1226 23:31:45.875267  6129 solver.cpp:253]     Train net output #1: loss = 0.602639 (* 1 = 0.602639 loss)
    I1226 23:31:45.875278  6129 sgd_solver.cpp:106] Iteration 83600, lr = 0.00013081
    I1226 23:31:56.792776  6129 solver.cpp:237] Iteration 83700, loss = 0.507359
    I1226 23:31:56.792815  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:31:56.792830  6129 solver.cpp:253]     Train net output #1: loss = 0.507359 (* 1 = 0.507359 loss)
    I1226 23:31:56.792840  6129 sgd_solver.cpp:106] Iteration 83700, lr = 0.000130705
    I1226 23:32:07.700599  6129 solver.cpp:237] Iteration 83800, loss = 0.571455
    I1226 23:32:07.700639  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:32:07.700654  6129 solver.cpp:253]     Train net output #1: loss = 0.571455 (* 1 = 0.571455 loss)
    I1226 23:32:07.700664  6129 sgd_solver.cpp:106] Iteration 83800, lr = 0.000130601
    I1226 23:32:18.563066  6129 solver.cpp:237] Iteration 83900, loss = 0.624085
    I1226 23:32:18.563155  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:32:18.563174  6129 solver.cpp:253]     Train net output #1: loss = 0.624085 (* 1 = 0.624085 loss)
    I1226 23:32:18.563185  6129 sgd_solver.cpp:106] Iteration 83900, lr = 0.000130496
    I1226 23:32:29.296743  6129 solver.cpp:341] Iteration 84000, Testing net (#0)
    I1226 23:32:33.736693  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7085
    I1226 23:32:33.736740  6129 solver.cpp:409]     Test net output #1: loss = 0.837508 (* 1 = 0.837508 loss)
    I1226 23:32:33.781122  6129 solver.cpp:237] Iteration 84000, loss = 0.521947
    I1226 23:32:33.781167  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:32:33.781178  6129 solver.cpp:253]     Train net output #1: loss = 0.521947 (* 1 = 0.521947 loss)
    I1226 23:32:33.781189  6129 sgd_solver.cpp:106] Iteration 84000, lr = 0.000130392
    I1226 23:32:44.673501  6129 solver.cpp:237] Iteration 84100, loss = 0.60553
    I1226 23:32:44.673537  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:32:44.673552  6129 solver.cpp:253]     Train net output #1: loss = 0.60553 (* 1 = 0.60553 loss)
    I1226 23:32:44.673562  6129 sgd_solver.cpp:106] Iteration 84100, lr = 0.000130288
    I1226 23:32:55.514642  6129 solver.cpp:237] Iteration 84200, loss = 0.506461
    I1226 23:32:55.514776  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:32:55.514799  6129 solver.cpp:253]     Train net output #1: loss = 0.506461 (* 1 = 0.506461 loss)
    I1226 23:32:55.514806  6129 sgd_solver.cpp:106] Iteration 84200, lr = 0.000130185
    I1226 23:33:06.607877  6129 solver.cpp:237] Iteration 84300, loss = 0.570749
    I1226 23:33:06.607924  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:33:06.607936  6129 solver.cpp:253]     Train net output #1: loss = 0.570749 (* 1 = 0.570749 loss)
    I1226 23:33:06.607945  6129 sgd_solver.cpp:106] Iteration 84300, lr = 0.000130081
    I1226 23:33:17.576508  6129 solver.cpp:237] Iteration 84400, loss = 0.623721
    I1226 23:33:17.576545  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:33:17.576557  6129 solver.cpp:253]     Train net output #1: loss = 0.623721 (* 1 = 0.623721 loss)
    I1226 23:33:17.576566  6129 sgd_solver.cpp:106] Iteration 84400, lr = 0.000129978
    I1226 23:33:28.499616  6129 solver.cpp:237] Iteration 84500, loss = 0.520937
    I1226 23:33:28.499771  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:33:28.499799  6129 solver.cpp:253]     Train net output #1: loss = 0.520937 (* 1 = 0.520937 loss)
    I1226 23:33:28.499809  6129 sgd_solver.cpp:106] Iteration 84500, lr = 0.000129875
    I1226 23:33:39.497894  6129 solver.cpp:237] Iteration 84600, loss = 0.619577
    I1226 23:33:39.497948  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:33:39.497970  6129 solver.cpp:253]     Train net output #1: loss = 0.619577 (* 1 = 0.619577 loss)
    I1226 23:33:39.497985  6129 sgd_solver.cpp:106] Iteration 84600, lr = 0.000129772
    I1226 23:33:50.446152  6129 solver.cpp:237] Iteration 84700, loss = 0.504563
    I1226 23:33:50.446198  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:33:50.446213  6129 solver.cpp:253]     Train net output #1: loss = 0.504563 (* 1 = 0.504563 loss)
    I1226 23:33:50.446226  6129 sgd_solver.cpp:106] Iteration 84700, lr = 0.000129669
    I1226 23:34:01.507375  6129 solver.cpp:237] Iteration 84800, loss = 0.569882
    I1226 23:34:01.507547  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:34:01.507562  6129 solver.cpp:253]     Train net output #1: loss = 0.569882 (* 1 = 0.569882 loss)
    I1226 23:34:01.507571  6129 sgd_solver.cpp:106] Iteration 84800, lr = 0.000129566
    I1226 23:34:12.479318  6129 solver.cpp:237] Iteration 84900, loss = 0.623298
    I1226 23:34:12.479354  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:34:12.479367  6129 solver.cpp:253]     Train net output #1: loss = 0.623298 (* 1 = 0.623298 loss)
    I1226 23:34:12.479375  6129 sgd_solver.cpp:106] Iteration 84900, lr = 0.000129464
    I1226 23:34:23.341311  6129 solver.cpp:341] Iteration 85000, Testing net (#0)
    I1226 23:34:27.786644  6129 solver.cpp:409]     Test net output #0: accuracy = 0.713
    I1226 23:34:27.786701  6129 solver.cpp:409]     Test net output #1: loss = 0.832651 (* 1 = 0.832651 loss)
    I1226 23:34:27.831279  6129 solver.cpp:237] Iteration 85000, loss = 0.520399
    I1226 23:34:27.831331  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:34:27.831348  6129 solver.cpp:253]     Train net output #1: loss = 0.520399 (* 1 = 0.520399 loss)
    I1226 23:34:27.831362  6129 sgd_solver.cpp:106] Iteration 85000, lr = 0.000129362
    I1226 23:34:38.688395  6129 solver.cpp:237] Iteration 85100, loss = 0.599503
    I1226 23:34:38.688555  6129 solver.cpp:253]     Train net output #0: accuracy = 0.82
    I1226 23:34:38.688572  6129 solver.cpp:253]     Train net output #1: loss = 0.599503 (* 1 = 0.599503 loss)
    I1226 23:34:38.688583  6129 sgd_solver.cpp:106] Iteration 85100, lr = 0.00012926
    I1226 23:34:49.553199  6129 solver.cpp:237] Iteration 85200, loss = 0.504491
    I1226 23:34:49.553248  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:34:49.553267  6129 solver.cpp:253]     Train net output #1: loss = 0.504491 (* 1 = 0.504491 loss)
    I1226 23:34:49.553282  6129 sgd_solver.cpp:106] Iteration 85200, lr = 0.000129158
    I1226 23:35:00.446099  6129 solver.cpp:237] Iteration 85300, loss = 0.570042
    I1226 23:35:00.446153  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:35:00.446171  6129 solver.cpp:253]     Train net output #1: loss = 0.570042 (* 1 = 0.570042 loss)
    I1226 23:35:00.446185  6129 sgd_solver.cpp:106] Iteration 85300, lr = 0.000129056
    I1226 23:35:11.301254  6129 solver.cpp:237] Iteration 85400, loss = 0.62361
    I1226 23:35:11.301424  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:35:11.301437  6129 solver.cpp:253]     Train net output #1: loss = 0.62361 (* 1 = 0.62361 loss)
    I1226 23:35:11.301446  6129 sgd_solver.cpp:106] Iteration 85400, lr = 0.000128955
    I1226 23:35:22.214977  6129 solver.cpp:237] Iteration 85500, loss = 0.51978
    I1226 23:35:22.215016  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:35:22.215030  6129 solver.cpp:253]     Train net output #1: loss = 0.51978 (* 1 = 0.51978 loss)
    I1226 23:35:22.215041  6129 sgd_solver.cpp:106] Iteration 85500, lr = 0.000128853
    I1226 23:35:33.714619  6129 solver.cpp:237] Iteration 85600, loss = 0.608071
    I1226 23:35:33.714658  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:35:33.714671  6129 solver.cpp:253]     Train net output #1: loss = 0.608071 (* 1 = 0.608071 loss)
    I1226 23:35:33.714682  6129 sgd_solver.cpp:106] Iteration 85600, lr = 0.000128752
    I1226 23:35:45.048390  6129 solver.cpp:237] Iteration 85700, loss = 0.503704
    I1226 23:35:45.048555  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:35:45.048569  6129 solver.cpp:253]     Train net output #1: loss = 0.503704 (* 1 = 0.503704 loss)
    I1226 23:35:45.048576  6129 sgd_solver.cpp:106] Iteration 85700, lr = 0.000128651
    I1226 23:35:56.658056  6129 solver.cpp:237] Iteration 85800, loss = 0.569018
    I1226 23:35:56.658094  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:35:56.658107  6129 solver.cpp:253]     Train net output #1: loss = 0.569018 (* 1 = 0.569018 loss)
    I1226 23:35:56.658115  6129 sgd_solver.cpp:106] Iteration 85800, lr = 0.000128551
    I1226 23:36:07.716096  6129 solver.cpp:237] Iteration 85900, loss = 0.62308
    I1226 23:36:07.716133  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:36:07.716145  6129 solver.cpp:253]     Train net output #1: loss = 0.62308 (* 1 = 0.62308 loss)
    I1226 23:36:07.716155  6129 sgd_solver.cpp:106] Iteration 85900, lr = 0.00012845
    I1226 23:36:18.314205  6129 solver.cpp:341] Iteration 86000, Testing net (#0)
    I1226 23:36:22.579524  6129 solver.cpp:409]     Test net output #0: accuracy = 0.709
    I1226 23:36:22.579572  6129 solver.cpp:409]     Test net output #1: loss = 0.837343 (* 1 = 0.837343 loss)
    I1226 23:36:22.624115  6129 solver.cpp:237] Iteration 86000, loss = 0.518656
    I1226 23:36:22.624164  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:36:22.624177  6129 solver.cpp:253]     Train net output #1: loss = 0.518656 (* 1 = 0.518656 loss)
    I1226 23:36:22.624187  6129 sgd_solver.cpp:106] Iteration 86000, lr = 0.00012835
    I1226 23:36:33.135947  6129 solver.cpp:237] Iteration 86100, loss = 0.603541
    I1226 23:36:33.135989  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:36:33.136004  6129 solver.cpp:253]     Train net output #1: loss = 0.603541 (* 1 = 0.603541 loss)
    I1226 23:36:33.136014  6129 sgd_solver.cpp:106] Iteration 86100, lr = 0.000128249
    I1226 23:36:43.620081  6129 solver.cpp:237] Iteration 86200, loss = 0.50282
    I1226 23:36:43.620127  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:36:43.620138  6129 solver.cpp:253]     Train net output #1: loss = 0.50282 (* 1 = 0.50282 loss)
    I1226 23:36:43.620148  6129 sgd_solver.cpp:106] Iteration 86200, lr = 0.000128149
    I1226 23:36:54.107199  6129 solver.cpp:237] Iteration 86300, loss = 0.568499
    I1226 23:36:54.107314  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:36:54.107332  6129 solver.cpp:253]     Train net output #1: loss = 0.568499 (* 1 = 0.568499 loss)
    I1226 23:36:54.107342  6129 sgd_solver.cpp:106] Iteration 86300, lr = 0.00012805
    I1226 23:37:04.633033  6129 solver.cpp:237] Iteration 86400, loss = 0.622806
    I1226 23:37:04.633069  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:37:04.633081  6129 solver.cpp:253]     Train net output #1: loss = 0.622806 (* 1 = 0.622806 loss)
    I1226 23:37:04.633090  6129 sgd_solver.cpp:106] Iteration 86400, lr = 0.00012795
    I1226 23:37:15.110724  6129 solver.cpp:237] Iteration 86500, loss = 0.518062
    I1226 23:37:15.110760  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:37:15.110771  6129 solver.cpp:253]     Train net output #1: loss = 0.518062 (* 1 = 0.518062 loss)
    I1226 23:37:15.110780  6129 sgd_solver.cpp:106] Iteration 86500, lr = 0.000127851
    I1226 23:37:25.671149  6129 solver.cpp:237] Iteration 86600, loss = 0.602933
    I1226 23:37:25.671278  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:37:25.671301  6129 solver.cpp:253]     Train net output #1: loss = 0.602933 (* 1 = 0.602933 loss)
    I1226 23:37:25.671311  6129 sgd_solver.cpp:106] Iteration 86600, lr = 0.000127751
    I1226 23:37:36.176343  6129 solver.cpp:237] Iteration 86700, loss = 0.501939
    I1226 23:37:36.176383  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:37:36.176398  6129 solver.cpp:253]     Train net output #1: loss = 0.501939 (* 1 = 0.501939 loss)
    I1226 23:37:36.176408  6129 sgd_solver.cpp:106] Iteration 86700, lr = 0.000127652
    I1226 23:37:46.643391  6129 solver.cpp:237] Iteration 86800, loss = 0.567754
    I1226 23:37:46.643426  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:37:46.643438  6129 solver.cpp:253]     Train net output #1: loss = 0.567754 (* 1 = 0.567754 loss)
    I1226 23:37:46.643450  6129 sgd_solver.cpp:106] Iteration 86800, lr = 0.000127553
    I1226 23:37:57.142150  6129 solver.cpp:237] Iteration 86900, loss = 0.622328
    I1226 23:37:57.142343  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:37:57.142361  6129 solver.cpp:253]     Train net output #1: loss = 0.622328 (* 1 = 0.622328 loss)
    I1226 23:37:57.142372  6129 sgd_solver.cpp:106] Iteration 86900, lr = 0.000127455
    I1226 23:38:07.528934  6129 solver.cpp:341] Iteration 87000, Testing net (#0)
    I1226 23:38:11.848176  6129 solver.cpp:409]     Test net output #0: accuracy = 0.711667
    I1226 23:38:11.848215  6129 solver.cpp:409]     Test net output #1: loss = 0.831656 (* 1 = 0.831656 loss)
    I1226 23:38:11.892726  6129 solver.cpp:237] Iteration 87000, loss = 0.517206
    I1226 23:38:11.892768  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:38:11.892781  6129 solver.cpp:253]     Train net output #1: loss = 0.517206 (* 1 = 0.517206 loss)
    I1226 23:38:11.892792  6129 sgd_solver.cpp:106] Iteration 87000, lr = 0.000127356
    I1226 23:38:22.409641  6129 solver.cpp:237] Iteration 87100, loss = 0.606797
    I1226 23:38:22.409694  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:38:22.409716  6129 solver.cpp:253]     Train net output #1: loss = 0.606797 (* 1 = 0.606797 loss)
    I1226 23:38:22.409732  6129 sgd_solver.cpp:106] Iteration 87100, lr = 0.000127258
    I1226 23:38:32.905390  6129 solver.cpp:237] Iteration 87200, loss = 0.502094
    I1226 23:38:32.905534  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:38:32.905547  6129 solver.cpp:253]     Train net output #1: loss = 0.502094 (* 1 = 0.502094 loss)
    I1226 23:38:32.905555  6129 sgd_solver.cpp:106] Iteration 87200, lr = 0.000127159
    I1226 23:38:43.920627  6129 solver.cpp:237] Iteration 87300, loss = 0.567273
    I1226 23:38:43.920665  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:38:43.920678  6129 solver.cpp:253]     Train net output #1: loss = 0.567273 (* 1 = 0.567273 loss)
    I1226 23:38:43.920687  6129 sgd_solver.cpp:106] Iteration 87300, lr = 0.000127061
    I1226 23:38:55.245280  6129 solver.cpp:237] Iteration 87400, loss = 0.621971
    I1226 23:38:55.245333  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:38:55.245355  6129 solver.cpp:253]     Train net output #1: loss = 0.621971 (* 1 = 0.621971 loss)
    I1226 23:38:55.245370  6129 sgd_solver.cpp:106] Iteration 87400, lr = 0.000126963
    I1226 23:39:05.739611  6129 solver.cpp:237] Iteration 87500, loss = 0.516766
    I1226 23:39:05.739750  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:39:05.739773  6129 solver.cpp:253]     Train net output #1: loss = 0.516766 (* 1 = 0.516766 loss)
    I1226 23:39:05.739779  6129 sgd_solver.cpp:106] Iteration 87500, lr = 0.000126866
    I1226 23:39:17.096132  6129 solver.cpp:237] Iteration 87600, loss = 0.601139
    I1226 23:39:17.096192  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:39:17.096213  6129 solver.cpp:253]     Train net output #1: loss = 0.601139 (* 1 = 0.601139 loss)
    I1226 23:39:17.096231  6129 sgd_solver.cpp:106] Iteration 87600, lr = 0.000126768
    I1226 23:39:29.620112  6129 solver.cpp:237] Iteration 87700, loss = 0.502341
    I1226 23:39:29.620168  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:39:29.620185  6129 solver.cpp:253]     Train net output #1: loss = 0.502341 (* 1 = 0.502341 loss)
    I1226 23:39:29.620199  6129 sgd_solver.cpp:106] Iteration 87700, lr = 0.000126671
    I1226 23:39:40.670459  6129 solver.cpp:237] Iteration 87800, loss = 0.566422
    I1226 23:39:40.670567  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:39:40.670583  6129 solver.cpp:253]     Train net output #1: loss = 0.566422 (* 1 = 0.566422 loss)
    I1226 23:39:40.670589  6129 sgd_solver.cpp:106] Iteration 87800, lr = 0.000126574
    I1226 23:39:52.583626  6129 solver.cpp:237] Iteration 87900, loss = 0.621547
    I1226 23:39:52.583680  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:39:52.583703  6129 solver.cpp:253]     Train net output #1: loss = 0.621547 (* 1 = 0.621547 loss)
    I1226 23:39:52.583719  6129 sgd_solver.cpp:106] Iteration 87900, lr = 0.000126477
    I1226 23:40:06.467890  6129 solver.cpp:341] Iteration 88000, Testing net (#0)
    I1226 23:40:11.662636  6129 solver.cpp:409]     Test net output #0: accuracy = 0.710417
    I1226 23:40:11.662827  6129 solver.cpp:409]     Test net output #1: loss = 0.831225 (* 1 = 0.831225 loss)
    I1226 23:40:11.749904  6129 solver.cpp:237] Iteration 88000, loss = 0.51598
    I1226 23:40:11.749956  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:40:11.749975  6129 solver.cpp:253]     Train net output #1: loss = 0.51598 (* 1 = 0.51598 loss)
    I1226 23:40:11.749992  6129 sgd_solver.cpp:106] Iteration 88000, lr = 0.00012638
    I1226 23:40:25.305809  6129 solver.cpp:237] Iteration 88100, loss = 0.595473
    I1226 23:40:25.305881  6129 solver.cpp:253]     Train net output #0: accuracy = 0.83
    I1226 23:40:25.305913  6129 solver.cpp:253]     Train net output #1: loss = 0.595473 (* 1 = 0.595473 loss)
    I1226 23:40:25.305938  6129 sgd_solver.cpp:106] Iteration 88100, lr = 0.000126283
    I1226 23:40:38.623975  6129 solver.cpp:237] Iteration 88200, loss = 0.501008
    I1226 23:40:38.624027  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:40:38.624040  6129 solver.cpp:253]     Train net output #1: loss = 0.501008 (* 1 = 0.501008 loss)
    I1226 23:40:38.624052  6129 sgd_solver.cpp:106] Iteration 88200, lr = 0.000126187
    I1226 23:40:50.604053  6129 solver.cpp:237] Iteration 88300, loss = 0.566294
    I1226 23:40:50.604199  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:40:50.604218  6129 solver.cpp:253]     Train net output #1: loss = 0.566294 (* 1 = 0.566294 loss)
    I1226 23:40:50.604229  6129 sgd_solver.cpp:106] Iteration 88300, lr = 0.000126091
    I1226 23:41:02.634146  6129 solver.cpp:237] Iteration 88400, loss = 0.621342
    I1226 23:41:02.634204  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:41:02.634227  6129 solver.cpp:253]     Train net output #1: loss = 0.621342 (* 1 = 0.621342 loss)
    I1226 23:41:02.634243  6129 sgd_solver.cpp:106] Iteration 88400, lr = 0.000125995
    I1226 23:41:13.965095  6129 solver.cpp:237] Iteration 88500, loss = 0.515243
    I1226 23:41:13.965134  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:41:13.965147  6129 solver.cpp:253]     Train net output #1: loss = 0.515243 (* 1 = 0.515243 loss)
    I1226 23:41:13.965157  6129 sgd_solver.cpp:106] Iteration 88500, lr = 0.000125899
    I1226 23:41:26.662103  6129 solver.cpp:237] Iteration 88600, loss = 0.599782
    I1226 23:41:26.662264  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:41:26.662277  6129 solver.cpp:253]     Train net output #1: loss = 0.599782 (* 1 = 0.599782 loss)
    I1226 23:41:26.662284  6129 sgd_solver.cpp:106] Iteration 88600, lr = 0.000125803
    I1226 23:41:37.641446  6129 solver.cpp:237] Iteration 88700, loss = 0.503516
    I1226 23:41:37.641489  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:41:37.641505  6129 solver.cpp:253]     Train net output #1: loss = 0.503516 (* 1 = 0.503516 loss)
    I1226 23:41:37.641516  6129 sgd_solver.cpp:106] Iteration 88700, lr = 0.000125707
    I1226 23:41:48.305907  6129 solver.cpp:237] Iteration 88800, loss = 0.565517
    I1226 23:41:48.305948  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:41:48.305963  6129 solver.cpp:253]     Train net output #1: loss = 0.565517 (* 1 = 0.565517 loss)
    I1226 23:41:48.305974  6129 sgd_solver.cpp:106] Iteration 88800, lr = 0.000125612
    I1226 23:41:59.916714  6129 solver.cpp:237] Iteration 88900, loss = 0.620076
    I1226 23:41:59.916820  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:41:59.916847  6129 solver.cpp:253]     Train net output #1: loss = 0.620076 (* 1 = 0.620076 loss)
    I1226 23:41:59.916853  6129 sgd_solver.cpp:106] Iteration 88900, lr = 0.000125516
    I1226 23:42:10.754709  6129 solver.cpp:341] Iteration 89000, Testing net (#0)
    I1226 23:42:15.440703  6129 solver.cpp:409]     Test net output #0: accuracy = 0.7085
    I1226 23:42:15.440743  6129 solver.cpp:409]     Test net output #1: loss = 0.836087 (* 1 = 0.836087 loss)
    I1226 23:42:15.488916  6129 solver.cpp:237] Iteration 89000, loss = 0.514698
    I1226 23:42:15.488956  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:42:15.488968  6129 solver.cpp:253]     Train net output #1: loss = 0.514698 (* 1 = 0.514698 loss)
    I1226 23:42:15.488979  6129 sgd_solver.cpp:106] Iteration 89000, lr = 0.000125421
    I1226 23:42:26.752146  6129 solver.cpp:237] Iteration 89100, loss = 0.606597
    I1226 23:42:26.752184  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:42:26.752197  6129 solver.cpp:253]     Train net output #1: loss = 0.606597 (* 1 = 0.606597 loss)
    I1226 23:42:26.752208  6129 sgd_solver.cpp:106] Iteration 89100, lr = 0.000125326
    I1226 23:42:39.270252  6129 solver.cpp:237] Iteration 89200, loss = 0.499701
    I1226 23:42:39.270426  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:42:39.270453  6129 solver.cpp:253]     Train net output #1: loss = 0.499701 (* 1 = 0.499701 loss)
    I1226 23:42:39.270472  6129 sgd_solver.cpp:106] Iteration 89200, lr = 0.000125232
    I1226 23:42:50.041127  6129 solver.cpp:237] Iteration 89300, loss = 0.565304
    I1226 23:42:50.041160  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:42:50.041172  6129 solver.cpp:253]     Train net output #1: loss = 0.565304 (* 1 = 0.565304 loss)
    I1226 23:42:50.041182  6129 sgd_solver.cpp:106] Iteration 89300, lr = 0.000125137
    I1226 23:43:01.428184  6129 solver.cpp:237] Iteration 89400, loss = 0.61969
    I1226 23:43:01.428216  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:43:01.428228  6129 solver.cpp:253]     Train net output #1: loss = 0.61969 (* 1 = 0.61969 loss)
    I1226 23:43:01.428236  6129 sgd_solver.cpp:106] Iteration 89400, lr = 0.000125043
    I1226 23:43:11.957751  6129 solver.cpp:237] Iteration 89500, loss = 0.514293
    I1226 23:43:11.957903  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:43:11.957928  6129 solver.cpp:253]     Train net output #1: loss = 0.514293 (* 1 = 0.514293 loss)
    I1226 23:43:11.957940  6129 sgd_solver.cpp:106] Iteration 89500, lr = 0.000124948
    I1226 23:43:22.934613  6129 solver.cpp:237] Iteration 89600, loss = 0.610612
    I1226 23:43:22.934650  6129 solver.cpp:253]     Train net output #0: accuracy = 0.84
    I1226 23:43:22.934662  6129 solver.cpp:253]     Train net output #1: loss = 0.610612 (* 1 = 0.610612 loss)
    I1226 23:43:22.934670  6129 sgd_solver.cpp:106] Iteration 89600, lr = 0.000124854
    I1226 23:43:33.782444  6129 solver.cpp:237] Iteration 89700, loss = 0.501409
    I1226 23:43:33.782500  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:43:33.782521  6129 solver.cpp:253]     Train net output #1: loss = 0.501409 (* 1 = 0.501409 loss)
    I1226 23:43:33.782536  6129 sgd_solver.cpp:106] Iteration 89700, lr = 0.00012476
    I1226 23:43:45.643019  6129 solver.cpp:237] Iteration 89800, loss = 0.564606
    I1226 23:43:45.643149  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:43:45.643173  6129 solver.cpp:253]     Train net output #1: loss = 0.564606 (* 1 = 0.564606 loss)
    I1226 23:43:45.643180  6129 sgd_solver.cpp:106] Iteration 89800, lr = 0.000124667
    I1226 23:43:56.286172  6129 solver.cpp:237] Iteration 89900, loss = 0.619036
    I1226 23:43:56.286216  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:43:56.286231  6129 solver.cpp:253]     Train net output #1: loss = 0.619036 (* 1 = 0.619036 loss)
    I1226 23:43:56.286242  6129 sgd_solver.cpp:106] Iteration 89900, lr = 0.000124573
    I1226 23:44:06.824926  6129 solver.cpp:341] Iteration 90000, Testing net (#0)
    I1226 23:44:11.195842  6129 solver.cpp:409]     Test net output #0: accuracy = 0.713333
    I1226 23:44:11.195904  6129 solver.cpp:409]     Test net output #1: loss = 0.8312 (* 1 = 0.8312 loss)
    I1226 23:44:11.240684  6129 solver.cpp:237] Iteration 90000, loss = 0.513265
    I1226 23:44:11.240736  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:44:11.240756  6129 solver.cpp:253]     Train net output #1: loss = 0.513265 (* 1 = 0.513265 loss)
    I1226 23:44:11.240772  6129 sgd_solver.cpp:106] Iteration 90000, lr = 0.00012448
    I1226 23:44:22.013339  6129 solver.cpp:237] Iteration 90100, loss = 0.595518
    I1226 23:44:22.013532  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:44:22.013557  6129 solver.cpp:253]     Train net output #1: loss = 0.595518 (* 1 = 0.595518 loss)
    I1226 23:44:22.013566  6129 sgd_solver.cpp:106] Iteration 90100, lr = 0.000124386
    I1226 23:44:33.239555  6129 solver.cpp:237] Iteration 90200, loss = 0.499685
    I1226 23:44:33.239596  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:44:33.239611  6129 solver.cpp:253]     Train net output #1: loss = 0.499685 (* 1 = 0.499685 loss)
    I1226 23:44:33.239621  6129 sgd_solver.cpp:106] Iteration 90200, lr = 0.000124293
    I1226 23:44:43.866436  6129 solver.cpp:237] Iteration 90300, loss = 0.563564
    I1226 23:44:43.866479  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:44:43.866492  6129 solver.cpp:253]     Train net output #1: loss = 0.563564 (* 1 = 0.563564 loss)
    I1226 23:44:43.866498  6129 sgd_solver.cpp:106] Iteration 90300, lr = 0.0001242
    I1226 23:44:54.759438  6129 solver.cpp:237] Iteration 90400, loss = 0.618629
    I1226 23:44:54.759577  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:44:54.759593  6129 solver.cpp:253]     Train net output #1: loss = 0.618629 (* 1 = 0.618629 loss)
    I1226 23:44:54.759603  6129 sgd_solver.cpp:106] Iteration 90400, lr = 0.000124107
    I1226 23:45:05.409744  6129 solver.cpp:237] Iteration 90500, loss = 0.512392
    I1226 23:45:05.409808  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:45:05.409835  6129 solver.cpp:253]     Train net output #1: loss = 0.512392 (* 1 = 0.512392 loss)
    I1226 23:45:05.409855  6129 sgd_solver.cpp:106] Iteration 90500, lr = 0.000124015
    I1226 23:45:16.023972  6129 solver.cpp:237] Iteration 90600, loss = 0.602071
    I1226 23:45:16.024013  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:45:16.024029  6129 solver.cpp:253]     Train net output #1: loss = 0.602071 (* 1 = 0.602071 loss)
    I1226 23:45:16.024041  6129 sgd_solver.cpp:106] Iteration 90600, lr = 0.000123922
    I1226 23:45:26.568917  6129 solver.cpp:237] Iteration 90700, loss = 0.500054
    I1226 23:45:26.569061  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:45:26.569077  6129 solver.cpp:253]     Train net output #1: loss = 0.500054 (* 1 = 0.500054 loss)
    I1226 23:45:26.569083  6129 sgd_solver.cpp:106] Iteration 90700, lr = 0.00012383
    I1226 23:45:37.433352  6129 solver.cpp:237] Iteration 90800, loss = 0.563476
    I1226 23:45:37.433393  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:45:37.433405  6129 solver.cpp:253]     Train net output #1: loss = 0.563476 (* 1 = 0.563476 loss)
    I1226 23:45:37.433416  6129 sgd_solver.cpp:106] Iteration 90800, lr = 0.000123738
    I1226 23:45:48.135879  6129 solver.cpp:237] Iteration 90900, loss = 0.617915
    I1226 23:45:48.135915  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:45:48.135926  6129 solver.cpp:253]     Train net output #1: loss = 0.617915 (* 1 = 0.617915 loss)
    I1226 23:45:48.135936  6129 sgd_solver.cpp:106] Iteration 90900, lr = 0.000123646
    I1226 23:45:58.594121  6129 solver.cpp:341] Iteration 91000, Testing net (#0)
    I1226 23:46:02.911825  6129 solver.cpp:409]     Test net output #0: accuracy = 0.710416
    I1226 23:46:02.911875  6129 solver.cpp:409]     Test net output #1: loss = 0.835748 (* 1 = 0.835748 loss)
    I1226 23:46:02.956328  6129 solver.cpp:237] Iteration 91000, loss = 0.511661
    I1226 23:46:02.956380  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:46:02.956393  6129 solver.cpp:253]     Train net output #1: loss = 0.511661 (* 1 = 0.511661 loss)
    I1226 23:46:02.956404  6129 sgd_solver.cpp:106] Iteration 91000, lr = 0.000123554
    I1226 23:46:13.834298  6129 solver.cpp:237] Iteration 91100, loss = 0.600608
    I1226 23:46:13.834334  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:46:13.834347  6129 solver.cpp:253]     Train net output #1: loss = 0.600608 (* 1 = 0.600608 loss)
    I1226 23:46:13.834354  6129 sgd_solver.cpp:106] Iteration 91100, lr = 0.000123462
    I1226 23:46:26.446727  6129 solver.cpp:237] Iteration 91200, loss = 0.498771
    I1226 23:46:26.446763  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:46:26.446774  6129 solver.cpp:253]     Train net output #1: loss = 0.498771 (* 1 = 0.498771 loss)
    I1226 23:46:26.446784  6129 sgd_solver.cpp:106] Iteration 91200, lr = 0.000123371
    I1226 23:46:36.981065  6129 solver.cpp:237] Iteration 91300, loss = 0.563435
    I1226 23:46:36.981201  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:46:36.981220  6129 solver.cpp:253]     Train net output #1: loss = 0.563435 (* 1 = 0.563435 loss)
    I1226 23:46:36.981231  6129 sgd_solver.cpp:106] Iteration 91300, lr = 0.00012328
    I1226 23:46:47.443541  6129 solver.cpp:237] Iteration 91400, loss = 0.617207
    I1226 23:46:47.443583  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:46:47.443595  6129 solver.cpp:253]     Train net output #1: loss = 0.617207 (* 1 = 0.617207 loss)
    I1226 23:46:47.443603  6129 sgd_solver.cpp:106] Iteration 91400, lr = 0.000123188
    I1226 23:46:58.236588  6129 solver.cpp:237] Iteration 91500, loss = 0.510901
    I1226 23:46:58.236629  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:46:58.236644  6129 solver.cpp:253]     Train net output #1: loss = 0.510901 (* 1 = 0.510901 loss)
    I1226 23:46:58.236654  6129 sgd_solver.cpp:106] Iteration 91500, lr = 0.000123097
    I1226 23:47:09.881567  6129 solver.cpp:237] Iteration 91600, loss = 0.597381
    I1226 23:47:09.881692  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:47:09.881717  6129 solver.cpp:253]     Train net output #1: loss = 0.597381 (* 1 = 0.597381 loss)
    I1226 23:47:09.881726  6129 sgd_solver.cpp:106] Iteration 91600, lr = 0.000123006
    I1226 23:47:20.716688  6129 solver.cpp:237] Iteration 91700, loss = 0.498077
    I1226 23:47:20.716727  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:47:20.716737  6129 solver.cpp:253]     Train net output #1: loss = 0.498077 (* 1 = 0.498077 loss)
    I1226 23:47:20.716747  6129 sgd_solver.cpp:106] Iteration 91700, lr = 0.000122916
    I1226 23:47:31.142462  6129 solver.cpp:237] Iteration 91800, loss = 0.563234
    I1226 23:47:31.142499  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:47:31.142511  6129 solver.cpp:253]     Train net output #1: loss = 0.563234 (* 1 = 0.563234 loss)
    I1226 23:47:31.142520  6129 sgd_solver.cpp:106] Iteration 91800, lr = 0.000122825
    I1226 23:47:41.649423  6129 solver.cpp:237] Iteration 91900, loss = 0.616379
    I1226 23:47:41.649586  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:47:41.649607  6129 solver.cpp:253]     Train net output #1: loss = 0.616379 (* 1 = 0.616379 loss)
    I1226 23:47:41.649619  6129 sgd_solver.cpp:106] Iteration 91900, lr = 0.000122735
    I1226 23:47:53.455611  6129 solver.cpp:341] Iteration 92000, Testing net (#0)
    I1226 23:47:58.255655  6129 solver.cpp:409]     Test net output #0: accuracy = 0.713166
    I1226 23:47:58.255697  6129 solver.cpp:409]     Test net output #1: loss = 0.83027 (* 1 = 0.83027 loss)
    I1226 23:47:58.300158  6129 solver.cpp:237] Iteration 92000, loss = 0.510005
    I1226 23:47:58.300197  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:47:58.300210  6129 solver.cpp:253]     Train net output #1: loss = 0.510005 (* 1 = 0.510005 loss)
    I1226 23:47:58.300221  6129 sgd_solver.cpp:106] Iteration 92000, lr = 0.000122644
    I1226 23:48:09.660145  6129 solver.cpp:237] Iteration 92100, loss = 0.592171
    I1226 23:48:09.660181  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:48:09.660192  6129 solver.cpp:253]     Train net output #1: loss = 0.592171 (* 1 = 0.592171 loss)
    I1226 23:48:09.660202  6129 sgd_solver.cpp:106] Iteration 92100, lr = 0.000122554
    I1226 23:48:20.613083  6129 solver.cpp:237] Iteration 92200, loss = 0.498587
    I1226 23:48:20.613239  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:48:20.613255  6129 solver.cpp:253]     Train net output #1: loss = 0.498587 (* 1 = 0.498587 loss)
    I1226 23:48:20.613266  6129 sgd_solver.cpp:106] Iteration 92200, lr = 0.000122464
    I1226 23:48:31.320807  6129 solver.cpp:237] Iteration 92300, loss = 0.562904
    I1226 23:48:31.320844  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:48:31.320857  6129 solver.cpp:253]     Train net output #1: loss = 0.562904 (* 1 = 0.562904 loss)
    I1226 23:48:31.320864  6129 sgd_solver.cpp:106] Iteration 92300, lr = 0.000122375
    I1226 23:48:41.957329  6129 solver.cpp:237] Iteration 92400, loss = 0.616379
    I1226 23:48:41.957387  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:48:41.957409  6129 solver.cpp:253]     Train net output #1: loss = 0.616379 (* 1 = 0.616379 loss)
    I1226 23:48:41.957425  6129 sgd_solver.cpp:106] Iteration 92400, lr = 0.000122285
    I1226 23:48:52.826228  6129 solver.cpp:237] Iteration 92500, loss = 0.508878
    I1226 23:48:52.826396  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:48:52.826411  6129 solver.cpp:253]     Train net output #1: loss = 0.508878 (* 1 = 0.508878 loss)
    I1226 23:48:52.826419  6129 sgd_solver.cpp:106] Iteration 92500, lr = 0.000122195
    I1226 23:49:03.743360  6129 solver.cpp:237] Iteration 92600, loss = 0.595823
    I1226 23:49:03.743415  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:49:03.743437  6129 solver.cpp:253]     Train net output #1: loss = 0.595823 (* 1 = 0.595823 loss)
    I1226 23:49:03.743451  6129 sgd_solver.cpp:106] Iteration 92600, lr = 0.000122106
    I1226 23:49:14.229962  6129 solver.cpp:237] Iteration 92700, loss = 0.498101
    I1226 23:49:14.230006  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:49:14.230018  6129 solver.cpp:253]     Train net output #1: loss = 0.498101 (* 1 = 0.498101 loss)
    I1226 23:49:14.230026  6129 sgd_solver.cpp:106] Iteration 92700, lr = 0.000122017
    I1226 23:49:24.722270  6129 solver.cpp:237] Iteration 92800, loss = 0.562092
    I1226 23:49:24.722375  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:49:24.722393  6129 solver.cpp:253]     Train net output #1: loss = 0.562092 (* 1 = 0.562092 loss)
    I1226 23:49:24.722404  6129 sgd_solver.cpp:106] Iteration 92800, lr = 0.000121928
    I1226 23:49:35.311769  6129 solver.cpp:237] Iteration 92900, loss = 0.616148
    I1226 23:49:35.311813  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:49:35.311825  6129 solver.cpp:253]     Train net output #1: loss = 0.616148 (* 1 = 0.616148 loss)
    I1226 23:49:35.311835  6129 sgd_solver.cpp:106] Iteration 92900, lr = 0.000121839
    I1226 23:49:46.301553  6129 solver.cpp:341] Iteration 93000, Testing net (#0)
    I1226 23:49:50.698134  6129 solver.cpp:409]     Test net output #0: accuracy = 0.712667
    I1226 23:49:50.698180  6129 solver.cpp:409]     Test net output #1: loss = 0.828918 (* 1 = 0.828918 loss)
    I1226 23:49:50.742705  6129 solver.cpp:237] Iteration 93000, loss = 0.508171
    I1226 23:49:50.742748  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:49:50.742763  6129 solver.cpp:253]     Train net output #1: loss = 0.508171 (* 1 = 0.508171 loss)
    I1226 23:49:50.742774  6129 sgd_solver.cpp:106] Iteration 93000, lr = 0.00012175
    I1226 23:50:01.668367  6129 solver.cpp:237] Iteration 93100, loss = 0.593304
    I1226 23:50:01.668489  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:50:01.668514  6129 solver.cpp:253]     Train net output #1: loss = 0.593304 (* 1 = 0.593304 loss)
    I1226 23:50:01.668530  6129 sgd_solver.cpp:106] Iteration 93100, lr = 0.000121662
    I1226 23:50:12.189927  6129 solver.cpp:237] Iteration 93200, loss = 0.497308
    I1226 23:50:12.189965  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:50:12.189976  6129 solver.cpp:253]     Train net output #1: loss = 0.497308 (* 1 = 0.497308 loss)
    I1226 23:50:12.189985  6129 sgd_solver.cpp:106] Iteration 93200, lr = 0.000121573
    I1226 23:50:22.815798  6129 solver.cpp:237] Iteration 93300, loss = 0.562713
    I1226 23:50:22.815843  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:50:22.815855  6129 solver.cpp:253]     Train net output #1: loss = 0.562713 (* 1 = 0.562713 loss)
    I1226 23:50:22.815863  6129 sgd_solver.cpp:106] Iteration 93300, lr = 0.000121485
    I1226 23:50:34.246736  6129 solver.cpp:237] Iteration 93400, loss = 0.615149
    I1226 23:50:34.247241  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:50:34.247267  6129 solver.cpp:253]     Train net output #1: loss = 0.615149 (* 1 = 0.615149 loss)
    I1226 23:50:34.247278  6129 sgd_solver.cpp:106] Iteration 93400, lr = 0.000121397
    I1226 23:50:48.017376  6129 solver.cpp:237] Iteration 93500, loss = 0.507402
    I1226 23:50:48.017442  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:50:48.017465  6129 solver.cpp:253]     Train net output #1: loss = 0.507402 (* 1 = 0.507402 loss)
    I1226 23:50:48.017484  6129 sgd_solver.cpp:106] Iteration 93500, lr = 0.000121309
    I1226 23:51:02.014075  6129 solver.cpp:237] Iteration 93600, loss = 0.595494
    I1226 23:51:02.014113  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:51:02.014125  6129 solver.cpp:253]     Train net output #1: loss = 0.595494 (* 1 = 0.595494 loss)
    I1226 23:51:02.014135  6129 sgd_solver.cpp:106] Iteration 93600, lr = 0.000121221
    I1226 23:51:15.122220  6129 solver.cpp:237] Iteration 93700, loss = 0.496744
    I1226 23:51:15.122385  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:51:15.122401  6129 solver.cpp:253]     Train net output #1: loss = 0.496744 (* 1 = 0.496744 loss)
    I1226 23:51:15.122411  6129 sgd_solver.cpp:106] Iteration 93700, lr = 0.000121133
    I1226 23:51:26.402468  6129 solver.cpp:237] Iteration 93800, loss = 0.56269
    I1226 23:51:26.402506  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:51:26.402518  6129 solver.cpp:253]     Train net output #1: loss = 0.56269 (* 1 = 0.56269 loss)
    I1226 23:51:26.402529  6129 sgd_solver.cpp:106] Iteration 93800, lr = 0.000121046
    I1226 23:51:41.427225  6129 solver.cpp:237] Iteration 93900, loss = 0.614128
    I1226 23:51:41.427268  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:51:41.427281  6129 solver.cpp:253]     Train net output #1: loss = 0.614128 (* 1 = 0.614128 loss)
    I1226 23:51:41.427292  6129 sgd_solver.cpp:106] Iteration 93900, lr = 0.000120958
    I1226 23:51:53.036667  6129 solver.cpp:341] Iteration 94000, Testing net (#0)
    I1226 23:51:58.042510  6129 solver.cpp:409]     Test net output #0: accuracy = 0.710917
    I1226 23:51:58.042559  6129 solver.cpp:409]     Test net output #1: loss = 0.834118 (* 1 = 0.834118 loss)
    I1226 23:51:58.094091  6129 solver.cpp:237] Iteration 94000, loss = 0.50689
    I1226 23:51:58.094138  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:51:58.094154  6129 solver.cpp:253]     Train net output #1: loss = 0.50689 (* 1 = 0.50689 loss)
    I1226 23:51:58.094167  6129 sgd_solver.cpp:106] Iteration 94000, lr = 0.000120871
    I1226 23:52:10.446974  6129 solver.cpp:237] Iteration 94100, loss = 0.598235
    I1226 23:52:10.447012  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:52:10.447024  6129 solver.cpp:253]     Train net output #1: loss = 0.598235 (* 1 = 0.598235 loss)
    I1226 23:52:10.447033  6129 sgd_solver.cpp:106] Iteration 94100, lr = 0.000120784
    I1226 23:52:22.902762  6129 solver.cpp:237] Iteration 94200, loss = 0.494873
    I1226 23:52:22.902797  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:52:22.902811  6129 solver.cpp:253]     Train net output #1: loss = 0.494873 (* 1 = 0.494873 loss)
    I1226 23:52:22.902820  6129 sgd_solver.cpp:106] Iteration 94200, lr = 0.000120697
    I1226 23:52:35.133530  6129 solver.cpp:237] Iteration 94300, loss = 0.5627
    I1226 23:52:35.133661  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:52:35.133677  6129 solver.cpp:253]     Train net output #1: loss = 0.5627 (* 1 = 0.5627 loss)
    I1226 23:52:35.133688  6129 sgd_solver.cpp:106] Iteration 94300, lr = 0.00012061
    I1226 23:52:47.557725  6129 solver.cpp:237] Iteration 94400, loss = 0.613626
    I1226 23:52:47.557768  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:52:47.557785  6129 solver.cpp:253]     Train net output #1: loss = 0.613626 (* 1 = 0.613626 loss)
    I1226 23:52:47.557796  6129 sgd_solver.cpp:106] Iteration 94400, lr = 0.000120524
    I1226 23:52:59.873226  6129 solver.cpp:237] Iteration 94500, loss = 0.506626
    I1226 23:52:59.873263  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:52:59.873275  6129 solver.cpp:253]     Train net output #1: loss = 0.506626 (* 1 = 0.506626 loss)
    I1226 23:52:59.873284  6129 sgd_solver.cpp:106] Iteration 94500, lr = 0.000120437
    I1226 23:53:11.078490  6129 solver.cpp:237] Iteration 94600, loss = 0.588565
    I1226 23:53:11.078662  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:53:11.078688  6129 solver.cpp:253]     Train net output #1: loss = 0.588565 (* 1 = 0.588565 loss)
    I1226 23:53:11.078701  6129 sgd_solver.cpp:106] Iteration 94600, lr = 0.000120351
    I1226 23:53:26.133028  6129 solver.cpp:237] Iteration 94700, loss = 0.495021
    I1226 23:53:26.133085  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:53:26.133106  6129 solver.cpp:253]     Train net output #1: loss = 0.495021 (* 1 = 0.495021 loss)
    I1226 23:53:26.133123  6129 sgd_solver.cpp:106] Iteration 94700, lr = 0.000120265
    I1226 23:53:39.629335  6129 solver.cpp:237] Iteration 94800, loss = 0.562697
    I1226 23:53:39.629376  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:53:39.629390  6129 solver.cpp:253]     Train net output #1: loss = 0.562697 (* 1 = 0.562697 loss)
    I1226 23:53:39.629402  6129 sgd_solver.cpp:106] Iteration 94800, lr = 0.000120179
    I1226 23:53:52.584064  6129 solver.cpp:237] Iteration 94900, loss = 0.61324
    I1226 23:53:52.584210  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:53:52.584226  6129 solver.cpp:253]     Train net output #1: loss = 0.61324 (* 1 = 0.61324 loss)
    I1226 23:53:52.584234  6129 sgd_solver.cpp:106] Iteration 94900, lr = 0.000120093
    I1226 23:54:08.176544  6129 solver.cpp:341] Iteration 95000, Testing net (#0)
    I1226 23:54:14.424024  6129 solver.cpp:409]     Test net output #0: accuracy = 0.715833
    I1226 23:54:14.424069  6129 solver.cpp:409]     Test net output #1: loss = 0.829106 (* 1 = 0.829106 loss)
    I1226 23:54:14.489320  6129 solver.cpp:237] Iteration 95000, loss = 0.505284
    I1226 23:54:14.489370  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:54:14.489384  6129 solver.cpp:253]     Train net output #1: loss = 0.505284 (* 1 = 0.505284 loss)
    I1226 23:54:14.489397  6129 sgd_solver.cpp:106] Iteration 95000, lr = 0.000120007
    I1226 23:54:26.463912  6129 solver.cpp:237] Iteration 95100, loss = 0.597334
    I1226 23:54:26.464069  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:54:26.464085  6129 solver.cpp:253]     Train net output #1: loss = 0.597334 (* 1 = 0.597334 loss)
    I1226 23:54:26.464094  6129 sgd_solver.cpp:106] Iteration 95100, lr = 0.000119921
    I1226 23:54:37.914182  6129 solver.cpp:237] Iteration 95200, loss = 0.493437
    I1226 23:54:37.914225  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:54:37.914242  6129 solver.cpp:253]     Train net output #1: loss = 0.493437 (* 1 = 0.493437 loss)
    I1226 23:54:37.914252  6129 sgd_solver.cpp:106] Iteration 95200, lr = 0.000119836
    I1226 23:54:50.077633  6129 solver.cpp:237] Iteration 95300, loss = 0.562109
    I1226 23:54:50.077688  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:54:50.077710  6129 solver.cpp:253]     Train net output #1: loss = 0.562109 (* 1 = 0.562109 loss)
    I1226 23:54:50.077728  6129 sgd_solver.cpp:106] Iteration 95300, lr = 0.00011975
    I1226 23:55:02.056329  6129 solver.cpp:237] Iteration 95400, loss = 0.612437
    I1226 23:55:02.056509  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:55:02.056545  6129 solver.cpp:253]     Train net output #1: loss = 0.612437 (* 1 = 0.612437 loss)
    I1226 23:55:02.056560  6129 sgd_solver.cpp:106] Iteration 95400, lr = 0.000119665
    I1226 23:55:13.461916  6129 solver.cpp:237] Iteration 95500, loss = 0.504868
    I1226 23:55:13.461953  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:55:13.461966  6129 solver.cpp:253]     Train net output #1: loss = 0.504868 (* 1 = 0.504868 loss)
    I1226 23:55:13.461974  6129 sgd_solver.cpp:106] Iteration 95500, lr = 0.00011958
    I1226 23:55:24.456383  6129 solver.cpp:237] Iteration 95600, loss = 0.590138
    I1226 23:55:24.456440  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:55:24.456470  6129 solver.cpp:253]     Train net output #1: loss = 0.590138 (* 1 = 0.590138 loss)
    I1226 23:55:24.456487  6129 sgd_solver.cpp:106] Iteration 95600, lr = 0.000119495
    I1226 23:55:36.087929  6129 solver.cpp:237] Iteration 95700, loss = 0.493802
    I1226 23:55:36.088075  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:55:36.088099  6129 solver.cpp:253]     Train net output #1: loss = 0.493802 (* 1 = 0.493802 loss)
    I1226 23:55:36.088105  6129 sgd_solver.cpp:106] Iteration 95700, lr = 0.00011941
    I1226 23:55:47.141052  6129 solver.cpp:237] Iteration 95800, loss = 0.561884
    I1226 23:55:47.141086  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:55:47.141098  6129 solver.cpp:253]     Train net output #1: loss = 0.561884 (* 1 = 0.561884 loss)
    I1226 23:55:47.141108  6129 sgd_solver.cpp:106] Iteration 95800, lr = 0.000119326
    I1226 23:55:58.496875  6129 solver.cpp:237] Iteration 95900, loss = 0.612016
    I1226 23:55:58.496917  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:55:58.496929  6129 solver.cpp:253]     Train net output #1: loss = 0.612016 (* 1 = 0.612016 loss)
    I1226 23:55:58.496939  6129 sgd_solver.cpp:106] Iteration 95900, lr = 0.000119241
    I1226 23:56:13.227967  6129 solver.cpp:341] Iteration 96000, Testing net (#0)
    I1226 23:56:17.635411  6129 solver.cpp:409]     Test net output #0: accuracy = 0.713083
    I1226 23:56:17.635457  6129 solver.cpp:409]     Test net output #1: loss = 0.833806 (* 1 = 0.833806 loss)
    I1226 23:56:17.680289  6129 solver.cpp:237] Iteration 96000, loss = 0.504252
    I1226 23:56:17.680335  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:56:17.680351  6129 solver.cpp:253]     Train net output #1: loss = 0.504252 (* 1 = 0.504252 loss)
    I1226 23:56:17.680362  6129 sgd_solver.cpp:106] Iteration 96000, lr = 0.000119157
    I1226 23:56:28.683260  6129 solver.cpp:237] Iteration 96100, loss = 0.591438
    I1226 23:56:28.683296  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:56:28.683308  6129 solver.cpp:253]     Train net output #1: loss = 0.591438 (* 1 = 0.591438 loss)
    I1226 23:56:28.683318  6129 sgd_solver.cpp:106] Iteration 96100, lr = 0.000119073
    I1226 23:56:39.342247  6129 solver.cpp:237] Iteration 96200, loss = 0.491685
    I1226 23:56:39.342283  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:56:39.342294  6129 solver.cpp:253]     Train net output #1: loss = 0.491685 (* 1 = 0.491685 loss)
    I1226 23:56:39.342303  6129 sgd_solver.cpp:106] Iteration 96200, lr = 0.000118988
    I1226 23:56:49.888973  6129 solver.cpp:237] Iteration 96300, loss = 0.561521
    I1226 23:56:49.889138  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:56:49.889154  6129 solver.cpp:253]     Train net output #1: loss = 0.561521 (* 1 = 0.561521 loss)
    I1226 23:56:49.889160  6129 sgd_solver.cpp:106] Iteration 96300, lr = 0.000118904
    I1226 23:57:00.923439  6129 solver.cpp:237] Iteration 96400, loss = 0.611266
    I1226 23:57:00.923477  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:57:00.923488  6129 solver.cpp:253]     Train net output #1: loss = 0.611266 (* 1 = 0.611266 loss)
    I1226 23:57:00.923498  6129 sgd_solver.cpp:106] Iteration 96400, lr = 0.000118821
    I1226 23:57:11.903110  6129 solver.cpp:237] Iteration 96500, loss = 0.503656
    I1226 23:57:11.903165  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:57:11.903187  6129 solver.cpp:253]     Train net output #1: loss = 0.503656 (* 1 = 0.503656 loss)
    I1226 23:57:11.903203  6129 sgd_solver.cpp:106] Iteration 96500, lr = 0.000118737
    I1226 23:57:24.805315  6129 solver.cpp:237] Iteration 96600, loss = 0.594675
    I1226 23:57:24.805487  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:57:24.805515  6129 solver.cpp:253]     Train net output #1: loss = 0.594675 (* 1 = 0.594675 loss)
    I1226 23:57:24.805532  6129 sgd_solver.cpp:106] Iteration 96600, lr = 0.000118653
    I1226 23:57:37.065376  6129 solver.cpp:237] Iteration 96700, loss = 0.490555
    I1226 23:57:37.065460  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:57:37.065492  6129 solver.cpp:253]     Train net output #1: loss = 0.490555 (* 1 = 0.490555 loss)
    I1226 23:57:37.065510  6129 sgd_solver.cpp:106] Iteration 96700, lr = 0.00011857
    I1226 23:57:47.600901  6129 solver.cpp:237] Iteration 96800, loss = 0.56121
    I1226 23:57:47.600989  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:57:47.601027  6129 solver.cpp:253]     Train net output #1: loss = 0.56121 (* 1 = 0.56121 loss)
    I1226 23:57:47.601052  6129 sgd_solver.cpp:106] Iteration 96800, lr = 0.000118487
    I1226 23:57:59.532044  6129 solver.cpp:237] Iteration 96900, loss = 0.610759
    I1226 23:57:59.532181  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:57:59.532207  6129 solver.cpp:253]     Train net output #1: loss = 0.610759 (* 1 = 0.610759 loss)
    I1226 23:57:59.532217  6129 sgd_solver.cpp:106] Iteration 96900, lr = 0.000118404
    I1226 23:58:12.774510  6129 solver.cpp:341] Iteration 97000, Testing net (#0)
    I1226 23:58:17.849723  6129 solver.cpp:409]     Test net output #0: accuracy = 0.714583
    I1226 23:58:17.849786  6129 solver.cpp:409]     Test net output #1: loss = 0.828302 (* 1 = 0.828302 loss)
    I1226 23:58:17.930132  6129 solver.cpp:237] Iteration 97000, loss = 0.503351
    I1226 23:58:17.930184  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:58:17.930204  6129 solver.cpp:253]     Train net output #1: loss = 0.503351 (* 1 = 0.503351 loss)
    I1226 23:58:17.930220  6129 sgd_solver.cpp:106] Iteration 97000, lr = 0.000118321
    I1226 23:58:29.302517  6129 solver.cpp:237] Iteration 97100, loss = 0.592104
    I1226 23:58:29.302575  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:58:29.302597  6129 solver.cpp:253]     Train net output #1: loss = 0.592104 (* 1 = 0.592104 loss)
    I1226 23:58:29.302613  6129 sgd_solver.cpp:106] Iteration 97100, lr = 0.000118238
    I1226 23:58:40.906188  6129 solver.cpp:237] Iteration 97200, loss = 0.489664
    I1226 23:58:40.906330  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1226 23:58:40.906354  6129 solver.cpp:253]     Train net output #1: loss = 0.489664 (* 1 = 0.489664 loss)
    I1226 23:58:40.906365  6129 sgd_solver.cpp:106] Iteration 97200, lr = 0.000118155
    I1226 23:58:53.515321  6129 solver.cpp:237] Iteration 97300, loss = 0.561098
    I1226 23:58:53.515377  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:58:53.515398  6129 solver.cpp:253]     Train net output #1: loss = 0.561098 (* 1 = 0.561098 loss)
    I1226 23:58:53.515415  6129 sgd_solver.cpp:106] Iteration 97300, lr = 0.000118072
    I1226 23:59:05.697074  6129 solver.cpp:237] Iteration 97400, loss = 0.610108
    I1226 23:59:05.697131  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1226 23:59:05.697154  6129 solver.cpp:253]     Train net output #1: loss = 0.610108 (* 1 = 0.610108 loss)
    I1226 23:59:05.697170  6129 sgd_solver.cpp:106] Iteration 97400, lr = 0.00011799
    I1226 23:59:17.272410  6129 solver.cpp:237] Iteration 97500, loss = 0.502546
    I1226 23:59:17.272521  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1226 23:59:17.272538  6129 solver.cpp:253]     Train net output #1: loss = 0.502546 (* 1 = 0.502546 loss)
    I1226 23:59:17.272549  6129 sgd_solver.cpp:106] Iteration 97500, lr = 0.000117908
    I1226 23:59:28.899231  6129 solver.cpp:237] Iteration 97600, loss = 0.589083
    I1226 23:59:28.899274  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1226 23:59:28.899288  6129 solver.cpp:253]     Train net output #1: loss = 0.589083 (* 1 = 0.589083 loss)
    I1226 23:59:28.899299  6129 sgd_solver.cpp:106] Iteration 97600, lr = 0.000117825
    I1226 23:59:40.710902  6129 solver.cpp:237] Iteration 97700, loss = 0.490592
    I1226 23:59:40.710948  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1226 23:59:40.710963  6129 solver.cpp:253]     Train net output #1: loss = 0.490592 (* 1 = 0.490592 loss)
    I1226 23:59:40.710975  6129 sgd_solver.cpp:106] Iteration 97700, lr = 0.000117743
    I1226 23:59:53.844665  6129 solver.cpp:237] Iteration 97800, loss = 0.560923
    I1226 23:59:53.844801  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1226 23:59:53.844820  6129 solver.cpp:253]     Train net output #1: loss = 0.560923 (* 1 = 0.560923 loss)
    I1226 23:59:53.844833  6129 sgd_solver.cpp:106] Iteration 97800, lr = 0.000117661
    I1227 00:00:04.921380  6129 solver.cpp:237] Iteration 97900, loss = 0.609531
    I1227 00:00:04.921418  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 00:00:04.921432  6129 solver.cpp:253]     Train net output #1: loss = 0.609531 (* 1 = 0.609531 loss)
    I1227 00:00:04.921443  6129 sgd_solver.cpp:106] Iteration 97900, lr = 0.00011758
    I1227 00:00:15.358300  6129 solver.cpp:341] Iteration 98000, Testing net (#0)
    I1227 00:00:21.044915  6129 solver.cpp:409]     Test net output #0: accuracy = 0.714
    I1227 00:00:21.044981  6129 solver.cpp:409]     Test net output #1: loss = 0.826624 (* 1 = 0.826624 loss)
    I1227 00:00:21.089812  6129 solver.cpp:237] Iteration 98000, loss = 0.502257
    I1227 00:00:21.089872  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1227 00:00:21.089898  6129 solver.cpp:253]     Train net output #1: loss = 0.502257 (* 1 = 0.502257 loss)
    I1227 00:00:21.089916  6129 sgd_solver.cpp:106] Iteration 98000, lr = 0.000117498
    I1227 00:00:32.935670  6129 solver.cpp:237] Iteration 98100, loss = 0.589859
    I1227 00:00:32.935817  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 00:00:32.935849  6129 solver.cpp:253]     Train net output #1: loss = 0.589859 (* 1 = 0.589859 loss)
    I1227 00:00:32.935859  6129 sgd_solver.cpp:106] Iteration 98100, lr = 0.000117416
    I1227 00:00:44.958943  6129 solver.cpp:237] Iteration 98200, loss = 0.489854
    I1227 00:00:44.958997  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 00:00:44.959009  6129 solver.cpp:253]     Train net output #1: loss = 0.489854 (* 1 = 0.489854 loss)
    I1227 00:00:44.959019  6129 sgd_solver.cpp:106] Iteration 98200, lr = 0.000117335
    I1227 00:00:57.608000  6129 solver.cpp:237] Iteration 98300, loss = 0.559973
    I1227 00:00:57.608042  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 00:00:57.608055  6129 solver.cpp:253]     Train net output #1: loss = 0.559973 (* 1 = 0.559973 loss)
    I1227 00:00:57.608067  6129 sgd_solver.cpp:106] Iteration 98300, lr = 0.000117254
    I1227 00:01:10.657541  6129 solver.cpp:237] Iteration 98400, loss = 0.609036
    I1227 00:01:10.657687  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 00:01:10.657706  6129 solver.cpp:253]     Train net output #1: loss = 0.609036 (* 1 = 0.609036 loss)
    I1227 00:01:10.657717  6129 sgd_solver.cpp:106] Iteration 98400, lr = 0.000117173
    I1227 00:01:23.042690  6129 solver.cpp:237] Iteration 98500, loss = 0.501871
    I1227 00:01:23.042768  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1227 00:01:23.042799  6129 solver.cpp:253]     Train net output #1: loss = 0.501871 (* 1 = 0.501871 loss)
    I1227 00:01:23.042817  6129 sgd_solver.cpp:106] Iteration 98500, lr = 0.000117092
    I1227 00:01:34.883312  6129 solver.cpp:237] Iteration 98600, loss = 0.592495
    I1227 00:01:34.883375  6129 solver.cpp:253]     Train net output #0: accuracy = 0.85
    I1227 00:01:34.883399  6129 solver.cpp:253]     Train net output #1: loss = 0.592495 (* 1 = 0.592495 loss)
    I1227 00:01:34.883419  6129 sgd_solver.cpp:106] Iteration 98600, lr = 0.000117011
    I1227 00:01:48.523562  6129 solver.cpp:237] Iteration 98700, loss = 0.488706
    I1227 00:01:48.523722  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 00:01:48.523740  6129 solver.cpp:253]     Train net output #1: loss = 0.488706 (* 1 = 0.488706 loss)
    I1227 00:01:48.523751  6129 sgd_solver.cpp:106] Iteration 98700, lr = 0.00011693
    I1227 00:02:00.345320  6129 solver.cpp:237] Iteration 98800, loss = 0.559942
    I1227 00:02:00.345361  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 00:02:00.345376  6129 solver.cpp:253]     Train net output #1: loss = 0.559942 (* 1 = 0.559942 loss)
    I1227 00:02:00.345386  6129 sgd_solver.cpp:106] Iteration 98800, lr = 0.000116849
    I1227 00:02:10.932096  6129 solver.cpp:237] Iteration 98900, loss = 0.608541
    I1227 00:02:10.932144  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 00:02:10.932163  6129 solver.cpp:253]     Train net output #1: loss = 0.608541 (* 1 = 0.608541 loss)
    I1227 00:02:10.932178  6129 sgd_solver.cpp:106] Iteration 98900, lr = 0.000116769
    I1227 00:02:21.749169  6129 solver.cpp:341] Iteration 99000, Testing net (#0)
    I1227 00:02:26.157024  6129 solver.cpp:409]     Test net output #0: accuracy = 0.711167
    I1227 00:02:26.157066  6129 solver.cpp:409]     Test net output #1: loss = 0.832446 (* 1 = 0.832446 loss)
    I1227 00:02:26.202139  6129 solver.cpp:237] Iteration 99000, loss = 0.501149
    I1227 00:02:26.202175  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1227 00:02:26.202186  6129 solver.cpp:253]     Train net output #1: loss = 0.501149 (* 1 = 0.501149 loss)
    I1227 00:02:26.202196  6129 sgd_solver.cpp:106] Iteration 99000, lr = 0.000116689
    I1227 00:02:36.928174  6129 solver.cpp:237] Iteration 99100, loss = 0.586287
    I1227 00:02:36.928210  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 00:02:36.928222  6129 solver.cpp:253]     Train net output #1: loss = 0.586287 (* 1 = 0.586287 loss)
    I1227 00:02:36.928231  6129 sgd_solver.cpp:106] Iteration 99100, lr = 0.000116608
    I1227 00:02:48.166069  6129 solver.cpp:237] Iteration 99200, loss = 0.48807
    I1227 00:02:48.166120  6129 solver.cpp:253]     Train net output #0: accuracy = 0.87
    I1227 00:02:48.166141  6129 solver.cpp:253]     Train net output #1: loss = 0.48807 (* 1 = 0.48807 loss)
    I1227 00:02:48.166158  6129 sgd_solver.cpp:106] Iteration 99200, lr = 0.000116528
    I1227 00:02:59.342063  6129 solver.cpp:237] Iteration 99300, loss = 0.55925
    I1227 00:02:59.342226  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 00:02:59.342242  6129 solver.cpp:253]     Train net output #1: loss = 0.55925 (* 1 = 0.55925 loss)
    I1227 00:02:59.342250  6129 sgd_solver.cpp:106] Iteration 99300, lr = 0.000116448
    I1227 00:03:10.304711  6129 solver.cpp:237] Iteration 99400, loss = 0.608683
    I1227 00:03:10.304776  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 00:03:10.304793  6129 solver.cpp:253]     Train net output #1: loss = 0.608683 (* 1 = 0.608683 loss)
    I1227 00:03:10.304807  6129 sgd_solver.cpp:106] Iteration 99400, lr = 0.000116368
    I1227 00:03:21.447975  6129 solver.cpp:237] Iteration 99500, loss = 0.500634
    I1227 00:03:21.448016  6129 solver.cpp:253]     Train net output #0: accuracy = 0.88
    I1227 00:03:21.448031  6129 solver.cpp:253]     Train net output #1: loss = 0.500634 (* 1 = 0.500634 loss)
    I1227 00:03:21.448042  6129 sgd_solver.cpp:106] Iteration 99500, lr = 0.000116289
    I1227 00:03:33.313370  6129 solver.cpp:237] Iteration 99600, loss = 0.585263
    I1227 00:03:33.313526  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 00:03:33.313555  6129 solver.cpp:253]     Train net output #1: loss = 0.585263 (* 1 = 0.585263 loss)
    I1227 00:03:33.313570  6129 sgd_solver.cpp:106] Iteration 99600, lr = 0.000116209
    I1227 00:03:44.558357  6129 solver.cpp:237] Iteration 99700, loss = 0.488142
    I1227 00:03:44.558403  6129 solver.cpp:253]     Train net output #0: accuracy = 0.86
    I1227 00:03:44.558418  6129 solver.cpp:253]     Train net output #1: loss = 0.488142 (* 1 = 0.488142 loss)
    I1227 00:03:44.558429  6129 sgd_solver.cpp:106] Iteration 99700, lr = 0.00011613
    I1227 00:03:55.930318  6129 solver.cpp:237] Iteration 99800, loss = 0.559244
    I1227 00:03:55.930399  6129 solver.cpp:253]     Train net output #0: accuracy = 0.79
    I1227 00:03:55.930423  6129 solver.cpp:253]     Train net output #1: loss = 0.559244 (* 1 = 0.559244 loss)
    I1227 00:03:55.930444  6129 sgd_solver.cpp:106] Iteration 99800, lr = 0.00011605
    I1227 00:04:08.106170  6129 solver.cpp:237] Iteration 99900, loss = 0.607724
    I1227 00:04:08.106504  6129 solver.cpp:253]     Train net output #0: accuracy = 0.78
    I1227 00:04:08.106562  6129 solver.cpp:253]     Train net output #1: loss = 0.607724 (* 1 = 0.607724 loss)
    I1227 00:04:08.106609  6129 sgd_solver.cpp:106] Iteration 99900, lr = 0.000115971
    I1227 00:04:19.627724  6129 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_100000.caffemodel
    I1227 00:04:19.685092  6129 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_100000.solverstate
    I1227 00:04:19.722662  6129 solver.cpp:321] Iteration 100000, loss = 0.500088
    I1227 00:04:19.722722  6129 solver.cpp:341] Iteration 100000, Testing net (#0)
    I1227 00:04:25.390161  6129 solver.cpp:409]     Test net output #0: accuracy = 0.716
    I1227 00:04:25.390275  6129 solver.cpp:409]     Test net output #1: loss = 0.82697 (* 1 = 0.82697 loss)
    I1227 00:04:25.390295  6129 solver.cpp:326] Optimization Done.
    I1227 00:04:25.390313  6129 caffe.cpp:215] Optimization Done.
    CPU times: user 30.4 s, sys: 4.03 s, total: 34.5 s
    Wall time: 3h 15min 22s


## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_100000.caffemodel -iterations 83
```

    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1227 00:04:25.706575 24337 caffe.cpp:234] Use CPU.
    I1227 00:04:25.928380 24337 net.cpp:49] Initializing net from parameters:
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
        num_output: 42
        kernel_size: 5
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
      name: "sig1"
      type: "Sigmoid"
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
      name: "sig2"
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
    I1227 00:04:25.928938 24337 layer_factory.hpp:77] Creating layer data
    I1227 00:04:25.928961 24337 net.cpp:106] Creating Layer data
    I1227 00:04:25.928971 24337 net.cpp:411] data -> data
    I1227 00:04:25.928992 24337 net.cpp:411] data -> label
    I1227 00:04:25.929007 24337 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_10_caffe_hdf5/test.txt
    I1227 00:04:25.929069 24337 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1227 00:04:25.930116 24337 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1227 00:04:26.925406 24337 net.cpp:150] Setting up data
    I1227 00:04:26.925448 24337 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1227 00:04:26.925457 24337 net.cpp:157] Top shape: 120 (120)
    I1227 00:04:26.925464 24337 net.cpp:165] Memory required for data: 1475040
    I1227 00:04:26.925475 24337 layer_factory.hpp:77] Creating layer label_data_1_split
    I1227 00:04:26.925496 24337 net.cpp:106] Creating Layer label_data_1_split
    I1227 00:04:26.925505 24337 net.cpp:454] label_data_1_split <- label
    I1227 00:04:26.925518 24337 net.cpp:411] label_data_1_split -> label_data_1_split_0
    I1227 00:04:26.925530 24337 net.cpp:411] label_data_1_split -> label_data_1_split_1
    I1227 00:04:26.925544 24337 net.cpp:150] Setting up label_data_1_split
    I1227 00:04:26.925551 24337 net.cpp:157] Top shape: 120 (120)
    I1227 00:04:26.925559 24337 net.cpp:157] Top shape: 120 (120)
    I1227 00:04:26.925564 24337 net.cpp:165] Memory required for data: 1476000
    I1227 00:04:26.925570 24337 layer_factory.hpp:77] Creating layer conv1
    I1227 00:04:26.925582 24337 net.cpp:106] Creating Layer conv1
    I1227 00:04:26.925590 24337 net.cpp:454] conv1 <- data
    I1227 00:04:26.925597 24337 net.cpp:411] conv1 -> conv1
    I1227 00:04:26.926013 24337 net.cpp:150] Setting up conv1
    I1227 00:04:26.926028 24337 net.cpp:157] Top shape: 120 32 30 30 (3456000)
    I1227 00:04:26.926034 24337 net.cpp:165] Memory required for data: 15300000
    I1227 00:04:26.926048 24337 layer_factory.hpp:77] Creating layer pool1
    I1227 00:04:26.926059 24337 net.cpp:106] Creating Layer pool1
    I1227 00:04:26.926065 24337 net.cpp:454] pool1 <- conv1
    I1227 00:04:26.926074 24337 net.cpp:411] pool1 -> pool1
    I1227 00:04:26.926132 24337 net.cpp:150] Setting up pool1
    I1227 00:04:26.926143 24337 net.cpp:157] Top shape: 120 32 15 15 (864000)
    I1227 00:04:26.926149 24337 net.cpp:165] Memory required for data: 18756000
    I1227 00:04:26.926156 24337 layer_factory.hpp:77] Creating layer relu1
    I1227 00:04:26.926167 24337 net.cpp:106] Creating Layer relu1
    I1227 00:04:26.926182 24337 net.cpp:454] relu1 <- pool1
    I1227 00:04:26.926189 24337 net.cpp:397] relu1 -> pool1 (in-place)
    I1227 00:04:26.926200 24337 net.cpp:150] Setting up relu1
    I1227 00:04:26.926208 24337 net.cpp:157] Top shape: 120 32 15 15 (864000)
    I1227 00:04:26.926213 24337 net.cpp:165] Memory required for data: 22212000
    I1227 00:04:26.926219 24337 layer_factory.hpp:77] Creating layer conv2
    I1227 00:04:26.926229 24337 net.cpp:106] Creating Layer conv2
    I1227 00:04:26.926234 24337 net.cpp:454] conv2 <- pool1
    I1227 00:04:26.926241 24337 net.cpp:411] conv2 -> conv2
    I1227 00:04:26.926501 24337 net.cpp:150] Setting up conv2
    I1227 00:04:26.926512 24337 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1227 00:04:26.926517 24337 net.cpp:165] Memory required for data: 24651360
    I1227 00:04:26.926527 24337 layer_factory.hpp:77] Creating layer pool2
    I1227 00:04:26.926537 24337 net.cpp:106] Creating Layer pool2
    I1227 00:04:26.926542 24337 net.cpp:454] pool2 <- conv2
    I1227 00:04:26.926549 24337 net.cpp:411] pool2 -> pool2
    I1227 00:04:26.926558 24337 net.cpp:150] Setting up pool2
    I1227 00:04:26.926566 24337 net.cpp:157] Top shape: 120 42 6 6 (181440)
    I1227 00:04:26.926571 24337 net.cpp:165] Memory required for data: 25377120
    I1227 00:04:26.926578 24337 layer_factory.hpp:77] Creating layer relu2
    I1227 00:04:26.926584 24337 net.cpp:106] Creating Layer relu2
    I1227 00:04:26.926590 24337 net.cpp:454] relu2 <- pool2
    I1227 00:04:26.926597 24337 net.cpp:397] relu2 -> pool2 (in-place)
    I1227 00:04:26.926604 24337 net.cpp:150] Setting up relu2
    I1227 00:04:26.926611 24337 net.cpp:157] Top shape: 120 42 6 6 (181440)
    I1227 00:04:26.926616 24337 net.cpp:165] Memory required for data: 26102880
    I1227 00:04:26.926622 24337 layer_factory.hpp:77] Creating layer conv3
    I1227 00:04:26.926631 24337 net.cpp:106] Creating Layer conv3
    I1227 00:04:26.926636 24337 net.cpp:454] conv3 <- pool2
    I1227 00:04:26.926645 24337 net.cpp:411] conv3 -> conv3
    I1227 00:04:26.927132 24337 net.cpp:150] Setting up conv3
    I1227 00:04:26.927144 24337 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 00:04:26.927150 24337 net.cpp:165] Memory required for data: 26225760
    I1227 00:04:26.927160 24337 layer_factory.hpp:77] Creating layer sig1
    I1227 00:04:26.927168 24337 net.cpp:106] Creating Layer sig1
    I1227 00:04:26.927175 24337 net.cpp:454] sig1 <- conv3
    I1227 00:04:26.927181 24337 net.cpp:397] sig1 -> conv3 (in-place)
    I1227 00:04:26.927189 24337 net.cpp:150] Setting up sig1
    I1227 00:04:26.927196 24337 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1227 00:04:26.927202 24337 net.cpp:165] Memory required for data: 26348640
    I1227 00:04:26.927208 24337 layer_factory.hpp:77] Creating layer ip1
    I1227 00:04:26.927220 24337 net.cpp:106] Creating Layer ip1
    I1227 00:04:26.927227 24337 net.cpp:454] ip1 <- conv3
    I1227 00:04:26.927233 24337 net.cpp:411] ip1 -> ip1
    I1227 00:04:26.928355 24337 net.cpp:150] Setting up ip1
    I1227 00:04:26.928383 24337 net.cpp:157] Top shape: 120 512 (61440)
    I1227 00:04:26.928390 24337 net.cpp:165] Memory required for data: 26594400
    I1227 00:04:26.928400 24337 layer_factory.hpp:77] Creating layer sig2
    I1227 00:04:26.928411 24337 net.cpp:106] Creating Layer sig2
    I1227 00:04:26.928417 24337 net.cpp:454] sig2 <- ip1
    I1227 00:04:26.928426 24337 net.cpp:397] sig2 -> ip1 (in-place)
    I1227 00:04:26.928436 24337 net.cpp:150] Setting up sig2
    I1227 00:04:26.928443 24337 net.cpp:157] Top shape: 120 512 (61440)
    I1227 00:04:26.928450 24337 net.cpp:165] Memory required for data: 26840160
    I1227 00:04:26.928506 24337 layer_factory.hpp:77] Creating layer ip2
    I1227 00:04:26.928516 24337 net.cpp:106] Creating Layer ip2
    I1227 00:04:26.928522 24337 net.cpp:454] ip2 <- ip1
    I1227 00:04:26.928531 24337 net.cpp:411] ip2 -> ip2
    I1227 00:04:26.928597 24337 net.cpp:150] Setting up ip2
    I1227 00:04:26.928622 24337 net.cpp:157] Top shape: 120 10 (1200)
    I1227 00:04:26.928632 24337 net.cpp:165] Memory required for data: 26844960
    I1227 00:04:26.928644 24337 layer_factory.hpp:77] Creating layer ip2_ip2_0_split
    I1227 00:04:26.928654 24337 net.cpp:106] Creating Layer ip2_ip2_0_split
    I1227 00:04:26.928661 24337 net.cpp:454] ip2_ip2_0_split <- ip2
    I1227 00:04:26.928669 24337 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_0
    I1227 00:04:26.928679 24337 net.cpp:411] ip2_ip2_0_split -> ip2_ip2_0_split_1
    I1227 00:04:26.928691 24337 net.cpp:150] Setting up ip2_ip2_0_split
    I1227 00:04:26.928699 24337 net.cpp:157] Top shape: 120 10 (1200)
    I1227 00:04:26.928707 24337 net.cpp:157] Top shape: 120 10 (1200)
    I1227 00:04:26.928714 24337 net.cpp:165] Memory required for data: 26854560
    I1227 00:04:26.928719 24337 layer_factory.hpp:77] Creating layer accuracy
    I1227 00:04:26.928728 24337 net.cpp:106] Creating Layer accuracy
    I1227 00:04:26.928735 24337 net.cpp:454] accuracy <- ip2_ip2_0_split_0
    I1227 00:04:26.928743 24337 net.cpp:454] accuracy <- label_data_1_split_0
    I1227 00:04:26.928751 24337 net.cpp:411] accuracy -> accuracy
    I1227 00:04:26.928762 24337 net.cpp:150] Setting up accuracy
    I1227 00:04:26.928771 24337 net.cpp:157] Top shape: (1)
    I1227 00:04:26.928776 24337 net.cpp:165] Memory required for data: 26854564
    I1227 00:04:26.928783 24337 layer_factory.hpp:77] Creating layer loss
    I1227 00:04:26.928792 24337 net.cpp:106] Creating Layer loss
    I1227 00:04:26.928799 24337 net.cpp:454] loss <- ip2_ip2_0_split_1
    I1227 00:04:26.928807 24337 net.cpp:454] loss <- label_data_1_split_1
    I1227 00:04:26.928814 24337 net.cpp:411] loss -> loss
    I1227 00:04:26.928831 24337 layer_factory.hpp:77] Creating layer loss
    I1227 00:04:26.928854 24337 net.cpp:150] Setting up loss
    I1227 00:04:26.928864 24337 net.cpp:157] Top shape: (1)
    I1227 00:04:26.928869 24337 net.cpp:160]     with loss weight 1
    I1227 00:04:26.928897 24337 net.cpp:165] Memory required for data: 26854568
    I1227 00:04:26.928905 24337 net.cpp:226] loss needs backward computation.
    I1227 00:04:26.928911 24337 net.cpp:228] accuracy does not need backward computation.
    I1227 00:04:26.928920 24337 net.cpp:226] ip2_ip2_0_split needs backward computation.
    I1227 00:04:26.928926 24337 net.cpp:226] ip2 needs backward computation.
    I1227 00:04:26.928933 24337 net.cpp:226] sig2 needs backward computation.
    I1227 00:04:26.928941 24337 net.cpp:226] ip1 needs backward computation.
    I1227 00:04:26.928947 24337 net.cpp:226] sig1 needs backward computation.
    I1227 00:04:26.928954 24337 net.cpp:226] conv3 needs backward computation.
    I1227 00:04:26.928961 24337 net.cpp:226] relu2 needs backward computation.
    I1227 00:04:26.928967 24337 net.cpp:226] pool2 needs backward computation.
    I1227 00:04:26.928974 24337 net.cpp:226] conv2 needs backward computation.
    I1227 00:04:26.928982 24337 net.cpp:226] relu1 needs backward computation.
    I1227 00:04:26.928988 24337 net.cpp:226] pool1 needs backward computation.
    I1227 00:04:26.928995 24337 net.cpp:226] conv1 needs backward computation.
    I1227 00:04:26.929003 24337 net.cpp:228] label_data_1_split does not need backward computation.
    I1227 00:04:26.929010 24337 net.cpp:228] data does not need backward computation.
    I1227 00:04:26.929018 24337 net.cpp:270] This network produces output accuracy
    I1227 00:04:26.929024 24337 net.cpp:270] This network produces output loss
    I1227 00:04:26.929042 24337 net.cpp:283] Network initialization done.
    I1227 00:04:26.931120 24337 caffe.cpp:240] Running for 83 iterations.
    I1227 00:04:30.735523 24337 caffe.cpp:264] Batch 0, accuracy = 0.708333
    I1227 00:04:30.735563 24337 caffe.cpp:264] Batch 0, loss = 0.710043
    I1227 00:04:33.917659 24337 caffe.cpp:264] Batch 1, accuracy = 0.725
    I1227 00:04:33.917707 24337 caffe.cpp:264] Batch 1, loss = 0.870084
    I1227 00:04:36.743369 24337 caffe.cpp:264] Batch 2, accuracy = 0.733333
    I1227 00:04:36.743415 24337 caffe.cpp:264] Batch 2, loss = 0.914649
    I1227 00:04:39.655344 24337 caffe.cpp:264] Batch 3, accuracy = 0.7
    I1227 00:04:39.655392 24337 caffe.cpp:264] Batch 3, loss = 0.869065
    I1227 00:04:42.508661 24337 caffe.cpp:264] Batch 4, accuracy = 0.758333
    I1227 00:04:42.508744 24337 caffe.cpp:264] Batch 4, loss = 0.925431
    I1227 00:04:45.465255 24337 caffe.cpp:264] Batch 5, accuracy = 0.675
    I1227 00:04:45.465302 24337 caffe.cpp:264] Batch 5, loss = 0.942123
    I1227 00:04:49.018954 24337 caffe.cpp:264] Batch 6, accuracy = 0.708333
    I1227 00:04:49.019001 24337 caffe.cpp:264] Batch 6, loss = 0.917058
    I1227 00:04:52.129842 24337 caffe.cpp:264] Batch 7, accuracy = 0.8
    I1227 00:04:52.129886 24337 caffe.cpp:264] Batch 7, loss = 0.692102
    I1227 00:04:55.066762 24337 caffe.cpp:264] Batch 8, accuracy = 0.75
    I1227 00:04:55.066807 24337 caffe.cpp:264] Batch 8, loss = 0.798371
    I1227 00:04:58.053598 24337 caffe.cpp:264] Batch 9, accuracy = 0.7
    I1227 00:04:58.053807 24337 caffe.cpp:264] Batch 9, loss = 0.837118
    I1227 00:05:01.057584 24337 caffe.cpp:264] Batch 10, accuracy = 0.675
    I1227 00:05:01.057632 24337 caffe.cpp:264] Batch 10, loss = 0.866327
    I1227 00:05:03.903832 24337 caffe.cpp:264] Batch 11, accuracy = 0.775
    I1227 00:05:03.903879 24337 caffe.cpp:264] Batch 11, loss = 0.700291
    I1227 00:05:06.792104 24337 caffe.cpp:264] Batch 12, accuracy = 0.716667
    I1227 00:05:06.792151 24337 caffe.cpp:264] Batch 12, loss = 0.783378
    I1227 00:05:09.781718 24337 caffe.cpp:264] Batch 13, accuracy = 0.733333
    I1227 00:05:09.781762 24337 caffe.cpp:264] Batch 13, loss = 0.695455
    I1227 00:05:12.695790 24337 caffe.cpp:264] Batch 14, accuracy = 0.75
    I1227 00:05:12.695834 24337 caffe.cpp:264] Batch 14, loss = 0.819243
    I1227 00:05:15.940016 24337 caffe.cpp:264] Batch 15, accuracy = 0.75
    I1227 00:05:15.940062 24337 caffe.cpp:264] Batch 15, loss = 0.759608
    I1227 00:05:18.901291 24337 caffe.cpp:264] Batch 16, accuracy = 0.675
    I1227 00:05:18.901337 24337 caffe.cpp:264] Batch 16, loss = 0.919274
    I1227 00:05:22.238922 24337 caffe.cpp:264] Batch 17, accuracy = 0.691667
    I1227 00:05:22.238965 24337 caffe.cpp:264] Batch 17, loss = 0.818299
    I1227 00:05:25.291561 24337 caffe.cpp:264] Batch 18, accuracy = 0.683333
    I1227 00:05:25.291607 24337 caffe.cpp:264] Batch 18, loss = 0.747129
    I1227 00:05:28.147450 24337 caffe.cpp:264] Batch 19, accuracy = 0.675
    I1227 00:05:28.147539 24337 caffe.cpp:264] Batch 19, loss = 0.917648
    I1227 00:05:31.018852 24337 caffe.cpp:264] Batch 20, accuracy = 0.7
    I1227 00:05:31.018896 24337 caffe.cpp:264] Batch 20, loss = 0.837081
    I1227 00:05:33.901108 24337 caffe.cpp:264] Batch 21, accuracy = 0.675
    I1227 00:05:33.901157 24337 caffe.cpp:264] Batch 21, loss = 0.891673
    I1227 00:05:36.896534 24337 caffe.cpp:264] Batch 22, accuracy = 0.75
    I1227 00:05:36.896584 24337 caffe.cpp:264] Batch 22, loss = 0.873731
    I1227 00:05:40.107789 24337 caffe.cpp:264] Batch 23, accuracy = 0.691667
    I1227 00:05:40.107837 24337 caffe.cpp:264] Batch 23, loss = 0.848864
    I1227 00:05:43.262225 24337 caffe.cpp:264] Batch 24, accuracy = 0.75
    I1227 00:05:43.262276 24337 caffe.cpp:264] Batch 24, loss = 0.728637
    I1227 00:05:46.280230 24337 caffe.cpp:264] Batch 25, accuracy = 0.675
    I1227 00:05:46.280282 24337 caffe.cpp:264] Batch 25, loss = 0.879669
    I1227 00:05:49.653578 24337 caffe.cpp:264] Batch 26, accuracy = 0.733333
    I1227 00:05:49.653625 24337 caffe.cpp:264] Batch 26, loss = 0.793071
    I1227 00:05:52.577879 24337 caffe.cpp:264] Batch 27, accuracy = 0.775
    I1227 00:05:52.577925 24337 caffe.cpp:264] Batch 27, loss = 0.784514
    I1227 00:05:55.556934 24337 caffe.cpp:264] Batch 28, accuracy = 0.733333
    I1227 00:05:55.556983 24337 caffe.cpp:264] Batch 28, loss = 0.868007
    I1227 00:05:58.475047 24337 caffe.cpp:264] Batch 29, accuracy = 0.7
    I1227 00:05:58.475133 24337 caffe.cpp:264] Batch 29, loss = 0.975071
    I1227 00:06:01.369321 24337 caffe.cpp:264] Batch 30, accuracy = 0.716667
    I1227 00:06:01.369371 24337 caffe.cpp:264] Batch 30, loss = 0.855769
    I1227 00:06:04.270915 24337 caffe.cpp:264] Batch 31, accuracy = 0.741667
    I1227 00:06:04.270961 24337 caffe.cpp:264] Batch 31, loss = 0.770391
    I1227 00:06:07.253733 24337 caffe.cpp:264] Batch 32, accuracy = 0.691667
    I1227 00:06:07.253782 24337 caffe.cpp:264] Batch 32, loss = 0.839115
    I1227 00:06:10.211302 24337 caffe.cpp:264] Batch 33, accuracy = 0.625
    I1227 00:06:10.211350 24337 caffe.cpp:264] Batch 33, loss = 0.96152
    I1227 00:06:13.323884 24337 caffe.cpp:264] Batch 34, accuracy = 0.725
    I1227 00:06:13.323935 24337 caffe.cpp:264] Batch 34, loss = 0.828116
    I1227 00:06:16.440819 24337 caffe.cpp:264] Batch 35, accuracy = 0.733333
    I1227 00:06:16.440865 24337 caffe.cpp:264] Batch 35, loss = 0.705592
    I1227 00:06:19.437077 24337 caffe.cpp:264] Batch 36, accuracy = 0.758333
    I1227 00:06:19.437122 24337 caffe.cpp:264] Batch 36, loss = 0.737309
    I1227 00:06:22.446817 24337 caffe.cpp:264] Batch 37, accuracy = 0.675
    I1227 00:06:22.446864 24337 caffe.cpp:264] Batch 37, loss = 0.900963
    I1227 00:06:25.548810 24337 caffe.cpp:264] Batch 38, accuracy = 0.691667
    I1227 00:06:25.548857 24337 caffe.cpp:264] Batch 38, loss = 0.848439
    I1227 00:06:28.796434 24337 caffe.cpp:264] Batch 39, accuracy = 0.758333
    I1227 00:06:28.796612 24337 caffe.cpp:264] Batch 39, loss = 0.798376
    I1227 00:06:31.990365 24337 caffe.cpp:264] Batch 40, accuracy = 0.791667
    I1227 00:06:31.990411 24337 caffe.cpp:264] Batch 40, loss = 0.666435
    I1227 00:06:35.131780 24337 caffe.cpp:264] Batch 41, accuracy = 0.675
    I1227 00:06:35.131829 24337 caffe.cpp:264] Batch 41, loss = 0.93582
    I1227 00:06:38.257607 24337 caffe.cpp:264] Batch 42, accuracy = 0.725
    I1227 00:06:38.257655 24337 caffe.cpp:264] Batch 42, loss = 0.783038
    I1227 00:06:41.283304 24337 caffe.cpp:264] Batch 43, accuracy = 0.716667
    I1227 00:06:41.283355 24337 caffe.cpp:264] Batch 43, loss = 0.85993
    I1227 00:06:44.422250 24337 caffe.cpp:264] Batch 44, accuracy = 0.616667
    I1227 00:06:44.422302 24337 caffe.cpp:264] Batch 44, loss = 1.02436
    I1227 00:06:47.705207 24337 caffe.cpp:264] Batch 45, accuracy = 0.7
    I1227 00:06:47.705255 24337 caffe.cpp:264] Batch 45, loss = 0.787656
    I1227 00:06:50.894853 24337 caffe.cpp:264] Batch 46, accuracy = 0.75
    I1227 00:06:50.894901 24337 caffe.cpp:264] Batch 46, loss = 0.776908
    I1227 00:06:53.956125 24337 caffe.cpp:264] Batch 47, accuracy = 0.691667
    I1227 00:06:53.956173 24337 caffe.cpp:264] Batch 47, loss = 0.867947
    I1227 00:06:56.944283 24337 caffe.cpp:264] Batch 48, accuracy = 0.741667
    I1227 00:06:56.944329 24337 caffe.cpp:264] Batch 48, loss = 0.761871
    I1227 00:06:59.834712 24337 caffe.cpp:264] Batch 49, accuracy = 0.775
    I1227 00:06:59.834789 24337 caffe.cpp:264] Batch 49, loss = 0.729589
    I1227 00:07:02.771567 24337 caffe.cpp:264] Batch 50, accuracy = 0.708333
    I1227 00:07:02.771613 24337 caffe.cpp:264] Batch 50, loss = 0.838298
    I1227 00:07:05.719566 24337 caffe.cpp:264] Batch 51, accuracy = 0.708333
    I1227 00:07:05.719612 24337 caffe.cpp:264] Batch 51, loss = 0.966379
    I1227 00:07:08.643625 24337 caffe.cpp:264] Batch 52, accuracy = 0.783333
    I1227 00:07:08.643662 24337 caffe.cpp:264] Batch 52, loss = 0.685364
    I1227 00:07:11.619843 24337 caffe.cpp:264] Batch 53, accuracy = 0.708333
    I1227 00:07:11.619892 24337 caffe.cpp:264] Batch 53, loss = 0.846044
    I1227 00:07:14.631521 24337 caffe.cpp:264] Batch 54, accuracy = 0.733333
    I1227 00:07:14.631569 24337 caffe.cpp:264] Batch 54, loss = 0.780715
    I1227 00:07:17.598484 24337 caffe.cpp:264] Batch 55, accuracy = 0.758333
    I1227 00:07:17.598531 24337 caffe.cpp:264] Batch 55, loss = 0.727294
    I1227 00:07:20.551568 24337 caffe.cpp:264] Batch 56, accuracy = 0.725
    I1227 00:07:20.551614 24337 caffe.cpp:264] Batch 56, loss = 0.887577
    I1227 00:07:23.417172 24337 caffe.cpp:264] Batch 57, accuracy = 0.641667
    I1227 00:07:23.417218 24337 caffe.cpp:264] Batch 57, loss = 0.932955
    I1227 00:07:26.334501 24337 caffe.cpp:264] Batch 58, accuracy = 0.708333
    I1227 00:07:26.334547 24337 caffe.cpp:264] Batch 58, loss = 0.737736
    I1227 00:07:29.209731 24337 caffe.cpp:264] Batch 59, accuracy = 0.691667
    I1227 00:07:29.209777 24337 caffe.cpp:264] Batch 59, loss = 0.861802
    I1227 00:07:32.126601 24337 caffe.cpp:264] Batch 60, accuracy = 0.625
    I1227 00:07:32.126679 24337 caffe.cpp:264] Batch 60, loss = 0.856307
    I1227 00:07:35.013510 24337 caffe.cpp:264] Batch 61, accuracy = 0.775
    I1227 00:07:35.013557 24337 caffe.cpp:264] Batch 61, loss = 0.709869
    I1227 00:07:37.975203 24337 caffe.cpp:264] Batch 62, accuracy = 0.708333
    I1227 00:07:37.975250 24337 caffe.cpp:264] Batch 62, loss = 0.840898
    I1227 00:07:40.867727 24337 caffe.cpp:264] Batch 63, accuracy = 0.708333
    I1227 00:07:40.867774 24337 caffe.cpp:264] Batch 63, loss = 0.784609
    I1227 00:07:43.753557 24337 caffe.cpp:264] Batch 64, accuracy = 0.691667
    I1227 00:07:43.753595 24337 caffe.cpp:264] Batch 64, loss = 0.818834
    I1227 00:07:46.712244 24337 caffe.cpp:264] Batch 65, accuracy = 0.741667
    I1227 00:07:46.712291 24337 caffe.cpp:264] Batch 65, loss = 0.737379
    I1227 00:07:49.692708 24337 caffe.cpp:264] Batch 66, accuracy = 0.675
    I1227 00:07:49.692752 24337 caffe.cpp:264] Batch 66, loss = 0.891518
    I1227 00:07:52.657735 24337 caffe.cpp:264] Batch 67, accuracy = 0.683333
    I1227 00:07:52.657781 24337 caffe.cpp:264] Batch 67, loss = 1.00494
    I1227 00:07:55.516261 24337 caffe.cpp:264] Batch 68, accuracy = 0.708333
    I1227 00:07:55.516299 24337 caffe.cpp:264] Batch 68, loss = 0.776541
    I1227 00:07:58.401654 24337 caffe.cpp:264] Batch 69, accuracy = 0.708333
    I1227 00:07:58.401698 24337 caffe.cpp:264] Batch 69, loss = 0.843098
    I1227 00:08:01.309397 24337 caffe.cpp:264] Batch 70, accuracy = 0.716667
    I1227 00:08:01.309455 24337 caffe.cpp:264] Batch 70, loss = 0.76641
    I1227 00:08:04.206987 24337 caffe.cpp:264] Batch 71, accuracy = 0.675
    I1227 00:08:04.207165 24337 caffe.cpp:264] Batch 71, loss = 0.911899
    I1227 00:08:07.115375 24337 caffe.cpp:264] Batch 72, accuracy = 0.691667
    I1227 00:08:07.115420 24337 caffe.cpp:264] Batch 72, loss = 0.934983
    I1227 00:08:10.072229 24337 caffe.cpp:264] Batch 73, accuracy = 0.7
    I1227 00:08:10.072275 24337 caffe.cpp:264] Batch 73, loss = 0.657652
    I1227 00:08:12.998819 24337 caffe.cpp:264] Batch 74, accuracy = 0.625
    I1227 00:08:12.998867 24337 caffe.cpp:264] Batch 74, loss = 0.97341
    I1227 00:08:15.879170 24337 caffe.cpp:264] Batch 75, accuracy = 0.775
    I1227 00:08:15.879215 24337 caffe.cpp:264] Batch 75, loss = 0.796351
    I1227 00:08:18.781627 24337 caffe.cpp:264] Batch 76, accuracy = 0.75
    I1227 00:08:18.781674 24337 caffe.cpp:264] Batch 76, loss = 0.717402
    I1227 00:08:21.728068 24337 caffe.cpp:264] Batch 77, accuracy = 0.708333
    I1227 00:08:21.728113 24337 caffe.cpp:264] Batch 77, loss = 0.780208
    I1227 00:08:24.703626 24337 caffe.cpp:264] Batch 78, accuracy = 0.741667
    I1227 00:08:24.703673 24337 caffe.cpp:264] Batch 78, loss = 0.737392
    I1227 00:08:27.629976 24337 caffe.cpp:264] Batch 79, accuracy = 0.691667
    I1227 00:08:27.630024 24337 caffe.cpp:264] Batch 79, loss = 0.870968
    I1227 00:08:30.521668 24337 caffe.cpp:264] Batch 80, accuracy = 0.725
    I1227 00:08:30.521721 24337 caffe.cpp:264] Batch 80, loss = 0.83058
    I1227 00:08:33.445896 24337 caffe.cpp:264] Batch 81, accuracy = 0.716667
    I1227 00:08:33.445945 24337 caffe.cpp:264] Batch 81, loss = 0.860486
    I1227 00:08:36.401772 24337 caffe.cpp:264] Batch 82, accuracy = 0.658333
    I1227 00:08:36.401859 24337 caffe.cpp:264] Batch 82, loss = 0.975929
    I1227 00:08:36.401870 24337 caffe.cpp:269] Loss: 0.828482
    I1227 00:08:36.401882 24337 caffe.cpp:281] accuracy = 0.713654
    I1227 00:08:36.401904 24337 caffe.cpp:281] loss = 0.828482 (* 1 = 0.828482 loss)
    CPU times: user 620 ms, sys: 112 ms, total: 732 ms
    Wall time: 4min 10s


### 71% accuracy
Coffe brewed. The net for sure could be fine tuned some more with better solver parameters.

Let's convert the notebook to github markdown:


```python
!jupyter nbconvert --to markdown custom-cifar-10.ipynb
!mv custom-cifar-10.md README.md
```

    [NbConvertApp] Converting notebook custom-cifar-10.ipynb to markdown
    [NbConvertApp] Writing 404667 bytes to custom-cifar-10.md
