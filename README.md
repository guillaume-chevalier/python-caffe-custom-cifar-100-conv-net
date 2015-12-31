
# Custom cifar-100 conv net with Caffe in Python (Pycaffe)

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
    --2015-12-30 23:48:28--  https://raw.githubusercontent.com/guillaume-chevalier/caffe-cifar-10-and-cifar-100-datasets-preprocessed-to-HDF5/master/download-and-convert-cifar-100.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 23.235.39.133
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|23.235.39.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3526 (3.4K) [text/plain]
    Saving to: ‘download-and-convert-cifar-100.py’

    100%[======================================>] 3,526       --.-K/s   in 0s      

    2015-12-30 23:48:28 (1.25 GB/s) - ‘download-and-convert-cifar-100.py’ saved [3526/3526]

    Downloaded script. Will execute to download and convert the cifar-100 dataset:

    Downloading...
    wget: /root/anaconda2/lib/libcrypto.so.1.0.0: no version information available (required by wget)
    wget: /root/anaconda2/lib/libssl.so.1.0.0: no version information available (required by wget)
    --2015-12-30 23:48:29--  http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30
    Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 169001437 (161M) [application/x-gzip]
    Saving to: ‘cifar-100-python.tar.gz’

    100%[======================================>] 169,001,437 1.23MB/s   in 2m 12s

    2015-12-30 23:50:41 (1.22 MB/s) - ‘cifar-100-python.tar.gz’ saved [169001437/169001437]

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

    CPU times: user 848 ms, sys: 84 ms, total: 932 ms
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
    n.data, n.label_coarse, n.label_fine = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=3)

    n.conv1 = L.Convolution(n.data, kernel_size=4, num_output=64, weight_filler=dict(type='xavier'))
    n.cccp1a = L.Convolution(n.conv1, kernel_size=1, num_output=42, weight_filler=dict(type='xavier'))
    n.relu1a = L.ReLU(n.cccp1a, in_place=True)
    n.cccp1b = L.Convolution(n.relu1a, kernel_size=1, num_output=32, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.cccp1b, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop1 = L.Dropout(n.pool1, in_place=True)
    n.relu1b = L.ReLU(n.drop1, in_place=True)

    n.conv2 = L.Convolution(n.relu1b, kernel_size=4, num_output=42, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.drop2 = L.Dropout(n.pool2, in_place=True)
    n.relu2 = L.ReLU(n.drop2, in_place=True)

    n.conv3 = L.Convolution(n.relu2, kernel_size=2, num_output=64, weight_filler=dict(type='xavier'))
    n.pool3 = L.Pooling(n.conv3, kernel_size=2, stride=2, pool=P.Pooling.AVE)
    n.relu3 = L.ReLU(n.pool3, in_place=True)

    n.ip1 = L.InnerProduct(n.relu3, num_output=768, weight_filler=dict(type='xavier'))
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
     ('cccp1a', (100, 42, 29, 29)),
     ('cccp1b', (100, 32, 29, 29)),
     ('pool1', (100, 32, 14, 14)),
     ('conv2', (100, 42, 11, 11)),
     ('pool2', (100, 42, 5, 5)),
     ('conv3', (100, 64, 4, 4)),
     ('pool3', (100, 64, 2, 2)),
     ('ip1', (100, 768)),
     ('ip1_sig1_0_split_0', (100, 768)),
     ('ip1_sig1_0_split_1', (100, 768)),
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
     ('cccp1a', (42, 64, 1, 1)),
     ('cccp1b', (32, 42, 1, 1)),
     ('conv2', (42, 32, 4, 4)),
     ('conv3', (64, 42, 2, 2)),
     ('ip1', (768, 256)),
     ('ip_c', (20, 768)),
     ('ip_f', (100, 768))]



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

    base_lr: 0.0006
    momentum: 0.0
    weight_decay: 0.001

    lr_policy: "inv"
    gamma: 0.0001
    power: 0.75

    display: 100

    max_iter: 150000

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
    I1230 23:53:02.863142  2138 caffe.cpp:184] Using GPUs 0
    I1230 23:53:03.078757  2138 solver.cpp:48] Initializing solver from parameters:
    train_net: "cnn_train.prototxt"
    test_net: "cnn_test.prototxt"
    test_iter: 100
    test_interval: 1000
    base_lr: 0.0006
    display: 100
    max_iter: 150000
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
    I1230 23:53:03.078974  2138 solver.cpp:81] Creating training net from train_net file: cnn_train.prototxt
    I1230 23:53:03.079375  2138 net.cpp:49] Initializing net from parameters:
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
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
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
      bottom: "cccp1b"
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
      name: "relu1b"
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
        num_output: 768
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
    I1230 23:53:03.080024  2138 layer_factory.hpp:77] Creating layer data
    I1230 23:53:03.080040  2138 net.cpp:106] Creating Layer data
    I1230 23:53:03.080049  2138 net.cpp:411] data -> data
    I1230 23:53:03.080066  2138 net.cpp:411] data -> label_coarse
    I1230 23:53:03.080076  2138 net.cpp:411] data -> label_fine
    I1230 23:53:03.080088  2138 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/train.txt
    I1230 23:53:03.080137  2138 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1230 23:53:03.081096  2138 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1230 23:53:05.449695  2138 net.cpp:150] Setting up data
    I1230 23:53:05.449764  2138 net.cpp:157] Top shape: 100 3 32 32 (307200)
    I1230 23:53:05.449777  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.449789  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.449797  2138 net.cpp:165] Memory required for data: 1229600
    I1230 23:53:05.449825  2138 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1230 23:53:05.449852  2138 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1230 23:53:05.449862  2138 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1230 23:53:05.449878  2138 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1230 23:53:05.449893  2138 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1230 23:53:05.449945  2138 net.cpp:150] Setting up label_coarse_data_1_split
    I1230 23:53:05.449956  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.449965  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.449971  2138 net.cpp:165] Memory required for data: 1230400
    I1230 23:53:05.449980  2138 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1230 23:53:05.449990  2138 net.cpp:106] Creating Layer label_fine_data_2_split
    I1230 23:53:05.449997  2138 net.cpp:454] label_fine_data_2_split <- label_fine
    I1230 23:53:05.450006  2138 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1230 23:53:05.450017  2138 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1230 23:53:05.450062  2138 net.cpp:150] Setting up label_fine_data_2_split
    I1230 23:53:05.450072  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.450078  2138 net.cpp:157] Top shape: 100 (100)
    I1230 23:53:05.450084  2138 net.cpp:165] Memory required for data: 1231200
    I1230 23:53:05.450090  2138 layer_factory.hpp:77] Creating layer conv1
    I1230 23:53:05.450103  2138 net.cpp:106] Creating Layer conv1
    I1230 23:53:05.450109  2138 net.cpp:454] conv1 <- data
    I1230 23:53:05.450116  2138 net.cpp:411] conv1 -> conv1
    I1230 23:53:05.451526  2138 net.cpp:150] Setting up conv1
    I1230 23:53:05.451545  2138 net.cpp:157] Top shape: 100 64 29 29 (5382400)
    I1230 23:53:05.451552  2138 net.cpp:165] Memory required for data: 22760800
    I1230 23:53:05.451567  2138 layer_factory.hpp:77] Creating layer cccp1a
    I1230 23:53:05.451578  2138 net.cpp:106] Creating Layer cccp1a
    I1230 23:53:05.451584  2138 net.cpp:454] cccp1a <- conv1
    I1230 23:53:05.451592  2138 net.cpp:411] cccp1a -> cccp1a
    I1230 23:53:05.451936  2138 net.cpp:150] Setting up cccp1a
    I1230 23:53:05.451962  2138 net.cpp:157] Top shape: 100 42 29 29 (3532200)
    I1230 23:53:05.451979  2138 net.cpp:165] Memory required for data: 36889600
    I1230 23:53:05.452003  2138 layer_factory.hpp:77] Creating layer relu1a
    I1230 23:53:05.452013  2138 net.cpp:106] Creating Layer relu1a
    I1230 23:53:05.452020  2138 net.cpp:454] relu1a <- cccp1a
    I1230 23:53:05.452029  2138 net.cpp:397] relu1a -> cccp1a (in-place)
    I1230 23:53:05.452046  2138 net.cpp:150] Setting up relu1a
    I1230 23:53:05.452055  2138 net.cpp:157] Top shape: 100 42 29 29 (3532200)
    I1230 23:53:05.452061  2138 net.cpp:165] Memory required for data: 51018400
    I1230 23:53:05.452069  2138 layer_factory.hpp:77] Creating layer cccp1b
    I1230 23:53:05.452080  2138 net.cpp:106] Creating Layer cccp1b
    I1230 23:53:05.452097  2138 net.cpp:454] cccp1b <- cccp1a
    I1230 23:53:05.452105  2138 net.cpp:411] cccp1b -> cccp1b
    I1230 23:53:05.452296  2138 net.cpp:150] Setting up cccp1b
    I1230 23:53:05.452307  2138 net.cpp:157] Top shape: 100 32 29 29 (2691200)
    I1230 23:53:05.452314  2138 net.cpp:165] Memory required for data: 61783200
    I1230 23:53:05.452325  2138 layer_factory.hpp:77] Creating layer pool1
    I1230 23:53:05.452335  2138 net.cpp:106] Creating Layer pool1
    I1230 23:53:05.452342  2138 net.cpp:454] pool1 <- cccp1b
    I1230 23:53:05.452350  2138 net.cpp:411] pool1 -> pool1
    I1230 23:53:05.452407  2138 net.cpp:150] Setting up pool1
    I1230 23:53:05.452415  2138 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 23:53:05.452441  2138 net.cpp:165] Memory required for data: 64292000
    I1230 23:53:05.452447  2138 layer_factory.hpp:77] Creating layer drop1
    I1230 23:53:05.452461  2138 net.cpp:106] Creating Layer drop1
    I1230 23:53:05.452466  2138 net.cpp:454] drop1 <- pool1
    I1230 23:53:05.452473  2138 net.cpp:397] drop1 -> pool1 (in-place)
    I1230 23:53:05.452496  2138 net.cpp:150] Setting up drop1
    I1230 23:53:05.452503  2138 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 23:53:05.452509  2138 net.cpp:165] Memory required for data: 66800800
    I1230 23:53:05.452525  2138 layer_factory.hpp:77] Creating layer relu1b
    I1230 23:53:05.452534  2138 net.cpp:106] Creating Layer relu1b
    I1230 23:53:05.452540  2138 net.cpp:454] relu1b <- pool1
    I1230 23:53:05.452548  2138 net.cpp:397] relu1b -> pool1 (in-place)
    I1230 23:53:05.452556  2138 net.cpp:150] Setting up relu1b
    I1230 23:53:05.452574  2138 net.cpp:157] Top shape: 100 32 14 14 (627200)
    I1230 23:53:05.452579  2138 net.cpp:165] Memory required for data: 69309600
    I1230 23:53:05.452584  2138 layer_factory.hpp:77] Creating layer conv2
    I1230 23:53:05.452594  2138 net.cpp:106] Creating Layer conv2
    I1230 23:53:05.452599  2138 net.cpp:454] conv2 <- pool1
    I1230 23:53:05.452605  2138 net.cpp:411] conv2 -> conv2
    I1230 23:53:05.452957  2138 net.cpp:150] Setting up conv2
    I1230 23:53:05.452980  2138 net.cpp:157] Top shape: 100 42 11 11 (508200)
    I1230 23:53:05.452986  2138 net.cpp:165] Memory required for data: 71342400
    I1230 23:53:05.452996  2138 layer_factory.hpp:77] Creating layer pool2
    I1230 23:53:05.453006  2138 net.cpp:106] Creating Layer pool2
    I1230 23:53:05.453011  2138 net.cpp:454] pool2 <- conv2
    I1230 23:53:05.453019  2138 net.cpp:411] pool2 -> pool2
    I1230 23:53:05.453060  2138 net.cpp:150] Setting up pool2
    I1230 23:53:05.453068  2138 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 23:53:05.453073  2138 net.cpp:165] Memory required for data: 71762400
    I1230 23:53:05.453079  2138 layer_factory.hpp:77] Creating layer drop2
    I1230 23:53:05.453086  2138 net.cpp:106] Creating Layer drop2
    I1230 23:53:05.453091  2138 net.cpp:454] drop2 <- pool2
    I1230 23:53:05.453099  2138 net.cpp:397] drop2 -> pool2 (in-place)
    I1230 23:53:05.453116  2138 net.cpp:150] Setting up drop2
    I1230 23:53:05.453125  2138 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 23:53:05.453130  2138 net.cpp:165] Memory required for data: 72182400
    I1230 23:53:05.453135  2138 layer_factory.hpp:77] Creating layer relu2
    I1230 23:53:05.453142  2138 net.cpp:106] Creating Layer relu2
    I1230 23:53:05.453147  2138 net.cpp:454] relu2 <- pool2
    I1230 23:53:05.453155  2138 net.cpp:397] relu2 -> pool2 (in-place)
    I1230 23:53:05.453171  2138 net.cpp:150] Setting up relu2
    I1230 23:53:05.453178  2138 net.cpp:157] Top shape: 100 42 5 5 (105000)
    I1230 23:53:05.453184  2138 net.cpp:165] Memory required for data: 72602400
    I1230 23:53:05.453191  2138 layer_factory.hpp:77] Creating layer conv3
    I1230 23:53:05.453199  2138 net.cpp:106] Creating Layer conv3
    I1230 23:53:05.453205  2138 net.cpp:454] conv3 <- pool2
    I1230 23:53:05.453224  2138 net.cpp:411] conv3 -> conv3
    I1230 23:53:05.454000  2138 net.cpp:150] Setting up conv3
    I1230 23:53:05.454030  2138 net.cpp:157] Top shape: 100 64 4 4 (102400)
    I1230 23:53:05.454038  2138 net.cpp:165] Memory required for data: 73012000
    I1230 23:53:05.454052  2138 layer_factory.hpp:77] Creating layer pool3
    I1230 23:53:05.454074  2138 net.cpp:106] Creating Layer pool3
    I1230 23:53:05.454082  2138 net.cpp:454] pool3 <- conv3
    I1230 23:53:05.454099  2138 net.cpp:411] pool3 -> pool3
    I1230 23:53:05.454129  2138 net.cpp:150] Setting up pool3
    I1230 23:53:05.454138  2138 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1230 23:53:05.454143  2138 net.cpp:165] Memory required for data: 73114400
    I1230 23:53:05.454147  2138 layer_factory.hpp:77] Creating layer relu3
    I1230 23:53:05.454155  2138 net.cpp:106] Creating Layer relu3
    I1230 23:53:05.454161  2138 net.cpp:454] relu3 <- pool3
    I1230 23:53:05.454169  2138 net.cpp:397] relu3 -> pool3 (in-place)
    I1230 23:53:05.454176  2138 net.cpp:150] Setting up relu3
    I1230 23:53:05.454193  2138 net.cpp:157] Top shape: 100 64 2 2 (25600)
    I1230 23:53:05.454210  2138 net.cpp:165] Memory required for data: 73216800
    I1230 23:53:05.454216  2138 layer_factory.hpp:77] Creating layer ip1
    I1230 23:53:05.454228  2138 net.cpp:106] Creating Layer ip1
    I1230 23:53:05.454244  2138 net.cpp:454] ip1 <- pool3
    I1230 23:53:05.454252  2138 net.cpp:411] ip1 -> ip1
    I1230 23:53:05.456027  2138 net.cpp:150] Setting up ip1
    I1230 23:53:05.456053  2138 net.cpp:157] Top shape: 100 768 (76800)
    I1230 23:53:05.456058  2138 net.cpp:165] Memory required for data: 73524000
    I1230 23:53:05.456068  2138 layer_factory.hpp:77] Creating layer sig1
    I1230 23:53:05.456086  2138 net.cpp:106] Creating Layer sig1
    I1230 23:53:05.456092  2138 net.cpp:454] sig1 <- ip1
    I1230 23:53:05.456099  2138 net.cpp:397] sig1 -> ip1 (in-place)
    I1230 23:53:05.456106  2138 net.cpp:150] Setting up sig1
    I1230 23:53:05.456112  2138 net.cpp:157] Top shape: 100 768 (76800)
    I1230 23:53:05.456117  2138 net.cpp:165] Memory required for data: 73831200
    I1230 23:53:05.456122  2138 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1230 23:53:05.456130  2138 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1230 23:53:05.456135  2138 net.cpp:454] ip1_sig1_0_split <- ip1
    I1230 23:53:05.456141  2138 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1230 23:53:05.456151  2138 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1230 23:53:05.456182  2138 net.cpp:150] Setting up ip1_sig1_0_split
    I1230 23:53:05.456190  2138 net.cpp:157] Top shape: 100 768 (76800)
    I1230 23:53:05.456197  2138 net.cpp:157] Top shape: 100 768 (76800)
    I1230 23:53:05.456202  2138 net.cpp:165] Memory required for data: 74445600
    I1230 23:53:05.456207  2138 layer_factory.hpp:77] Creating layer ip_c
    I1230 23:53:05.456213  2138 net.cpp:106] Creating Layer ip_c
    I1230 23:53:05.456229  2138 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1230 23:53:05.456236  2138 net.cpp:411] ip_c -> ip_c
    I1230 23:53:05.456428  2138 net.cpp:150] Setting up ip_c
    I1230 23:53:05.456437  2138 net.cpp:157] Top shape: 100 20 (2000)
    I1230 23:53:05.456444  2138 net.cpp:165] Memory required for data: 74453600
    I1230 23:53:05.456451  2138 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1230 23:53:05.456459  2138 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1230 23:53:05.456475  2138 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1230 23:53:05.456481  2138 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1230 23:53:05.456488  2138 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1230 23:53:05.456516  2138 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1230 23:53:05.456523  2138 net.cpp:157] Top shape: 100 20 (2000)
    I1230 23:53:05.456529  2138 net.cpp:157] Top shape: 100 20 (2000)
    I1230 23:53:05.456534  2138 net.cpp:165] Memory required for data: 74469600
    I1230 23:53:05.456540  2138 layer_factory.hpp:77] Creating layer accuracy_c
    I1230 23:53:05.456547  2138 net.cpp:106] Creating Layer accuracy_c
    I1230 23:53:05.456553  2138 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1230 23:53:05.456559  2138 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1230 23:53:05.456567  2138 net.cpp:411] accuracy_c -> accuracy_c
    I1230 23:53:05.456574  2138 net.cpp:150] Setting up accuracy_c
    I1230 23:53:05.456581  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.456586  2138 net.cpp:165] Memory required for data: 74469604
    I1230 23:53:05.456591  2138 layer_factory.hpp:77] Creating layer loss_c
    I1230 23:53:05.456598  2138 net.cpp:106] Creating Layer loss_c
    I1230 23:53:05.456614  2138 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1230 23:53:05.456619  2138 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1230 23:53:05.456626  2138 net.cpp:411] loss_c -> loss_c
    I1230 23:53:05.456637  2138 layer_factory.hpp:77] Creating layer loss_c
    I1230 23:53:05.456724  2138 net.cpp:150] Setting up loss_c
    I1230 23:53:05.456732  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.456738  2138 net.cpp:160]     with loss weight 1
    I1230 23:53:05.456754  2138 net.cpp:165] Memory required for data: 74469608
    I1230 23:53:05.456759  2138 layer_factory.hpp:77] Creating layer ip_f
    I1230 23:53:05.456779  2138 net.cpp:106] Creating Layer ip_f
    I1230 23:53:05.456795  2138 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1230 23:53:05.456802  2138 net.cpp:411] ip_f -> ip_f
    I1230 23:53:05.457996  2138 net.cpp:150] Setting up ip_f
    I1230 23:53:05.458011  2138 net.cpp:157] Top shape: 100 100 (10000)
    I1230 23:53:05.458017  2138 net.cpp:165] Memory required for data: 74509608
    I1230 23:53:05.458025  2138 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1230 23:53:05.458034  2138 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1230 23:53:05.458051  2138 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1230 23:53:05.458060  2138 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1230 23:53:05.458081  2138 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1230 23:53:05.458132  2138 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1230 23:53:05.458140  2138 net.cpp:157] Top shape: 100 100 (10000)
    I1230 23:53:05.458148  2138 net.cpp:157] Top shape: 100 100 (10000)
    I1230 23:53:05.458153  2138 net.cpp:165] Memory required for data: 74589608
    I1230 23:53:05.458159  2138 layer_factory.hpp:77] Creating layer accuracy_f
    I1230 23:53:05.458169  2138 net.cpp:106] Creating Layer accuracy_f
    I1230 23:53:05.458175  2138 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1230 23:53:05.458184  2138 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1230 23:53:05.458191  2138 net.cpp:411] accuracy_f -> accuracy_f
    I1230 23:53:05.458201  2138 net.cpp:150] Setting up accuracy_f
    I1230 23:53:05.458209  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.458215  2138 net.cpp:165] Memory required for data: 74589612
    I1230 23:53:05.458222  2138 layer_factory.hpp:77] Creating layer loss_f
    I1230 23:53:05.458230  2138 net.cpp:106] Creating Layer loss_f
    I1230 23:53:05.458236  2138 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1230 23:53:05.458245  2138 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1230 23:53:05.458251  2138 net.cpp:411] loss_f -> loss_f
    I1230 23:53:05.458262  2138 layer_factory.hpp:77] Creating layer loss_f
    I1230 23:53:05.458350  2138 net.cpp:150] Setting up loss_f
    I1230 23:53:05.458360  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.458366  2138 net.cpp:160]     with loss weight 1
    I1230 23:53:05.458377  2138 net.cpp:165] Memory required for data: 74589616
    I1230 23:53:05.458384  2138 net.cpp:226] loss_f needs backward computation.
    I1230 23:53:05.458390  2138 net.cpp:228] accuracy_f does not need backward computation.
    I1230 23:53:05.458397  2138 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1230 23:53:05.458405  2138 net.cpp:226] ip_f needs backward computation.
    I1230 23:53:05.458411  2138 net.cpp:226] loss_c needs backward computation.
    I1230 23:53:05.458418  2138 net.cpp:228] accuracy_c does not need backward computation.
    I1230 23:53:05.458425  2138 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1230 23:53:05.458432  2138 net.cpp:226] ip_c needs backward computation.
    I1230 23:53:05.458438  2138 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1230 23:53:05.458446  2138 net.cpp:226] sig1 needs backward computation.
    I1230 23:53:05.458451  2138 net.cpp:226] ip1 needs backward computation.
    I1230 23:53:05.458458  2138 net.cpp:226] relu3 needs backward computation.
    I1230 23:53:05.458464  2138 net.cpp:226] pool3 needs backward computation.
    I1230 23:53:05.458472  2138 net.cpp:226] conv3 needs backward computation.
    I1230 23:53:05.458478  2138 net.cpp:226] relu2 needs backward computation.
    I1230 23:53:05.458485  2138 net.cpp:226] drop2 needs backward computation.
    I1230 23:53:05.458492  2138 net.cpp:226] pool2 needs backward computation.
    I1230 23:53:05.458498  2138 net.cpp:226] conv2 needs backward computation.
    I1230 23:53:05.458505  2138 net.cpp:226] relu1b needs backward computation.
    I1230 23:53:05.458513  2138 net.cpp:226] drop1 needs backward computation.
    I1230 23:53:05.458519  2138 net.cpp:226] pool1 needs backward computation.
    I1230 23:53:05.458530  2138 net.cpp:226] cccp1b needs backward computation.
    I1230 23:53:05.458540  2138 net.cpp:226] relu1a needs backward computation.
    I1230 23:53:05.458569  2138 net.cpp:226] cccp1a needs backward computation.
    I1230 23:53:05.458590  2138 net.cpp:226] conv1 needs backward computation.
    I1230 23:53:05.458605  2138 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1230 23:53:05.458619  2138 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1230 23:53:05.458629  2138 net.cpp:228] data does not need backward computation.
    I1230 23:53:05.458636  2138 net.cpp:270] This network produces output accuracy_c
    I1230 23:53:05.458644  2138 net.cpp:270] This network produces output accuracy_f
    I1230 23:53:05.458663  2138 net.cpp:270] This network produces output loss_c
    I1230 23:53:05.458669  2138 net.cpp:270] This network produces output loss_f
    I1230 23:53:05.458696  2138 net.cpp:283] Network initialization done.
    I1230 23:53:05.459130  2138 solver.cpp:181] Creating test net (#0) specified by test_net file: cnn_test.prototxt
    I1230 23:53:05.459292  2138 net.cpp:49] Initializing net from parameters:
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
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
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
      bottom: "cccp1b"
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
      name: "relu1b"
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
        num_output: 768
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
    I1230 23:53:05.460078  2138 layer_factory.hpp:77] Creating layer data
    I1230 23:53:05.460093  2138 net.cpp:106] Creating Layer data
    I1230 23:53:05.460100  2138 net.cpp:411] data -> data
    I1230 23:53:05.460114  2138 net.cpp:411] data -> label_coarse
    I1230 23:53:05.460122  2138 net.cpp:411] data -> label_fine
    I1230 23:53:05.460134  2138 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/test.txt
    I1230 23:53:05.460160  2138 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1230 23:53:05.847424  2138 net.cpp:150] Setting up data
    I1230 23:53:05.847476  2138 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1230 23:53:05.847486  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847494  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847501  2138 net.cpp:165] Memory required for data: 1475520
    I1230 23:53:05.847512  2138 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1230 23:53:05.847528  2138 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1230 23:53:05.847537  2138 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1230 23:53:05.847549  2138 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1230 23:53:05.847563  2138 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1230 23:53:05.847609  2138 net.cpp:150] Setting up label_coarse_data_1_split
    I1230 23:53:05.847620  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847628  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847635  2138 net.cpp:165] Memory required for data: 1476480
    I1230 23:53:05.847642  2138 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1230 23:53:05.847664  2138 net.cpp:106] Creating Layer label_fine_data_2_split
    I1230 23:53:05.847676  2138 net.cpp:454] label_fine_data_2_split <- label_fine
    I1230 23:53:05.847687  2138 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1230 23:53:05.847697  2138 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1230 23:53:05.847735  2138 net.cpp:150] Setting up label_fine_data_2_split
    I1230 23:53:05.847746  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847754  2138 net.cpp:157] Top shape: 120 (120)
    I1230 23:53:05.847760  2138 net.cpp:165] Memory required for data: 1477440
    I1230 23:53:05.847767  2138 layer_factory.hpp:77] Creating layer conv1
    I1230 23:53:05.847780  2138 net.cpp:106] Creating Layer conv1
    I1230 23:53:05.847787  2138 net.cpp:454] conv1 <- data
    I1230 23:53:05.847797  2138 net.cpp:411] conv1 -> conv1
    I1230 23:53:05.848098  2138 net.cpp:150] Setting up conv1
    I1230 23:53:05.848124  2138 net.cpp:157] Top shape: 120 64 29 29 (6458880)
    I1230 23:53:05.848136  2138 net.cpp:165] Memory required for data: 27312960
    I1230 23:53:05.848160  2138 layer_factory.hpp:77] Creating layer cccp1a
    I1230 23:53:05.848181  2138 net.cpp:106] Creating Layer cccp1a
    I1230 23:53:05.848194  2138 net.cpp:454] cccp1a <- conv1
    I1230 23:53:05.848212  2138 net.cpp:411] cccp1a -> cccp1a
    I1230 23:53:05.848475  2138 net.cpp:150] Setting up cccp1a
    I1230 23:53:05.848500  2138 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1230 23:53:05.848510  2138 net.cpp:165] Memory required for data: 44267520
    I1230 23:53:05.848529  2138 layer_factory.hpp:77] Creating layer relu1a
    I1230 23:53:05.848546  2138 net.cpp:106] Creating Layer relu1a
    I1230 23:53:05.848558  2138 net.cpp:454] relu1a <- cccp1a
    I1230 23:53:05.848573  2138 net.cpp:397] relu1a -> cccp1a (in-place)
    I1230 23:53:05.848592  2138 net.cpp:150] Setting up relu1a
    I1230 23:53:05.848605  2138 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1230 23:53:05.848618  2138 net.cpp:165] Memory required for data: 61222080
    I1230 23:53:05.848628  2138 layer_factory.hpp:77] Creating layer cccp1b
    I1230 23:53:05.848645  2138 net.cpp:106] Creating Layer cccp1b
    I1230 23:53:05.848657  2138 net.cpp:454] cccp1b <- cccp1a
    I1230 23:53:05.848673  2138 net.cpp:411] cccp1b -> cccp1b
    I1230 23:53:05.848997  2138 net.cpp:150] Setting up cccp1b
    I1230 23:53:05.849027  2138 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1230 23:53:05.849040  2138 net.cpp:165] Memory required for data: 74139840
    I1230 23:53:05.849089  2138 layer_factory.hpp:77] Creating layer pool1
    I1230 23:53:05.849109  2138 net.cpp:106] Creating Layer pool1
    I1230 23:53:05.849122  2138 net.cpp:454] pool1 <- cccp1b
    I1230 23:53:05.849138  2138 net.cpp:411] pool1 -> pool1
    I1230 23:53:05.849205  2138 net.cpp:150] Setting up pool1
    I1230 23:53:05.849222  2138 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 23:53:05.849236  2138 net.cpp:165] Memory required for data: 77150400
    I1230 23:53:05.849246  2138 layer_factory.hpp:77] Creating layer drop1
    I1230 23:53:05.849266  2138 net.cpp:106] Creating Layer drop1
    I1230 23:53:05.849278  2138 net.cpp:454] drop1 <- pool1
    I1230 23:53:05.849294  2138 net.cpp:397] drop1 -> pool1 (in-place)
    I1230 23:53:05.849339  2138 net.cpp:150] Setting up drop1
    I1230 23:53:05.849360  2138 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 23:53:05.849373  2138 net.cpp:165] Memory required for data: 80160960
    I1230 23:53:05.849385  2138 layer_factory.hpp:77] Creating layer relu1b
    I1230 23:53:05.849400  2138 net.cpp:106] Creating Layer relu1b
    I1230 23:53:05.849412  2138 net.cpp:454] relu1b <- pool1
    I1230 23:53:05.849429  2138 net.cpp:397] relu1b -> pool1 (in-place)
    I1230 23:53:05.849448  2138 net.cpp:150] Setting up relu1b
    I1230 23:53:05.849475  2138 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1230 23:53:05.849485  2138 net.cpp:165] Memory required for data: 83171520
    I1230 23:53:05.849498  2138 layer_factory.hpp:77] Creating layer conv2
    I1230 23:53:05.849519  2138 net.cpp:106] Creating Layer conv2
    I1230 23:53:05.849532  2138 net.cpp:454] conv2 <- pool1
    I1230 23:53:05.849551  2138 net.cpp:411] conv2 -> conv2
    I1230 23:53:05.850190  2138 net.cpp:150] Setting up conv2
    I1230 23:53:05.850247  2138 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1230 23:53:05.850260  2138 net.cpp:165] Memory required for data: 85610880
    I1230 23:53:05.850285  2138 layer_factory.hpp:77] Creating layer pool2
    I1230 23:53:05.850307  2138 net.cpp:106] Creating Layer pool2
    I1230 23:53:05.850324  2138 net.cpp:454] pool2 <- conv2
    I1230 23:53:05.850345  2138 net.cpp:411] pool2 -> pool2
    I1230 23:53:05.850427  2138 net.cpp:150] Setting up pool2
    I1230 23:53:05.850450  2138 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 23:53:05.850462  2138 net.cpp:165] Memory required for data: 86114880
    I1230 23:53:05.850473  2138 layer_factory.hpp:77] Creating layer drop2
    I1230 23:53:05.850489  2138 net.cpp:106] Creating Layer drop2
    I1230 23:53:05.850502  2138 net.cpp:454] drop2 <- pool2
    I1230 23:53:05.850515  2138 net.cpp:397] drop2 -> pool2 (in-place)
    I1230 23:53:05.850559  2138 net.cpp:150] Setting up drop2
    I1230 23:53:05.850575  2138 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 23:53:05.850587  2138 net.cpp:165] Memory required for data: 86618880
    I1230 23:53:05.850600  2138 layer_factory.hpp:77] Creating layer relu2
    I1230 23:53:05.850617  2138 net.cpp:106] Creating Layer relu2
    I1230 23:53:05.850628  2138 net.cpp:454] relu2 <- pool2
    I1230 23:53:05.850641  2138 net.cpp:397] relu2 -> pool2 (in-place)
    I1230 23:53:05.850654  2138 net.cpp:150] Setting up relu2
    I1230 23:53:05.850667  2138 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1230 23:53:05.850677  2138 net.cpp:165] Memory required for data: 87122880
    I1230 23:53:05.850687  2138 layer_factory.hpp:77] Creating layer conv3
    I1230 23:53:05.850704  2138 net.cpp:106] Creating Layer conv3
    I1230 23:53:05.850715  2138 net.cpp:454] conv3 <- pool2
    I1230 23:53:05.850729  2138 net.cpp:411] conv3 -> conv3
    I1230 23:53:05.851186  2138 net.cpp:150] Setting up conv3
    I1230 23:53:05.851227  2138 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1230 23:53:05.851238  2138 net.cpp:165] Memory required for data: 87614400
    I1230 23:53:05.851265  2138 layer_factory.hpp:77] Creating layer pool3
    I1230 23:53:05.851285  2138 net.cpp:106] Creating Layer pool3
    I1230 23:53:05.851297  2138 net.cpp:454] pool3 <- conv3
    I1230 23:53:05.851315  2138 net.cpp:411] pool3 -> pool3
    I1230 23:53:05.851369  2138 net.cpp:150] Setting up pool3
    I1230 23:53:05.851389  2138 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 23:53:05.851402  2138 net.cpp:165] Memory required for data: 87737280
    I1230 23:53:05.851444  2138 layer_factory.hpp:77] Creating layer relu3
    I1230 23:53:05.851466  2138 net.cpp:106] Creating Layer relu3
    I1230 23:53:05.851480  2138 net.cpp:454] relu3 <- pool3
    I1230 23:53:05.851495  2138 net.cpp:397] relu3 -> pool3 (in-place)
    I1230 23:53:05.851512  2138 net.cpp:150] Setting up relu3
    I1230 23:53:05.851526  2138 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1230 23:53:05.851536  2138 net.cpp:165] Memory required for data: 87860160
    I1230 23:53:05.851548  2138 layer_factory.hpp:77] Creating layer ip1
    I1230 23:53:05.851565  2138 net.cpp:106] Creating Layer ip1
    I1230 23:53:05.851578  2138 net.cpp:454] ip1 <- pool3
    I1230 23:53:05.851594  2138 net.cpp:411] ip1 -> ip1
    I1230 23:53:05.855173  2138 net.cpp:150] Setting up ip1
    I1230 23:53:05.855253  2138 net.cpp:157] Top shape: 120 768 (92160)
    I1230 23:53:05.855278  2138 net.cpp:165] Memory required for data: 88228800
    I1230 23:53:05.855307  2138 layer_factory.hpp:77] Creating layer sig1
    I1230 23:53:05.855342  2138 net.cpp:106] Creating Layer sig1
    I1230 23:53:05.855365  2138 net.cpp:454] sig1 <- ip1
    I1230 23:53:05.855387  2138 net.cpp:397] sig1 -> ip1 (in-place)
    I1230 23:53:05.855414  2138 net.cpp:150] Setting up sig1
    I1230 23:53:05.855434  2138 net.cpp:157] Top shape: 120 768 (92160)
    I1230 23:53:05.855448  2138 net.cpp:165] Memory required for data: 88597440
    I1230 23:53:05.855475  2138 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1230 23:53:05.855504  2138 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1230 23:53:05.855514  2138 net.cpp:454] ip1_sig1_0_split <- ip1
    I1230 23:53:05.855540  2138 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1230 23:53:05.855566  2138 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1230 23:53:05.855695  2138 net.cpp:150] Setting up ip1_sig1_0_split
    I1230 23:53:05.855710  2138 net.cpp:157] Top shape: 120 768 (92160)
    I1230 23:53:05.855718  2138 net.cpp:157] Top shape: 120 768 (92160)
    I1230 23:53:05.855736  2138 net.cpp:165] Memory required for data: 89334720
    I1230 23:53:05.855748  2138 layer_factory.hpp:77] Creating layer ip_c
    I1230 23:53:05.855778  2138 net.cpp:106] Creating Layer ip_c
    I1230 23:53:05.855792  2138 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1230 23:53:05.855811  2138 net.cpp:411] ip_c -> ip_c
    I1230 23:53:05.856233  2138 net.cpp:150] Setting up ip_c
    I1230 23:53:05.856254  2138 net.cpp:157] Top shape: 120 20 (2400)
    I1230 23:53:05.856262  2138 net.cpp:165] Memory required for data: 89344320
    I1230 23:53:05.856274  2138 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1230 23:53:05.856286  2138 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1230 23:53:05.856293  2138 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1230 23:53:05.856302  2138 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1230 23:53:05.856324  2138 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1230 23:53:05.856359  2138 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1230 23:53:05.856369  2138 net.cpp:157] Top shape: 120 20 (2400)
    I1230 23:53:05.856375  2138 net.cpp:157] Top shape: 120 20 (2400)
    I1230 23:53:05.856382  2138 net.cpp:165] Memory required for data: 89363520
    I1230 23:53:05.856389  2138 layer_factory.hpp:77] Creating layer accuracy_c
    I1230 23:53:05.856398  2138 net.cpp:106] Creating Layer accuracy_c
    I1230 23:53:05.856405  2138 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1230 23:53:05.856413  2138 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1230 23:53:05.856421  2138 net.cpp:411] accuracy_c -> accuracy_c
    I1230 23:53:05.856431  2138 net.cpp:150] Setting up accuracy_c
    I1230 23:53:05.856439  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.856446  2138 net.cpp:165] Memory required for data: 89363524
    I1230 23:53:05.856451  2138 layer_factory.hpp:77] Creating layer loss_c
    I1230 23:53:05.856462  2138 net.cpp:106] Creating Layer loss_c
    I1230 23:53:05.856468  2138 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1230 23:53:05.856475  2138 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1230 23:53:05.856483  2138 net.cpp:411] loss_c -> loss_c
    I1230 23:53:05.856494  2138 layer_factory.hpp:77] Creating layer loss_c
    I1230 23:53:05.856616  2138 net.cpp:150] Setting up loss_c
    I1230 23:53:05.856626  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.856632  2138 net.cpp:160]     with loss weight 1
    I1230 23:53:05.856645  2138 net.cpp:165] Memory required for data: 89363528
    I1230 23:53:05.856652  2138 layer_factory.hpp:77] Creating layer ip_f
    I1230 23:53:05.856662  2138 net.cpp:106] Creating Layer ip_f
    I1230 23:53:05.856668  2138 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1230 23:53:05.856678  2138 net.cpp:411] ip_f -> ip_f
    I1230 23:53:05.857385  2138 net.cpp:150] Setting up ip_f
    I1230 23:53:05.857400  2138 net.cpp:157] Top shape: 120 100 (12000)
    I1230 23:53:05.857406  2138 net.cpp:165] Memory required for data: 89411528
    I1230 23:53:05.857417  2138 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1230 23:53:05.857426  2138 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1230 23:53:05.857434  2138 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1230 23:53:05.857441  2138 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1230 23:53:05.857462  2138 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1230 23:53:05.857502  2138 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1230 23:53:05.857522  2138 net.cpp:157] Top shape: 120 100 (12000)
    I1230 23:53:05.857528  2138 net.cpp:157] Top shape: 120 100 (12000)
    I1230 23:53:05.857534  2138 net.cpp:165] Memory required for data: 89507528
    I1230 23:53:05.857542  2138 layer_factory.hpp:77] Creating layer accuracy_f
    I1230 23:53:05.857550  2138 net.cpp:106] Creating Layer accuracy_f
    I1230 23:53:05.857558  2138 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1230 23:53:05.857565  2138 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1230 23:53:05.857573  2138 net.cpp:411] accuracy_f -> accuracy_f
    I1230 23:53:05.857583  2138 net.cpp:150] Setting up accuracy_f
    I1230 23:53:05.857602  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.857609  2138 net.cpp:165] Memory required for data: 89507532
    I1230 23:53:05.857616  2138 layer_factory.hpp:77] Creating layer loss_f
    I1230 23:53:05.857625  2138 net.cpp:106] Creating Layer loss_f
    I1230 23:53:05.857632  2138 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1230 23:53:05.857641  2138 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1230 23:53:05.857650  2138 net.cpp:411] loss_f -> loss_f
    I1230 23:53:05.857661  2138 layer_factory.hpp:77] Creating layer loss_f
    I1230 23:53:05.857780  2138 net.cpp:150] Setting up loss_f
    I1230 23:53:05.857791  2138 net.cpp:157] Top shape: (1)
    I1230 23:53:05.857797  2138 net.cpp:160]     with loss weight 1
    I1230 23:53:05.857807  2138 net.cpp:165] Memory required for data: 89507536
    I1230 23:53:05.857815  2138 net.cpp:226] loss_f needs backward computation.
    I1230 23:53:05.857821  2138 net.cpp:228] accuracy_f does not need backward computation.
    I1230 23:53:05.857830  2138 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1230 23:53:05.857836  2138 net.cpp:226] ip_f needs backward computation.
    I1230 23:53:05.857843  2138 net.cpp:226] loss_c needs backward computation.
    I1230 23:53:05.857851  2138 net.cpp:228] accuracy_c does not need backward computation.
    I1230 23:53:05.857858  2138 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1230 23:53:05.857866  2138 net.cpp:226] ip_c needs backward computation.
    I1230 23:53:05.857872  2138 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1230 23:53:05.857879  2138 net.cpp:226] sig1 needs backward computation.
    I1230 23:53:05.857887  2138 net.cpp:226] ip1 needs backward computation.
    I1230 23:53:05.857893  2138 net.cpp:226] relu3 needs backward computation.
    I1230 23:53:05.857900  2138 net.cpp:226] pool3 needs backward computation.
    I1230 23:53:05.857908  2138 net.cpp:226] conv3 needs backward computation.
    I1230 23:53:05.857913  2138 net.cpp:226] relu2 needs backward computation.
    I1230 23:53:05.857945  2138 net.cpp:226] drop2 needs backward computation.
    I1230 23:53:05.857954  2138 net.cpp:226] pool2 needs backward computation.
    I1230 23:53:05.857960  2138 net.cpp:226] conv2 needs backward computation.
    I1230 23:53:05.857967  2138 net.cpp:226] relu1b needs backward computation.
    I1230 23:53:05.857991  2138 net.cpp:226] drop1 needs backward computation.
    I1230 23:53:05.857998  2138 net.cpp:226] pool1 needs backward computation.
    I1230 23:53:05.858006  2138 net.cpp:226] cccp1b needs backward computation.
    I1230 23:53:05.858013  2138 net.cpp:226] relu1a needs backward computation.
    I1230 23:53:05.858021  2138 net.cpp:226] cccp1a needs backward computation.
    I1230 23:53:05.858027  2138 net.cpp:226] conv1 needs backward computation.
    I1230 23:53:05.858034  2138 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1230 23:53:05.858042  2138 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1230 23:53:05.858052  2138 net.cpp:228] data does not need backward computation.
    I1230 23:53:05.858057  2138 net.cpp:270] This network produces output accuracy_c
    I1230 23:53:05.858065  2138 net.cpp:270] This network produces output accuracy_f
    I1230 23:53:05.858072  2138 net.cpp:270] This network produces output loss_c
    I1230 23:53:05.858079  2138 net.cpp:270] This network produces output loss_f
    I1230 23:53:05.858108  2138 net.cpp:283] Network initialization done.
    I1230 23:53:05.858211  2138 solver.cpp:60] Solver scaffolding done.
    I1230 23:53:05.858958  2138 caffe.cpp:212] Starting Optimization
    I1230 23:53:05.858989  2138 solver.cpp:288] Solving
    I1230 23:53:05.858996  2138 solver.cpp:289] Learning Rate Policy: inv
    I1230 23:53:05.860535  2138 solver.cpp:341] Iteration 0, Testing net (#0)
    I1230 23:53:13.773147  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.0495
    I1230 23:53:13.773216  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.0103333
    I1230 23:53:13.773233  2138 solver.cpp:409]     Test net output #2: loss_c = 3.28457 (* 1 = 3.28457 loss)
    I1230 23:53:13.773246  2138 solver.cpp:409]     Test net output #3: loss_f = 4.8514 (* 1 = 4.8514 loss)
    I1230 23:53:13.933205  2138 solver.cpp:237] Iteration 0, loss = 8.25779
    I1230 23:53:13.933256  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.07
    I1230 23:53:13.933267  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.02
    I1230 23:53:13.933279  2138 solver.cpp:253]     Train net output #2: loss_c = 3.35899 (* 1 = 3.35899 loss)
    I1230 23:53:13.933290  2138 solver.cpp:253]     Train net output #3: loss_f = 4.8988 (* 1 = 4.8988 loss)
    I1230 23:53:13.933315  2138 sgd_solver.cpp:106] Iteration 0, lr = 0.0006
    I1230 23:53:34.218413  2138 solver.cpp:237] Iteration 100, loss = 7.58237
    I1230 23:53:34.218582  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.06
    I1230 23:53:34.218602  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0
    I1230 23:53:34.218621  2138 solver.cpp:253]     Train net output #2: loss_c = 2.98283 (* 1 = 2.98283 loss)
    I1230 23:53:34.218634  2138 solver.cpp:253]     Train net output #3: loss_f = 4.59954 (* 1 = 4.59954 loss)
    I1230 23:53:34.218648  2138 sgd_solver.cpp:106] Iteration 100, lr = 0.000595539
    I1230 23:53:54.519860  2138 solver.cpp:237] Iteration 200, loss = 7.68019
    I1230 23:53:54.519917  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.05
    I1230 23:53:54.519932  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 23:53:54.519948  2138 solver.cpp:253]     Train net output #2: loss_c = 3.03685 (* 1 = 3.03685 loss)
    I1230 23:53:54.519961  2138 solver.cpp:253]     Train net output #3: loss_f = 4.64334 (* 1 = 4.64334 loss)
    I1230 23:53:54.519974  2138 sgd_solver.cpp:106] Iteration 200, lr = 0.000591155
    I1230 23:54:15.143225  2138 solver.cpp:237] Iteration 300, loss = 7.6897
    I1230 23:54:15.143345  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.05
    I1230 23:54:15.143378  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.03
    I1230 23:54:15.143410  2138 solver.cpp:253]     Train net output #2: loss_c = 3.02311 (* 1 = 3.02311 loss)
    I1230 23:54:15.143435  2138 solver.cpp:253]     Train net output #3: loss_f = 4.66659 (* 1 = 4.66659 loss)
    I1230 23:54:15.143458  2138 sgd_solver.cpp:106] Iteration 300, lr = 0.000586845
    I1230 23:54:35.507261  2138 solver.cpp:237] Iteration 400, loss = 7.67128
    I1230 23:54:35.507323  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.05
    I1230 23:54:35.507336  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0
    I1230 23:54:35.507350  2138 solver.cpp:253]     Train net output #2: loss_c = 3.01924 (* 1 = 3.01924 loss)
    I1230 23:54:35.507364  2138 solver.cpp:253]     Train net output #3: loss_f = 4.65204 (* 1 = 4.65204 loss)
    I1230 23:54:35.507386  2138 sgd_solver.cpp:106] Iteration 400, lr = 0.000582608
    I1230 23:54:56.354081  2138 solver.cpp:237] Iteration 500, loss = 7.63208
    I1230 23:54:56.354272  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.05
    I1230 23:54:56.354297  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 23:54:56.354317  2138 solver.cpp:253]     Train net output #2: loss_c = 3.0089 (* 1 = 3.0089 loss)
    I1230 23:54:56.354333  2138 solver.cpp:253]     Train net output #3: loss_f = 4.62318 (* 1 = 4.62318 loss)
    I1230 23:54:56.354349  2138 sgd_solver.cpp:106] Iteration 500, lr = 0.000578441
    I1230 23:55:16.827519  2138 solver.cpp:237] Iteration 600, loss = 7.39545
    I1230 23:55:16.827566  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.11
    I1230 23:55:16.827579  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 23:55:16.827590  2138 solver.cpp:253]     Train net output #2: loss_c = 2.86711 (* 1 = 2.86711 loss)
    I1230 23:55:16.827600  2138 solver.cpp:253]     Train net output #3: loss_f = 4.52834 (* 1 = 4.52834 loss)
    I1230 23:55:16.827610  2138 sgd_solver.cpp:106] Iteration 600, lr = 0.000574344
    I1230 23:55:37.538697  2138 solver.cpp:237] Iteration 700, loss = 6.98586
    I1230 23:55:37.538831  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.17
    I1230 23:55:37.538844  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.06
    I1230 23:55:37.538856  2138 solver.cpp:253]     Train net output #2: loss_c = 2.69125 (* 1 = 2.69125 loss)
    I1230 23:55:37.538866  2138 solver.cpp:253]     Train net output #3: loss_f = 4.29461 (* 1 = 4.29461 loss)
    I1230 23:55:37.538875  2138 sgd_solver.cpp:106] Iteration 700, lr = 0.000570313
    I1230 23:55:57.255702  2138 solver.cpp:237] Iteration 800, loss = 7.1337
    I1230 23:55:57.255767  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.1
    I1230 23:55:57.255790  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.01
    I1230 23:55:57.255812  2138 solver.cpp:253]     Train net output #2: loss_c = 2.76999 (* 1 = 2.76999 loss)
    I1230 23:55:57.255827  2138 solver.cpp:253]     Train net output #3: loss_f = 4.36371 (* 1 = 4.36371 loss)
    I1230 23:55:57.255843  2138 sgd_solver.cpp:106] Iteration 800, lr = 0.000566348
    I1230 23:56:17.418638  2138 solver.cpp:237] Iteration 900, loss = 6.91595
    I1230 23:56:17.418735  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.08
    I1230 23:56:17.418751  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.05
    I1230 23:56:17.418764  2138 solver.cpp:253]     Train net output #2: loss_c = 2.74066 (* 1 = 2.74066 loss)
    I1230 23:56:17.418776  2138 solver.cpp:253]     Train net output #3: loss_f = 4.17529 (* 1 = 4.17529 loss)
    I1230 23:56:17.418787  2138 sgd_solver.cpp:106] Iteration 900, lr = 0.000562447
    I1230 23:56:37.461427  2138 solver.cpp:341] Iteration 1000, Testing net (#0)
    I1230 23:56:45.418112  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.175167
    I1230 23:56:45.418174  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.05475
    I1230 23:56:45.418190  2138 solver.cpp:409]     Test net output #2: loss_c = 2.63685 (* 1 = 2.63685 loss)
    I1230 23:56:45.418200  2138 solver.cpp:409]     Test net output #3: loss_f = 4.13944 (* 1 = 4.13944 loss)
    I1230 23:56:45.500344  2138 solver.cpp:237] Iteration 1000, loss = 6.71198
    I1230 23:56:45.500396  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.18
    I1230 23:56:45.500408  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.06
    I1230 23:56:45.500423  2138 solver.cpp:253]     Train net output #2: loss_c = 2.60938 (* 1 = 2.60938 loss)
    I1230 23:56:45.500435  2138 solver.cpp:253]     Train net output #3: loss_f = 4.1026 (* 1 = 4.1026 loss)
    I1230 23:56:45.500447  2138 sgd_solver.cpp:106] Iteration 1000, lr = 0.000558608
    I1230 23:57:05.331466  2138 solver.cpp:237] Iteration 1100, loss = 6.52994
    I1230 23:57:05.331640  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.19
    I1230 23:57:05.331663  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.1
    I1230 23:57:05.331677  2138 solver.cpp:253]     Train net output #2: loss_c = 2.58351 (* 1 = 2.58351 loss)
    I1230 23:57:05.331691  2138 solver.cpp:253]     Train net output #3: loss_f = 3.94643 (* 1 = 3.94643 loss)
    I1230 23:57:05.331701  2138 sgd_solver.cpp:106] Iteration 1100, lr = 0.000554829
    I1230 23:57:25.569013  2138 solver.cpp:237] Iteration 1200, loss = 6.4408
    I1230 23:57:25.569074  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.27
    I1230 23:57:25.569087  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.07
    I1230 23:57:25.569100  2138 solver.cpp:253]     Train net output #2: loss_c = 2.46677 (* 1 = 2.46677 loss)
    I1230 23:57:25.569110  2138 solver.cpp:253]     Train net output #3: loss_f = 3.97404 (* 1 = 3.97404 loss)
    I1230 23:57:25.569123  2138 sgd_solver.cpp:106] Iteration 1200, lr = 0.000551109
    I1230 23:57:45.659368  2138 solver.cpp:237] Iteration 1300, loss = 6.63577
    I1230 23:57:45.659519  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.21
    I1230 23:57:45.659543  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.08
    I1230 23:57:45.659564  2138 solver.cpp:253]     Train net output #2: loss_c = 2.58761 (* 1 = 2.58761 loss)
    I1230 23:57:45.659580  2138 solver.cpp:253]     Train net output #3: loss_f = 4.04815 (* 1 = 4.04815 loss)
    I1230 23:57:45.659597  2138 sgd_solver.cpp:106] Iteration 1300, lr = 0.000547447
    I1230 23:58:05.881971  2138 solver.cpp:237] Iteration 1400, loss = 6.22094
    I1230 23:58:05.882010  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.19
    I1230 23:58:05.882022  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1230 23:58:05.882035  2138 solver.cpp:253]     Train net output #2: loss_c = 2.45446 (* 1 = 2.45446 loss)
    I1230 23:58:05.882043  2138 solver.cpp:253]     Train net output #3: loss_f = 3.76648 (* 1 = 3.76648 loss)
    I1230 23:58:05.882052  2138 sgd_solver.cpp:106] Iteration 1400, lr = 0.000543842
    I1230 23:58:25.982110  2138 solver.cpp:237] Iteration 1500, loss = 6.30764
    I1230 23:58:25.982256  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.27
    I1230 23:58:25.982278  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.08
    I1230 23:58:25.982298  2138 solver.cpp:253]     Train net output #2: loss_c = 2.45159 (* 1 = 2.45159 loss)
    I1230 23:58:25.982314  2138 solver.cpp:253]     Train net output #3: loss_f = 3.85605 (* 1 = 3.85605 loss)
    I1230 23:58:25.982328  2138 sgd_solver.cpp:106] Iteration 1500, lr = 0.000540291
    I1230 23:58:46.120713  2138 solver.cpp:237] Iteration 1600, loss = 6.03829
    I1230 23:58:46.120766  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.25
    I1230 23:58:46.120789  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1230 23:58:46.120800  2138 solver.cpp:253]     Train net output #2: loss_c = 2.39091 (* 1 = 2.39091 loss)
    I1230 23:58:46.120810  2138 solver.cpp:253]     Train net output #3: loss_f = 3.64738 (* 1 = 3.64738 loss)
    I1230 23:58:46.120820  2138 sgd_solver.cpp:106] Iteration 1600, lr = 0.000536794
    I1230 23:59:06.315742  2138 solver.cpp:237] Iteration 1700, loss = 6.19891
    I1230 23:59:06.315884  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.25
    I1230 23:59:06.315899  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.11
    I1230 23:59:06.315909  2138 solver.cpp:253]     Train net output #2: loss_c = 2.37832 (* 1 = 2.37832 loss)
    I1230 23:59:06.315918  2138 solver.cpp:253]     Train net output #3: loss_f = 3.8206 (* 1 = 3.8206 loss)
    I1230 23:59:06.315930  2138 sgd_solver.cpp:106] Iteration 1700, lr = 0.00053335
    I1230 23:59:26.445183  2138 solver.cpp:237] Iteration 1800, loss = 6.18983
    I1230 23:59:26.445253  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.24
    I1230 23:59:26.445276  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.15
    I1230 23:59:26.445298  2138 solver.cpp:253]     Train net output #2: loss_c = 2.43027 (* 1 = 2.43027 loss)
    I1230 23:59:26.445317  2138 solver.cpp:253]     Train net output #3: loss_f = 3.75957 (* 1 = 3.75957 loss)
    I1230 23:59:26.445338  2138 sgd_solver.cpp:106] Iteration 1800, lr = 0.000529956
    I1230 23:59:46.555738  2138 solver.cpp:237] Iteration 1900, loss = 5.60266
    I1230 23:59:46.555902  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.3
    I1230 23:59:46.555924  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.12
    I1230 23:59:46.555943  2138 solver.cpp:253]     Train net output #2: loss_c = 2.21368 (* 1 = 2.21368 loss)
    I1230 23:59:46.555959  2138 solver.cpp:253]     Train net output #3: loss_f = 3.38898 (* 1 = 3.38898 loss)
    I1230 23:59:46.555976  2138 sgd_solver.cpp:106] Iteration 1900, lr = 0.000526612
    I1231 00:00:06.504546  2138 solver.cpp:341] Iteration 2000, Testing net (#0)
    I1231 00:00:14.097273  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.285833
    I1231 00:00:14.097337  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.145417
    I1231 00:00:14.097352  2138 solver.cpp:409]     Test net output #2: loss_c = 2.30261 (* 1 = 2.30261 loss)
    I1231 00:00:14.097362  2138 solver.cpp:409]     Test net output #3: loss_f = 3.57398 (* 1 = 3.57398 loss)
    I1231 00:00:14.189960  2138 solver.cpp:237] Iteration 2000, loss = 6.06265
    I1231 00:00:14.190017  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.26
    I1231 00:00:14.190028  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.13
    I1231 00:00:14.190042  2138 solver.cpp:253]     Train net output #2: loss_c = 2.39453 (* 1 = 2.39453 loss)
    I1231 00:00:14.190052  2138 solver.cpp:253]     Train net output #3: loss_f = 3.66812 (* 1 = 3.66812 loss)
    I1231 00:00:14.190064  2138 sgd_solver.cpp:106] Iteration 2000, lr = 0.000523318
    I1231 00:00:34.180815  2138 solver.cpp:237] Iteration 2100, loss = 5.80773
    I1231 00:00:34.180974  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.26
    I1231 00:00:34.180999  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1231 00:00:34.181021  2138 solver.cpp:253]     Train net output #2: loss_c = 2.31623 (* 1 = 2.31623 loss)
    I1231 00:00:34.181040  2138 solver.cpp:253]     Train net output #3: loss_f = 3.4915 (* 1 = 3.4915 loss)
    I1231 00:00:34.181057  2138 sgd_solver.cpp:106] Iteration 2100, lr = 0.00052007
    I1231 00:00:54.377974  2138 solver.cpp:237] Iteration 2200, loss = 5.90506
    I1231 00:00:54.378022  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1231 00:00:54.378034  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1231 00:00:54.378048  2138 solver.cpp:253]     Train net output #2: loss_c = 2.24687 (* 1 = 2.24687 loss)
    I1231 00:00:54.378059  2138 solver.cpp:253]     Train net output #3: loss_f = 3.6582 (* 1 = 3.6582 loss)
    I1231 00:00:54.378072  2138 sgd_solver.cpp:106] Iteration 2200, lr = 0.00051687
    I1231 00:01:14.500301  2138 solver.cpp:237] Iteration 2300, loss = 5.7426
    I1231 00:01:14.500399  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.24
    I1231 00:01:14.500416  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1231 00:01:14.500427  2138 solver.cpp:253]     Train net output #2: loss_c = 2.28028 (* 1 = 2.28028 loss)
    I1231 00:01:14.500438  2138 solver.cpp:253]     Train net output #3: loss_f = 3.46232 (* 1 = 3.46232 loss)
    I1231 00:01:14.500449  2138 sgd_solver.cpp:106] Iteration 2300, lr = 0.000513715
    I1231 00:01:34.554718  2138 solver.cpp:237] Iteration 2400, loss = 5.24205
    I1231 00:01:34.554790  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1231 00:01:34.554811  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:01:34.554834  2138 solver.cpp:253]     Train net output #2: loss_c = 2.08808 (* 1 = 2.08808 loss)
    I1231 00:01:34.554854  2138 solver.cpp:253]     Train net output #3: loss_f = 3.15397 (* 1 = 3.15397 loss)
    I1231 00:01:34.554873  2138 sgd_solver.cpp:106] Iteration 2400, lr = 0.000510605
    I1231 00:01:54.370311  2138 solver.cpp:237] Iteration 2500, loss = 5.88314
    I1231 00:01:54.370509  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.28
    I1231 00:01:54.370538  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.16
    I1231 00:01:54.370561  2138 solver.cpp:253]     Train net output #2: loss_c = 2.31615 (* 1 = 2.31615 loss)
    I1231 00:01:54.370580  2138 solver.cpp:253]     Train net output #3: loss_f = 3.56699 (* 1 = 3.56699 loss)
    I1231 00:01:54.370599  2138 sgd_solver.cpp:106] Iteration 2500, lr = 0.000507538
    I1231 00:02:14.251708  2138 solver.cpp:237] Iteration 2600, loss = 5.72017
    I1231 00:02:14.251761  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1231 00:02:14.251775  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.19
    I1231 00:02:14.251786  2138 solver.cpp:253]     Train net output #2: loss_c = 2.3121 (* 1 = 2.3121 loss)
    I1231 00:02:14.251796  2138 solver.cpp:253]     Train net output #3: loss_f = 3.40807 (* 1 = 3.40807 loss)
    I1231 00:02:14.251807  2138 sgd_solver.cpp:106] Iteration 2600, lr = 0.000504514
    I1231 00:02:34.289662  2138 solver.cpp:237] Iteration 2700, loss = 5.76095
    I1231 00:02:34.289813  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1231 00:02:34.289835  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:02:34.289855  2138 solver.cpp:253]     Train net output #2: loss_c = 2.24453 (* 1 = 2.24453 loss)
    I1231 00:02:34.289870  2138 solver.cpp:253]     Train net output #3: loss_f = 3.51643 (* 1 = 3.51643 loss)
    I1231 00:02:34.289885  2138 sgd_solver.cpp:106] Iteration 2700, lr = 0.000501532
    I1231 00:02:54.177814  2138 solver.cpp:237] Iteration 2800, loss = 5.60803
    I1231 00:02:54.177855  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.28
    I1231 00:02:54.177865  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.17
    I1231 00:02:54.177876  2138 solver.cpp:253]     Train net output #2: loss_c = 2.23709 (* 1 = 2.23709 loss)
    I1231 00:02:54.177886  2138 solver.cpp:253]     Train net output #3: loss_f = 3.37093 (* 1 = 3.37093 loss)
    I1231 00:02:54.177896  2138 sgd_solver.cpp:106] Iteration 2800, lr = 0.00049859
    I1231 00:03:14.002027  2138 solver.cpp:237] Iteration 2900, loss = 5.10867
    I1231 00:03:14.002183  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1231 00:03:14.002198  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.23
    I1231 00:03:14.002210  2138 solver.cpp:253]     Train net output #2: loss_c = 2.06478 (* 1 = 2.06478 loss)
    I1231 00:03:14.002220  2138 solver.cpp:253]     Train net output #3: loss_f = 3.04388 (* 1 = 3.04388 loss)
    I1231 00:03:14.002230  2138 sgd_solver.cpp:106] Iteration 2900, lr = 0.000495689
    I1231 00:03:33.822412  2138 solver.cpp:341] Iteration 3000, Testing net (#0)
    I1231 00:03:41.621206  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.33025
    I1231 00:03:41.621290  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.1995
    I1231 00:03:41.621315  2138 solver.cpp:409]     Test net output #2: loss_c = 2.15724 (* 1 = 2.15724 loss)
    I1231 00:03:41.621330  2138 solver.cpp:409]     Test net output #3: loss_f = 3.30451 (* 1 = 3.30451 loss)
    I1231 00:03:41.725824  2138 solver.cpp:237] Iteration 3000, loss = 5.66194
    I1231 00:03:41.725868  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1231 00:03:41.725879  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.14
    I1231 00:03:41.725893  2138 solver.cpp:253]     Train net output #2: loss_c = 2.24803 (* 1 = 2.24803 loss)
    I1231 00:03:41.725903  2138 solver.cpp:253]     Train net output #3: loss_f = 3.41391 (* 1 = 3.41391 loss)
    I1231 00:03:41.725915  2138 sgd_solver.cpp:106] Iteration 3000, lr = 0.000492826
    I1231 00:04:02.423810  2138 solver.cpp:237] Iteration 3100, loss = 5.44345
    I1231 00:04:02.423900  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.3
    I1231 00:04:02.423916  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:04:02.423928  2138 solver.cpp:253]     Train net output #2: loss_c = 2.21517 (* 1 = 2.21517 loss)
    I1231 00:04:02.423939  2138 solver.cpp:253]     Train net output #3: loss_f = 3.22828 (* 1 = 3.22828 loss)
    I1231 00:04:02.423949  2138 sgd_solver.cpp:106] Iteration 3100, lr = 0.000490002
    I1231 00:04:21.236203  2138 solver.cpp:237] Iteration 3200, loss = 5.52063
    I1231 00:04:21.236263  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.3
    I1231 00:04:21.236294  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:04:21.236307  2138 solver.cpp:253]     Train net output #2: loss_c = 2.13733 (* 1 = 2.13733 loss)
    I1231 00:04:21.236318  2138 solver.cpp:253]     Train net output #3: loss_f = 3.3833 (* 1 = 3.3833 loss)
    I1231 00:04:21.236330  2138 sgd_solver.cpp:106] Iteration 3200, lr = 0.000487215
    I1231 00:04:40.593997  2138 solver.cpp:237] Iteration 3300, loss = 5.29822
    I1231 00:04:40.594182  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1231 00:04:40.594208  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:04:40.594229  2138 solver.cpp:253]     Train net output #2: loss_c = 2.12396 (* 1 = 2.12396 loss)
    I1231 00:04:40.594246  2138 solver.cpp:253]     Train net output #3: loss_f = 3.17426 (* 1 = 3.17426 loss)
    I1231 00:04:40.594265  2138 sgd_solver.cpp:106] Iteration 3300, lr = 0.000484465
    I1231 00:05:00.088093  2138 solver.cpp:237] Iteration 3400, loss = 5.23021
    I1231 00:05:00.088155  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.31
    I1231 00:05:00.088172  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.19
    I1231 00:05:00.088191  2138 solver.cpp:253]     Train net output #2: loss_c = 2.14554 (* 1 = 2.14554 loss)
    I1231 00:05:00.088207  2138 solver.cpp:253]     Train net output #3: loss_f = 3.08467 (* 1 = 3.08467 loss)
    I1231 00:05:00.088223  2138 sgd_solver.cpp:106] Iteration 3400, lr = 0.000481751
    I1231 00:05:20.706063  2138 solver.cpp:237] Iteration 3500, loss = 5.27436
    I1231 00:05:20.706154  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.29
    I1231 00:05:20.706168  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:05:20.706182  2138 solver.cpp:253]     Train net output #2: loss_c = 2.07476 (* 1 = 2.07476 loss)
    I1231 00:05:20.706192  2138 solver.cpp:253]     Train net output #3: loss_f = 3.19961 (* 1 = 3.19961 loss)
    I1231 00:05:20.706204  2138 sgd_solver.cpp:106] Iteration 3500, lr = 0.000479072
    I1231 00:05:40.583398  2138 solver.cpp:237] Iteration 3600, loss = 5.12261
    I1231 00:05:40.583457  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1231 00:05:40.583470  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1231 00:05:40.583482  2138 solver.cpp:253]     Train net output #2: loss_c = 2.07468 (* 1 = 2.07468 loss)
    I1231 00:05:40.583492  2138 solver.cpp:253]     Train net output #3: loss_f = 3.04793 (* 1 = 3.04793 loss)
    I1231 00:05:40.583503  2138 sgd_solver.cpp:106] Iteration 3600, lr = 0.000476428
    I1231 00:06:00.588161  2138 solver.cpp:237] Iteration 3700, loss = 5.13906
    I1231 00:06:00.588270  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1231 00:06:00.588285  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1231 00:06:00.588299  2138 solver.cpp:253]     Train net output #2: loss_c = 1.97287 (* 1 = 1.97287 loss)
    I1231 00:06:00.588310  2138 solver.cpp:253]     Train net output #3: loss_f = 3.16619 (* 1 = 3.16619 loss)
    I1231 00:06:00.588320  2138 sgd_solver.cpp:106] Iteration 3700, lr = 0.000473817
    I1231 00:06:20.866381  2138 solver.cpp:237] Iteration 3800, loss = 5.18724
    I1231 00:06:20.866431  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.28
    I1231 00:06:20.866443  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.16
    I1231 00:06:20.866456  2138 solver.cpp:253]     Train net output #2: loss_c = 2.08642 (* 1 = 2.08642 loss)
    I1231 00:06:20.866466  2138 solver.cpp:253]     Train net output #3: loss_f = 3.10081 (* 1 = 3.10081 loss)
    I1231 00:06:20.866479  2138 sgd_solver.cpp:106] Iteration 3800, lr = 0.00047124
    I1231 00:06:41.013377  2138 solver.cpp:237] Iteration 3900, loss = 4.69109
    I1231 00:06:41.013558  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1231 00:06:41.013574  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1231 00:06:41.013589  2138 solver.cpp:253]     Train net output #2: loss_c = 1.91922 (* 1 = 1.91922 loss)
    I1231 00:06:41.013602  2138 solver.cpp:253]     Train net output #3: loss_f = 2.77187 (* 1 = 2.77187 loss)
    I1231 00:06:41.013615  2138 sgd_solver.cpp:106] Iteration 3900, lr = 0.000468695
    I1231 00:07:00.890699  2138 solver.cpp:341] Iteration 4000, Testing net (#0)
    I1231 00:07:08.602891  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.372333
    I1231 00:07:08.602967  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.241417
    I1231 00:07:08.602982  2138 solver.cpp:409]     Test net output #2: loss_c = 2.03733 (* 1 = 2.03733 loss)
    I1231 00:07:08.602993  2138 solver.cpp:409]     Test net output #3: loss_f = 3.10276 (* 1 = 3.10276 loss)
    I1231 00:07:08.693734  2138 solver.cpp:237] Iteration 4000, loss = 5.24987
    I1231 00:07:08.693778  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1231 00:07:08.693790  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.2
    I1231 00:07:08.693804  2138 solver.cpp:253]     Train net output #2: loss_c = 2.06367 (* 1 = 2.06367 loss)
    I1231 00:07:08.693814  2138 solver.cpp:253]     Train net output #3: loss_f = 3.1862 (* 1 = 3.1862 loss)
    I1231 00:07:08.693825  2138 sgd_solver.cpp:106] Iteration 4000, lr = 0.000466182
    I1231 00:07:28.857923  2138 solver.cpp:237] Iteration 4100, loss = 5.25732
    I1231 00:07:28.858075  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.34
    I1231 00:07:28.858101  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:07:28.858122  2138 solver.cpp:253]     Train net output #2: loss_c = 2.17671 (* 1 = 2.17671 loss)
    I1231 00:07:28.858139  2138 solver.cpp:253]     Train net output #3: loss_f = 3.08061 (* 1 = 3.08061 loss)
    I1231 00:07:28.858156  2138 sgd_solver.cpp:106] Iteration 4100, lr = 0.0004637
    I1231 00:07:49.054364  2138 solver.cpp:237] Iteration 4200, loss = 5.09103
    I1231 00:07:49.054425  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1231 00:07:49.054442  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1231 00:07:49.054461  2138 solver.cpp:253]     Train net output #2: loss_c = 1.95192 (* 1 = 1.95192 loss)
    I1231 00:07:49.054477  2138 solver.cpp:253]     Train net output #3: loss_f = 3.13911 (* 1 = 3.13911 loss)
    I1231 00:07:49.054492  2138 sgd_solver.cpp:106] Iteration 4200, lr = 0.000461249
    I1231 00:08:09.205281  2138 solver.cpp:237] Iteration 4300, loss = 5.11852
    I1231 00:08:09.205379  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.33
    I1231 00:08:09.205394  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:08:09.205406  2138 solver.cpp:253]     Train net output #2: loss_c = 2.06226 (* 1 = 2.06226 loss)
    I1231 00:08:09.205416  2138 solver.cpp:253]     Train net output #3: loss_f = 3.05626 (* 1 = 3.05626 loss)
    I1231 00:08:09.205427  2138 sgd_solver.cpp:106] Iteration 4300, lr = 0.000458827
    I1231 00:08:32.948310  2138 solver.cpp:237] Iteration 4400, loss = 4.55937
    I1231 00:08:32.948354  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:08:32.948364  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.33
    I1231 00:08:32.948374  2138 solver.cpp:253]     Train net output #2: loss_c = 1.86002 (* 1 = 1.86002 loss)
    I1231 00:08:32.948385  2138 solver.cpp:253]     Train net output #3: loss_f = 2.69935 (* 1 = 2.69935 loss)
    I1231 00:08:32.948395  2138 sgd_solver.cpp:106] Iteration 4400, lr = 0.000456435
    I1231 00:08:54.413588  2138 solver.cpp:237] Iteration 4500, loss = 5.23566
    I1231 00:08:54.413717  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1231 00:08:54.413738  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1231 00:08:54.413756  2138 solver.cpp:253]     Train net output #2: loss_c = 2.1437 (* 1 = 2.1437 loss)
    I1231 00:08:54.413772  2138 solver.cpp:253]     Train net output #3: loss_f = 3.09196 (* 1 = 3.09196 loss)
    I1231 00:08:54.413787  2138 sgd_solver.cpp:106] Iteration 4500, lr = 0.000454073
    I1231 00:09:13.726876  2138 solver.cpp:237] Iteration 4600, loss = 4.91976
    I1231 00:09:13.726935  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1231 00:09:13.726953  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:09:13.726972  2138 solver.cpp:253]     Train net output #2: loss_c = 2.01037 (* 1 = 2.01037 loss)
    I1231 00:09:13.726989  2138 solver.cpp:253]     Train net output #3: loss_f = 2.9094 (* 1 = 2.9094 loss)
    I1231 00:09:13.727005  2138 sgd_solver.cpp:106] Iteration 4600, lr = 0.000451738
    I1231 00:09:32.941186  2138 solver.cpp:237] Iteration 4700, loss = 5.06068
    I1231 00:09:32.941371  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.37
    I1231 00:09:32.941397  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:09:32.941416  2138 solver.cpp:253]     Train net output #2: loss_c = 1.96839 (* 1 = 1.96839 loss)
    I1231 00:09:32.941431  2138 solver.cpp:253]     Train net output #3: loss_f = 3.09228 (* 1 = 3.09228 loss)
    I1231 00:09:32.941447  2138 sgd_solver.cpp:106] Iteration 4700, lr = 0.000449431
    I1231 00:09:52.101300  2138 solver.cpp:237] Iteration 4800, loss = 4.79813
    I1231 00:09:52.101357  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1231 00:09:52.101374  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.27
    I1231 00:09:52.101392  2138 solver.cpp:253]     Train net output #2: loss_c = 1.96539 (* 1 = 1.96539 loss)
    I1231 00:09:52.101408  2138 solver.cpp:253]     Train net output #3: loss_f = 2.83274 (* 1 = 2.83274 loss)
    I1231 00:09:52.101423  2138 sgd_solver.cpp:106] Iteration 4800, lr = 0.000447152
    I1231 00:10:11.314023  2138 solver.cpp:237] Iteration 4900, loss = 4.47358
    I1231 00:10:11.314157  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.43
    I1231 00:10:11.314178  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1231 00:10:11.314198  2138 solver.cpp:253]     Train net output #2: loss_c = 1.8061 (* 1 = 1.8061 loss)
    I1231 00:10:11.314214  2138 solver.cpp:253]     Train net output #3: loss_f = 2.66749 (* 1 = 2.66749 loss)
    I1231 00:10:11.314229  2138 sgd_solver.cpp:106] Iteration 4900, lr = 0.000444899
    I1231 00:10:30.332945  2138 solver.cpp:341] Iteration 5000, Testing net (#0)
    I1231 00:10:38.237726  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.4095
    I1231 00:10:38.237787  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.274333
    I1231 00:10:38.237807  2138 solver.cpp:409]     Test net output #2: loss_c = 1.92452 (* 1 = 1.92452 loss)
    I1231 00:10:38.237823  2138 solver.cpp:409]     Test net output #3: loss_f = 2.92962 (* 1 = 2.92962 loss)
    I1231 00:10:38.337940  2138 solver.cpp:237] Iteration 5000, loss = 4.81319
    I1231 00:10:38.337995  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.42
    I1231 00:10:38.338012  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1231 00:10:38.338032  2138 solver.cpp:253]     Train net output #2: loss_c = 1.94146 (* 1 = 1.94146 loss)
    I1231 00:10:38.338048  2138 solver.cpp:253]     Train net output #3: loss_f = 2.87173 (* 1 = 2.87173 loss)
    I1231 00:10:38.338064  2138 sgd_solver.cpp:106] Iteration 5000, lr = 0.000442673
    I1231 00:10:57.646738  2138 solver.cpp:237] Iteration 5100, loss = 4.72833
    I1231 00:10:57.646874  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:10:57.646898  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.3
    I1231 00:10:57.646917  2138 solver.cpp:253]     Train net output #2: loss_c = 1.95028 (* 1 = 1.95028 loss)
    I1231 00:10:57.646934  2138 solver.cpp:253]     Train net output #3: loss_f = 2.77805 (* 1 = 2.77805 loss)
    I1231 00:10:57.646950  2138 sgd_solver.cpp:106] Iteration 5100, lr = 0.000440472
    I1231 00:11:16.970507  2138 solver.cpp:237] Iteration 5200, loss = 4.72109
    I1231 00:11:16.970566  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:11:16.970584  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:11:16.970604  2138 solver.cpp:253]     Train net output #2: loss_c = 1.83884 (* 1 = 1.83884 loss)
    I1231 00:11:16.970620  2138 solver.cpp:253]     Train net output #3: loss_f = 2.88224 (* 1 = 2.88224 loss)
    I1231 00:11:16.970638  2138 sgd_solver.cpp:106] Iteration 5200, lr = 0.000438297
    I1231 00:11:36.228305  2138 solver.cpp:237] Iteration 5300, loss = 4.72028
    I1231 00:11:36.228466  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1231 00:11:36.228487  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1231 00:11:36.228507  2138 solver.cpp:253]     Train net output #2: loss_c = 1.93728 (* 1 = 1.93728 loss)
    I1231 00:11:36.228523  2138 solver.cpp:253]     Train net output #3: loss_f = 2.783 (* 1 = 2.783 loss)
    I1231 00:11:36.228538  2138 sgd_solver.cpp:106] Iteration 5300, lr = 0.000436147
    I1231 00:11:55.190723  2138 solver.cpp:237] Iteration 5400, loss = 4.44658
    I1231 00:11:55.190783  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.35
    I1231 00:11:55.190796  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.32
    I1231 00:11:55.190809  2138 solver.cpp:253]     Train net output #2: loss_c = 1.80768 (* 1 = 1.80768 loss)
    I1231 00:11:55.190820  2138 solver.cpp:253]     Train net output #3: loss_f = 2.6389 (* 1 = 2.6389 loss)
    I1231 00:11:55.190834  2138 sgd_solver.cpp:106] Iteration 5400, lr = 0.000434021
    I1231 00:12:14.075893  2138 solver.cpp:237] Iteration 5500, loss = 4.66436
    I1231 00:12:14.075997  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:12:14.076012  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.26
    I1231 00:12:14.076025  2138 solver.cpp:253]     Train net output #2: loss_c = 1.80794 (* 1 = 1.80794 loss)
    I1231 00:12:14.076036  2138 solver.cpp:253]     Train net output #3: loss_f = 2.85642 (* 1 = 2.85642 loss)
    I1231 00:12:14.076047  2138 sgd_solver.cpp:106] Iteration 5500, lr = 0.000431919
    I1231 00:12:33.045637  2138 solver.cpp:237] Iteration 5600, loss = 4.99731
    I1231 00:12:33.045707  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:12:33.045722  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:12:33.045750  2138 solver.cpp:253]     Train net output #2: loss_c = 2.08094 (* 1 = 2.08094 loss)
    I1231 00:12:33.045764  2138 solver.cpp:253]     Train net output #3: loss_f = 2.91637 (* 1 = 2.91637 loss)
    I1231 00:12:33.045778  2138 sgd_solver.cpp:106] Iteration 5600, lr = 0.000429841
    I1231 00:12:52.078433  2138 solver.cpp:237] Iteration 5700, loss = 5.01406
    I1231 00:12:52.078536  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1231 00:12:52.078560  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.19
    I1231 00:12:52.078572  2138 solver.cpp:253]     Train net output #2: loss_c = 1.98122 (* 1 = 1.98122 loss)
    I1231 00:12:52.078580  2138 solver.cpp:253]     Train net output #3: loss_f = 3.03284 (* 1 = 3.03284 loss)
    I1231 00:12:52.078589  2138 sgd_solver.cpp:106] Iteration 5700, lr = 0.000427786
    I1231 00:13:11.219907  2138 solver.cpp:237] Iteration 5800, loss = 4.61523
    I1231 00:13:11.219969  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:13:11.219987  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1231 00:13:11.220006  2138 solver.cpp:253]     Train net output #2: loss_c = 1.90537 (* 1 = 1.90537 loss)
    I1231 00:13:11.220023  2138 solver.cpp:253]     Train net output #3: loss_f = 2.70986 (* 1 = 2.70986 loss)
    I1231 00:13:11.220039  2138 sgd_solver.cpp:106] Iteration 5800, lr = 0.000425754
    I1231 00:13:30.515661  2138 solver.cpp:237] Iteration 5900, loss = 4.38586
    I1231 00:13:30.515785  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.44
    I1231 00:13:30.515808  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.31
    I1231 00:13:30.515828  2138 solver.cpp:253]     Train net output #2: loss_c = 1.80026 (* 1 = 1.80026 loss)
    I1231 00:13:30.515846  2138 solver.cpp:253]     Train net output #3: loss_f = 2.5856 (* 1 = 2.5856 loss)
    I1231 00:13:30.515861  2138 sgd_solver.cpp:106] Iteration 5900, lr = 0.000423744
    I1231 00:13:49.254819  2138 solver.cpp:341] Iteration 6000, Testing net (#0)
    I1231 00:13:56.365422  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.420917
    I1231 00:13:56.365484  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.2855
    I1231 00:13:56.365499  2138 solver.cpp:409]     Test net output #2: loss_c = 1.88961 (* 1 = 1.88961 loss)
    I1231 00:13:56.365509  2138 solver.cpp:409]     Test net output #3: loss_f = 2.87295 (* 1 = 2.87295 loss)
    I1231 00:13:56.455998  2138 solver.cpp:237] Iteration 6000, loss = 4.78796
    I1231 00:13:56.456069  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.41
    I1231 00:13:56.456079  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.25
    I1231 00:13:56.456090  2138 solver.cpp:253]     Train net output #2: loss_c = 1.90747 (* 1 = 1.90747 loss)
    I1231 00:13:56.456099  2138 solver.cpp:253]     Train net output #3: loss_f = 2.88049 (* 1 = 2.88049 loss)
    I1231 00:13:56.456110  2138 sgd_solver.cpp:106] Iteration 6000, lr = 0.000421756
    I1231 00:14:15.626996  2138 solver.cpp:237] Iteration 6100, loss = 4.72526
    I1231 00:14:15.627154  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.4
    I1231 00:14:15.627179  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.29
    I1231 00:14:15.627192  2138 solver.cpp:253]     Train net output #2: loss_c = 1.97153 (* 1 = 1.97153 loss)
    I1231 00:14:15.627200  2138 solver.cpp:253]     Train net output #3: loss_f = 2.75373 (* 1 = 2.75373 loss)
    I1231 00:14:15.627209  2138 sgd_solver.cpp:106] Iteration 6100, lr = 0.00041979
    I1231 00:14:34.700146  2138 solver.cpp:237] Iteration 6200, loss = 4.63793
    I1231 00:14:34.700186  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.38
    I1231 00:14:34.700197  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.26
    I1231 00:14:34.700209  2138 solver.cpp:253]     Train net output #2: loss_c = 1.81514 (* 1 = 1.81514 loss)
    I1231 00:14:34.700218  2138 solver.cpp:253]     Train net output #3: loss_f = 2.82278 (* 1 = 2.82278 loss)
    I1231 00:14:34.700228  2138 sgd_solver.cpp:106] Iteration 6200, lr = 0.000417845
    I1231 00:14:53.844827  2138 solver.cpp:237] Iteration 6300, loss = 4.72601
    I1231 00:14:53.846292  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.36
    I1231 00:14:53.846335  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.28
    I1231 00:14:53.846360  2138 solver.cpp:253]     Train net output #2: loss_c = 1.9608 (* 1 = 1.9608 loss)
    I1231 00:14:53.846379  2138 solver.cpp:253]     Train net output #3: loss_f = 2.7652 (* 1 = 2.7652 loss)
    I1231 00:14:53.846396  2138 sgd_solver.cpp:106] Iteration 6300, lr = 0.000415921
    I1231 00:15:13.068171  2138 solver.cpp:237] Iteration 6400, loss = 4.21712
    I1231 00:15:13.068240  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.49
    I1231 00:15:13.068266  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.37
    I1231 00:15:13.068295  2138 solver.cpp:253]     Train net output #2: loss_c = 1.69107 (* 1 = 1.69107 loss)
    I1231 00:15:13.068318  2138 solver.cpp:253]     Train net output #3: loss_f = 2.52605 (* 1 = 2.52605 loss)
    I1231 00:15:13.068341  2138 sgd_solver.cpp:106] Iteration 6400, lr = 0.000414017
    I1231 00:15:32.415386  2138 solver.cpp:237] Iteration 6500, loss = 4.73194
    I1231 00:15:32.415531  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.32
    I1231 00:15:32.415552  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.21
    I1231 00:15:32.415573  2138 solver.cpp:253]     Train net output #2: loss_c = 1.88969 (* 1 = 1.88969 loss)
    I1231 00:15:32.415591  2138 solver.cpp:253]     Train net output #3: loss_f = 2.84224 (* 1 = 2.84224 loss)
    I1231 00:15:32.415607  2138 sgd_solver.cpp:106] Iteration 6500, lr = 0.000412134
    # [...]
    I1231 06:05:24.423212  2138 solver.cpp:237] Iteration 146300, loss = 2.7319
    I1231 06:05:24.423246  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1231 06:05:24.423255  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.56
    I1231 06:05:24.423265  2138 solver.cpp:253]     Train net output #2: loss_c = 1.1136 (* 1 = 1.1136 loss)
    I1231 06:05:24.423274  2138 solver.cpp:253]     Train net output #3: loss_f = 1.6183 (* 1 = 1.6183 loss)
    I1231 06:05:24.423283  2138 sgd_solver.cpp:106] Iteration 146300, lr = 7.63277e-05
    I1231 06:05:37.820432  2138 solver.cpp:237] Iteration 146400, loss = 2.55879
    I1231 06:05:37.820529  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.69
    I1231 06:05:37.820543  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.62
    I1231 06:05:37.820554  2138 solver.cpp:253]     Train net output #2: loss_c = 1.03966 (* 1 = 1.03966 loss)
    I1231 06:05:37.820565  2138 solver.cpp:253]     Train net output #3: loss_f = 1.51913 (* 1 = 1.51913 loss)
    I1231 06:05:37.820575  2138 sgd_solver.cpp:106] Iteration 146400, lr = 7.62911e-05
    I1231 06:05:51.180083  2138 solver.cpp:237] Iteration 146500, loss = 2.73653
    I1231 06:05:51.180119  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.67
    I1231 06:05:51.180129  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.49
    I1231 06:05:51.180138  2138 solver.cpp:253]     Train net output #2: loss_c = 1.01435 (* 1 = 1.01435 loss)
    I1231 06:05:51.180146  2138 solver.cpp:253]     Train net output #3: loss_f = 1.72217 (* 1 = 1.72217 loss)
    I1231 06:05:51.180155  2138 sgd_solver.cpp:106] Iteration 146500, lr = 7.62545e-05
    I1231 06:06:04.595229  2138 solver.cpp:237] Iteration 146600, loss = 2.81641
    I1231 06:06:04.595273  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1231 06:06:04.595281  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1231 06:06:04.595291  2138 solver.cpp:253]     Train net output #2: loss_c = 1.11449 (* 1 = 1.11449 loss)
    I1231 06:06:04.595299  2138 solver.cpp:253]     Train net output #3: loss_f = 1.70192 (* 1 = 1.70192 loss)
    I1231 06:06:04.595309  2138 sgd_solver.cpp:106] Iteration 146600, lr = 7.6218e-05
    I1231 06:06:17.984534  2138 solver.cpp:237] Iteration 146700, loss = 2.62631
    I1231 06:06:17.984632  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.72
    I1231 06:06:17.984645  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.55
    I1231 06:06:17.984658  2138 solver.cpp:253]     Train net output #2: loss_c = 0.941709 (* 1 = 0.941709 loss)
    I1231 06:06:17.984668  2138 solver.cpp:253]     Train net output #3: loss_f = 1.6846 (* 1 = 1.6846 loss)
    I1231 06:06:17.984678  2138 sgd_solver.cpp:106] Iteration 146700, lr = 7.61815e-05
    I1231 06:06:31.359715  2138 solver.cpp:237] Iteration 146800, loss = 2.88624
    I1231 06:06:31.359758  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1231 06:06:31.359767  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1231 06:06:31.359776  2138 solver.cpp:253]     Train net output #2: loss_c = 1.19534 (* 1 = 1.19534 loss)
    I1231 06:06:31.359784  2138 solver.cpp:253]     Train net output #3: loss_f = 1.69091 (* 1 = 1.69091 loss)
    I1231 06:06:31.359792  2138 sgd_solver.cpp:106] Iteration 146800, lr = 7.61451e-05
    I1231 06:06:44.704105  2138 solver.cpp:237] Iteration 146900, loss = 2.58872
    I1231 06:06:44.704140  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1231 06:06:44.704149  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.59
    I1231 06:06:44.704160  2138 solver.cpp:253]     Train net output #2: loss_c = 1.06249 (* 1 = 1.06249 loss)
    I1231 06:06:44.704169  2138 solver.cpp:253]     Train net output #3: loss_f = 1.52623 (* 1 = 1.52623 loss)
    I1231 06:06:44.704179  2138 sgd_solver.cpp:106] Iteration 146900, lr = 7.61087e-05
    I1231 06:06:57.977560  2138 solver.cpp:341] Iteration 147000, Testing net (#0)
    I1231 06:07:03.050328  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.587333
    I1231 06:07:03.050371  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.471167
    I1231 06:07:03.050384  2138 solver.cpp:409]     Test net output #2: loss_c = 1.30179 (* 1 = 1.30179 loss)
    I1231 06:07:03.050391  2138 solver.cpp:409]     Test net output #3: loss_f = 1.98733 (* 1 = 1.98733 loss)
    I1231 06:07:03.119812  2138 solver.cpp:237] Iteration 147000, loss = 2.56085
    I1231 06:07:03.119848  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.76
    I1231 06:07:03.119858  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.56
    I1231 06:07:03.119868  2138 solver.cpp:253]     Train net output #2: loss_c = 0.987198 (* 1 = 0.987198 loss)
    I1231 06:07:03.119877  2138 solver.cpp:253]     Train net output #3: loss_f = 1.57365 (* 1 = 1.57365 loss)
    I1231 06:07:03.119887  2138 sgd_solver.cpp:106] Iteration 147000, lr = 7.60723e-05
    I1231 06:07:16.618124  2138 solver.cpp:237] Iteration 147100, loss = 2.84686
    I1231 06:07:16.618165  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.68
    I1231 06:07:16.618176  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1231 06:07:16.618190  2138 solver.cpp:253]     Train net output #2: loss_c = 1.12777 (* 1 = 1.12777 loss)
    I1231 06:07:16.618198  2138 solver.cpp:253]     Train net output #3: loss_f = 1.71909 (* 1 = 1.71909 loss)
    I1231 06:07:16.618209  2138 sgd_solver.cpp:106] Iteration 147100, lr = 7.6036e-05
    I1231 06:07:29.954643  2138 solver.cpp:237] Iteration 147200, loss = 2.72291
    I1231 06:07:29.954772  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.72
    I1231 06:07:29.954798  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.47
    I1231 06:07:29.954814  2138 solver.cpp:253]     Train net output #2: loss_c = 0.942553 (* 1 = 0.942553 loss)
    I1231 06:07:29.954828  2138 solver.cpp:253]     Train net output #3: loss_f = 1.78035 (* 1 = 1.78035 loss)
    I1231 06:07:29.954850  2138 sgd_solver.cpp:106] Iteration 147200, lr = 7.59997e-05
    I1231 06:07:43.302773  2138 solver.cpp:237] Iteration 147300, loss = 2.97918
    I1231 06:07:43.302809  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1231 06:07:43.302819  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.56
    I1231 06:07:43.302829  2138 solver.cpp:253]     Train net output #2: loss_c = 1.24342 (* 1 = 1.24342 loss)
    I1231 06:07:43.302836  2138 solver.cpp:253]     Train net output #3: loss_f = 1.73576 (* 1 = 1.73576 loss)
    I1231 06:07:43.302845  2138 sgd_solver.cpp:106] Iteration 147300, lr = 7.59635e-05
    I1231 06:07:56.670791  2138 solver.cpp:237] Iteration 147400, loss = 2.34985
    I1231 06:07:56.670837  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.72
    I1231 06:07:56.670847  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.6
    I1231 06:07:56.670860  2138 solver.cpp:253]     Train net output #2: loss_c = 0.93196 (* 1 = 0.93196 loss)
    I1231 06:07:56.670869  2138 solver.cpp:253]     Train net output #3: loss_f = 1.41789 (* 1 = 1.41789 loss)
    I1231 06:07:56.670879  2138 sgd_solver.cpp:106] Iteration 147400, lr = 7.59273e-05
    I1231 06:08:10.049121  2138 solver.cpp:237] Iteration 147500, loss = 2.41439
    I1231 06:08:10.049271  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.71
    I1231 06:08:10.049293  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.62
    I1231 06:08:10.049302  2138 solver.cpp:253]     Train net output #2: loss_c = 0.925961 (* 1 = 0.925961 loss)
    I1231 06:08:10.049310  2138 solver.cpp:253]     Train net output #3: loss_f = 1.48843 (* 1 = 1.48843 loss)
    I1231 06:08:10.049319  2138 sgd_solver.cpp:106] Iteration 147500, lr = 7.58911e-05
    I1231 06:08:23.433985  2138 solver.cpp:237] Iteration 147600, loss = 2.83468
    I1231 06:08:23.434020  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.61
    I1231 06:08:23.434029  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.57
    I1231 06:08:23.434041  2138 solver.cpp:253]     Train net output #2: loss_c = 1.11712 (* 1 = 1.11712 loss)
    I1231 06:08:23.434048  2138 solver.cpp:253]     Train net output #3: loss_f = 1.71757 (* 1 = 1.71757 loss)
    I1231 06:08:23.434056  2138 sgd_solver.cpp:106] Iteration 147600, lr = 7.5855e-05
    I1231 06:08:36.766068  2138 solver.cpp:237] Iteration 147700, loss = 2.62481
    I1231 06:08:36.766103  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.68
    I1231 06:08:36.766113  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.6
    I1231 06:08:36.766122  2138 solver.cpp:253]     Train net output #2: loss_c = 0.975592 (* 1 = 0.975592 loss)
    I1231 06:08:36.766130  2138 solver.cpp:253]     Train net output #3: loss_f = 1.64922 (* 1 = 1.64922 loss)
    I1231 06:08:36.766139  2138 sgd_solver.cpp:106] Iteration 147700, lr = 7.58189e-05
    I1231 06:08:50.089320  2138 solver.cpp:237] Iteration 147800, loss = 2.66334
    I1231 06:08:50.089443  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.67
    I1231 06:08:50.089464  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.61
    I1231 06:08:50.089475  2138 solver.cpp:253]     Train net output #2: loss_c = 1.05528 (* 1 = 1.05528 loss)
    I1231 06:08:50.089483  2138 solver.cpp:253]     Train net output #3: loss_f = 1.60806 (* 1 = 1.60806 loss)
    I1231 06:08:50.089493  2138 sgd_solver.cpp:106] Iteration 147800, lr = 7.57829e-05
    I1231 06:09:03.492894  2138 solver.cpp:237] Iteration 147900, loss = 2.2202
    I1231 06:09:03.492949  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.7
    I1231 06:09:03.492964  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.66
    I1231 06:09:03.492982  2138 solver.cpp:253]     Train net output #2: loss_c = 0.870057 (* 1 = 0.870057 loss)
    I1231 06:09:03.492998  2138 solver.cpp:253]     Train net output #3: loss_f = 1.35014 (* 1 = 1.35014 loss)
    I1231 06:09:03.493012  2138 sgd_solver.cpp:106] Iteration 147900, lr = 7.57469e-05
    I1231 06:09:16.779404  2138 solver.cpp:341] Iteration 148000, Testing net (#0)
    I1231 06:09:21.843276  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.588667
    I1231 06:09:21.843400  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.466917
    I1231 06:09:21.843413  2138 solver.cpp:409]     Test net output #2: loss_c = 1.3118 (* 1 = 1.3118 loss)
    I1231 06:09:21.843422  2138 solver.cpp:409]     Test net output #3: loss_f = 1.99368 (* 1 = 1.99368 loss)
    I1231 06:09:21.903728  2138 solver.cpp:237] Iteration 148000, loss = 2.50215
    I1231 06:09:21.903774  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.71
    I1231 06:09:21.903784  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.55
    I1231 06:09:21.903796  2138 solver.cpp:253]     Train net output #2: loss_c = 0.907243 (* 1 = 0.907243 loss)
    I1231 06:09:21.903806  2138 solver.cpp:253]     Train net output #3: loss_f = 1.59491 (* 1 = 1.59491 loss)
    I1231 06:09:21.903817  2138 sgd_solver.cpp:106] Iteration 148000, lr = 7.57109e-05
    I1231 06:09:35.382504  2138 solver.cpp:237] Iteration 148100, loss = 3.21983
    I1231 06:09:35.382555  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.58
    I1231 06:09:35.382572  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.51
    I1231 06:09:35.382591  2138 solver.cpp:253]     Train net output #2: loss_c = 1.32573 (* 1 = 1.32573 loss)
    I1231 06:09:35.382606  2138 solver.cpp:253]     Train net output #3: loss_f = 1.8941 (* 1 = 1.8941 loss)
    I1231 06:09:35.382622  2138 sgd_solver.cpp:106] Iteration 148100, lr = 7.5675e-05
    I1231 06:09:48.837824  2138 solver.cpp:237] Iteration 148200, loss = 2.68891
    I1231 06:09:48.837868  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.73
    I1231 06:09:48.837877  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.59
    I1231 06:09:48.837887  2138 solver.cpp:253]     Train net output #2: loss_c = 0.980896 (* 1 = 0.980896 loss)
    I1231 06:09:48.837894  2138 solver.cpp:253]     Train net output #3: loss_f = 1.70802 (* 1 = 1.70802 loss)
    I1231 06:09:48.837903  2138 sgd_solver.cpp:106] Iteration 148200, lr = 7.56391e-05
    I1231 06:10:02.211529  2138 solver.cpp:237] Iteration 148300, loss = 2.752
    I1231 06:10:02.211696  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1231 06:10:02.211709  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.61
    I1231 06:10:02.211719  2138 solver.cpp:253]     Train net output #2: loss_c = 1.08816 (* 1 = 1.08816 loss)
    I1231 06:10:02.211726  2138 solver.cpp:253]     Train net output #3: loss_f = 1.66385 (* 1 = 1.66385 loss)
    I1231 06:10:02.211735  2138 sgd_solver.cpp:106] Iteration 148300, lr = 7.56033e-05
    I1231 06:10:15.552367  2138 solver.cpp:237] Iteration 148400, loss = 2.43593
    I1231 06:10:15.552404  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.64
    I1231 06:10:15.552414  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.61
    I1231 06:10:15.552425  2138 solver.cpp:253]     Train net output #2: loss_c = 0.973249 (* 1 = 0.973249 loss)
    I1231 06:10:15.552434  2138 solver.cpp:253]     Train net output #3: loss_f = 1.46268 (* 1 = 1.46268 loss)
    I1231 06:10:15.552443  2138 sgd_solver.cpp:106] Iteration 148400, lr = 7.55675e-05
    I1231 06:10:28.936435  2138 solver.cpp:237] Iteration 148500, loss = 2.58353
    I1231 06:10:28.936471  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.71
    I1231 06:10:28.936480  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.61
    I1231 06:10:28.936491  2138 solver.cpp:253]     Train net output #2: loss_c = 1.02229 (* 1 = 1.02229 loss)
    I1231 06:10:28.936499  2138 solver.cpp:253]     Train net output #3: loss_f = 1.56124 (* 1 = 1.56124 loss)
    I1231 06:10:28.936508  2138 sgd_solver.cpp:106] Iteration 148500, lr = 7.55317e-05
    I1231 06:10:42.256090  2138 solver.cpp:237] Iteration 148600, loss = 2.99594
    I1231 06:10:42.256220  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1231 06:10:42.256232  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.57
    I1231 06:10:42.256242  2138 solver.cpp:253]     Train net output #2: loss_c = 1.1926 (* 1 = 1.1926 loss)
    I1231 06:10:42.256250  2138 solver.cpp:253]     Train net output #3: loss_f = 1.80334 (* 1 = 1.80334 loss)
    I1231 06:10:42.256258  2138 sgd_solver.cpp:106] Iteration 148600, lr = 7.5496e-05
    I1231 06:10:55.679983  2138 solver.cpp:237] Iteration 148700, loss = 2.6756
    I1231 06:10:55.680030  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.67
    I1231 06:10:55.680042  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.54
    I1231 06:10:55.680052  2138 solver.cpp:253]     Train net output #2: loss_c = 0.987313 (* 1 = 0.987313 loss)
    I1231 06:10:55.680061  2138 solver.cpp:253]     Train net output #3: loss_f = 1.68828 (* 1 = 1.68828 loss)
    I1231 06:10:55.680071  2138 sgd_solver.cpp:106] Iteration 148700, lr = 7.54603e-05
    I1231 06:11:09.074872  2138 solver.cpp:237] Iteration 148800, loss = 2.68766
    I1231 06:11:09.074908  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1231 06:11:09.074916  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.54
    I1231 06:11:09.074926  2138 solver.cpp:253]     Train net output #2: loss_c = 1.09246 (* 1 = 1.09246 loss)
    I1231 06:11:09.074934  2138 solver.cpp:253]     Train net output #3: loss_f = 1.5952 (* 1 = 1.5952 loss)
    I1231 06:11:09.074944  2138 sgd_solver.cpp:106] Iteration 148800, lr = 7.54247e-05
    I1231 06:11:22.532462  2138 solver.cpp:237] Iteration 148900, loss = 2.51225
    I1231 06:11:22.532587  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1231 06:11:22.532600  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.64
    I1231 06:11:22.532613  2138 solver.cpp:253]     Train net output #2: loss_c = 1.0352 (* 1 = 1.0352 loss)
    I1231 06:11:22.532624  2138 solver.cpp:253]     Train net output #3: loss_f = 1.47705 (* 1 = 1.47705 loss)
    I1231 06:11:22.532632  2138 sgd_solver.cpp:106] Iteration 148900, lr = 7.53891e-05
    I1231 06:11:35.759099  2138 solver.cpp:341] Iteration 149000, Testing net (#0)
    I1231 06:11:40.808928  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.585333
    I1231 06:11:40.808969  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.460417
    I1231 06:11:40.808984  2138 solver.cpp:409]     Test net output #2: loss_c = 1.32166 (* 1 = 1.32166 loss)
    I1231 06:11:40.808995  2138 solver.cpp:409]     Test net output #3: loss_f = 2.00916 (* 1 = 2.00916 loss)
    I1231 06:11:40.869251  2138 solver.cpp:237] Iteration 149000, loss = 2.55914
    I1231 06:11:40.869290  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.73
    I1231 06:11:40.869302  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.6
    I1231 06:11:40.869315  2138 solver.cpp:253]     Train net output #2: loss_c = 0.979388 (* 1 = 0.979388 loss)
    I1231 06:11:40.869325  2138 solver.cpp:253]     Train net output #3: loss_f = 1.57975 (* 1 = 1.57975 loss)
    I1231 06:11:40.869336  2138 sgd_solver.cpp:106] Iteration 149000, lr = 7.53535e-05
    I1231 06:11:54.315853  2138 solver.cpp:237] Iteration 149100, loss = 2.82864
    I1231 06:11:54.315953  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1231 06:11:54.315965  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.56
    I1231 06:11:54.315979  2138 solver.cpp:253]     Train net output #2: loss_c = 1.08207 (* 1 = 1.08207 loss)
    I1231 06:11:54.315989  2138 solver.cpp:253]     Train net output #3: loss_f = 1.74657 (* 1 = 1.74657 loss)
    I1231 06:11:54.315997  2138 sgd_solver.cpp:106] Iteration 149100, lr = 7.5318e-05
    I1231 06:12:07.734880  2138 solver.cpp:237] Iteration 149200, loss = 2.51091
    I1231 06:12:07.734921  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.76
    I1231 06:12:07.734933  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.59
    I1231 06:12:07.734946  2138 solver.cpp:253]     Train net output #2: loss_c = 0.882446 (* 1 = 0.882446 loss)
    I1231 06:12:07.734956  2138 solver.cpp:253]     Train net output #3: loss_f = 1.62846 (* 1 = 1.62846 loss)
    I1231 06:12:07.734967  2138 sgd_solver.cpp:106] Iteration 149200, lr = 7.52825e-05
    I1231 06:12:21.080442  2138 solver.cpp:237] Iteration 149300, loss = 2.72498
    I1231 06:12:21.080482  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.68
    I1231 06:12:21.080493  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.55
    I1231 06:12:21.080505  2138 solver.cpp:253]     Train net output #2: loss_c = 1.07087 (* 1 = 1.07087 loss)
    I1231 06:12:21.080515  2138 solver.cpp:253]     Train net output #3: loss_f = 1.65411 (* 1 = 1.65411 loss)
    I1231 06:12:21.080526  2138 sgd_solver.cpp:106] Iteration 149300, lr = 7.5247e-05
    I1231 06:12:34.434806  2138 solver.cpp:237] Iteration 149400, loss = 2.30494
    I1231 06:12:34.434902  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.66
    I1231 06:12:34.434916  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.64
    I1231 06:12:34.434927  2138 solver.cpp:253]     Train net output #2: loss_c = 0.951722 (* 1 = 0.951722 loss)
    I1231 06:12:34.434938  2138 solver.cpp:253]     Train net output #3: loss_f = 1.35322 (* 1 = 1.35322 loss)
    I1231 06:12:34.434948  2138 sgd_solver.cpp:106] Iteration 149400, lr = 7.52116e-05
    I1231 06:12:47.867127  2138 solver.cpp:237] Iteration 149500, loss = 2.465
    I1231 06:12:47.867172  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.69
    I1231 06:12:47.867182  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.54
    I1231 06:12:47.867195  2138 solver.cpp:253]     Train net output #2: loss_c = 0.889394 (* 1 = 0.889394 loss)
    I1231 06:12:47.867207  2138 solver.cpp:253]     Train net output #3: loss_f = 1.5756 (* 1 = 1.5756 loss)
    I1231 06:12:47.867218  2138 sgd_solver.cpp:106] Iteration 149500, lr = 7.51763e-05
    I1231 06:13:01.263911  2138 solver.cpp:237] Iteration 149600, loss = 2.53617
    I1231 06:13:01.263952  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.67
    I1231 06:13:01.263963  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.57
    I1231 06:13:01.263977  2138 solver.cpp:253]     Train net output #2: loss_c = 0.989398 (* 1 = 0.989398 loss)
    I1231 06:13:01.263988  2138 solver.cpp:253]     Train net output #3: loss_f = 1.54677 (* 1 = 1.54677 loss)
    I1231 06:13:01.263998  2138 sgd_solver.cpp:106] Iteration 149600, lr = 7.51409e-05
    I1231 06:13:14.685462  2138 solver.cpp:237] Iteration 149700, loss = 2.60986
    I1231 06:13:14.685585  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.78
    I1231 06:13:14.685600  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.61
    I1231 06:13:14.685612  2138 solver.cpp:253]     Train net output #2: loss_c = 0.942807 (* 1 = 0.942807 loss)
    I1231 06:13:14.685622  2138 solver.cpp:253]     Train net output #3: loss_f = 1.66705 (* 1 = 1.66705 loss)
    I1231 06:13:14.685632  2138 sgd_solver.cpp:106] Iteration 149700, lr = 7.51057e-05
    I1231 06:13:28.007290  2138 solver.cpp:237] Iteration 149800, loss = 2.71886
    I1231 06:13:28.007330  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.63
    I1231 06:13:28.007341  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.53
    I1231 06:13:28.007354  2138 solver.cpp:253]     Train net output #2: loss_c = 1.07193 (* 1 = 1.07193 loss)
    I1231 06:13:28.007364  2138 solver.cpp:253]     Train net output #3: loss_f = 1.64693 (* 1 = 1.64693 loss)
    I1231 06:13:28.007374  2138 sgd_solver.cpp:106] Iteration 149800, lr = 7.50704e-05
    I1231 06:13:41.355706  2138 solver.cpp:237] Iteration 149900, loss = 2.38288
    I1231 06:13:41.355746  2138 solver.cpp:253]     Train net output #0: accuracy_c = 0.65
    I1231 06:13:41.355757  2138 solver.cpp:253]     Train net output #1: accuracy_f = 0.62
    I1231 06:13:41.355770  2138 solver.cpp:253]     Train net output #2: loss_c = 0.957244 (* 1 = 0.957244 loss)
    I1231 06:13:41.355782  2138 solver.cpp:253]     Train net output #3: loss_f = 1.42564 (* 1 = 1.42564 loss)
    I1231 06:13:41.355792  2138 sgd_solver.cpp:106] Iteration 149900, lr = 7.50352e-05
    I1231 06:13:54.628958  2138 solver.cpp:459] Snapshotting to binary proto file cnn_snapshot_iter_150000.caffemodel
    I1231 06:13:54.710508  2138 sgd_solver.cpp:269] Snapshotting solver state to binary proto file cnn_snapshot_iter_150000.solverstate
    I1231 06:13:54.762184  2138 solver.cpp:321] Iteration 150000, loss = 2.71164
    I1231 06:13:54.762255  2138 solver.cpp:341] Iteration 150000, Testing net (#0)
    I1231 06:13:59.879096  2138 solver.cpp:409]     Test net output #0: accuracy_c = 0.58575
    I1231 06:13:59.879139  2138 solver.cpp:409]     Test net output #1: accuracy_f = 0.46975
    I1231 06:13:59.879153  2138 solver.cpp:409]     Test net output #2: loss_c = 1.31659 (* 1 = 1.31659 loss)
    I1231 06:13:59.879165  2138 solver.cpp:409]     Test net output #3: loss_f = 2.00393 (* 1 = 2.00393 loss)
    I1231 06:13:59.879174  2138 solver.cpp:326] Optimization Done.
    I1231 06:13:59.879182  2138 caffe.cpp:215] Optimization Done.
    CPU times: user 1min 6s, sys: 7.82 s, total: 1min 14s
    Wall time: 6h 20min 57s


Caffe brewed.
## Test the model completely on test data
Let's test directly in command-line:


```python
%%time
!$CAFFE_ROOT/build/tools/caffe test -model cnn_test.prototxt -weights cnn_snapshot_iter_150000.caffemodel -iterations 83
```

    /root/caffe/build/tools/caffe: /root/anaconda2/lib/liblzma.so.5: no version information available (required by /usr/lib/x86_64-linux-gnu/libunwind.so.8)
    I1231 10:31:19.907760  9759 caffe.cpp:234] Use CPU.
    I1231 10:31:20.073982  9759 net.cpp:49] Initializing net from parameters:
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
      name: "cccp1a"
      type: "Convolution"
      bottom: "conv1"
      top: "cccp1a"
      convolution_param {
        num_output: 42
        kernel_size: 1
        weight_filler {
          type: "xavier"
        }
      }
    }
    layer {
      name: "relu1a"
      type: "ReLU"
      bottom: "cccp1a"
      top: "cccp1a"
    }
    layer {
      name: "cccp1b"
      type: "Convolution"
      bottom: "cccp1a"
      top: "cccp1b"
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
      bottom: "cccp1b"
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
      name: "relu1b"
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
        num_output: 768
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
    I1231 10:31:20.074616  9759 layer_factory.hpp:77] Creating layer data
    I1231 10:31:20.074633  9759 net.cpp:106] Creating Layer data
    I1231 10:31:20.074641  9759 net.cpp:411] data -> data
    I1231 10:31:20.074658  9759 net.cpp:411] data -> label_coarse
    I1231 10:31:20.074669  9759 net.cpp:411] data -> label_fine
    I1231 10:31:20.074681  9759 hdf5_data_layer.cpp:79] Loading list of HDF5 filenames from: cifar_100_caffe_hdf5/test.txt
    I1231 10:31:20.074714  9759 hdf5_data_layer.cpp:93] Number of HDF5 files: 1
    I1231 10:31:20.075601  9759 hdf5.cpp:35] Datatype class: H5T_INTEGER
    I1231 10:31:23.082351  9759 net.cpp:150] Setting up data
    I1231 10:31:23.082401  9759 net.cpp:157] Top shape: 120 3 32 32 (368640)
    I1231 10:31:23.082411  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082417  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082423  9759 net.cpp:165] Memory required for data: 1475520
    I1231 10:31:23.082434  9759 layer_factory.hpp:77] Creating layer label_coarse_data_1_split
    I1231 10:31:23.082486  9759 net.cpp:106] Creating Layer label_coarse_data_1_split
    I1231 10:31:23.082494  9759 net.cpp:454] label_coarse_data_1_split <- label_coarse
    I1231 10:31:23.082506  9759 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_0
    I1231 10:31:23.082516  9759 net.cpp:411] label_coarse_data_1_split -> label_coarse_data_1_split_1
    I1231 10:31:23.082528  9759 net.cpp:150] Setting up label_coarse_data_1_split
    I1231 10:31:23.082535  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082541  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082546  9759 net.cpp:165] Memory required for data: 1476480
    I1231 10:31:23.082551  9759 layer_factory.hpp:77] Creating layer label_fine_data_2_split
    I1231 10:31:23.082559  9759 net.cpp:106] Creating Layer label_fine_data_2_split
    I1231 10:31:23.082566  9759 net.cpp:454] label_fine_data_2_split <- label_fine
    I1231 10:31:23.082571  9759 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_0
    I1231 10:31:23.082578  9759 net.cpp:411] label_fine_data_2_split -> label_fine_data_2_split_1
    I1231 10:31:23.082597  9759 net.cpp:150] Setting up label_fine_data_2_split
    I1231 10:31:23.082604  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082610  9759 net.cpp:157] Top shape: 120 (120)
    I1231 10:31:23.082615  9759 net.cpp:165] Memory required for data: 1477440
    I1231 10:31:23.082622  9759 layer_factory.hpp:77] Creating layer conv1
    I1231 10:31:23.082633  9759 net.cpp:106] Creating Layer conv1
    I1231 10:31:23.082649  9759 net.cpp:454] conv1 <- data
    I1231 10:31:23.082656  9759 net.cpp:411] conv1 -> conv1
    I1231 10:31:23.083060  9759 net.cpp:150] Setting up conv1
    I1231 10:31:23.083083  9759 net.cpp:157] Top shape: 120 64 29 29 (6458880)
    I1231 10:31:23.083089  9759 net.cpp:165] Memory required for data: 27312960
    I1231 10:31:23.083104  9759 layer_factory.hpp:77] Creating layer cccp1a
    I1231 10:31:23.083115  9759 net.cpp:106] Creating Layer cccp1a
    I1231 10:31:23.083132  9759 net.cpp:454] cccp1a <- conv1
    I1231 10:31:23.083138  9759 net.cpp:411] cccp1a -> cccp1a
    I1231 10:31:23.083173  9759 net.cpp:150] Setting up cccp1a
    I1231 10:31:23.083179  9759 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1231 10:31:23.083185  9759 net.cpp:165] Memory required for data: 44267520
    I1231 10:31:23.083194  9759 layer_factory.hpp:77] Creating layer relu1a
    I1231 10:31:23.083201  9759 net.cpp:106] Creating Layer relu1a
    I1231 10:31:23.083206  9759 net.cpp:454] relu1a <- cccp1a
    I1231 10:31:23.083214  9759 net.cpp:397] relu1a -> cccp1a (in-place)
    I1231 10:31:23.083223  9759 net.cpp:150] Setting up relu1a
    I1231 10:31:23.083230  9759 net.cpp:157] Top shape: 120 42 29 29 (4238640)
    I1231 10:31:23.083235  9759 net.cpp:165] Memory required for data: 61222080
    I1231 10:31:23.083240  9759 layer_factory.hpp:77] Creating layer cccp1b
    I1231 10:31:23.083248  9759 net.cpp:106] Creating Layer cccp1b
    I1231 10:31:23.083253  9759 net.cpp:454] cccp1b <- cccp1a
    I1231 10:31:23.083271  9759 net.cpp:411] cccp1b -> cccp1b
    I1231 10:31:23.083295  9759 net.cpp:150] Setting up cccp1b
    I1231 10:31:23.083303  9759 net.cpp:157] Top shape: 120 32 29 29 (3229440)
    I1231 10:31:23.083319  9759 net.cpp:165] Memory required for data: 74139840
    I1231 10:31:23.083328  9759 layer_factory.hpp:77] Creating layer pool1
    I1231 10:31:23.083336  9759 net.cpp:106] Creating Layer pool1
    I1231 10:31:23.083341  9759 net.cpp:454] pool1 <- cccp1b
    I1231 10:31:23.083348  9759 net.cpp:411] pool1 -> pool1
    I1231 10:31:23.083365  9759 net.cpp:150] Setting up pool1
    I1231 10:31:23.083372  9759 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1231 10:31:23.083377  9759 net.cpp:165] Memory required for data: 77150400
    I1231 10:31:23.083384  9759 layer_factory.hpp:77] Creating layer drop1
    I1231 10:31:23.083395  9759 net.cpp:106] Creating Layer drop1
    I1231 10:31:23.083400  9759 net.cpp:454] drop1 <- pool1
    I1231 10:31:23.083406  9759 net.cpp:397] drop1 -> pool1 (in-place)
    I1231 10:31:23.083415  9759 net.cpp:150] Setting up drop1
    I1231 10:31:23.083421  9759 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1231 10:31:23.083434  9759 net.cpp:165] Memory required for data: 80160960
    I1231 10:31:23.083441  9759 layer_factory.hpp:77] Creating layer relu1b
    I1231 10:31:23.083457  9759 net.cpp:106] Creating Layer relu1b
    I1231 10:31:23.083463  9759 net.cpp:454] relu1b <- pool1
    I1231 10:31:23.083470  9759 net.cpp:397] relu1b -> pool1 (in-place)
    I1231 10:31:23.083478  9759 net.cpp:150] Setting up relu1b
    I1231 10:31:23.083484  9759 net.cpp:157] Top shape: 120 32 14 14 (752640)
    I1231 10:31:23.083500  9759 net.cpp:165] Memory required for data: 83171520
    I1231 10:31:23.083505  9759 layer_factory.hpp:77] Creating layer conv2
    I1231 10:31:23.083513  9759 net.cpp:106] Creating Layer conv2
    I1231 10:31:23.083518  9759 net.cpp:454] conv2 <- pool1
    I1231 10:31:23.083525  9759 net.cpp:411] conv2 -> conv2
    I1231 10:31:23.083693  9759 net.cpp:150] Setting up conv2
    I1231 10:31:23.083716  9759 net.cpp:157] Top shape: 120 42 11 11 (609840)
    I1231 10:31:23.083731  9759 net.cpp:165] Memory required for data: 85610880
    I1231 10:31:23.083739  9759 layer_factory.hpp:77] Creating layer pool2
    I1231 10:31:23.083745  9759 net.cpp:106] Creating Layer pool2
    I1231 10:31:23.083751  9759 net.cpp:454] pool2 <- conv2
    I1231 10:31:23.083757  9759 net.cpp:411] pool2 -> pool2
    I1231 10:31:23.083766  9759 net.cpp:150] Setting up pool2
    I1231 10:31:23.083772  9759 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1231 10:31:23.083778  9759 net.cpp:165] Memory required for data: 86114880
    I1231 10:31:23.083783  9759 layer_factory.hpp:77] Creating layer drop2
    I1231 10:31:23.083791  9759 net.cpp:106] Creating Layer drop2
    I1231 10:31:23.083796  9759 net.cpp:454] drop2 <- pool2
    I1231 10:31:23.083801  9759 net.cpp:397] drop2 -> pool2 (in-place)
    I1231 10:31:23.083808  9759 net.cpp:150] Setting up drop2
    I1231 10:31:23.083814  9759 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1231 10:31:23.083820  9759 net.cpp:165] Memory required for data: 86618880
    I1231 10:31:23.083825  9759 layer_factory.hpp:77] Creating layer relu2
    I1231 10:31:23.083832  9759 net.cpp:106] Creating Layer relu2
    I1231 10:31:23.083847  9759 net.cpp:454] relu2 <- pool2
    I1231 10:31:23.083853  9759 net.cpp:397] relu2 -> pool2 (in-place)
    I1231 10:31:23.083861  9759 net.cpp:150] Setting up relu2
    I1231 10:31:23.083868  9759 net.cpp:157] Top shape: 120 42 5 5 (126000)
    I1231 10:31:23.083884  9759 net.cpp:165] Memory required for data: 87122880
    I1231 10:31:23.083889  9759 layer_factory.hpp:77] Creating layer conv3
    I1231 10:31:23.083895  9759 net.cpp:106] Creating Layer conv3
    I1231 10:31:23.083901  9759 net.cpp:454] conv3 <- pool2
    I1231 10:31:23.083907  9759 net.cpp:411] conv3 -> conv3
    I1231 10:31:23.083983  9759 net.cpp:150] Setting up conv3
    I1231 10:31:23.083991  9759 net.cpp:157] Top shape: 120 64 4 4 (122880)
    I1231 10:31:23.083997  9759 net.cpp:165] Memory required for data: 87614400
    I1231 10:31:23.084005  9759 layer_factory.hpp:77] Creating layer pool3
    I1231 10:31:23.084012  9759 net.cpp:106] Creating Layer pool3
    I1231 10:31:23.084028  9759 net.cpp:454] pool3 <- conv3
    I1231 10:31:23.084035  9759 net.cpp:411] pool3 -> pool3
    I1231 10:31:23.084043  9759 net.cpp:150] Setting up pool3
    I1231 10:31:23.084050  9759 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1231 10:31:23.084065  9759 net.cpp:165] Memory required for data: 87737280
    I1231 10:31:23.084071  9759 layer_factory.hpp:77] Creating layer relu3
    I1231 10:31:23.084079  9759 net.cpp:106] Creating Layer relu3
    I1231 10:31:23.084084  9759 net.cpp:454] relu3 <- pool3
    I1231 10:31:23.084089  9759 net.cpp:397] relu3 -> pool3 (in-place)
    I1231 10:31:23.084096  9759 net.cpp:150] Setting up relu3
    I1231 10:31:23.084102  9759 net.cpp:157] Top shape: 120 64 2 2 (30720)
    I1231 10:31:23.084107  9759 net.cpp:165] Memory required for data: 87860160
    I1231 10:31:23.084112  9759 layer_factory.hpp:77] Creating layer ip1
    I1231 10:31:23.084120  9759 net.cpp:106] Creating Layer ip1
    I1231 10:31:23.084125  9759 net.cpp:454] ip1 <- pool3
    I1231 10:31:23.084132  9759 net.cpp:411] ip1 -> ip1
    I1231 10:31:23.085372  9759 net.cpp:150] Setting up ip1
    I1231 10:31:23.085392  9759 net.cpp:157] Top shape: 120 768 (92160)
    I1231 10:31:23.085408  9759 net.cpp:165] Memory required for data: 88228800
    I1231 10:31:23.085427  9759 layer_factory.hpp:77] Creating layer sig1
    I1231 10:31:23.085434  9759 net.cpp:106] Creating Layer sig1
    I1231 10:31:23.085439  9759 net.cpp:454] sig1 <- ip1
    I1231 10:31:23.085446  9759 net.cpp:397] sig1 -> ip1 (in-place)
    I1231 10:31:23.085453  9759 net.cpp:150] Setting up sig1
    I1231 10:31:23.085459  9759 net.cpp:157] Top shape: 120 768 (92160)
    I1231 10:31:23.085464  9759 net.cpp:165] Memory required for data: 88597440
    I1231 10:31:23.085469  9759 layer_factory.hpp:77] Creating layer ip1_sig1_0_split
    I1231 10:31:23.085476  9759 net.cpp:106] Creating Layer ip1_sig1_0_split
    I1231 10:31:23.085481  9759 net.cpp:454] ip1_sig1_0_split <- ip1
    I1231 10:31:23.085487  9759 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_0
    I1231 10:31:23.085495  9759 net.cpp:411] ip1_sig1_0_split -> ip1_sig1_0_split_1
    I1231 10:31:23.085503  9759 net.cpp:150] Setting up ip1_sig1_0_split
    I1231 10:31:23.085510  9759 net.cpp:157] Top shape: 120 768 (92160)
    I1231 10:31:23.085516  9759 net.cpp:157] Top shape: 120 768 (92160)
    I1231 10:31:23.085521  9759 net.cpp:165] Memory required for data: 89334720
    I1231 10:31:23.085526  9759 layer_factory.hpp:77] Creating layer ip_c
    I1231 10:31:23.085533  9759 net.cpp:106] Creating Layer ip_c
    I1231 10:31:23.085538  9759 net.cpp:454] ip_c <- ip1_sig1_0_split_0
    I1231 10:31:23.085546  9759 net.cpp:411] ip_c -> ip_c
    I1231 10:31:23.085667  9759 net.cpp:150] Setting up ip_c
    I1231 10:31:23.085675  9759 net.cpp:157] Top shape: 120 20 (2400)
    I1231 10:31:23.085680  9759 net.cpp:165] Memory required for data: 89344320
    I1231 10:31:23.085687  9759 layer_factory.hpp:77] Creating layer ip_c_ip_c_0_split
    I1231 10:31:23.085695  9759 net.cpp:106] Creating Layer ip_c_ip_c_0_split
    I1231 10:31:23.085700  9759 net.cpp:454] ip_c_ip_c_0_split <- ip_c
    I1231 10:31:23.085705  9759 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_0
    I1231 10:31:23.085713  9759 net.cpp:411] ip_c_ip_c_0_split -> ip_c_ip_c_0_split_1
    I1231 10:31:23.085721  9759 net.cpp:150] Setting up ip_c_ip_c_0_split
    I1231 10:31:23.085727  9759 net.cpp:157] Top shape: 120 20 (2400)
    I1231 10:31:23.085732  9759 net.cpp:157] Top shape: 120 20 (2400)
    I1231 10:31:23.085737  9759 net.cpp:165] Memory required for data: 89363520
    I1231 10:31:23.085753  9759 layer_factory.hpp:77] Creating layer accuracy_c
    I1231 10:31:23.085765  9759 net.cpp:106] Creating Layer accuracy_c
    I1231 10:31:23.085772  9759 net.cpp:454] accuracy_c <- ip_c_ip_c_0_split_0
    I1231 10:31:23.085779  9759 net.cpp:454] accuracy_c <- label_coarse_data_1_split_0
    I1231 10:31:23.085795  9759 net.cpp:411] accuracy_c -> accuracy_c
    I1231 10:31:23.085803  9759 net.cpp:150] Setting up accuracy_c
    I1231 10:31:23.085809  9759 net.cpp:157] Top shape: (1)
    I1231 10:31:23.085814  9759 net.cpp:165] Memory required for data: 89363524
    I1231 10:31:23.085820  9759 layer_factory.hpp:77] Creating layer loss_c
    I1231 10:31:23.085826  9759 net.cpp:106] Creating Layer loss_c
    I1231 10:31:23.085832  9759 net.cpp:454] loss_c <- ip_c_ip_c_0_split_1
    I1231 10:31:23.085839  9759 net.cpp:454] loss_c <- label_coarse_data_1_split_1
    I1231 10:31:23.085844  9759 net.cpp:411] loss_c -> loss_c
    I1231 10:31:23.085855  9759 layer_factory.hpp:77] Creating layer loss_c
    I1231 10:31:23.085868  9759 net.cpp:150] Setting up loss_c
    I1231 10:31:23.085875  9759 net.cpp:157] Top shape: (1)
    I1231 10:31:23.085880  9759 net.cpp:160]     with loss weight 1
    I1231 10:31:23.085899  9759 net.cpp:165] Memory required for data: 89363528
    I1231 10:31:23.085904  9759 layer_factory.hpp:77] Creating layer ip_f
    I1231 10:31:23.085911  9759 net.cpp:106] Creating Layer ip_f
    I1231 10:31:23.085917  9759 net.cpp:454] ip_f <- ip1_sig1_0_split_1
    I1231 10:31:23.085933  9759 net.cpp:411] ip_f -> ip_f
    I1231 10:31:23.086418  9759 net.cpp:150] Setting up ip_f
    I1231 10:31:23.086438  9759 net.cpp:157] Top shape: 120 100 (12000)
    I1231 10:31:23.086444  9759 net.cpp:165] Memory required for data: 89411528
    I1231 10:31:23.086452  9759 layer_factory.hpp:77] Creating layer ip_f_ip_f_0_split
    I1231 10:31:23.086459  9759 net.cpp:106] Creating Layer ip_f_ip_f_0_split
    I1231 10:31:23.086482  9759 net.cpp:454] ip_f_ip_f_0_split <- ip_f
    I1231 10:31:23.086489  9759 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_0
    I1231 10:31:23.086498  9759 net.cpp:411] ip_f_ip_f_0_split -> ip_f_ip_f_0_split_1
    I1231 10:31:23.086505  9759 net.cpp:150] Setting up ip_f_ip_f_0_split
    I1231 10:31:23.086511  9759 net.cpp:157] Top shape: 120 100 (12000)
    I1231 10:31:23.086518  9759 net.cpp:157] Top shape: 120 100 (12000)
    I1231 10:31:23.086522  9759 net.cpp:165] Memory required for data: 89507528
    I1231 10:31:23.086529  9759 layer_factory.hpp:77] Creating layer accuracy_f
    I1231 10:31:23.086534  9759 net.cpp:106] Creating Layer accuracy_f
    I1231 10:31:23.086540  9759 net.cpp:454] accuracy_f <- ip_f_ip_f_0_split_0
    I1231 10:31:23.086546  9759 net.cpp:454] accuracy_f <- label_fine_data_2_split_0
    I1231 10:31:23.086552  9759 net.cpp:411] accuracy_f -> accuracy_f
    I1231 10:31:23.086560  9759 net.cpp:150] Setting up accuracy_f
    I1231 10:31:23.086565  9759 net.cpp:157] Top shape: (1)
    I1231 10:31:23.086571  9759 net.cpp:165] Memory required for data: 89507532
    I1231 10:31:23.086576  9759 layer_factory.hpp:77] Creating layer loss_f
    I1231 10:31:23.086582  9759 net.cpp:106] Creating Layer loss_f
    I1231 10:31:23.086588  9759 net.cpp:454] loss_f <- ip_f_ip_f_0_split_1
    I1231 10:31:23.086594  9759 net.cpp:454] loss_f <- label_fine_data_2_split_1
    I1231 10:31:23.086601  9759 net.cpp:411] loss_f -> loss_f
    I1231 10:31:23.086618  9759 layer_factory.hpp:77] Creating layer loss_f
    I1231 10:31:23.086639  9759 net.cpp:150] Setting up loss_f
    I1231 10:31:23.086657  9759 net.cpp:157] Top shape: (1)
    I1231 10:31:23.086661  9759 net.cpp:160]     with loss weight 1
    I1231 10:31:23.086668  9759 net.cpp:165] Memory required for data: 89507536
    I1231 10:31:23.086673  9759 net.cpp:226] loss_f needs backward computation.
    I1231 10:31:23.086679  9759 net.cpp:228] accuracy_f does not need backward computation.
    I1231 10:31:23.086685  9759 net.cpp:226] ip_f_ip_f_0_split needs backward computation.
    I1231 10:31:23.086690  9759 net.cpp:226] ip_f needs backward computation.
    I1231 10:31:23.086696  9759 net.cpp:226] loss_c needs backward computation.
    I1231 10:31:23.086702  9759 net.cpp:228] accuracy_c does not need backward computation.
    I1231 10:31:23.086709  9759 net.cpp:226] ip_c_ip_c_0_split needs backward computation.
    I1231 10:31:23.086714  9759 net.cpp:226] ip_c needs backward computation.
    I1231 10:31:23.086719  9759 net.cpp:226] ip1_sig1_0_split needs backward computation.
    I1231 10:31:23.086724  9759 net.cpp:226] sig1 needs backward computation.
    I1231 10:31:23.086730  9759 net.cpp:226] ip1 needs backward computation.
    I1231 10:31:23.086735  9759 net.cpp:226] relu3 needs backward computation.
    I1231 10:31:23.086740  9759 net.cpp:226] pool3 needs backward computation.
    I1231 10:31:23.086745  9759 net.cpp:226] conv3 needs backward computation.
    I1231 10:31:23.086751  9759 net.cpp:226] relu2 needs backward computation.
    I1231 10:31:23.086756  9759 net.cpp:226] drop2 needs backward computation.
    I1231 10:31:23.086762  9759 net.cpp:226] pool2 needs backward computation.
    I1231 10:31:23.086767  9759 net.cpp:226] conv2 needs backward computation.
    I1231 10:31:23.086773  9759 net.cpp:226] relu1b needs backward computation.
    I1231 10:31:23.086778  9759 net.cpp:226] drop1 needs backward computation.
    I1231 10:31:23.086793  9759 net.cpp:226] pool1 needs backward computation.
    I1231 10:31:23.086799  9759 net.cpp:226] cccp1b needs backward computation.
    I1231 10:31:23.086805  9759 net.cpp:226] relu1a needs backward computation.
    I1231 10:31:23.086812  9759 net.cpp:226] cccp1a needs backward computation.
    I1231 10:31:23.086817  9759 net.cpp:226] conv1 needs backward computation.
    I1231 10:31:23.086833  9759 net.cpp:228] label_fine_data_2_split does not need backward computation.
    I1231 10:31:23.086840  9759 net.cpp:228] label_coarse_data_1_split does not need backward computation.
    I1231 10:31:23.086846  9759 net.cpp:228] data does not need backward computation.
    I1231 10:31:23.086853  9759 net.cpp:270] This network produces output accuracy_c
    I1231 10:31:23.086863  9759 net.cpp:270] This network produces output accuracy_f
    I1231 10:31:23.086869  9759 net.cpp:270] This network produces output loss_c
    I1231 10:31:23.086875  9759 net.cpp:270] This network produces output loss_f
    I1231 10:31:23.086894  9759 net.cpp:283] Network initialization done.
    I1231 10:31:23.088927  9759 caffe.cpp:240] Running for 83 iterations.
    I1231 10:31:32.818135  9759 caffe.cpp:264] Batch 0, accuracy_c = 0.608333
    I1231 10:31:32.818186  9759 caffe.cpp:264] Batch 0, accuracy_f = 0.508333
    I1231 10:31:32.818194  9759 caffe.cpp:264] Batch 0, loss_c = 1.21104
    I1231 10:31:32.818202  9759 caffe.cpp:264] Batch 0, loss_f = 1.8871
    # [...]
    I1231 10:32:02.289006  9759 caffe.cpp:264] Batch 82, accuracy_c = 0.575
    I1231 10:32:02.289054  9759 caffe.cpp:264] Batch 82, accuracy_f = 0.416667
    I1231 10:32:02.289063  9759 caffe.cpp:264] Batch 82, loss_c = 1.35587
    I1231 10:32:02.289070  9759 caffe.cpp:264] Batch 82, loss_f = 2.04268
    I1231 10:32:02.289077  9759 caffe.cpp:269] Loss: 3.30597
    I1231 10:32:02.289089  9759 caffe.cpp:281] accuracy_c = 0.58745
    I1231 10:32:02.289099  9759 caffe.cpp:281] accuracy_f = 0.470884
    I1231 10:32:02.289110  9759 caffe.cpp:281] loss_c = 1.30973 (* 1 = 1.30973 loss)
    I1231 10:32:02.289119  9759 caffe.cpp:281] loss_f = 1.99623 (* 1 = 1.99623 loss)
    CPU times: user 188 ms, sys: 64 ms, total: 252 ms
    Wall time: 42.6 s


## The model achieved near 58% accuracy on the 20 coarse labels and 47% accuracy on fine labels.
This means that upon showing the neural network a picture it had never seen, it will correctly classify it in one of the 20 coarse categories 58% of the time or it will classify it correctly in the fine categories 47% of the time right, and ignoring the coarse label. This is amazing, but the neural network for sure could be fine tuned with better solver parameters.

It would  be also possible to have two more loss layers on top of the existing loss, to recombine the predictions made and synchronize with the fact that coarse and fine labels influence on each other and are related.

This neural network training could be compared to the results listed here: http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#494c5356524332303132207461736b2031

Let's convert the notebook to github markdown:


```python
!jupyter nbconvert --to markdown custom-cifar-100.ipynb
!mv custom-cifar-100.md README.md
```

    [NbConvertApp] Converting notebook custom-cifar-100.ipynb to markdown
    [NbConvertApp] Writing 413451 bytes to custom-cifar-100.md
