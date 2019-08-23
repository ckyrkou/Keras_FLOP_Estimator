# Performance Estimator for Keras Models
*WARNING - Under Construction - Code will Follow

Developed and tested on keras 2.2.4.

# Keras FLOP Estimator

This is a function for estimating the floating point operations (FLOPS) of deep learning models developed with keras. It supports some basic layers such as Convolutional, Separable Convolution, Depthwise Convolution, BatchNormalization, Activations, and Merge Layers (Add, Max, Concatenate)

### Usage

```python

from keras.applications.mobilenet import MobileNet

model = MobileNet(weights=None, include_top=True, pooling=None,input_shape=(224,224,3))
model.summary()

#Prints a table with the FLOPS at each layer and total FLOPs
net_flops(model,table=True)

```

### Output
```
               Layer Name |      Input Shape |     Output Shape |      Kernel Size |          Filters | Strides |  FLOPS
-----------------------------------------------------------------------------------------------------------------------------------
                  input_1 |    [224, 224, 3] |    [224, 224, 3] |           [0, 0] |           [0, 0] | [1, 1] | 0.0000
                conv1_pad |     ['', '', ''] |     ['', '', ''] |           [0, 0] |           [0, 0] | [1, 1] | 0.0000
                    conv1 |    [225, 225, 3] |   [112, 112, 32] |           (3, 3) |               32 | (2, 2) | 21870000.0000
                 conv1_bn |   [112, 112, 32] |   [112, 112, 32] |           [0, 0] |           [0, 0] | [1, 1] | 0.4014
               conv1_relu |   [112, 112, 32] |   [112, 112, 32] |           [0, 0] |           [0, 0] | [1, 1] | 0.4014
                conv_dw_1 |   [112, 112, 32] |   [112, 112, 32] |           (3, 3) |               32 | (1, 1) | 7.2253
             conv_dw_1_bn |   [112, 112, 32] |   [112, 112, 32] |           [0, 0] |           [0, 0] | [1, 1] | 0.4014
           conv_dw_1_relu |   [112, 112, 32] |   [112, 112, 32] |           [0, 0] |           [0, 0] | [1, 1] | 0.4014
                conv_pw_1 |   [112, 112, 32] |   [112, 112, 64] |           (1, 1) |               64 | (1, 1) | 51380224.0000
.
.
.

Total FLOPS (x 10^-6): 551.47646256
```

# Keras Model Timing Performannce Per Layer

This is a function for estimating the timing performance of each leayer in a neural network. It can be used to identify the bottlenecks in computation when run on the target device. The function iterates over the network by runninng an input image through it by removing each of the layers. The layer time is found by subtracting the current run without the last layer from the previous run that contained the layer. There are some timing issues where the timings are off a bit thus some times may appear as negative. In such, case the layer compute time can be considered as negligible.

### Usage

```python

from keras.applications.vgg16 import VGG16
model = VGG16(weights='imagenet', include_top=False)

times = time_per_layer(model)

# Visualize

import matplotlib.pyplot as plt

plt.style.use('ggplot')
x = [model.layers[-i].name for i in range(1,len(model.layers))]
#x = [i for i in range(1,len(model.layers))]
g = [times[i,0] for i in range(1,len(times))]
x_pos = np.arange(len(x))
plt.bar(x, g, color='#7ed6df')
plt.xlabel("Layers")
plt.ylabel("Processing Time")
plt.title("Processing Time of each Layer")
plt.xticks(x_pos, x,rotation=90)

plt.show()

```
### Output Graph

<img src="./Figures/VGG16_timings.png" width="512">

# Disclaimer:

This code is provided as is and there might be some errors especially with the timing as it depends on many factors. In many papers the same number can be reporter under either FLOPs or MACCs. By definition these two quantities are not the same and care must be taken as to which one you want to report and compare against. For example MobileNetV1 paper it is reported to have ~569 MACCs in the paper. However, many leaderboarding sites put this metric under FLOPS which may also include other operations. In the provided function, FLOPS can be any operation such as multiply  

# Resources:
1. [Convolutional Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
2. [How fast is my model?](https://machinethink.net/blog/how-fast-is-my-model/)
3. [3 Small But Powerful Convolutional Networks](https://towardsdatascience.com/3-small-but-powerful-convolutional-networks-27ef86faa42d)
