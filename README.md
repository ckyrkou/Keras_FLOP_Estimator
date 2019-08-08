# Performance Estimator for Keras Models
*WARNING Not a complete solution yet

This is a function for estimating the floating point operations (FLOPS) of deep learning models developed with keras. It supports some basic layers such as Convolutional, Separable Convolution, Depthwise Convolution, BatchNormalization, Activations, and Merge Layers (Add, Max, Concatenate)

# Usage

```python

from keras.applications.mobilenetv2 import MobileNetV2

model = MobileNetV2(weights=None, include_top=True, pooling=None,input_shape=(224,224,3))
model.summary()

#Prints a table with the FLOPS at each layer and total FLOPs
net_flops(model)

```


# Resources:
1. [Convolutional Neural Networks Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
2. [How fast is my model?](https://machinethink.net/blog/how-fast-is-my-model/)
3. [3 Small But Powerful Convolutional Networks](https://towardsdatascience.com/3-small-but-powerful-convolutional-networks-27ef86faa42d)
