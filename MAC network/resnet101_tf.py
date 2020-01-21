from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
std = np.array([0.229, 0.224, 0.224]).reshape(1, 1, 1, 3)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  if stride == 1:
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        rate=rate,
        padding='VALID',
        activation_fn=None,
        scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = array_ops.pad(
        inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return layers_lib.conv2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=stride,
        rate=rate,
        padding='VALID',
        activation_fn=None,
        scope=scope)

def identity_block(input_tensor, kernel_size, filters, scope = None):
    filters1, filters2, filters3 = filters
    with tf.variable_scope(scope) as scope:
         net = layers_lib.conv2d(input_tensor,filters1, 1, stride=1, padding='VALID',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         net = tf.nn.relu(net)
         net = layers_lib.conv2d(net, filters2, kernel_size, stride=1, padding='SAME',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         net = tf.nn.relu(net)
         net = layers_lib.conv2d(net,filters3, 1, stride=1, padding='VALID',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         net = net + input_tensor
         net = tf.nn.relu(net)
    return net

def conv_block(input_tensor,kernel_size,filters,strides, scope=None):
    filters1, filters2, filters3 = filters
    with tf.variable_scope(scope) as scope:
         net = layers_lib.conv2d(input_tensor,filters1, 1, stride=strides[0], padding='VALID',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         net = tf.nn.relu(net)
         net = tf.pad(net,tf.constant([[0,0],[1, 1,], [1, 1], [0,0]]))
         net = layers_lib.conv2d(net, filters2, kernel_size, stride=strides[1], padding='VALID',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         net = tf.nn.relu(net)
         net = layers_lib.conv2d(net,filters3, 1,stride=strides[2], padding='VALID',activation_fn=None)
         net = tf.layers.batch_normalization(net,epsilon = 1e-05)
         shortcut = layers_lib.conv2d(input_tensor,filters3, 1, stride=strides[3], padding='VALID',activation_fn=None)
         shortcut = tf.layers.batch_normalization(shortcut,epsilon = 1e-05)
         net = net + shortcut
         net = tf.nn.relu(net)
    return net

class resnet101:
     def __init__(self, input_batch, masking_batch, grad_scaling,scope="resnet101",reuse=None):
          with tf.variable_scope(scope, reuse=reuse):
               with tf.variable_scope("adversarial_layer"):
                    #W_adv = tf.get_variable('adv_weights', [1] + [224,224,3] ,initializer=tf.contrib.layers.xavier_initializer())
                    W_adv = tf.get_variable('adv_weights', [1] + [224,224,3] ,initializer=tf.zeros_initializer())
                    W_adv1 = tf.where(masking_batch,tf.zeros(tf.shape(W_adv),dtype=tf.float32),W_adv)
                    W_adv2 = tf.where(tf.greater(input_batch+grad_scaling*W_adv1,255.0), (1./grad_scaling)*(255.0 - input_batch), W_adv1)
                    W_adv3 = tf.where(tf.less(input_batch+grad_scaling*W_adv2,0.0), (-1./grad_scaling)*input_batch , W_adv2)
                    adv_img = input_batch + grad_scaling*W_adv3
                    #adv_img = input_batch + 0.0*grad_scaling*W_adv3
                    #adv_img_scaled = adv_img
                    #adv_img = (adv_img / 255.0 - mean) / std
                    adv_img_scaled = adv_img/255.0
                    adv_img_scaled = adv_img_scaled - mean
                    adv_img_scaled = adv_img_scaled / std
                    adv_img_scaled.set_shape([1,224,224, 3])
                    #self.org_img = input_batch
                    self.adv_input = adv_img
                    self.adv_img_scaled = adv_img_scaled
               with tf.variable_scope("preblock"):
                    net = conv2d_same(adv_img_scaled, 64, 7, stride=2, scope='conv1')
                    net = tf.layers.batch_normalization(net,epsilon = 1e-05)
                    net = tf.nn.relu(net)
                    net = tf.pad(net,tf.constant([[0,0],[1, 1,], [1, 1], [0,0]]))
                    net = tf.layers.max_pooling2d(net,pool_size=(3,3),strides=(2,2),padding='valid')
               with tf.variable_scope("block1"):
                    net = conv_block(net,3,[64,64,256],strides=[1,1,1,1],scope='conv_block')
                    net = identity_block(net, 3, [64, 64, 256], scope='ib_1')
                    net = identity_block(net, 3, [64, 64, 256], scope='ib_2')
               with tf.variable_scope("block2"):
                    net = conv_block(net,3,[128, 128, 512],strides=[1,2,1,2],scope='conv_block')
                    for i in range(1, 4):
                         net = identity_block(net, 3, [128, 128, 512], scope ='ib_'+str(i))
               with tf.variable_scope("block3"):
                    net = conv_block(net,3,[256, 256, 1024],strides=[1,2,1,2],scope='conv_block')
                    for i in range(1, 23):
                         net = identity_block(net, 3, [256, 256, 1024], scope ='ib_'+str(i))
               self.imagefeatures = net

