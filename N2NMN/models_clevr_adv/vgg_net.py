from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

# components
from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import pooling_layer as pool
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu

channel_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
input_shape = [320,480,3]

class vgg_pool5:
    def __init__(self,input_batch, masking_batch, grad_scaling, scope='vgg_net', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
             # Adversarial layer
             W_adv = tf.get_variable('adv_weights', [1] + input_shape,initializer=tf.contrib.layers.xavier_initializer())
             # channel mean is not subtracted in input_batch so it's the original masked image
             W_adv1 = tf.where(masking_batch,tf.zeros(tf.shape(W_adv),dtype=tf.float32),W_adv)
             W_adv2 = tf.where(tf.greater(input_batch+grad_scaling*W_adv1,255.0), (1./grad_scaling)*(255.0 - input_batch), W_adv1)
             W_adv3 = tf.where(tf.less(input_batch+grad_scaling*W_adv2,0.0), (-1./grad_scaling)*input_batch , W_adv2)
             # we subtract the channel mean before passing it to vgg16
             adv_img = input_batch + grad_scaling*W_adv3 - channel_mean
             self.W_adv3 = W_adv3
             self.adv_input = adv_img
             # layer 1
             conv1_1 = conv_relu('conv1_1', adv_img,
                            kernel_size=3, stride=1, output_dim=64)
             conv1_2 = conv_relu('conv1_2', conv1_1,
                            kernel_size=3, stride=1, output_dim=64)
             pool1 = pool('pool1', conv1_2, kernel_size=2, stride=2)
             # layer 2
             conv2_1 = conv_relu('conv2_1', pool1,
                            kernel_size=3, stride=1, output_dim=128)
             conv2_2 = conv_relu('conv2_2', conv2_1,
                            kernel_size=3, stride=1, output_dim=128)
             pool2 = pool('pool2', conv2_2, kernel_size=2, stride=2)
             # layer 3
             conv3_1 = conv_relu('conv3_1', pool2,
                            kernel_size=3, stride=1, output_dim=256)
             conv3_2 = conv_relu('conv3_2', conv3_1,
                            kernel_size=3, stride=1, output_dim=256)
             conv3_3 = conv_relu('conv3_3', conv3_2,
                            kernel_size=3, stride=1, output_dim=256)
             pool3 = pool('pool3', conv3_3, kernel_size=2, stride=2)
             # layer 4
             conv4_1 = conv_relu('conv4_1', pool3,
                            kernel_size=3, stride=1, output_dim=512)
             conv4_2 = conv_relu('conv4_2', conv4_1,
                            kernel_size=3, stride=1, output_dim=512)
             conv4_3 = conv_relu('conv4_3', conv4_2,
                            kernel_size=3, stride=1, output_dim=512)
             pool4 = pool('pool4', conv4_3, kernel_size=2, stride=2)
             # layer 5
             conv5_1 = conv_relu('conv5_1', pool4,
                            kernel_size=3, stride=1, output_dim=512)
             conv5_2 = conv_relu('conv5_2', conv5_1,
                            kernel_size=3, stride=1, output_dim=512)
             conv5_3 = conv_relu('conv5_3', conv5_2,
                            kernel_size=3, stride=1, output_dim=512)
             pool5 = pool('pool5', conv5_3, kernel_size=2, stride=2)
             self.image_feat_grid = pool5
