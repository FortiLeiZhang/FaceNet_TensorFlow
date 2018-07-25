from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import numpy as np
import tensorflow as tf
import cv2
import os
       
class PNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='PReLU1', shared_axes=[1, 2])
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='PReLU2', shared_axes=[1, 2])
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='PReLU3', shared_axes=[1, 2])
        self.conv4_1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv4-1')
        self.softmax = tf.keras.layers.Softmax(axis=3, name='prob1')
        self.conv4_2 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv4-2')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        _x = tf.identity(x)
        x = self.conv4_1(x)
        out_1 = self.softmax(x)
        out_2 = self.conv4_2(_x)
        return x

class RNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=28, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='prelu1', shared_axes=[1, 2])
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='prelu2', shared_axes=[1, 2])
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding='valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='prelu3', shared_axes=[1, 2])
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, name='conv4')
        self.prelu4 = tf.keras.layers.PReLU(name='prelu4')
        self.fc2_1 = tf.keras.layers.Dense(2, name='conv5-1')
        self.softmax = tf.keras.layers.Softmax(axis=1, name='prob1')
        self.fc2_2 = tf.keras.layers.Dense(4, name='conv5-2')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.prelu4(x)
        _x = tf.identity(x)
        x = self.fc2_1(x)
        out_1 = self.softmax(x)
        out_2 = self.fc2_2(_x)
        return x
        
class ONet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv1')
        self.prelu1 = tf.keras.layers.PReLU(name='prelu1', shared_axes=[1, 2])
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')
        self.prelu2 = tf.keras.layers.PReLU(name='prelu2', shared_axes=[1, 2])
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv3')
        self.prelu3 = tf.keras.layers.PReLU(name='prelu3', shared_axes=[1, 2])
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=(1, 1), padding='valid', name='conv4')
        self.prelu4 = tf.keras.layers.PReLU(name='prelu4', shared_axes=[1, 2])
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, name='conv5')
        self.prelu5 = tf.keras.layers.PReLU(name='prelu5')
        
        self.fc2_1 = tf.keras.layers.Dense(2, name='conv6-1')
        self.softmax = tf.keras.layers.Softmax(axis=1, name='prob1')

        self.fc2_2 = tf.keras.layers.Dense(4, name='conv6-2')
        
        self.fc2_3 = tf.keras.layers.Dense(10, name='conv6-3')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.prelu5(x)
        x2 = tf.identity(x)
        x3 = tf.identity(x)
        x = self.fc2_1(x)
        out_1 = self.softmax(x)
        out_2 = self.fc2_2(x2)
        out_3 = self.fc2_3(x3)

        return x

def load_param(data_path, session, net_name):
    data_dict = np.load(data_path, encoding='latin1').item()
    
    for op_name in data_dict:
        for param_name, data in iteritems(data_dict[op_name]):
            if param_name == 'weights':
                var_name = net_name + '/' + op_name + '/' + 'kernel'
            if param_name == 'biases':
                var_name = net_name + '/' + op_name + '/' + 'bias'
            if param_name == 'alpha':
                var_name = net_name + '/' + op_name + '/' + 'alpha'
                var = tf.get_variable(var_name)
                if var.get_shape().ndims == 3:
                    data = data[np.newaxis, np.newaxis, :]

            var = tf.get_variable(var_name)
            session.run(var.assign(data))

def create_mtcnn(sess, model_path):
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))
        
    with tf.variable_scope('pnet', reuse=tf.AUTO_REUSE):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = PNet()
        pnet(data)
        load_param(os.path.join(model_path, 'det1.npy'), sess, 'p_net')
        
    with tf.variable_scope('rnet', reuse=tf.AUTO_REUSE):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet()
        rnet(data)
        load_param(os.path.join(model_path, 'det2.npy'), sess, 'r_net')
        
    with tf.variable_scope('onet', reuse=tf.AUTO_REUSE):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet()
        onet(data)
        load_param(os.path.join(model_path, 'det3.npy'), sess, 'o_net')

    with open('/home/lzhang/tensorflow_debug.txt', 'w') as f:
        for var in tf.global_variables():
            f.write(str(var))
        for op in tf.get_default_graph().get_operations():
            f.write(str(op))
        
    pnet_fun = lambda img : sess.run(('pnet/p_net/conv4-2/bias:0', 'pnet/p_net/prob1/truediv'), feed_dict={'pnet/input:0':img})
    rnet_fun = lambda img : sess.run(('rnet/r_net/conv5-2/bias:0', 'rnet/r_net/prob1:0'), feed_dict={'rnet/input:0':img})
    onet_fun = lambda img : sess.run(('onet/o_net/conv6-2/bias:0', 'onet/o_net/conv6-3/bias:0', 'onet/o_net/prob1:0'), feed_dict={'onet/input:0':img})
    return pnet_fun, rnet_fun, onet_fun
        
def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    factor_cnt = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    h, w = img.shape[0], img.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_cnt)]
        minl = minl * factor
        factor_cnt += 1
        
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        
        print(len(out))
        print(out[0].shape)
        print(out[1].shape)
        
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))
        
        print(out0.shape)
        print(out1.shape)
    
    
    
    
    
    
    
    
    
    
    
    
def imresample(img, sz):
    return cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)

