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
        
    pnet_fun = lambda img : sess.run(('pnet/p_net/conv4-2/BiasAdd:0', 'pnet/p_net/prob1/truediv:0'), feed_dict={'pnet/input:0':img})
    rnet_fun = lambda img : sess.run(('rnet/r_net/conv5-2/BiasAdd:0', 'rnet/r_net/prob1/Softmax:0'), feed_dict={'rnet/input:0':img})
    onet_fun = lambda img : sess.run(('onet/o_net/conv6-2/BiasAdd:0', 'onet/o_net/conv6-3/BiasAdd:0', 'onet/o_net/prob1/Softmax:0'), feed_dict={'onet/input:0':img})
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
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))
        
        boxes = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])
        
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)
            
    if total_boxes.shape[0] > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        
        reg_w = total_boxes[:, 2] - total_boxes[:, 0]
        reg_h = total_boxes[:, 3] - total_boxes[:, 1]
        
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * reg_w
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * reg_h
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * reg_w
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * reg_h
        
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, dy_h, dx, dx_w, y1, y2, x1, x2, tmp_w, tmp_h = pad(total_boxes.copy(), w, h)
    
    num_box = total_boxes.shape[0]
    if num_box > 0:
        temp_img = np.zeros((24, 24, 3, num_box))
        for k in range(num_box):
            tmp = np.zeros((int(tmp_h[k]), int(tmp_w[k]), 3))
            tmp[dy[k]-1:dy_h[k], dx[k]-1:dx_w[k], :] = img[y1[k]-1:y2[k], x1[k]-1:x2[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                temp_img[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
            
        temp_img = (temp_img - 127.5) * 0.0078125
        temp_img1 = np.transpose(temp_img, (3, 1, 0, 2))
        out = rnet(temp_img1)
        
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1,:]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        offset = out0[:, ipass[0]]

        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(offset[:, pick]))
            total_boxes = rerec(total_boxes.copy())
    
    num_box = total_boxes.shape[0]
    if num_box > 0:
        temp_img = np.zeros((48, 48, 3, num_box))
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, dy_h, dx, dx_w, y1, y2, x1, x2, tmp_w, tmp_h = pad(total_boxes.copy(), w, h)
        for k in range(num_box):
            tmp = np.zeros((int(tmp_h[k]), int(tmp_w[k]), 3))
            tmp[dy[k]-1:dy_h[k], dx[k]-1:dx_w[k], :] = img[y1[k]-1:y2[k], x1[k]-1:x2[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                temp_img[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        temp_img = (temp_img - 127.5) * 0.0078125
        temp_img1 = np.transpose(temp_img, (3, 1, 0, 2))
        out = onet(temp_img1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1, :]
        points = out1
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0],0:4].copy(), np.expand_dims(score[ipass].copy(),1)])
        offset = out0[:, ipass[0]]
        
        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1

        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(offset))
            pick = nms(total_boxes, 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points    

def bbreg(boundingbox, offset):
    if offset.shape[1] == 1:
        offset = np.reshape(offset, (offset.shape[2], offset.shape[3]))
        
    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + offset[:, 0] * w
    b2 = boundingbox[:, 1] + offset[:, 1] * h
    b3 = boundingbox[:, 2] + offset[:, 2] * w
    b4 = boundingbox[:, 3] + offset[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

def pad(total_boxes, w, h):
#     make sure the bbox border is not exceed the image border
    tmp_w = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmp_h = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    num_box = total_boxes.shape[0]
    
    dx = np.ones((num_box), dtype=np.int32)
    dy = np.ones((num_box), dtype=np.int32)
    dx_w = tmp_w.copy().astype(np.int32)
    dy_h = tmp_h.copy().astype(np.int32)
        
    x1 = total_boxes[:, 0].copy().astype(np.int32)
    y1 = total_boxes[:, 1].copy().astype(np.int32)
    x2 = total_boxes[:, 2].copy().astype(np.int32)
    y2 = total_boxes[:, 3].copy().astype(np.int32)
        
    tmp = np.where(x2 > w)
    dx_w.flat[tmp] = np.expand_dims(-x2[tmp] + w + tmp_w[tmp], 1)
    x2[tmp] = w
    
    tmp = np.where(y2 > w)
    dy_h.flat[tmp] = np.expand_dims(-y2[tmp] + h + tmp_h[tmp], 1)
    y2[tmp] = h
    
    tmp = np.where(x1 < 1)
    dx.flat[tmp] = np.expand_dims(2 - x1[tmp], 1)
    x1[tmp] = 1
    
    tmp = np.where(y1 < 1)
    dy.flat[tmp] = np.expand_dims(2 - y1[tmp], 1)
    y1[tmp] = 1
    
    return dy, dy_h, dx, dx_w, y1, y2, x1, x2, tmp_w, tmp_h
   
def rerec(bboxA):
#     make the bbox to square shape
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]

    l = np.maximum(w, h)
    bboxA[:, 0] += (w - l) * 0.5
    bboxA[:, 1] += (h - l) * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))

    return bboxA

def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.uint16)
    cnt = 0
    
    while I.size > 0:
        i = I[-1]
        pick[cnt] = i
        cnt += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        I = I[np.where(o <= threshold)]

    return pick[0:cnt]

def generateBoundingBox(prob, coord, scale, threshold):
    stride = 2
    cellsize = 12
    
    prob = np.transpose(prob)
    dx1 = np.transpose(coord[:, :, 0])
    dy1 = np.transpose(coord[:, :, 1])
    dx2 = np.transpose(coord[:, :, 2])
    dy2 = np.transpose(coord[:, :, 3])
    y, x = np.where(prob >= threshold)
    
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    
    score = prob[(y, x)]
    coord = np.transpose(np.vstack([dx1[y, x], dy1[y, x], dx2[y, x], dy2[y, x]]))
    if coord.size == 0:
        coord = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), coord])

    return boundingbox 
  
def imresample(img, sz):
    return cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)

