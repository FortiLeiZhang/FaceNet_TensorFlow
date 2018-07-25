from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
import math
from six import iteritems

def triplet_loss(anchor, positive, negative, alpha):
    pass
  
def center_loss(features, label, alfa, nrof_classes):
    pass

def get_image_paths_and_labels(dataset):
    pass

def shuffle_examples(image_paths, labels):
    pass

def random_rotate_image(image):
    pass
  
# 1: Random rotate 2: Random crop  4: Random flip  8:  Fixed image standardization  16: Flip
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder):
    pass

def get_control_flag(control, field):
    pass
  
def _add_loss_summaries(total_loss):
    pass

def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True):
    pass

def prewhiten(x):
    pass

def crop(image, random_crop, image_size):
    pass
  
def flip(image, random_flip):
    pass

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  
def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    pass

def get_label_batch(label_data, batch_size, batch_index):
    pass

def get_batch(image_data, batch_size, batch_index):
    pass

def get_triplet_batch(triplets, batch_index, batch_size):
    pass

def get_learning_rate_from_file(filename, epoch):
    pass

class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ',' + str(len(self.image_paths)) + 'images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    n_classes = len(classes)
    for i in range(n_classes):
        class_name = classes[i]
        face_dir = os.path.join(path_exp, class_name)
        img_path = get_image_paths(face_dir)
        dataset.append(ImageClass(class_name, img_path))
    return dataset    

def get_image_paths(facedir):
    img_path = []
    if os.path.isdir(facedir):
        imgs = os.listdir(facedir)
        img_path = [os.path.join(facedir, img) for img in imgs]
    return img_path
  
def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    pass

def load_model(model, input_map=None):
    pass
    
def get_model_filenames(model_dir):
    pass
  
def distance(embeddings1, embeddings2, distance_metric=0):
    pass

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    pass

def calculate_accuracy(threshold, dist, actual_issame):
    pass
  
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    pass

def calculate_val_far(threshold, dist, actual_issame):
    pass

def store_revision_info(src_path, output_dir, arg_string):
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSErros as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror
        
    try:
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout = PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSErros as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror
        
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, 'w') as f:
        f.write('arguments: %s\n--------------------\n' % arg_string)
        f.write('tensorflow version: %s\n--------------------\n' % tf.__version__)
        f.write('git hash: %s\n--------------------\n' % git_hash)
        f.write('%s' % git_diff)

def list_variables(filename):
    pass

def put_images_on_grid(images, shape=(16,8)):
    pass

def write_arguments_to_file(args, filename):
    pass
