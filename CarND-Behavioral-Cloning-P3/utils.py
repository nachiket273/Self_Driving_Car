import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

IMG_HT, IMG_WIDTH, IMG_CH = 66, 200, 3

def read_img(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

def crop_and_resize(img, width, height):
    img = img[40:-20, :, :]
    return cv2.resize(img, (width, height), cv2.INTER_AREA)

def preprocess(img):
    img = crop_and_resize(img, IMG_WIDTH, IMG_HT)
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

def read_csv_and_rename_cols(file_path, columns=None):
    path = Path(file_path)
    if path.is_file() and path.exists():
        if columns:
            df = pd.read_csv(file_path, header=None)
            df = df.rename(columns=columns)
        else :
            df = pd.read_csv(file_path)
        return df

def get_relative_path(img_path):
    relative_path = './' + img_path.split('/')[-3] + '/' + img_path.split('/')[-2] + '/' + img_path.split('/')[-1]
    return relative_path

def get_kernel(no):
    if no < 2 :
        return tf.constant(np.ones((3,3,1,1)), tf.float32)
    return tf.constant(np.ones((5,5,1,1)), tf.float32)

def get_stride(no):
    if no < 2 :
        return [1,1,1,1]
    return [1,2,2,1]

def reverse(lst): 
    return [ele for ele in reversed(lst)]

def get_salient_feature_mask(ops):
    i = 0
    upscaled_conv = np.ones((1, 18))
    layer_ops = reverse(ops)
    for layer in layer_ops:
        avg_actvn = np.mean(layer, axis=3).squeeze(axis=0)
        avg_actvn = avg_actvn * upscaled_conv
        if i == 4 :
            output_shape = (IMG_HT, IMG_WIDTH)
        else :
            output_shape = (layer_ops[i+1].shape[1], layer_ops[i+1].shape[2])
        x = tf.constant(np.reshape(avg_actvn, (1, avg_actvn.shape[0], avg_actvn.shape[1], 1)), tf.float32)
        deconv = tf.nn.conv2d_transpose(x, get_kernel(i),(1, output_shape[0], output_shape[1], 1), get_stride(i), padding='VALID')
        with tf.Session() as session:
            res = session.run(deconv)
        deconv_actn = np.reshape(res, output_shape)
        upscaled_conv = deconv_actn
        i += 1
    mask = (upscaled_conv - np.min(upscaled_conv)) / (np.max(upscaled_conv) - np.min(upscaled_conv))
    return mask

def display_multiple_images(img_list, label_list=[], cmap='', is_yuv=True):
    assert(len(img_list) % 3 == 0)
    if len(label_list) > 0 :
        assert(len(img_list) == len(label_list))
    
    cols = 3
    rows = int(len(img_list) / 3)
    
    fig = plt.figure(figsize=(16,16))
    
    for i in range(1, cols*rows +1):
        fig.add_subplot(rows, cols, i)       
        img = img_list[i-1]
        if len(label_list) > 0:
            title = label_list[i-1]
            plt.title(title)
        if is_yuv:
            img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
        if cmap != '':
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
    plt.show()