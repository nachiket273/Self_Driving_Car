import numpy as np
import cv2

def random_shift(img, steering, shift_range=20):
    ht, wd, ch = img.shape
    
    shift_x = shift_range * (np.random.rand()- 0.5)
    shift_y = shift_range * (np.random.rand()- 0.5)
    shift_m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    img = cv2.warpAffine(img, shift_m, (wd, ht))
    
    steering += shift_x * 0.002
    return img, steering

def random_shadow(img):
    ht, wd , ch = img.shape
    x1, y1 = wd * np.random.rand(), 0
    x2, y2 = wd * np.random.rand(), ht
    xm, ym = np.mgrid[0:ht, 0:wd]
    
    mask = np.zeros_like(img[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.6, high=0.9)
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def adjust_brightness(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv_img[:,:,2] =  hsv_img[:,:,2] * ratio
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

def flip(img, steering):
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        if(steering != 0):
            steering = -steering 
    return img, steering