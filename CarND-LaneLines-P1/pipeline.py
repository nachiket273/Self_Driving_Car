from collections import deque
from helper import *
import cv2
import numpy as np


class LaneMemory:
    def __init__(self, max_entries=50):
        self.max_entries = max_entries
        self.left_lanes = deque(maxlen=self.max_entries)
        self.right_lanes = deque(maxlen=self.max_entries)
        
    def process(self, img):
        result, left_lane , right_lane = process_image(img, calc_mean=True, left_mem= self.left_lanes, right_mem = self.right_lanes)
        self.left_lanes.append(left_lane)
        self.right_lanes.append(right_lane)
        return result


def process_image(image, calc_mean=False, **kwargs):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    if calc_mean:
        assert('left_mem' in kwargs.keys())
        assert('right_mem' in kwargs.keys())
    
    original_img = np.copy(image)
    
    # convert to grayscale
    gray_img = grayscale(image)
    
    # darken the grayscale
    darkened_img = adjust_gamma(gray_img, 0.5)
    
    # Color Selection
    white_mask = isolate_color_mask(to_hls(image), np.array([0, 200, 0], dtype=np.uint8), np.array([200, 255, 255], dtype=np.uint8))
    yellow_mask = isolate_color_mask(to_hls(image), np.array([10, 0, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    colored_img = cv2.bitwise_and(darkened_img, darkened_img, mask=mask)
    
    # Apply Gaussian Blur
    blurred_img = gaussian_blur(colored_img, kernel_size=7)
    
    # Apply Canny edge filter
    canny_img = canny(blurred_img, low_threshold=70, high_threshold=140)
    
    # Get Area of Interest
    aoi_img = get_aoi(canny_img)
    
    # Apply Hough lines
    hough_lines = get_hough_lines(aoi_img)
    hough_img = draw_lines(original_img, hough_lines)
    
    # Extrapolation and averaging
    left_lane, right_lane = get_lane_lines(original_img, hough_lines)
    
    if calc_mean:
        if left_lane is not None and right_lane is not None:
            kwargs['left_mem'].append(left_lane)
            kwargs['right_mem'].append(right_lane)
        left_mean = np.mean(kwargs['left_mem'], axis=0, dtype=np.int32)
        right_mean = np.mean(kwargs['right_mem'], axis=0, dtype=np.int32)
        left_lane_avg = tuple(map(tuple, left_mean))
        right_lane_avg = tuple(map(tuple, right_mean))
        result = draw_weighted_lines(original_img, [left_lane_avg, right_lane_avg], thickness= 10)
        return result, left_lane, right_lane
    
    result = draw_weighted_lines(original_img, [left_lane, right_lane], thickness= 10)
       
    return result, left_lane, right_lane



