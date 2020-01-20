#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:46:42 2020

@author: y56
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper 
import cv2
import numpy as np
from scipy.optimize import curve_fit


def process_image(image):

    # grayscale
    gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    # darken the grayscale
    darkened_img = helper.adjust_gamma(gray_img, 0.5)
    
    # Color Selection
    white_mask = helper.isolate_color_mask(helper.to_hls(image), 
                                    np.array([0, 200, 0], dtype=np.uint8), 
                                    np.array([200, 255, 255], dtype=np.uint8))
    yellow_mask = helper.isolate_color_mask(helper.to_hls(image), 
                                     np.array([10, 0, 100], dtype=np.uint8), 
                                     np.array([40, 255, 255], dtype=np.uint8))
    color_selection_mask = cv2.bitwise_or(white_mask, yellow_mask)
    color_selected_img = cv2.bitwise_and(darkened_img, color_selection_mask)
    
    # gaussian blurring
    blur_img = cv2.GaussianBlur(color_selected_img,(5, 5),0)

    # canny edge 
    edge_points = helper.canny(blur_img, 10 , 10)
    
    # area selection
    imshape = image.shape
    vertices = np.array([[(100/960*imshape[1], imshape[0] * 0.9999),
                          (445/960*imshape[1], 320/540*imshape[0]),
                          (520/960*imshape[1], 320/540*imshape[0]),
                          ((960-30)/960*imshape[1], imshape[0] * 0.9999)]],
                        dtype=np.int32)

    area_selection_mask = np.zeros_like(edge_points)
    noneed = cv2.fillPoly(area_selection_mask, vertices, 255)
    
    masked_edges = cv2.bitwise_and(edge_points, area_selection_mask)
    
    # Hough Line Detection
    line_img = helper.hough_lines(masked_edges, 1, np.pi/360, 4, 12, 4)
    
    # curve fitting
    tmp_y, tmp_x, noneed = np.nonzero(line_img)
    
    vertical_cut = 490/960 * imshape[1]
    
    x_left = tmp_x[tmp_x < vertical_cut]
    y_left = tmp_y[tmp_x < vertical_cut]
    
    x_right = tmp_x[tmp_x >= vertical_cut]
    y_right = tmp_y[tmp_x >= vertical_cut]
     
    def line_func(x, m, b):
        return m*x+b
    
    popt_left, pcov_left = curve_fit(line_func, x_left, y_left)
    popt_right, pcov_right = curve_fit(line_func, x_right, y_right)
    
    whole_x = np.linspace(0,imshape[1],100)
    
    whole_left_y = line_func(whole_x, *popt_left)
    whole_right_y = line_func(whole_x, *popt_right)
   
    whole_left_x = whole_x[(imshape[0]*0.6<=whole_left_y) & (whole_left_y<imshape[0])]
    whole_right_x = whole_x[(imshape[0]*0.6<=whole_right_y) & (whole_right_y<imshape[0])]
    
    whole_left_y = whole_left_y[(imshape[0]*0.6<=whole_left_y) & (whole_left_y<=imshape[0])]
    whole_right_y = whole_right_y[(imshape[0]*0.6<=whole_right_y) & (whole_right_y<=imshape[0])]
    
    
    noneed = cv2.line(image, 
             (int(whole_right_x[-1]), int(whole_right_y[-1])), 
             (int(whole_right_x[0]), int(whole_right_y[0])),
             color=[255, 0, 0], thickness=2)
        
    noneed = cv2.line(image, 
            (int(whole_left_x[-1]), int(whole_left_y[-1])), 
             (int(whole_left_x[0]), int(whole_left_y[0])),
             color=[0, 255, 0], thickness=2)
    return image
    
if __name__ == '__main__':
    if 0:
        import os
        filenames = os.listdir("test_images/")
        for filename  in ["vlcsnap-2020-01-06-02h40m50s838.jpg"]:
    #    for filename in filenames[2:3]:
        # for filename in filenames:
            image = mpimg.imread('test_images/' + filename)
            output = process_image(image)
            plt.figure()
            plt.imshow(output)
    else:
        from moviepy.editor import VideoFileClip
        from IPython.display import HTML
        import os
        filenames = os.listdir("test_videos/")
        for filename in filenames:
        # for filename in [filenames[0]]:
            output_filename = 'test_videos_output/' + filename
    
            clip1 = VideoFileClip("test_videos/" + filename)#.subclip(3,3.01)
            white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
            white_clip.write_videofile(output_filename, audio=False)