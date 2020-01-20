#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 23:24:53 2020

to learn gamma correction

@author: y56
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# build a lookup table mapping the pixel values [0, 255] to
# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # plot the correction curve
    plt.figure()
    plt.title("gamma "+str(gamma))
    plt.plot(table)
    
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

image = mpimg.imread('exit-ramp.jpg')

plt.figure()
plt.title('original')
plt.imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.figure()
plt.title('gray')
plt.imshow(gray, cmap='gray')

gamma05 = adjust_gamma(gray, 0.5)

plt.figure()
plt.title('gamma 0.5')
plt.imshow(gamma05, cmap='gray')

gamma05 = adjust_gamma(gray, 2)

plt.figure()
plt.title('gamma 2')
plt.imshow(gamma05, cmap='gray')


