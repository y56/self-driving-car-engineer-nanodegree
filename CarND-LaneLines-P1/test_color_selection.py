#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 03:16:10 2020

@author: y56
"""

original = np.copy(image)
    
plt.figure()
plt.title("original")
plt.imshow(original)

# grayscale
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

plt.figure()
plt.title("gray // cv2.COLOR_RGB2GRAY")
plt.imshow(gray, cmap='gray')

# darken the grayscale
darkened_img05 = helper.adjust_gamma(gray, 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(gray, 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

# check HLS
hls = helper.to_hls(image)

plt.figure()
plt.title("HLS H")
plt.imshow(hls[:,:,0], cmap='gray')

darkened_img05 = helper.adjust_gamma(hls[:,:,0], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hls[:,:,0], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

plt.figure()
plt.title("HLS L")
plt.imshow(hls[:,:,1], cmap='gray')

darkened_img05 = helper.adjust_gamma(hls[:,:,1], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hls[:,:,1], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

plt.figure()
plt.title("HLS S")
plt.imshow(hls[:,:,2], cmap='gray')

darkened_img05 = helper.adjust_gamma(hls[:,:,2], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hls[:,:,2], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

# check HSV
hsv = helper.to_hsv(image)

plt.figure()
plt.title("hsv H")
plt.imshow(hsv[:,:,0], cmap='gray')

darkened_img05 = helper.adjust_gamma(hsv[:,:,0], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hsv[:,:,0], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

plt.figure()
plt.title("hsv S")
plt.imshow(hsv[:,:,1], cmap='gray')

darkened_img05 = helper.adjust_gamma(hsv[:,:,1], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hsv[:,:,1], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')

plt.figure()
plt.title("hsv V")
plt.imshow(hsv[:,:,2], cmap='gray')

darkened_img05 = helper.adjust_gamma(hsv[:,:,2], 0.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 0.5")
plt.imshow(darkened_img05, cmap='gray')

darkened_img15 = helper.adjust_gamma(hsv[:,:,2], 1.5)

plt.figure()
plt.title("darkened_img // adjust_gamma // 1.5")
plt.imshow(darkened_img15, cmap='gray')