#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:44:26 2020

@author: y56
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


"""
Area-selection makes sharp drops.
Do it in a late stage.

(color-enhancement -->) gray--> Canny/edge --> area-selecion --> Hough
"""
global line_img, mask
def process_image(image):

    global line_img, mask

    # print('This image is:', type(image), 'with dimensions:', image.shape)

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(gray, cmap='gray')

    blur_gray = cv2.GaussianBlur(gray,(5, 5),0)


    edges = canny(blur_gray, 10 , 10)
    plt.figure()
    plt.imshow(edges, cmap='gray')

    # area selection
    imshape = image.shape
    # print(imshape[0]) # 540
    # print(imshape[1]) # 960
    vertices = np.array([[(120/960*imshape[1], imshape[0] * 0.92),
                          (445/960*imshape[1], 320/540*imshape[0]),
                          (520/960*imshape[1], 320/540*imshape[0]),
                          ((960-70)/960*imshape[1], imshape[0] * 0.91)]],
                        dtype=np.int32)

    mask = np.zeros_like(edges)
    mask3 = np.ones_like(image) * 100

    cv2.fillPoly(mask, vertices, 255)
    cv2.fillPoly(mask3, vertices, 0)
    masked_edges = cv2.bitwise_and(edges, mask)
    plt.figure()
    plt.imshow(masked_edges, cmap='gray')

    # Display the image and show region and color selections
    plt.figure()
    masked_gray = cv2.bitwise_and(gray, mask)
    plt.imshow(masked_gray)
    x = [vertices[0][0][0], vertices[0][1][0], vertices[0][2][0],
          vertices[0][3][0], vertices[0][0][0]]
    y = [vertices[0][0][1], vertices[0][1][1], vertices[0][2][1],
          vertices[0][3][1], vertices[0][0][1]]
    plt.plot(x, y, 'b--', lw=4)

    line_img = hough_lines(masked_edges, 1, np.pi/360, 4, 12, 4)
    plt.figure()
    plt.imshow(line_img, cmap='gray')

    my_weighted_img = weighted_img(line_img, image, α=0.8, β=1., γ=0.)
    plt.figure()
    plt.imshow(line_img)

    output = weighted_img(mask3, my_weighted_img, α=0.8, β=1., γ=0.)

    return output

if 1:
    import os
    filenames = os.listdir("test_images/")
    for filename in filenames[2:4]:
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

        clip1 = VideoFileClip("test_videos/" + filename)#.subclip(3,6)
        white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
        white_clip.write_videofile(output_filename, audio=False)
