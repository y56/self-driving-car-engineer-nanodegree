###### tags: `self-driving-car-engineer-nanodegree` `computer-vision` `learning`
# Line Detection by Hough Transformation

## there are two kinds of resolutions
- resolution of the Hough space
- resolution of the accumulator

## a useful tutorial
http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/HoughTrans_lines_09.pdf
### how to define a valid point in Hough space
As it is obvious from Figure 4e, several entrances in the accumulator around one true line in the edge map will have large values. Therefore a simple threshold has a tendency to detect several (almost identical) lines for each true line. To avoid this, a ***suppression neighborhood*** can be defined, so that two lines must be significantly different before both are detected.

### from infinite lines to finite lines
The classical Hough transform detects lines given only by the parameters r and θ and no information with regards to length. Thus, all detected lines are infinite in length. If finite lines are
desired, some additional analysis must be performed to determine which areas of the image that
contributes to each line. Several algorithms for doing this exist. 

One way is to store coordinate information for all points in the accumulator, and use this information to limit the lines. How-
ever, this would cause the accumulator to use much more memory. 

Another way is to search along the infinite lines in the edge image to find finite lines. A variant of this approach known as
the Progressive Probabilistic Hough Transform is discussed in Section 6.

#### parameters 
- hough: Performs the Hough transform on a binary edge image, and returns the accumulator. The resolution of the accumulator used in this worksheet is 1 for both r and θ.
- houghpeaks: Detects lines by interpreting the accumulator. In this worksheet the threshold was set to 20% of the maximum value in the accumulator, and the ***suppression neighbourhood*** was set to approximately 5% of the resolution of r and θ respectively.
- houghlines: Converts infinite lines to finite lines. In this worksheet, the ***minimum length of a line*** was set to 30 pixels, and the algorithm was allowed ***to connect lines through holes of up to 30 pixels***.
### Progressive Probabilistic Hough Transform (PPHT)

## python openCV
```python=
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

```