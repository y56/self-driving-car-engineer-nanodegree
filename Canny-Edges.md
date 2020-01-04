# Canny Edges

ref = https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html

-   _Hysteresis_: The final step. Canny does use two thresholds (upper and lower):
    
    1.  If a pixel gradient is higher than the **_upper_ threshold**, the pixel is accepted as an edge
    2.  If a pixel gradient value is below the **_lower_ threshold**, then it is rejected.
    3.  If the pixel gradient is **between** the two thresholds, then it will be accepted only if it is **connected** to a pixel that is above the _upper_ threshold.
    
    **Canny recommended a _upper_:_lower_ ratio between 2:1 and 3:1.**
