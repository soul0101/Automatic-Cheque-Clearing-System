import cv2
import math
import numpy as np 

def get_size(img):
    """Return the size of the image in pixels."""
    ih, iw = img.shape[:2]
    return iw * ih

def white_percent(img):
    """Return the percentage of the thresholded image that's white."""
    return cv2.countNonZero(img) / get_size(img)

def near_edge(img, contour):
    """Check if a contour is near the edge in the given image."""
    x, y, w, h = cv2.boundingRect(contour)
    ih, iw = img.shape[:2]
    mm = 80 # margin in pixels
    return (x < mm
            or x + w > iw - mm
            or y < mm
            or y + h > ih - mm)

def contourOK(img, cc):
    ih, iw = img.shape[:2]
    image_area = ih*iw
    contour_area = cv2.contourArea(cc)

    if contour_area > image_area * 0.75:
        return False
    if contour_area < image_area * 1e-5:
        return False
    return True

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def get_scaled_xywh(img, cheque_width, cheque_height, x, y, w, h):
    ih, iw = img.shape[:2]
    x = iw * x / cheque_width
    w = iw * w / cheque_width
    y = ih * y / cheque_height
    h = ih * h / cheque_height
    return math.floor(x), math.floor(y), math.floor(w), math.floor(h)