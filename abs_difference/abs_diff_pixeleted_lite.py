"""
This code gives the mean abf difference of the orginal image and pixelated image.
"""


import cv2
import numpy as np

# read original and pixelated image
img1 = cv2.imread(r"C:\Users\reddy\Desktop\INTEL UNNATI\clear.png")
img2 = cv2.imread(r"C:\Users\reddy\Desktop\INTEL UNNATI\blured.png")

# decimate images by some skip factor (2) for two different offsets (0 and 1)
dec1A = img1[::2, ::2]
dec1B = img1[1::2, 1::2]
dec2A = img2[::2, ::2]
dec2B = img2[1::2, 1::2]

# get mean of absolute difference
diff1 = cv2.absdiff(dec1A, dec1B)
mean1 = np.mean(diff1)
diff2 = cv2.absdiff(dec2A, dec2B)
mean2 = np.mean(diff2)
# The absolute difference between original and pixelated image .
print('mean absdiff original image:', mean1)
print('mean absdiff pixelated image:', mean2)

# convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# gray1.inshow(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
# gray1.inshow(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

"""
mean absdiff original image: 8.341491666666666
mean absdiff pixelated image: 3.068625

"""

