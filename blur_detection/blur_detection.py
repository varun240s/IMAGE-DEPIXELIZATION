
import sys
import cv2
import numpy as np


def variance_of_image(image):
    """
    Calculates the variance of the Laplacian of an image, a measure of sharpness.

    It Returns The variance of the Laplacian.
    """

    # using grey scale version we can calculate the Laplacian for sharpness detection of an image 
# this is hsv color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]  # Use the V channel
    image2 = cv2.Laplacian(value_channel, cv2.CV_64F)
    variance = image2.var()

# This is grayscale color
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #image2 = cv2.Laplacian(gray, cv2.CV_64F)  # Calculate the Laplacian
    #variance = image2.var()  # Calculate variance

    return variance

if __name__ == "__main__":
    image_path = r"C:\Users\reddy\Desktop\INTEL UNNATI\clear.png"

    try:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Could not read image '{image_path}'.")
            sys.exit(1)

        image1 = variance_of_image(image)
        print(f"image_Variance: {image1:.2f}")  # Format output with 2 decimal places

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
"""
It is a statistical measure, quantifies the spread of pixel intensity values in an image.
if the image have high intensity it proves it has high contrast and sharp edges or else low intensity.

"""