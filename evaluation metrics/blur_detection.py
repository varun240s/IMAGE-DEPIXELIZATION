import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Load the original and depixelized images
original_image = cv2.imread(r"C:\Users\Varun Reddy\Desktop\evaluation metrics\clear.png")
depixelized_image = cv2.imread(r"C:\Users\Varun Reddy\Desktop\evaluation metrics\blured.jpg")

# Convert the images to grayscale (if needed)
original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
depixelized_image_gray = cv2.cvtColor(depixelized_image, cv2.COLOR_BGR2GRAY)

# Calculate PSNR
psnr_value = psnr(original_image_gray, depixelized_image_gray)

# Calculate SSIM
ssim_value, _ = ssim(original_image_gray, depixelized_image_gray, full=True)

print(f"PSNR: {psnr_value}")
print(f"SSIM: {ssim_value}")
