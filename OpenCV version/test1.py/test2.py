import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_images_from_folder(folder):
    images = []  # Initialize an empty list to store images
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        else:
            print(f"Warning: Failed to load image: {os.path.join(folder, filename)}")
    return images

def pixelate_image(image, scale_factor=6):
    height, width = image.shape[:2]
    # Downscale image
    small_image = cv2.resize(image, (width // scale_factor, height // scale_factor), interpolation=cv2.INTER_NEAREST)
    # Upscale image
    pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image

def process_images(images, scale_factor=6):
    pixelated_images = [pixelate_image(img, scale_factor) for img in images]
    return pixelated_images

# Image Correction
def correct_pixelation(image, d=15, sigmaColor=250, sigmaSpace=250):
    corrected_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return corrected_image

# Evaluation Metrics
def evaluate_metrics(original_image, corrected_image):
    psnr_value = psnr(original_image, corrected_image)
    ssim_value = ssim(original_image, corrected_image, win_size=3, channel_axis=-1)  # Setting win_size to 3 and specifying the channel axis
    return psnr_value, ssim_value

# Load and process images
folder_path = r"C:\Users\reddy\Desktop\INTEL UNNATI\pixel images"
images = load_images_from_folder(folder_path)

if not images:
    print("Error: No images loaded. Please check the folder path and image files.")
else:
    pixelated_images = process_images(images, scale_factor=6)

    # Correct pixelation and evaluate
    for i in range(len(images)):
        original_image = images[i]
        pixelated_image = pixelated_images[i]
        corrected_image = correct_pixelation(pixelated_image)

        psnr_value, ssim_value = evaluate_metrics(original_image, corrected_image)
        print(f'Image {i+1}: PSNR: {psnr_value}, SSIM: {ssim_value}')

        # Display the results
        cv2.imshow('Original Image', original_image)
        cv2.imshow('Pixelated Image', pixelated_image)
        cv2.imshow('Corrected Image', corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
