import os
import cv2

# Directory paths
input_dir = 'data/train/pixelated_images/'
results_dir = 'data/train/detected_pixelated/'

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Function to detect pixelation
def is_pixelated(image):
    # Example detection logic (replace with your detection algorithm)
    # Here, we check if the image dimensions are smaller than expected due to pixelation
    if image.shape[0] < 50 or image.shape[1] < 50:
        return True
    else:
        return False

# Iterate through each image in input_dir
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    
    # Check if image is pixelated
    if is_pixelated(img):
        # Save detected pixelated image
        output_path = os.path.join(results_dir, filename)
        cv2.imwrite(output_path, img)

print("Pixelated images detected and saved.")
