import cv2
import os
import glob

def create_pixelated_images(input_folder, output_folder, jpeg_qualities=[10, 20], downscale_factors=[5, 6]):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of images in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    if not image_paths:
        print(f"No images found in the input folder '{input_folder}'.")
        return
    
    # Process each image
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image '{img_path}'.")
            continue
        
        filename = os.path.basename(img_path)
        for quality in jpeg_qualities:
            for factor in downscale_factors:
                # Downscale the image
                downscaled_img = cv2.resize(img, 
                                            (img.shape[1] // factor, img.shape[0] // factor), 
                                            interpolation=cv2.INTER_LINEAR)
                # Upscale back to original size
                pixelated_img = cv2.resize(downscaled_img, 
                                           (img.shape[1], img.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                
                # Create directory for this combination if it doesn't exist
                output_dir = os.path.join(output_folder, f"quality_{quality}_factor_{factor}")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the pixelated image with JPEG compression
                pixelated_img_path = os.path.join(output_dir, filename)
                cv2.imwrite(pixelated_img_path, pixelated_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                print(f"Saved: {pixelated_img_path}")

if __name__ == "__main__":
    input_folder = r"E:\INTEL VARUN\data\train\high_quality_images"
    output_folder = r"E:\INTEL VARUN\data\train\pixelated_images"
    create_pixelated_images(input_folder, output_folder, jpeg_qualities=[10, 20], downscale_factors=[5, 6])
