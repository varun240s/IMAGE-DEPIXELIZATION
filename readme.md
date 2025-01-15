
## Setup

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

2. Prepare the dataset:

    ```bash
    python scripts/create_pixelated_images.py
    ```

3. Train the model:

    ```bash
    python scripts/train_dncnn.py
    ```

4. Evaluate the model:

    ```bash
    python scripts/evaluate_model.py
    ```
    
## Scripts

- `create_pixelated_images.py`: Generates pixelated images from high-quality images.
- `train_dncnn.py`: Trains the DnCNN model on the dataset.
- `evaluate_model.py`: Evaluates the trained model on test data.
- `detect_pixelated_images.py`: Detects pixelated images in a given dataset.



# Image Depixelization

This project focuses on correcting pixelated images using machine learning techniques. The primary objective is to develop a lightweight machine learning model (less than 10MB) capable of detecting and correcting pixelated images with high efficiency. The model should be able to handle images with a resolution of at least 1920x1080 and run at a minimum of 30 FPS.

## Data Collection and Preparation

### Step 1: Upscaling Pixelated Images
- **Goal**: Increase the image size to observe blockiness (pixelation). This process is known as upscaling.
- **Process**: 
  - Use various datasets containing images with different characteristics.
  - Ensure that all images are resized uniformly.

### Step 2: Model Comparison
- **Goal**: Compare various models based on:
  - Weight
  - Accuracy
  - Precision
  - Recall
- **Outcome**: Select models that show the best performance while remaining lightweight.

### Step 3: Dataset Collection
- **Types of Data**:
  - Nature images
  - Bird images
  - Gaming screenshots
  - Animated scenes
  - Text-based images
- **Tips**:
  - Ensure uniform image size and format.
  - Split datasets into training, validation, and test sets.
  - Use NumPy for image processing tasks such as resizing, cropping, and converting to grayscale.
  - Save and load datasets in formats such as `.npy`, `.jpg`, and `.png`.

## Data Augmentation

### Step 4: Pixelation Process
- **Goal**: Create pixelated images from original images.
- **Method**:
  - Downscale the original image by a factor of 5x or 6x.
  - Upscale it back using:
    - Nearest neighbor interpolation
    - Bilinear interpolation
- **Outcome**: Generate multiple datasets by varying upscaling factors and interpolation methods.

## Model Development

### Step 5: Design a Lightweight Model
- **Requirements**:
  1. Model size should be less than 10MB.
  2. It should run at a minimum of 30 FPS, preferably 60 FPS.
  3. Performance should be evaluated using metrics like:
     - F1 score
     - Precision
     - Recall
- **Objective**: The model should detect pixelated images even if only 1% of the dataset contains pixelated images and correct them effectively.

### Step 6: Pixelated Image Correction
- **Rules**:
  1. Model size must be under 10MB with 32-bit floating point precision.
  2. The model should run efficiently at 30 FPS or higher.
  3. Evaluation should be based on both subjective analysis and metric evaluation.
  4. The model should handle images of resolution 1920x1080.
  5. If a non-pixelated image is provided, the model should not distort the original image.

## Evaluation and Analysis

### Step 7: Subjective Analysis
- **Goal**: Use new datasets (different from the training set) for subjective evaluation.
- **Process**:
  - Evaluate the model's performance using different types of images.

### Step 8: Quantitative Analysis and Metric Evaluation
- **Metrics Used**:
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **LPIPS (Learned Perceptual Image Patch Similarity)**
  - **SSIM (Structural Similarity Index)**

### Final Comparison
- Compare the **MOS (Mean Opinion Score)** with the metric evaluation scores to cross-verify the performance of the developed model.

## Results
- Provide a detailed comparison of the selected models based on:
  - Weight
  - Accuracy
  - Precision
  - Recall
  - PSNR
  - LPIPS
  - SSIM
  - FPS (Frames Per Second)

## Conclusion
Summarize the key findings of the project, highlighting the best-performing model and its capabilities in correcting pixelated images efficiently.

## Future Work
- Explore more advanced interpolation methods.
- Experiment with additional datasets and edge cases.
- Further optimize the model for real-time performance.

## Repository Structure
```
IMAGE-DEPIXELIZATION/
├── data/                   # Collected datasets
├── models/                 # Trained models
├── scripts/                # Scripts for training and evaluation
├── results/                # Evaluation results
└── README.md               # Project documentation (this file)
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/varun240s/IMAGE-DEPIXELIZATION.git
   cd IMAGE-DEPIXELIZATION
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python scripts/train.py
   ```
4. Evaluate the model:
   ```bash
   python scripts/evaluate.py
   ```

## Acknowledgments
Special thanks to the contributors and supporters who made this project possible.

## License
This project is licensed under the MIT License.




