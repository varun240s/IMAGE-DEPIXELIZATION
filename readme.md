
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

## Description

This project aims to detect and correct pixelated images using machine learning models. The model is trained using high-quality images and their pixelated versions.

## Scripts

- `create_pixelated_images.py`: Generates pixelated images from high-quality images.
- `train_dncnn.py`: Trains the DnCNN model on the dataset.
- `evaluate_model.py`: Evaluates the trained model on test data.
- `detect_pixelated_images.py`: Detects pixelated images in a given dataset.
