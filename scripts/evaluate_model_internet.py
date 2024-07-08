import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pickle

# Add current directory to sys.path for correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DnCNN.model import DnCNN  # Assuming DnCNN model is defined in DnCNN/model.py
from DnCNN.utils import CustomDataset  # Assuming CustomDataset is defined in DnCNN/utils.py

# Define paths to your test dataset and model weights
test_data_path = 'data/test/pixelated_images/'
model_path = 'models/model_DnCNN_sigma=50_epoch_46.pth'

# Hyperparameters
batch_size = 8  # Adjust based on your GPU capacity

# AverageMeter class to keep track of loss
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Custom Unpickler to remap module names
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'models_DnCNN':
            module = 'DnCNN.model'  # Adjust based on the module path in your current environment
        return super().find_class(module, name)

def custom_load(file_path):
    with open(file_path, 'rb') as f:
        unpickler = CustomUnpickler(f)
        state_dict = unpickler.load()
    return state_dict

# Load the test dataset
test_dataset = CustomDataset(test_data_path, transform=ToTensor())

# DataLoader setup
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
model = DnCNN()  # Initialize your DnCNN model

# Load the trained model
state_dict = custom_load(model_path)
model.load_state_dict(state_dict)
model.eval()

# Define the loss function for evaluation (adjust as needed)
criterion = torch.nn.MSELoss()

# Evaluation
test_loss_meter = AverageMeter()

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss_meter.update(loss.item(), inputs.size(0))

print(f'Test Loss: {test_loss_meter.avg:.6f}')

# Function to display images side by side
def show_images(input_img, output_img, target_img, title1, title2, title3):
    input_img = input_img.detach().cpu().numpy().transpose((1, 2, 0))
    output_img = output_img.detach().cpu().numpy().transpose((1, 2, 0))
    target_img = target_img.detach().cpu().numpy().transpose((1, 2, 0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_img)
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(output_img)
    axes[1].set_title(title2)
    axes[1].axis('off')
    axes[2].imshow(target_img)
    axes[2].set_title(title3)
    axes[2].axis('off')
    plt.show()

# Display a few test results
num_examples = 5
for i in range(num_examples):
    inputs, targets = next(iter(test_loader))
    outputs = model(inputs)
    print(f"Example {i+1}")
    show_images(inputs[0], outputs[0], targets[0], "Input Image", "Output Image", "Target Image")
