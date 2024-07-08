import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
# Add current directory to sys.path for correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from DnCNN.model import DnCNN  # Assuming DnCNN model is defined in DnCNN/model.py
from DnCNN.utils import AverageMeter,CustomDataset  # Assuming AverageMeter is defined in DnCNN/utils.py
# from DnCNN.utils import CustomDataset  # Uncomment if you implement CustomDataset

# Example training configuration
# Replace placeholders with actual values or variables from your project

# Define paths to your datasets
train_data_path = r"E:\INTEL VARUN\data\train"
val_data_path = r"E:\INTEL VARUN\data\val"

# Hyperparameters
batch_size = 12
learning_rate = 0.001
num_epochs = 10

# Initialize AverageMeters for tracking loss and any other metrics
train_loss_meter = AverageMeter()
val_loss_meter = AverageMeter()

# Load datasets
train_dataset = CustomDataset(train_data_path, transform=ToTensor())
val_dataset = CustomDataset(val_data_path, transform=ToTensor())

print(f"Length of train_dataset: {len(train_dataset)}")
print(f"Length of val_dataset: {len(val_dataset)}")

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
model = DnCNN()  # Initialize your DnCNN model

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # Example loss function, adjust as needed

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss_meter.reset()

    # Iterate over batches
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update AverageMeter
        train_loss_meter.update(loss.item(), inputs.size(0))

        # Print progress every few batches
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {train_loss_meter.val:.6f} ({train_loss_meter.avg:.6f})')

    # Validation
    model.eval()
    val_loss_meter.reset()

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, targets)
            val_loss_meter.update(val_loss.item(), inputs.size(0))

    print(f'Validation Epoch: {epoch}\t'
          f'Loss: {val_loss_meter.avg:.6f}')

# Save trained model
torch.save(model.state_dict(),r"E:\INTEL VARUN\models\dncnn_model.pth")
print('Training finished and model saved.')
