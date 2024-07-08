import os
from PIL import Image
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
from torchvision.transforms import ToTensor, CenterCrop


class AverageMeter(object):
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


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images = self.load_images()  # Load images upon initialization

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply CenterCrop transformation
        image = CenterCrop((256, 256))(image)

        target = image.copy()  # Example: Use image as target for denoising

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

    def load_images(self):
        images = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    images.append(os.path.join(root, file))
        return images
