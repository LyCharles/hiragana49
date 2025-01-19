import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class HiraganaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image)  # Convert numpy array to PIL image
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data():
    train_images = np.load('/dataset/k49-train-imgs.npz')['arr_0']
    train_labels = np.load('/dataset/k49-train-labels.npz')['arr_0']
    test_images = np.load('/dataset/k49-test-imgs.npz')['arr_0']
    test_labels = np.load('/dataset/k49-test-labels.npz')['arr_0']
    return train_images, train_labels, test_images, test_labels

# Define data transformations
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_data_loaders(batch_size=8192):
    train_images, train_labels, test_images, test_labels = load_data()
    train_dataset = HiraganaDataset(train_images, train_labels, transform=train_transforms)
    val_dataset = HiraganaDataset(test_images, test_labels, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader
