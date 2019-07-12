import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

BATCH_SIZE = 1
data_transforms = (
    transforms.Compose([
        # transforms.Resize((800, 800)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))


class VOC2007DataLoader:
    def __init__(self):
        train_set = torchvision.datasets.VOCDetection(
            './dataloaders/datasets/voc2007',
            year='2007',
            image_set='train',
            download=True,
            transform=data_transforms)
        self.train = DataLoader(
            train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
