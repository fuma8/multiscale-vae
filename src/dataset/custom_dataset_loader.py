import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Any

class CustomDatasetLoader:
    def __init__(self, image_size: int = 32, batch_size: int = 32, train: bool = True, download: bool = True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.train = train
        self.download = download

        # 前処理
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _get_dataset(self):
        raise NotImplementedError("This method should be overridden in child classes")

    def get_dataloader(self) -> DataLoader:
        dataset = self._get_dataset()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)