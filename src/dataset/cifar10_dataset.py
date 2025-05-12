from torchvision import datasets
from torch.utils.data import DataLoader

from src.dataset.custom_dataset_loader import CustomDatasetLoader
from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='dataset', name='cifar10')
class CIFAR10Loader(CustomDatasetLoader):
    def __init__(self, image_size: int = 64, batch_size: int = 32, train: bool = True, download: bool = True):
        super().__init__(image_size, batch_size, train, download)

    def _get_dataset(self):
        dataset = datasets.CIFAR10(
            root='./data',
            train=self.train,
            download=self.download,
            transform=self.transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.train,
            num_workers=1
        )