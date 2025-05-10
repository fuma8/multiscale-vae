from torchvision import datasets

from src.dataset.custom_dataset_loader import CustomDatasetLoader
from src.registry.registry import GLOBAL_REGISTRY

@GLOBAL_REGISTRY.register(category='dataset', name='cifar10')
class CIFAR10Loader(CustomDatasetLoader):
    def __init__(self, image_size: int = 64, batch_size: int = 32, train: bool = True, download: bool = True):
        super().__init__(image_size, batch_size, train, download)

    def _get_dataset(self):
        return datasets.CIFAR10(
            root='./data',
            train=self.train,
            download=self.download,
            transform=self.transform
        )