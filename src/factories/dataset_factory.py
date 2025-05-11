from src.dataset.cifar10_dataset import CIFAR10Loader
from src.registry.registry import GLOBAL_REGISTRY

def get_dataloader(dataset_name, **dataset_args):
    GLOBAL_REGISTRY.get_instance(category='dataset', name=dataset_name, **dataset_args)