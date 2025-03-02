from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
import torch


class DataLoaderBuilderInterface(ABC):
    """Interface for building DataLoaders."""
    
    @abstractmethod
    def build(self, tensor: torch.Tensor) -> DataLoader:
        pass


class DefaultDataLoaderBuilder(DataLoaderBuilderInterface):
    """Builds a DataLoader with specified batch size and worker settings."""
    
    def __init__(self, batch_size: int, num_workers: int, shuffle: bool = False):
        """
        Initialize the DataLoader builder.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses for data loading.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def build(self, tensor: torch.Tensor) -> DataLoader:
        """
        Build a DataLoader from a tensor.

        Args:
            tensor (torch.Tensor): Tensor to wrap in a DataLoader.

        Returns:
            DataLoader: Configured DataLoader instance.
        """
        return DataLoader(
            tensor,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available()  # Otimiza para GPU se disponível
        )