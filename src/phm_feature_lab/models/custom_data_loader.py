import torch
from torch.utils.data import DataLoader
from phm_feature_lab.utils.data_processing.tensor_converter import DefaultTensorConverter
from phm_feature_lab.utils.data_processing.data_loader_builder import DefaultDataLoaderBuilder
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

class CustomDataLoader:
    """Factory class to create DataLoaders for training, validation, and test datasets."""

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        device: torch.device,
        shuffle: bool = False,
    ):
        """
        Initialize the DataLoaderFactory.

        Args:
            tensor_converter (DefaultTensorConverter): Object to convert data to tensors.
            dataloader_builder (DefaultDataLoaderBuilder): Object to build DataLoaders.
            device (torch.device): Device to load the data onto (e.g., 'cpu', 'cuda').
        """
        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__shuffle = shuffle
        self.__device = device
        
        self.__tensor_converter = DefaultTensorConverter()
        self.__dataloader_builder = DefaultDataLoaderBuilder(
            batch_size=self.__batch_size,
            num_workers=self.__num_workers,
            shuffle=self.__shuffle,
        )

    def create(self, X_train, X_val, X_test) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for training, validation, and test datasets.

        Args:
            X_train (array-like): Training data.
            X_val (array-like): Validation data.
            X_test (array-like): Test data.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader]: Tuple of (train_loader, val_loader, test_loader).
        """
        logger.info('Creating DataLoaders...')

        # Convert data to tensors
        train_tensor = self.__tensor_converter.convert(X_train, self.__device)
        val_tensor = self.__tensor_converter.convert(X_val, self.__device)
        test_tensor = self.__tensor_converter.convert(X_test, self.__device)

        # Build DataLoaders
        train_loader = self.__dataloader_builder.build(train_tensor)
        val_loader = self.__dataloader_builder.build(val_tensor)
        test_loader = self.__dataloader_builder.build(test_tensor)

        logger.info(f"DataLoaders created: \ntrain={len(train_loader)}, \nval={len(val_loader)}, \ntest={len(test_loader)}")
        return train_loader, val_loader, test_loader
