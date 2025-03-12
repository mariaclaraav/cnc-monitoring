from abc import ABC, abstractmethod
import torch
from typing import Optional, Any
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

class TensorConverterInterface(ABC):
    """Interface for converting data into PyTorch tensors."""
    
    @abstractmethod
    def convert(self, data: Any, device: torch.device) -> torch.Tensor:
        pass


class DefaultTensorConverter(TensorConverterInterface):
    """Converts array-like data into PyTorch tensors and moves them to a device."""
    
    def __init__(self):
        pass

    def convert(self, data: Optional[Any], device: torch.device) -> torch.Tensor:
        """
        Convert data to a PyTorch tensor and move it to the specified device.

        Args:
            data (Optional[Any]): Data to convert (e.g., numpy array, list).
            device (torch.device): Device to move the tensor to (e.g., 'cpu', 'cuda').

        Returns:
            torch.Tensor: Converted tensor on the specified device.
        """
        if data is None:
            logger.warning("Input data is None, returning empty tensor.")
            return torch.tensor([], dtype=torch.float32).to(device)
        
        try:
            tensor = torch.tensor(data, dtype=torch.float32).to(device)
            return tensor
        except Exception as e:
            logger.error(f"Failed to convert data to tensor: {e}")
            raise