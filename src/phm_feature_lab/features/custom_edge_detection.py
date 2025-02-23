import cv2
from typing import Tuple
import numpy as np
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()


class CustomEdgeDetection:
    def __init__(self, blur_kernel: Tuple[int, int] = (5, 5), threshold1: int = 100, threshold2: int = 200) -> None:
        """
        Initializes the CustomEdgeDetection class.

        Parameters:
        - blur_kernel: Kernel size for Gaussian blur (default is (5, 5)).
        - threshold1: First threshold for Canny edge detection (default is 100).
        - threshold2: Second threshold for Canny edge detection (default is 200).
        """
        self.blur_kernel = blur_kernel
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        logger.info("CustomEdgeDetection initialized with given parameters.")


    def __check_image_validity(self, image: np.ndarray) -> None:
        """
        Validates the image format and properties.

        Parameters:
        - image: Input image.

        Raises:
        - TypeError: If the image is not a NumPy array.
        - ValueError: If the image is not grayscale or square.
        """
        if not isinstance(image, np.ndarray):
            logger.error("Invalid image type: Expected a NumPy array.")
            raise TypeError("Image must be a NumPy array.")

        if len(image.shape) != 2:
            logger.error("Invalid image format: Expected a 2D grayscale image.")
            raise ValueError("Image must be a 2D grayscale array.")

        if image.shape[0] != image.shape[1]:
            logger.error("Invalid image dimensions: Expected a square image.")
            raise ValueError("Image must be square (width equals height).")

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur to reduce noise in the image.

        Parameters:
        - image: Input image (in grayscale).

        Returns:
        - blurred_image: The image after applying Gaussian blur.
        """
        try:
            blurred_image = cv2.GaussianBlur(image, self.blur_kernel, 0)
            return blurred_image
        
        except Exception as e:
            logger.error("Failed to apply Gaussian blur.", exc_info=True)
            raise ValueError("Error applying Gaussian blur.") from e

    def _apply_canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the Canny edge detection to find edges in the image.

        Parameters:
        - image: Input image (in grayscale).

        Returns:
        - edges: The detected edges after applying Canny edge detection.
        """
        try:
            edges = cv2.Canny(image, self.threshold1, self.threshold2)
            return edges
        except Exception as e:
            logger.error("Failed to apply Canny edge detection.", exc_info=True)
            raise ValueError("Error applying Canny edge detection.") from e

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian blur and Canny edge detection if the image is valid.

        Parameters:
        - image: Input image (in grayscale).

        Returns:
        - processed_image: The final processed image with detected edges.
        """
        try:
            self.__check_image_validity(image)
            blurred_image = self._apply_gaussian_blur(image)
            processed_image = self._apply_canny_edge_detection(blurred_image)
            #logger.info("Edge detection completed successfully.")
            return processed_image
        except (TypeError, ValueError) as e:
            logger.error(f"Validation or processing error: {e}")
            raise
        except Exception as e:
            logger.error("Unexpected error during edge detection.", exc_info=True)
            raise


