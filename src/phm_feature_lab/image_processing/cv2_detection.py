import cv2
import numpy as np
from typing import List

class CV2Detection:
    """
    Detects anomalies in grayscale spectrogram images using computer vision strategies.
    """

    def __image_norm(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Normalizes the grayscale image to the range [0, 1].

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.

        Returns:
            np.ndarray: Normalized grayscale image.
        """
        return img_gray.astype(np.float32) / 255.0

    def __apply_threshold(self, img_norm: np.ndarray, threshold: float) -> np.ndarray:
        """
        Applies a binary threshold to a normalized image.

        Args:
            img_norm (np.ndarray): Image with values normalized between 0 and 1.
            threshold (float): Threshold value between 0 and 1.

        Returns:
            np.ndarray: Binary mask (uint8) with values 0 or 255.
        """
        _, mask = cv2.threshold(img_norm, threshold, 1.0, cv2.THRESH_BINARY)
        mask_uint8 = (mask * 255).astype(np.uint8)
        return mask_uint8

    def __extract_contours_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Converts a binary mask to contours by thresholding and extracting shapes.

        Args:
            mask (np.ndarray): Binary mask (uint8).

        Returns:
            List[np.ndarray]: List of contours.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def __sobel_grad(self, img_norm: np.ndarray, dx: int = 1, dy: int = 1, ksize: int = 5) -> np.ndarray:
        """
        Computes the Sobel gradient magnitude from a normalized image.

        Args:
            img_norm (np.ndarray): Normalized image.
            dx (int): Derivative order in x direction.
            dy (int): Derivative order in y direction.
            ksize (int): Kernel size. Must be odd and positive.

        Returns:
            np.ndarray: Normalized gradient magnitude (0 to 1).
        """
        sobel_x = cv2.Sobel(img_norm, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
        sobel_y = cv2.Sobel(img_norm, cv2.CV_64F, dx=0, dy=1, ksize=ksize)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_mag /= grad_mag.max() if grad_mag.max() > 0 else 1
        return grad_mag

    def detect_by_intensity(self, img_gray: np.ndarray, threshold: float = 0.8) -> List[np.ndarray]:
        """
        Detects anomalies based on high normalized intensity values.

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.
            threshold (float): Normalized intensity threshold (0 to 1).

        Returns:
            List[np.ndarray]: Contours of detected regions.
        """
        img_norm = self.__image_norm(img_gray)
        mask = self.__apply_threshold(img_norm, threshold)
        contours = self.__extract_contours_from_mask(mask)
        return contours

    def detect_by_gradient(self, img_gray: np.ndarray, threshold: float = 0.3) -> List[np.ndarray]:
        """
        Detects anomalies based on spatial and temporal gradients.

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.
            threshold (float): Gradient magnitude threshold (0 to 1).

        Returns:
            List[np.ndarray]: Contours of detected regions.
        """
        img_norm = self.__image_norm(img_gray)
        grad_mag = self.__sobel_grad(img_norm)
        mask = self.__apply_threshold(grad_mag, threshold)
        contours = self.__extract_contours_from_mask(mask)
        return contours

    # def detect_combined(self, img_gray: np.ndarray, intensity_thresh: float = 0.8, gradient_thresh: float = 0.3) -> List[np.ndarray]:
    #     """
    #     Detects anomalies that are both intense and have high gradient values.

    #     Args:
    #         img_gray (np.ndarray): Grayscale spectrogram image.
    #         intensity_thresh (float): Intensity threshold (0 to 1).
    #         gradient_thresh (float): Gradient magnitude threshold (0 to 1).

    #     Returns:
    #         List[np.ndarray]: Contours of detected regions.
    #     """
    #     img_norm = self.__image_norm(img_gray)
    #     mask_int = self.__apply_threshold(img_norm, intensity_thresh)
    #     grad_mag = self.__sobel_grad(img_norm)
    #     mask_grad = self.__apply_threshold(grad_mag, gradient_thresh)
    #     combined_mask = cv2.bitwise_and(mask_int, mask_grad)
    #     contours = self.__extract_contours_from_mask(combined_mask)
    #     return contours
    def detect_combined(self, img_gray: np.ndarray, intensity_thresh: float = 0.8, gradient_thresh: float = 0.3):
        """
        Detects anomalies with weighted scoring:
        - 1: Anomaly from intensity OR gradient.
        - 2: Anomaly from both.

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.
            intensity_thresh (float): Intensity threshold (0 to 1).
            gradient_thresh (float): Gradient threshold (0 to 1).

        Returns:
            Tuple[np.ndarray, List[np.ndarray], List[int]]:
                - Score mask with values 0, 1, or 2
                - List of contours
                - Corresponding score for each contour (1 or 2)
        """
        img_norm = self.__image_norm(img_gray)

        mask_int = self.__apply_threshold(img_norm, intensity_thresh)
        grad_mag = self.__sobel_grad(img_norm)
        mask_grad = self.__apply_threshold(grad_mag, gradient_thresh)

        # Cria mapa de pontuação
        combined_score_mask = (mask_int > 0).astype(np.uint8) + (mask_grad > 0).astype(np.uint8)

        # Máscara binária para extração de contornos (regiões com pontuação 1 ou 2)
        mask_bin = (combined_score_mask > 0).astype(np.uint8) * 255

        # Extrai contornos da máscara binária
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Atribui uma pontuação para cada contorno com base na média dos pixels dentro do contorno
        contour_scores = []
        for contour in contours:
            mask_contour = np.zeros_like(combined_score_mask, dtype=np.uint8)
            cv2.drawContours(mask_contour, [contour], -1, 1, thickness=cv2.FILLED)
            score_region = combined_score_mask[mask_contour == 1]
            avg_score = int(score_region.max())
            contour_scores.append(avg_score)

        return combined_score_mask, contours, contour_scores
