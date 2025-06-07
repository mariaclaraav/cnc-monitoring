import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Optional


class CV2DrawContours:
    """
    Class responsible for processing and plotting anomalies detected
    from grayscale spectrograms using the CV2Detection pipeline.
    """

    def __init__(self, cv2_detector, df, base_dir: str, freq_range: Tuple[int, int] = (100, 600), cmap: str = 'seismic'):
        """
        Initializes the class.

        Args:
            detector: Instance of CV2Detection class.
            df: DataFrame containing 'Unique_Code' and 'Time' columns.
            base_dir (str): Directory where original RGB images are stored.
            freq_range (Tuple[int, int]): Frequency range for the Y axis.
            cmap (str): Colormap for the original spectrogram image.
        """
        self.__cv2_detector = cv2_detector
        self.__df = df
        self.__base_dir = base_dir
        self.__freq_range = freq_range
        self.__cmap = cmap
        
        # Default plot params
        self.__plot_params = {
            'intensity': {'color': (255, 0, 0), 'thickness': 5},  # Blue
            'gradient': {'color': (255, 0, 0), 'thickness': 5},   # Green
            'combined_1': {'color': (0, 140, 255), 'thickness': 5},  # Orange
            'combined_2': {'color': (0, 255, 0), 'thickness': 7}     # Green
        }

    def set_plot_params(self, 
                    intensity_color: str = '#FF0000', intensity_thickness: int = 5,
                    gradient_color: str = '#FF0000', gradient_thickness: int = 5,
                    combined_score1_color: str = '#FFA500', combined_score1_thickness: int = 5,
                    combined_score2_color: str = '#00FF00', combined_score2_thickness: int = 7,
                    use_rgb_hex: bool = True):
        """
        Sets plotting parameters using HEX color codes.

        Args:
            intensity_color (str): HEX color for intensity contours.
            intensity_thickness (int): Line thickness for intensity contours.
            gradient_color (str): HEX color for gradient contours.
            gradient_thickness (int): Line thickness for gradient contours.
            combined_score1_color (str): HEX color for combined contours with score 1.
            combined_score1_thickness (int): Thickness for combined score 1.
            combined_score2_color (str): HEX color for combined contours with score 2.
            combined_score2_thickness (int): Thickness for combined score 2.
            use_rgb_hex (bool): Whether the hex code is interpreted as RGB (True) or BGR (False).
        """
        self.__plot_params['intensity'] = {
            'color': self.__hex_to_bgr(intensity_color, use_rgb_hex),
            'thickness': intensity_thickness
        }
        self.__plot_params['gradient'] = {
            'color': self.__hex_to_bgr(gradient_color, use_rgb_hex),
            'thickness': gradient_thickness
        }
        self.__plot_params['combined_1'] = {
            'color': self.__hex_to_bgr(combined_score1_color, use_rgb_hex),
            'thickness': combined_score1_thickness
        }
        self.__plot_params['combined_2'] = {
            'color': self.__hex_to_bgr(combined_score2_color, use_rgb_hex),
            'thickness': combined_score2_thickness
        }

    def __hex_to_bgr(self, hex_color: str, use_rgb_hex: bool = True) -> Tuple[int, int, int]:
        """
        Converts a HEX color code to BGR tuple.

        Args:
            hex_color (str): Color in HEX format (e.g. '#FF0000').
            use_rgb_hex (bool): If True, HEX is treated as RGB (standard), otherwise BGR.

        Returns:
            Tuple[int, int, int]: BGR color for use with OpenCV.
        """
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r) if use_rgb_hex else (r, g, b)

    def __detect_anomalies(self, img_gray: np.ndarray, intensity_thresh: float, gradient_thresh: float):
        """
        Applies intensity, gradient, and combined anomaly detection.

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.
            intensity_thresh (float): Threshold for intensity.
            gradient_thresh (float): Threshold for gradient.

        Returns:
            Tuple containing:
                - intensity contours (List[np.ndarray])
                - gradient contours (List[np.ndarray])
                - combined contours (List[np.ndarray])
                - contour scores (List[int])
        """
        intensity = self.__cv2_detector.detect_by_intensity(img_gray, threshold=intensity_thresh)
        gradient = self.__cv2_detector.detect_by_gradient(img_gray, threshold=gradient_thresh)
        score_mask, combined_contours, contour_scores = self.__cv2_detector.detect_combined(
            img_gray,
            intensity_thresh=intensity_thresh,
            gradient_thresh=gradient_thresh
        )
        return intensity, gradient, combined_contours, contour_scores

    def __draw_contours(self, img_gray: np.ndarray, img_original: np.ndarray,
                          intensity, gradient, combined_contours, contour_scores):
        """
        Draws contours on the original image.

        Args:
            img_gray (np.ndarray): Grayscale spectrogram image.
            img_original (np.ndarray): RGB image.
            intensity: Contours detected by intensity.
            gradient: Contours detected by gradient.
            combined_contours: Contours detected by combined method.
            contour_scores: Scores per contour.

        Returns:
            Tuple of 3 images: intensity image, gradient image, combined image with bounding boxes.
        """
        img_rgb_intensity = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_rgb_gradient = img_rgb_intensity.copy()
        img_combined = img_original.copy()

        for contour in intensity:
            x, y, w, h = cv2.boundingRect(contour)
            color = self.__plot_params['intensity']['color']
            thickness = self.__plot_params['intensity']['thickness']
            cv2.rectangle(img_rgb_intensity, (x, y), (x + w, y + h), color, thickness)

        for contour in gradient:
            x, y, w, h = cv2.boundingRect(contour)
            color = self.__plot_params['gradient']['color']
            thickness = self.__plot_params['gradient']['thickness']
            cv2.rectangle(img_rgb_gradient, (x, y), (x + w, y + h), color, thickness)

        for contour, score in zip(combined_contours, contour_scores):
            x, y, w, h = cv2.boundingRect(contour)
            key = f'combined_{score}'
            color = self.__plot_params[key]['color']
            thickness = self.__plot_params[key]['thickness']
            cv2.rectangle(img_combined, (x, y), (x + w, y + h), color, thickness)

        return img_rgb_intensity, img_rgb_gradient, img_combined
    def __get_time_extent(self, name: str) -> Optional[List[float]]:
        """
        Retrieves the time extent for plotting based on metadata.

        Args:
            name (str): Image file name (without extension).

        Returns:
            List[float]: [time_start, time_end, freq_min, freq_max] or None if not found.
        """
        unique_code = name.rsplit('_', 2)[0]
        time_range = self.__df.loc[self.__df['Unique_Code'] == unique_code, 'Time'].values
        if len(time_range) == 0:
            return None
        time_max = float(time_range[-1])
        return [0, time_max, *self.__freq_range]

    def __plot_anomaly(self, name: str, img_original: np.ndarray, img_rgb_intensity: np.ndarray,
                            img_rgb_gradient: np.ndarray, img_combined: np.ndarray,
                            intensity, gradient, contour_scores, extent):
        """
        Plots the four-panel visualization with original and annotated images.

        Args:
            name (str): Image name.
            img_original (np.ndarray): Original RGB image.
            img_rgb_intensity (np.ndarray): Image with intensity contours.
            img_rgb_gradient (np.ndarray): Image with gradient contours.
            img_combined (np.ndarray): Image with combined annotations.
            intensity: List of intensity contours.
            gradient: List of gradient contours.
            contour_scores: List of scores per combined contour.
            extent: List defining the image extent for matplotlib.
        """
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))

        axs[0].imshow(img_original, extent=extent, cmap=self.__cmap, aspect='auto')
        axs[0].set_title(f'{name}', fontsize=14)
        axs[0].set_xlabel("Time [s]", fontsize=14)
        axs[0].set_ylabel("Frequency [Hz]", fontsize=14)

        axs[1].imshow(img_rgb_intensity, extent=extent, aspect='auto')
        axs[1].set_title(f'Intensity ({len(intensity)})', fontsize=14)
        axs[1].set_xlabel("Time [s]", fontsize=14)

        axs[2].imshow(img_rgb_gradient, extent=extent, aspect='auto')
        axs[2].set_title(f'Gradient ({len(gradient)})', fontsize=14)
        axs[2].set_xlabel("Time [s]", fontsize=14)

        axs[3].imshow(img_combined, extent=extent, aspect='auto')
        axs[3].set_title(f'Combined ({sum(contour_scores)})', fontsize=14)
        axs[3].set_xlabel("Time [s]", fontsize=14)

        for ax in axs:
            ax.grid(False)
            ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        plt.show()

    def plot(self, img_list: List[np.ndarray], img_info,
                           intensity_thresh: float = 0.5, gradient_thresh: float = 0.6):
        """
        Orchestrates detection and plotting for all images in the dataset.

        Args:
            img_list (List[np.ndarray]): List of grayscale images.
            img_info (DataFrame): DataFrame with 'image_name' column.
            intensity_thresh (float): Threshold for intensity-based detection.
            gradient_thresh (float): Threshold for gradient-based detection.
        """
        self.__anomaly_scores = []
        self.__img_info = img_info.copy()

        for img_gray, name in zip(img_list, img_info['image_name']):
            print(f"Processing: {name}")

            intensity, gradient, combined_contours, contour_scores = self.__detect_anomalies(
                img_gray, intensity_thresh, gradient_thresh
            )

            image_path = os.path.join(self.__base_dir, name + '.png')
            img_original = np.asarray(Image.open(image_path).convert("RGB"))

            img_rgb_intensity, img_rgb_gradient, img_combined = self.__draw_contours(
                img_gray, img_original, intensity, gradient, combined_contours, contour_scores
            )

            extent = self.__get_time_extent(name)
            if extent is None:
                print(f"Time not found for {name}, skipping...")
                continue
            
            self.__anomaly_scores.append(sum(contour_scores))
            
            self.__plot_anomaly(
                name, img_original, img_rgb_intensity, img_rgb_gradient,
                img_combined, intensity, gradient, contour_scores, extent
            )
            
    def evaluate_anomaly_prediction(self) -> None:
        
        img_info = self.__img_info.copy()
        img_info['Anomaly_score'] = self.__anomaly_scores
        img_info.sort_values(by='Anomaly_score', ascending=False, inplace=True)
        img_info['Label'] = img_info['Label'].astype('str')
  
        img_info['Label'] = img_info['Label'].replace({'1': 'Anomaly', '0': 'Normal'})

        # Impressão dos resultados
        for _, row in img_info.iterrows():
            print(f"Unique_Code: {row['Unique_Code']} (axis: {row['axis']}), Anomaly Score: {row['Anomaly_score']}, true_label: {row['Label']}")
    
                
  