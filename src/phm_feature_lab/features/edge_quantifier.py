import numpy as np
from typing import List, Tuple
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()


class EdgeQuantifier:
    def __init__(self) -> None:
        """
        Initializes the EdgeQuantifier class.
        """
        logger.info("EdgeQuantifier initialized.")

    def _count_edge_points(self, image: np.ndarray) -> np.ndarray:
        """
        Counts the coordinates of edge points in the image.

        Parameters:
        - image: A processed image (grayscale with detected edges).

        Returns:
        - edge_points_coords: An array of coordinates where edges are detected.
        """
        try:
            edge_points_coords = np.column_stack(np.where(image > 0))
            return edge_points_coords
        except Exception as e:
            logger.error("Error counting edge points.", exc_info=True)
            raise ValueError("Error counting edge points.") from e
        
    def __img_check(self, processed_imgs: List[np.ndarray], img_names: List[str]) -> None:
        """
        Validates that the number of processed images matches the number of image names.

        Parameters:
        - processed_imgs: List of processed grayscale images.
        - img_names: List of image names.

        Raises:
        - ValueError: If the lengths of processed images and image names do not match.
        """
        if len(processed_imgs) != len(img_names):
            logger.error("Mismatch between number of images and names.")
            raise ValueError("Number of processed images and image names must match.")

    def rank_by_edges(self, processed_imgs: List[np.ndarray], img_names: List[str]) -> List[Tuple[str, int, np.ndarray]]:
        """
        Ranks images by the number of detected edge points.

        Parameters:
        - processed_imgs: List of processed grayscale images.
        - img_names: List of image names.

        Returns:
        - ranked_images: List of tuples (image name, number of edge points, edge points coordinates),
                         sorted by the number of edge points in descending order.
        """
        self.__img_check(processed_imgs, img_names)

        edges_count = []

        try:
            for i, image in enumerate(processed_imgs):
                # Count edge points for the current image
                edge_points_coords = self._count_edge_points(image)
                num_edges = len(edge_points_coords)

                # Append the result with image name, edge count, and edge coordinates
                edges_count.append((img_names[i], num_edges, edge_points_coords))

            # Sort the results by the number of edge points in descending order
            ranked_images = sorted(edges_count, key=lambda x: x[1], reverse=True)

            logger.info("Edge quantification completed successfully.")
            return ranked_images
        
        except Exception as e:
            logger.error("Error quantifying edges.", exc_info=True)
            raise ValueError("Error quantifying edges.") from e
