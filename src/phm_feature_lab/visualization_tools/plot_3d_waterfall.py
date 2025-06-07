import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from typing import Optional, List, Tuple

def plot_3d_waterfall(
    data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    figsize: Optional[Tuple[int, int]] = (12, 14),
    cmap: str = "coolwarm",
    alpha: float = 0.75,
    cmap_axis: str = "Y",
    z_lim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Generates a 3D waterfall plot with colors based on either the Y-axis (dataset index) or the Z-axis (maximum amplitude).

    Args:
        data (list of tuples):
            A list of tuples containing (x_values, y_values, z_values), where:
            - x_values: X-axis values (array-like).
            - y_values: Y-axis values (array-like).
            - z_values: Z-axis values (array-like).
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        z_label (str): Label for the Z-axis.
        title (str): Title of the plot.
        figsize (tuple, optional): Figure size (width, height). Default is (12, 14).
        cmap (str, optional): Colormap used to map values. Default is 'coolwarm'.
        alpha (float, optional): Transparency of the polygons in the plot. Default is 0.75.
        cmap_axis (str, optional): Axis to which the colormap is applied. Can be 'Y' (dataset index) or 'Z' (maximum amplitude). Default is 'Y'.
        z_lim (tuple, optional): Limits for the Z-axis (min, max). If not provided, limits are adjusted automatically.

    Returns:
        None: Displays the 3D waterfall plot.
    """
    # Extract data for the plot
    x_values = [item[0] for item in data]
    y_values = [item[1] for item in data]
    z_values = [item[2] for item in data]

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Create a list of polygons for the waterfall plot
    waterfall = [list(zip(x_values[i], z_values[i])) for i in range(len(data))]

    # Define colors based on the specified axis
    if cmap_axis == "Y":
        colors = [plt.get_cmap(cmap)(i / float(len(data) - 1)) for i in range(len(data))]
    elif cmap_axis == "Z":
        max_z_per_dataset = [max(z) for z in z_values]
        z_min, z_max = min(max_z_per_dataset), max(max_z_per_dataset)
        norm = plt.Normalize(z_min, z_max)
        colors = [plt.get_cmap(cmap)(norm(z)) for z in max_z_per_dataset]
    else:
        raise ValueError("cmap_axis must be 'Y' or 'Z'")

    # # Create the polygon collection
    # col = PolyCollection(waterfall, facecolors=colors, alpha=alpha)
    # ax.add_collection3d(col, zs=np.arange(len(data)), zdir="y")
    
    # Plot each FFT as a 3D line
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        # Normalizar o índice para o mapa de cores (opcional, para diferenciar linhas)
        color = plt.get_cmap(cmap)(i / float(len(data) - 1))
        ax.plot(x, np.full_like(x, i), z, 'b-', color=color, alpha=0.75)  # Linha 3D 

    # Set axis limits
    ax.set_xlim(min(map(min, x_values)), max(map(max, x_values)))
    ax.set_ylim(0, len(data))
    if z_lim:
        ax.set_zlim(z_lim[0], z_lim[1])
    else:
        ax.set_zlim(min(map(min, z_values)), max(map(max, z_values)))

    # Label the axes
    ax.set_xlabel(x_label, fontsize=15, labelpad=8)
    ax.set_ylabel(y_label, fontsize=15, labelpad=8)
    ax.set_zlabel(z_label, fontsize=15, labelpad=8)

    # Add custom Y-axis labels
    y_ticks = np.arange(0, len(data), 3)
    ax.set_yticks(y_ticks)

    # Set the plot title
    plt.title(title)
    plt.tight_layout()
    plt.show()
