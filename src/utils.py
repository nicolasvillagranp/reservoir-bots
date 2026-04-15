from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def resize_image(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Resize the input image to the specified size.
    Args:
        image: A PIL Image to be resized.
        size: A tuple (width, height) specifying the target size.
    Returns:
        The resized PIL Image.
    """
    return image.resize(size)


def visualize_depth_map(image: Image.Image, depth_map: list[float]) -> None:
    """
    Visualize the original image and its corresponding depth map side by side.
    Args:
        image: The original PIL Image.
        depth_map: A 2D list representing the depth map corresponding to the image.
    """
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    sns.heatmap(depth_map, ax=axs[1], cbar=False)
    axs[0].set_title("Original Image")
    axs[1].set_title("Depth Map")
    plt.show()
