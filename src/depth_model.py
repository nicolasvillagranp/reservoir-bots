from transformers import pipeline
from PIL import Image


def get_model(model_name: str = "Intel/zoedepth-nyu-kitti") -> pipeline:
    """
    Load and return a depth estimation model from Hugging Face.
    Args:
        model_name: The name of the model to load (default is "Intel/zoedepth-nyu-kitti").
    Returns:
        A Hugging Face pipeline for depth estimation.
    """
    return pipeline(task="depth-estimation", model=model_name)


def estimate_depth(model: pipeline, image: Image.Image) -> list[float]:
    """
    Estimate the depth map for a given image using the provided model.
    Args:
        model: A Hugging Face pipeline for depth estimation.
        image: A PIL Image for which to estimate the depth map.
    Returns:
        A 2D list representing the estimated depth map corresponding to the input image.
    """
    outputs = model(image)
    return outputs["depth"]


if __name__ == "__main__":
    # Example usage
    from src.utils import resize_image, visualize_depth_map

    model = get_model()
    image_path = "data/images/train/coco2017_000000000632.jpg"
    image = Image.open(image_path)
    image = resize_image(image)
    depth_map = estimate_depth(model, image)
    visualize_depth_map(image, depth_map)
