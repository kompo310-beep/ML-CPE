import cv2
import numpy as np

# CLAHE evens out contrast, so images from different machines look alike
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess_image(image, img_size=150):
    """Resize one image to img_size x img_size grayscale. None if unusable."""

    if image is None or image.size == 0:
        return None

    # X-ray images are grayscale; convert if a file was read with 3 channels
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # INTER_AREA is the right filter for shrinking
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return _clahe.apply(image)


def to_features(images):
    """(N, H, W) uint8 -> (N, H, W, 1) uint8, the shape the CNN expects."""

    images = np.ascontiguousarray(images, dtype=np.uint8)
    return images[..., np.newaxis]