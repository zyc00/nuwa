from PIL import Image, ImageOps, ImageFilter
import numpy as np


def sharpness(image: Image.Image | str):
    if isinstance(image, str):
        image = Image.open(image)

    gray = ImageOps.grayscale(image)
    fm = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var()
    return fm
