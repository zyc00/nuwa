from PIL import Image, ImageOps, ImageFilter
import numpy as np


def sharpness(image: Image.Image | str):
    if isinstance(image, str):
        image = Image.open(image)

    gray = ImageOps.grayscale(image)
    fm = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var()
    return fm


def center_crop_and_update_intrinsics(image, fxfycxcy, crop_size):
    # Get the original image dimensions
    width, height = image.size

    # Unpack the crop size
    if isinstance(crop_size, int):
        crop_width = crop_size
        crop_height = crop_size
    else:
        crop_width, crop_height = crop_size

    # Calculate the left, top, right, and bottom coordinates for cropping
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Perform center cropping
    cropped_image = image.crop((left, top, right, bottom))

    # Update the intrinsic values
    fx, fy, cx, cy = fxfycxcy
    new_cx = cx - left
    new_cy = cy - top

    # Create the updated intrinsic values array
    updated_fxfycxcy = np.array([fx, fy, new_cx, new_cy])

    return cropped_image, updated_fxfycxcy
