import io
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .cv2_util import findContours


def image_grid(
        images,
        rows=None,
        cols=None,
        fill: bool = True,
        show_axes: bool = False,
        rgb=None,
        show=True,
        label=None,
        begin_i=0,
        **kwargs
):
    """
    A util function for plotting a grid of images.
    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.
    Returns:
        None
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if len(images[0].shape) == 2:
        rgb = False
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = int(len(images) ** 0.5)
        cols = math.ceil(len(images) / rows)

    if len(images) < 50:
        figsize = (10, 10)
    else:
        figsize = (15, 15)
    plt.figure(figsize=figsize)
    if label:
        plt.suptitle(label, fontsize=30)

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        if rgb:
            # only render RGB channels
            plt.imshow(images[i][..., :3], **kwargs)
        else:
            plt.imshow(images[i], **kwargs)
        if not show_axes:
            plt.axis('off')
        plt.title(f'{i + begin_i}', fontsize=30)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        ret = np.array(Image.open(img_buf))
        plt.close("all")
        return ret


def vis_mask(img,
             mask,
             color=[255, 255, 255],
             alpha=0.4,
             show_border=True,
             border_alpha=0.5,
             border_thick=1,
             border_color=None):
    """
    Visualizes a single binary mask.
    :param img: H,W,3, uint8, np.ndarray
    :param mask: H,W, bool or uint8(0,1). np.ndarray
    :param color:
    :param alpha:
    :param show_border:
    :param border_alpha:
    :param border_thick:
    :param border_color:
    :return:
    """
    img = img.astype(np.float32)
    mask = (mask > 0).astype(np.uint8)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += [alpha * x for x in color]

    if show_border:
        contours, _ = findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours = [c for c in contours if c.shape[0] > 10]
        if border_color is None:
            border_color = color
        if not isinstance(border_color, list):
            border_color = border_color.tolist()
        if border_alpha < 1:
            with_border = img.copy()
            cv2.drawContours(with_border, contours, -1, border_color,
                             border_thick, cv2.LINE_AA)
            img = (1 - border_alpha) * img + border_alpha * with_border
        else:
            cv2.drawContours(img, contours, -1, border_color, border_thick,
                             cv2.LINE_AA)

    return img.astype(np.uint8)
