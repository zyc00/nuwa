# Adapted from linghao's code

import os
import os.path as osp

from PIL.Image import Resampling

from rembg import new_session, remove
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Please install segment-anything: "
          "pip install git+https://github.com/facebookresearch/segment-anything.git")
    raise

from .dmv_utils import plt_utils, utils_3d
from .dmv_utils.backproject import generate_grid

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class RembgHelper:
    session = None

    @staticmethod
    def getinstance():
        if RembgHelper.session is None:
            # providers = cfg.single_img_infer.rembg_backends
            # if len(providers) == 0:
            #     providers = None
            session = new_session()
            RembgHelper.session = session
        return RembgHelper.session

    @staticmethod
    def remove(img):
        session = RembgHelper.getinstance()
        return remove(img, session=session)


def scene_carving(masks, Ks, camera_poses):
    """
    Assume normalized camera poses

    :param masks: N,H,W
    :param Ks: N,3,3
    :param camera_poses: N,4,4, opencv format
    :return:
    """
    coords = generate_grid([64, 64, 64], 1)[0].T.cpu().numpy()
    origin = np.array([-1, -1, -1])
    voxel_size = 2 / 64
    pts = coords * voxel_size + origin
    occ = np.zeros((64 ** 3), dtype=np.uint8)
    for i in range(len(masks)):
        pts_cam = utils_3d.transform_points(pts, np.linalg.inv(camera_poses[i]))
        pts_img = utils_3d.rect_to_img(Ks[i], pts_cam)
        ys = pts_img[:, 1].astype(int).clip(0, masks.shape[1] - 1)
        xs = pts_img[:, 0].astype(int).clip(0, masks.shape[2] - 1)

        valid = np.logical_and.reduce([pts_img[:, 0] >= 0, pts_img[:, 0] < masks.shape[2],
                                       pts_img[:, 1] >= 0, pts_img[:, 1] < masks.shape[1],
                                       pts_cam[:, 2] > 0, masks[i][ys, xs]])
        occ += valid
        # wis3d.add_point_cloud(pts[valid])
        # print()
    valid = occ > masks.shape[0] // 2
    pts_valid = pts[valid]
    center = pts_valid.mean(0)
    pts_valid = pts_valid - center
    camera_poses[:, :3, 3] -= center
    # scale = 1.0 / np.linalg.norm(pts_valid, axis=-1).max() # normalized by radius
    scale = 0.9 / np.abs(pts_valid).max()  # normalized by bbox
    pts_valid *= scale
    # print(pts_valid.max(0), pts_valid.min(0))
    camera_poses[:, :3, 3] *= scale

    return camera_poses


def crop_images(images, masks, Ks, out_size=512, pad_margin=0.1):
    new_images, new_Ks, new_masks = [], [], []
    for i in range(len(masks)):
        mask = masks[i]
        ys, xs = mask.nonzero()
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        image = images[i][ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
        K = Ks[i].copy()
        K[0, 2] -= xmin
        K[1, 2] -= ymin
        # pad to have same width and height
        h, w = image.shape[:2]
        if h > w:
            pad = (h - w) // 2
            image = np.pad(image, ((0, 0), (pad, pad), (0, 0)))
            mask = np.pad(mask, ((0, 0), (pad, pad)))
            K[0, 2] += pad
        elif w > h:
            pad = (w - h) // 2
            image = np.pad(image, ((pad, pad), (0, 0), (0, 0)))
            mask = np.pad(mask, ((pad, pad), (0, 0)))
            K[1, 2] += pad

        h, w = image.shape[:2]
        segmented_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        segmented_img.paste(Image.fromarray(image), mask=Image.fromarray(mask))
        mask = Image.fromarray(mask)

        pad = int(pad_margin * segmented_img.size[0])
        img2 = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        img2.paste(segmented_img, (pad, pad))

        mask2 = Image.new(mask.mode, (w + 2 * pad, h + 2 * pad), (0,))
        mask2.paste(mask, (pad, pad))

        K[0, 2] += pad
        K[1, 2] += pad

        w, h = img2.size
        img2 = img2.resize((out_size, out_size), Resampling.LANCZOS)
        mask2 = mask2.resize((out_size, out_size), Resampling.NEAREST)

        K[0] *= out_size / w
        K[1] *= out_size / h

        new_images.append(np.array(img2))
        new_Ks.append(K)
        new_masks.append(np.array(mask2))

    new_images = np.stack(new_images)
    new_Ks = np.stack(new_Ks)
    new_masks = np.stack(new_masks)

    return new_images, new_Ks, new_masks


def segment_img(img: Image):
    output = RembgHelper.remove(img)
    mask = np.array(output)[:, :, 3] > 0
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(mask))
    return segmented_img, mask


def sam(img, mask):
    sam_mask = SAMAPI.segment_api(np.array(img)[:, :, :3], mask)
    segmented_img = Image.new("RGBA", img.size, (0, 0, 0, 0))
    segmented_img.paste(img, mask=Image.fromarray(sam_mask))
    return segmented_img, sam_mask


class SAMAPI:
    predictor = None

    @staticmethod
    def get_instance(sam_checkpoint=None):
        default_ckpt_path = osp.expanduser("~/.cache/sam_vit_h_4b8939.pth")

        if SAMAPI.predictor is None:
            if sam_checkpoint is None:
                sam_checkpoint = default_ckpt_path
            if not osp.exists(sam_checkpoint):
                os.system(f'wget '
                          f'-O {default_ckpt_path} '
                          f'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth ')
            device = "cuda"
            model_type = "default"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(rgb, mask=None, bbox=None,
                    point_coords=None, point_labels=None,
                    sam_checkpoint=None,
                    dbg=False):
        """

        Parameters
        ----------
        rgb : np.ndarray h,w,3 uint8
        mask: np.ndarray h,w bool
        dbg

        Returns
        -------

        """
        predictor = SAMAPI.get_instance(sam_checkpoint)
        predictor.set_image(rgb)
        if mask is None and bbox is None:
            box_input = None
        else:
            # mask to bbox
            if bbox is None:
                y1, y2, x1, x2 = np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max(), np.nonzero(mask)[1].min(), \
                                 np.nonzero(mask)[1].max()
            else:
                x1, y1, x2, y2 = bbox
            box_input = np.array([[x1, y1, x2, y2]])
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_input,
            # mask_input=None,
            multimask_output=True,
            return_logits=False,
        )
        mask = masks[-1]
        if dbg:
            plt.subplot(1, 2, 1)
            plt.imshow(rgb)
            if box_input is not None:
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=3))
            if point_coords is not None:
                for i in range(len(point_coords)):
                    plt.scatter(point_coords[i, 0], point_coords[i, 1], c='g' if point_labels[i] == 1 else 'r', s=2)
            plt.subplot(1, 2, 2)
            plt.title("sam output")
            plt.imshow(plt_utils.vis_mask(rgb, mask.astype(np.uint8), [0, 255, 0]))
            plt.show()
        return mask
