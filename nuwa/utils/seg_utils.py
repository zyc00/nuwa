# Adapted from linghao's code

import os
import os.path as osp
from typing import List

import cv2
from PIL.Image import Resampling

import nuwa
from nuwa.utils import utils_3d, raft_api

try:
    import torch
    from rembg import new_session, remove
    from segment_anything import sam_model_registry, SamPredictor
    from nuwa.utils.cutie_utils import get_default_model_for_cutie
    from cutie.inference.inference_core import InferenceCore
    from torchvision.transforms.functional import to_tensor
except ImportError:
    nuwa.get_logger().error(
        "Please install rembg and segment_anything for processing objects:"
        '`pip install "rembg>=2.0.57" "torch>=2.0.0" "torchvision>=0.16.0" '
        "git+https://github.com/facebookresearch/segment-anything.git`"
    )
    raise

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tqdm


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


def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError("cv2 must be either version 3 or 4 to call this method")

    return contours, hierarchy


def vis_mask(
    img,
    mask,
    color=[255, 255, 255],
    alpha=0.4,
    show_border=True,
    border_alpha=0.5,
    border_thick=1,
    border_color=None,
):
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
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # contours = [c for c in contours if c.shape[0] > 10]
        if border_color is None:
            border_color = color
        if not isinstance(border_color, list):
            border_color = border_color.tolist()
        if border_alpha < 1:
            with_border = img.copy()
            cv2.drawContours(
                with_border, contours, -1, border_color, border_thick, cv2.LINE_AA
            )
            img = (1 - border_alpha) * img + border_alpha * with_border
        else:
            cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


@torch.no_grad()
def generate_grid(n_vox, interval, dtype=torch.float32, device="cuda"):
    # global grid_
    # if grid_ is None:
    grid_range = [
        torch.arange(0, n_vox[axis], interval, device=device) for axis in range(3)
    ]
    grid = torch.stack(
        torch.meshgrid(grid_range[0], grid_range[1], grid_range[2], indexing="ij")
    )
    grid = grid.unsqueeze(0).to(dtype=dtype)  # 1 3 dx dy dz
    grid = grid.view(1, 3, -1)
    grid_ = grid
    # else:
    #     pass
    # print("reuse grid")
    return grid_


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
    occ = np.zeros((64**3), dtype=np.uint8)
    for i in range(len(masks)):
        pts_cam = utils_3d.transform_points(pts, np.linalg.inv(camera_poses[i]))
        pts_img = utils_3d.rect_to_img(Ks[i], pts_cam)
        ys = pts_img[:, 1].astype(int).clip(0, masks.shape[1] - 1)
        xs = pts_img[:, 0].astype(int).clip(0, masks.shape[2] - 1)

        valid = np.logical_and.reduce(
            [
                pts_img[:, 0] >= 0,
                pts_img[:, 0] < masks.shape[2],
                pts_img[:, 1] >= 0,
                pts_img[:, 1] < masks.shape[1],
                pts_cam[:, 2] > 0,
                masks[i][ys, xs],
            ]
        )
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

    return camera_poses, center, scale


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
                os.system(
                    f"wget "
                    f"-O {default_ckpt_path} "
                    f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "
                )
            device = "cuda"
            model_type = "default"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            predictor = SamPredictor(sam)
            SAMAPI.predictor = predictor
        return SAMAPI.predictor

    @staticmethod
    def segment_api(
        rgb,
        mask=None,
        bbox=None,
        point_coords=None,
        point_labels=None,
        sam_checkpoint=None,
        dbg=False,
    ):
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
                y1, y2, x1, x2 = (
                    np.nonzero(mask)[0].min(),
                    np.nonzero(mask)[0].max(),
                    np.nonzero(mask)[1].min(),
                    np.nonzero(mask)[1].max(),
                )
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
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="r",
                        linewidth=3,
                    )
                )
            if point_coords is not None:
                for i in range(len(point_coords)):
                    plt.scatter(
                        point_coords[i, 0],
                        point_coords[i, 1],
                        c="g" if point_labels[i] == 1 else "r",
                        s=2,
                    )
            plt.subplot(1, 2, 2)
            plt.title("sam output")
            plt.imshow(vis_mask(rgb, mask.astype(np.uint8), [0, 255, 0]))
            plt.show()
        return mask


@torch.inference_mode()
def segment_fg(
    images: List[Image.Image],
    use_flow=True,
    use_tracking=True,
    sam_ckpt_path=None,
    reduce_factor=2,
    shrink=0.02,
):
    new_images = []
    masks = []

    nuwa.get_logger().info(
        f"nseg - starting to generate masks for {len(images)} frames, {use_flow=}"
    )

    if use_tracking:
        cutie = get_default_model_for_cutie()
        processor = InferenceCore(cutie, cfg=cutie.cfg)
        processor.max_internal_size = 768

    for i in tqdm.trange(len(images), desc="masking"):
        img = images[i]

        if ((not use_flow) and (not use_tracking)) or i == 0:
            _, rembg_mask = segment_img(img)
            _, mask = sam(img, rembg_mask)

            if use_tracking:
                img_tensor = to_tensor(img).cuda().float()
                mask_tensor = to_tensor(mask).cuda().float()
                objects = np.unique(mask)
                objects = objects[objects != 0].tolist()
                output_prob = processor.step(img_tensor, mask_tensor[0], objects)
                mask = processor.output_prob_to_mask(output_prob)
                mask = mask.cpu().numpy().astype(bool)

        elif not use_tracking:
            flow_rgb_img0 = images[i - 1].reduce(reduce_factor)
            flow_rgb_img1 = img.reduce(reduce_factor)
            flow = (
                raft_api.raft_optical_flow_api(flow_rgb_img0, flow_rgb_img1)
                .cpu()
                .numpy()
            )

            ref_mask = Image.fromarray(masks[i - 1].astype(np.uint8)).reduce(
                reduce_factor
            )
            ref_mask = np.array(ref_mask) > 0

            ys, xs = ref_mask.nonzero()
            fg_pixels = np.concatenate([xs[:, None], ys[:, None]], axis=1)
            fg_pixels = fg_pixels.astype(np.float32)
            fg_pixels = fg_pixels + flow[ys, xs]
            xmin, xmax = fg_pixels[:, 0].min(), fg_pixels[:, 0].max()
            ymin, ymax = fg_pixels[:, 1].min(), fg_pixels[:, 1].max()
            w, h = xmax - xmin, ymax - ymin
            xmin = int(xmin - w * shrink)
            xmax = int(xmax + w * shrink)
            ymin = int(ymin - h * shrink)
            ymax = int(ymax + h * shrink)
            bbox = xmin, ymin, xmax, ymax
            bbox = [int(x * reduce_factor) for x in bbox]
            mask = SAMAPI.segment_api(
                np.array(img), bbox=bbox, sam_checkpoint=sam_ckpt_path
            )
            retval, labels, stats, cent = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8)
            )
            try:
                maxcomp = np.argmax(stats[1:, 4]) + 1
                mask = labels == maxcomp
            except ValueError:
                nuwa.get_logger().warning(
                    f"nseg - fail to process image {i}, no object found..."
                )
                _, rembg_mask = segment_img(img)
                _, mask = sam(img, rembg_mask)
        else:
            img_tensor = to_tensor(img).cuda().float()
            output_prob = processor.step(img_tensor)
            mask = processor.output_prob_to_mask(output_prob)
            mask = mask.cpu().numpy().astype(bool)

        new_images.append(np.array(img))
        masks.append(mask)

    return new_images, masks
