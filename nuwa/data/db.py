import json
import os
import shutil
from typing import List

import cv2
import numpy as np
import tqdm
from PIL import Image

from nuwa.data.frame import Frame


class NuwaDB:
    source: str = ""
    frames: List[Frame] = []

    _colmap_dir = ""

    def get_up(self):
        up = np.zeros(3)
        for f in self.frames:
            up += -f.pose[:3, 1]
        up = up / np.linalg.norm(up)
        return up

    def dump(self, out_json_path, copy_images_to=None, copy_masks_to=None):
        if copy_images_to is not None:
            os.makedirs(copy_images_to, exist_ok=True)
            copy_images_to = os.path.abspath(copy_images_to)

            for f in self.frames:
                new_path = os.path.join(copy_images_to, os.path.basename(f.image_path))
                shutil.copy2(f.image_path, new_path)
                f.image_path = new_path

        if copy_masks_to is not None:
            os.makedirs(copy_masks_to, exist_ok=True)
            copy_masks_to = os.path.abspath(copy_masks_to)

            for f in self.frames:
                new_path = os.path.join(copy_masks_to, os.path.basename(f.mask_path))
                shutil.copy2(f.mask_path, new_path)
                f.mask_path = new_path

        frames = [f.to_dict() for f in self.frames]
        for f in frames:
            f["file_path"] = os.path.relpath(
                f["file_path"], start=os.path.dirname(out_json_path))
            f["org_path"] = os.path.relpath(
                f["org_path"], start=os.path.dirname(out_json_path))
            if f["mask_path"]:
                f["mask_path"] = os.path.relpath(
                    f["mask_path"], start=os.path.dirname(out_json_path))

        with open(out_json_path, "w") as outfile:
            json.dump({
                "source": self.source,
                "up": self.get_up().tolist(),
                "frames": frames
            }, outfile, indent=4)

        with open(out_json_path + ".txt", "w") as outfile:
            outfile.write(f"{os.path.basename(out_json_path)}")

    def calculate_object_mask(
            self,
            save_dir,
            reduce_factor=2,
            shrink=0.02,
            sam_ckpt_path=None,
            adjust_cameras=True,
            copy_org=True
    ):
        from nuwa.utils.seg_utils import segment_img, sam, scene_carving, crop_images, SAMAPI
        from nuwa.utils.dmv_utils import raft_api

        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.abspath(save_dir)

        masks = []
        images = []
        for i, frame in tqdm.tqdm(enumerate(self.frames), desc='masking'):
            img = Image.open(frame.image_path)

            if i == 0:
                _, rembg_mask = segment_img(img)
                _, mask = sam(img, rembg_mask)
            else:
                flow_rgb_img0 = Image.open(self.frames[i - 1].image_path).reduce(reduce_factor)
                flow_rgb_img1 = img.reduce(reduce_factor)
                flow = raft_api.raft_optical_flow_api(flow_rgb_img0, flow_rgb_img1).cpu().numpy()

                ref_mask = Image.fromarray(masks[i - 1].astype(np.uint8)).reduce(reduce_factor)
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
                mask = SAMAPI.segment_api(np.array(img), bbox=bbox, sam_checkpoint=sam_ckpt_path)
                retval, labels, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
                try:
                    maxcomp = np.argmax(stats[1:, 4]) + 1
                    mask = (labels == maxcomp)
                except ValueError:
                    mask = np.ones_like(mask)

            images.append(np.array(img))
            masks.append(mask)

        if copy_org:
            for i, f in enumerate(self.frames):
                new_path = os.path.join(save_dir, f"{i:06d}_org.png")
                shutil.copy2(f.org_path, new_path)
                f.org_path = new_path

        if adjust_cameras:
            masks = np.array(masks)
            ks = np.array([f.camera.intrinsic_matrix for f in self.frames])
            camera_poses = np.array([f.pose for f in self.frames])
            camera_poses = scene_carving(masks, ks, camera_poses)
            images, ks, masks = crop_images(images, masks, ks)

            h, w = images[0].shape[:2]

            # dump images
            for i, img in enumerate(images):
                Image.fromarray(img).save(os.path.join(save_dir, f"{i:06d}.png"))

            # update info
            for i in range(len(self.frames)):
                assert self.frames[i].camera.to_dict()["camera_param_model"] == "PINHOLE"
                self.frames[i].pose = camera_poses[i]
                self.frames[i].image_path = os.path.join(save_dir, f"{i:06d}.png")
                self.frames[i].camera.w = w
                self.frames[i].camera.h = h
                self.frames[i].camera.fx = ks[i][0, 0]
                self.frames[i].camera.fy = ks[i][1, 1]
                self.frames[i].camera.cx = ks[i][0, 2]
                self.frames[i].camera.cy = ks[i][1, 2]

        for i, mask in enumerate(masks):
            mask_path = os.path.join(save_dir, f"{i:06d}_mask.png")
            Image.fromarray(mask).save(mask_path)
            self.frames[i].mask_path = mask_path

        return masks

    def undistort_images(self):
        raise NotImplementedError

    def export_3dgs(self, out_dir):
        """
        Export data to 3DGS format

        <location>
        |---images
        |   |---<image 0>
        |   |---<image 1>
        |   |---...
        |---sparse
            |---0
                |---cameras.bin
                |---images.bin
                |---points3D.bin

        :param out_dir:
        :return: None
        """

        if self._colmap_dir == "":
            raise NotImplementedError("db is not imported from colmap")

        if self.frames[0].mask_path != "":
            raise NotImplementedError("export is not supported after masking")

        if not os.path.exists(self._colmap_dir):
            raise FileNotFoundError(f"colmap dir {self._colmap_dir} not found")

        img_dir = os.path.join(out_dir, "images")
        sparse_dir = os.path.join(out_dir, "sparse")

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)

        # copy images
        for i, f in enumerate(self.frames):
            shutil.copy2(f.image_path, img_dir)

        # copy sparse
        shutil.copytree(self._colmap_dir, os.path.join(sparse_dir, "0"))
