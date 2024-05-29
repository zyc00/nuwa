import json
import os
import shutil
from typing import List
import abc

import math

import cv2
import numpy as np
import tqdm
from PIL import Image

from nuwa.utils.image_utils import sharpness
from nuwa.utils.pose_utils import convert_camera_pose, qvec2rotmat


class _Camera:
    w: float
    h: float
    fx: float
    fy: float
    cx: float
    cy: float
    is_fisheye: bool = False

    k1: float
    k2: float
    p1: float
    p2: float

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @property
    def intrinsic_matrix(self):
        return [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]


class OpenCvCamera(_Camera):
    def __init__(self, w, h, fx, fy, cx, cy, k1=0., k2=0., p1=0., p2=0.):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def to_dict(self):
        return {
            "w": self.w,
            "h": self.h,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,

            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,

            "fl_x": self.fx,
            "fl_y": self.fy,
            "camera_angle_x": math.atan(self.w / (self.fx * 2)) * 2,
            "camera_angle_y": math.atan(self.h / (self.fy * 2)) * 2,
            "fovx": math.atan(self.w / (self.fx * 2)) * 2 * 180 / math.pi,
            "foxy": math.atan(self.h / (self.fy * 2)) * 2 * 180 / math.pi,

            "intrinsic_matrix": self.intrinsic_matrix,

            "camera_param_model": "OPENCV"
        }


class PinholeCamera(_Camera):
    def __init__(self, w, h, fx, fy, cx, cy):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def to_dict(self):
        return {
            "w": self.w,
            "h": self.h,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "is_fisheye": self.is_fisheye,

            "fl_x": self.fx,
            "fl_y": self.fy,
            "camera_angle_x": math.atan(self.w / (self.fx * 2)) * 2,
            "camera_angle_y": math.atan(self.h / (self.fy * 2)) * 2,
            "fovx": math.atan(self.w / (self.fx * 2)) * 2 * 180 / math.pi,
            "foxy": math.atan(self.h / (self.fy * 2)) * 2 * 180 / math.pi,

            "intrinsic_matrix": self.intrinsic_matrix,

            "camera_param_model": "PINHOLE"
        }


class Frame:
    camera: _Camera
    image_path: str  # abs path
    pose: np.ndarray  # OpenCV Convention
    sharpness: float
    seq_id: int = -1
    mask_path: str = ""  # abs path
    org_path: str = ""  # abs path

    def __init__(
            self,
            camera: _Camera,
            image_path: str,
            pose: np.ndarray,
            seq_id: int = -1,
            sharpness_score: float | None = None,
            mask_path: str = "",
            org_path: str = ""
    ):
        self.camera = camera
        self.image_path = image_path
        self.pose = pose
        self.seq_id = seq_id

        if sharpness_score is None:
            sharpness_score = sharpness(image_path)

        self.sharpness = sharpness_score
        self.mask_path = mask_path
        self.org_path = org_path

    def to_dict(self):
        ret = {
            "file_path": self.image_path,
            "mask_path": self.mask_path,
            "org_path": self.org_path,

            "c2w": self.pose.tolist(),
            "w2c": np.linalg.inv(self.pose).tolist(),
            "transform_matrix": convert_camera_pose(self.pose, "cv", "blender").tolist(),

            "sharpness": float(self.sharpness),

            "seq_id": self.seq_id,
            "camera_matrices_hints": {
                "c2w": "OPENCV",
                "w2c": "OPENCV_inv",
                "transform_matrix": "BLENDER"
            }
        }
        ret.update(self.camera.to_dict())
        return ret


class NuwaDB:
    source: str = ""
    frames: List[Frame] = []

    @classmethod
    def from_polycam(cls, polycam_dir):
        seq_dir = os.path.join(polycam_dir, "keyframes")
        assert os.path.exists(seq_dir), f"Directory {seq_dir} does not exist"

        uids = [a[:-4] for a in sorted(os.listdir(os.path.join(seq_dir, "corrected_images")))]
        nimgs = len(uids)

        frames = []

        for i in range(nimgs):
            camera_json = json.load(open(os.path.join(seq_dir, f"corrected_cameras/{uids[i]}.json")))
            camera = PinholeCamera(
                w=camera_json['width'],
                h=camera_json['height'],
                fx=camera_json['fx'],
                fy=camera_json['fy'],
                cx=camera_json['cx'],
                cy=camera_json['cy']
            )

            pose = np.array([
                [camera_json['t_00'], camera_json['t_01'], camera_json['t_02'], camera_json['t_03']],
                [camera_json['t_10'], camera_json['t_11'], camera_json['t_12'], camera_json['t_13']],
                [camera_json['t_20'], camera_json['t_21'], camera_json['t_22'], camera_json['t_23']],
                [0, 0, 0, 1]
            ])
            pose = convert_camera_pose(pose, "blender", "cv")
            image_path = os.path.abspath(os.path.join(seq_dir, f"corrected_images/{uids[i]}.jpg"))

            frame = Frame(
                camera=camera,
                image_path=image_path,
                org_path=image_path,
                pose=pose,
                seq_id=i,
                sharpness_score=camera_json['blur_score']
            )

            frames.append(frame)

        ret = cls()
        ret.frames = sorted(frames, key=lambda x: x.image_path)
        ret.source = "polycam"
        return ret

    @classmethod
    def from_colmap(
            cls,
            img_dir,
            colmap_dir,
            skip_early=0,
            max_frames=-1
    ):
        cameras = {}
        with open(os.path.join(colmap_dir, "cameras.txt"), "r") as f:
            for line in f:
                # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
                # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
                # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
                if line[0] == "#":
                    continue
                els = line.split(" ")

                camera_id = int(els[0])
                camera_model = els[1]

                if camera_model == "OPENCV":
                    camera = OpenCvCamera(
                        w=float(els[2]),
                        h=float(els[3]),
                        fx=float(els[4]),
                        fy=float(els[5]),
                        cx=float(els[6]),
                        cy=float(els[7]),
                        k1=float(els[8]),
                        k2=float(els[9]),
                        p1=float(els[10]),
                        p2=float(els[11])
                    )
                elif camera_model == "PINHOLE":
                    camera = PinholeCamera(
                        w=float(els[2]),
                        h=float(els[3]),
                        fx=float(els[4]),
                        fy=float(els[5]),
                        cx=float(els[6]),
                        cy=float(els[7])
                    )
                else:
                    raise ValueError(f"Unknown camera model: {camera_model}")

                cameras[camera_id] = camera

        if len(cameras) == 0:
            raise ValueError("No cameras found in cameras.txt")

        with open(os.path.join(colmap_dir, "images.txt"), "r") as f:
            i = 0
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])

            frames = []
            frames_abs = []

            # up = np.zeros(3)
            for line in f:
                if max_frames > 0 and len(frames) >= max_frames:
                    break

                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1
                if i < skip_early * 2:
                    continue
                if i % 2 == 1:
                    # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                    elems = line.split(" ")
                    image_abs = os.path.abspath(os.path.join(img_dir, '_'.join(elems[9:])))
                    frames_abs.append(image_abs)
                    image_id = int(elems[0])
                    q_vec = np.array(tuple(map(float, elems[1:5])))
                    t_vec = np.array(tuple(map(float, elems[5:8])))
                    r = qvec2rotmat(-q_vec)
                    t = t_vec.reshape([3, 1])
                    w2c = np.concatenate([np.concatenate([r, t], 1), bottom], 0)
                    c2w = np.linalg.inv(w2c)

                    frame = Frame(
                        camera=cameras[int(elems[8])],
                        image_path=image_abs,
                        org_path=image_abs,
                        pose=c2w,
                        seq_id=image_id
                    )
                    frames.append(frame)

        ret = cls()
        ret.frames = sorted(frames, key=lambda x: x.image_path)
        ret.source = "colmap"
        return ret

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
            json.dump({"frames": frames}, outfile, indent=4)

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
                maxcomp = np.argmax(stats[1:, 4]) + 1
                mask = (labels == maxcomp)

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
