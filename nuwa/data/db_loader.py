import json
import os
from copy import deepcopy

import numpy as np
from nuwa.data.db import NuwaDB
from nuwa.data.camera import OpenCvCamera, PinholeCamera
from nuwa.data.frame import Frame
from nuwa.utils.pose_utils import qvec2rotmat, convert_camera_pose


def from_colmap(
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
            if 0 < max_frames <= len(frames):
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
                    camera=deepcopy(cameras[int(elems[8])]),
                    image_path=image_abs,
                    org_path=image_abs,
                    pose=c2w,
                    seq_id=image_id
                )
                frames.append(frame)

    ret = NuwaDB()
    ret.frames = sorted(frames, key=lambda x: x.image_path)
    ret.source = "colmap"
    return ret


def from_polycam(polycam_dir):
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

    ret = NuwaDB()
    ret.frames = sorted(frames, key=lambda x: x.image_path)
    ret.source = "polycam"
    return ret


def from_image_folder(img_dir):
    raise NotImplementedError
