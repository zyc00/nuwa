import os
from typing import Dict, List

import numpy as np
from copy import deepcopy as copy

import nuwa
from nuwa.data.frame import Frame
from nuwa.utils.colmap_utils import colmap_convert_model
from nuwa.utils.pose_utils import pose2qt, qt2pose, rt2pose
from nuwa.utils.utils_3d import save_ply


class Reconstruction:
    cameras: Dict[int, Dict[str, str | int | np.ndarray]]
    points: Dict[int, Dict[str, float | np.ndarray]]
    images: Dict[int, Dict[str, str | int | np.ndarray]]
    image_dir: str | None

    def __init__(self, folder_path: str, image_dir: str | None = None):
        camera_path = os.path.join(folder_path, "cameras.txt")

        if not os.path.exists(camera_path):
            camera_bin_path = camera_path.replace("txt", "bin")
            if os.path.exists(camera_bin_path):
                colmap_convert_model(os.path.dirname(camera_path))
            else:
                raise ValueError(f"Camera file {camera_path} does not exist")

        self.cameras = self._load_cameras(folder_path)
        self.points = self._load_points(folder_path)
        self.images = self._load_images(folder_path)
        self.image_dir = image_dir

        nuwa.get_logger().info(f"nrecon - "
                               f"{len(self.cameras)} cameras, "
                               f"{len(self.points)} points, "
                               f"{len(self.images)} images "
                               f"loaded to colmap reconstruction.")

    @classmethod
    def from_colmap(cls, folder_path: str, image_dir: str | None = None):
        return cls(folder_path, image_dir)

    @classmethod
    def from_data(
            cls,
            cameras: Dict[int, Dict[str, str | int | np.ndarray]],
            points: Dict[int, Dict[str, float | np.ndarray]],
            images: Dict[int, Dict[str, str | int | np.ndarray]],
            image_dir: str | None = None
    ):
        obj = cls.__new__(cls)
        obj.cameras = cameras
        obj.points = points
        obj.images = images
        obj.image_dir = image_dir

        nuwa.get_logger().info(f"nrecon - {len(cameras)} cameras, {len(points)} points, {len(images)} images "
                               f"loaded to colmap reconstruction.")

        return obj

    @classmethod
    def from_frames(cls, frames: List[Frame]):
        nuwa_cameras = [f.camera for f in frames]

        cameras = []
        f2c = {}
        for i, camera in enumerate(nuwa_cameras):
            if camera in cameras:
                f2c[i] = cameras.index(camera)
            else:
                cameras.append(camera)
                f2c[i] = len(cameras) - 1

        points = {}

        images = {}
        image_dir = None

        for i, frame in enumerate(frames):
            if image_dir is None:
                image_dir = os.path.dirname(frame.image_path)
            else:
                assert image_dir == os.path.dirname(frame.image_path), "All images should be in the same directory"

            q, t = pose2qt(frame.pose)

            images[i] = {
                "camera_id": f2c[i],
                "name": os.path.basename(frame.image_path),
                "xys": np.zeros((0, 2), dtype=float),
                "point_ids": np.array([], dtype=int),
                "tvec": t,
                "qvec": q
            }

        image_dir = os.path.abspath(image_dir)

        return cls.from_data({i: {"model": c.type, "width": c.w, "height": c.h, "params": c.params}
                              for i, c in enumerate(cameras)}, points, images, image_dir)

    def update_poses_from_frames(self, frames: List[Frame]):
        for i, frame in enumerate(frames):
            assert os.path.basename(frame.org_path) == self.images[i]["name"]
            q, t = pose2qt(frame.pose)
            self.images[i]["tvec"] = t
            self.images[i]["qvec"] = q

    def __repr__(self):
        return {
            "cameras": f"[...{len(self.cameras)}...]",
            "points": f"[...{len(self.points)}...]",
            "images": f"[...{len(self.images)}...]"
        }.__repr__()

    @staticmethod
    def _load_cameras(folder_path):
        cameras_path = os.path.join(folder_path, 'cameras.txt')
        cameras = {}
        with open(cameras_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = np.array([float(p) for p in parts[4:]])
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
        if len(cameras) == 0:
            raise ValueError('No cameras found in the file')

        return cameras

    @staticmethod
    def _load_points(folder_path):
        points_path = os.path.join(folder_path, 'points3D.txt')
        points = {}
        with open(points_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                point_id = int(parts[0])
                xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])])
                error = float(parts[7])
                track = np.array([int(p) for p in parts[8:]])
                points[point_id] = {
                    'xyz': xyz,
                    'rgb': rgb,
                    'error': error,
                    'track': track
                }
        if len(points) == 0:
            nuwa.get_logger().warning(f"No points found in {folder_path}")

        return points

    @staticmethod
    def _load_images(folder_path):
        images_path = os.path.join(folder_path, 'images.txt')
        images = {}
        with open(images_path, 'r') as file:
            lines = file.read().strip().splitlines()

        idx = 0
        while idx < len(lines):
            if lines[idx].startswith('#'):
                idx += 1
                continue

            parts = lines[idx].strip().split() + lines[idx + 1].strip().split()
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            camera_id = int(parts[8])
            name = parts[9]
            xys = []
            point_ids = []

            for i in range(10, len(parts), 3):
                xys.append([float(parts[i]), float(parts[i+1])])
                point_ids.append(int(parts[i+2]))

            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': name,
                'xys': np.array(xys),
                'point_ids': np.array(point_ids)
            }

            idx += 2

        if len(images) == 0:
            raise ValueError('No images found in the file')

        return images

    def dump(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._dump_cameras(output_folder)
        self._dump_points(output_folder)
        self._dump_images(output_folder)

        if os.path.exists(os.path.join(output_folder, 'cameras.bin')):
            try:
                os.remove(os.path.join(output_folder, 'cameras.bin'))
                os.remove(os.path.join(output_folder, 'images.bin'))
                os.remove(os.path.join(output_folder, 'points3D.bin'))
            except FileNotFoundError:
                pass

        colmap_convert_model(output_folder, out_type="BIN")
        self.export_point_cloud(os.path.join(output_folder, 'vis.ply'))

    def export_point_cloud(self, path, write_text=False):
        if path.endswith('.ply'):
            points = np.array([point['xyz'] for point in self.points.values()])
            colors = np.array([point['rgb'] for point in self.points.values()])
            points = np.hstack([points, colors])

            if len(points) > 0:
                save_ply(points, path, write_text=write_text)

        elif path.endswith('.xyz'):
            with open(path, 'w') as file:
                for point in self.points.values():
                    xyz = point['xyz']
                    rgb = point['rgb']
                    file.write(f'{xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n')

        else:
            raise ValueError('Unsupported file format')

    def world_translate(self, translation):
        for image in self.images.values():
            pose = qt2pose(q=image['qvec'], t=image['tvec'])
            pose[:3, 3] += translation
            _, image['tvec'] = pose2qt(pose)

        for point in self.points.values():
            point['xyz'] += translation

    def world_scale(self, scale):
        for image in self.images.values():
            image['tvec'] *= scale

        for point in self.points.values():
            point['xyz'] *= scale

    def world_transform(self, R, t):
        for image in self.images.values():
            pose = qt2pose(q=image['qvec'], t=image['tvec'])
            pose = rt2pose(R, t) @ pose
            image['qvec'], image['tvec'] = pose2qt(pose)

        for point in self.points.values():
            point['xyz'] = point['xyz'] @ R.T + t

    def _dump_cameras(self, output_folder):
        cameras_path = os.path.join(output_folder, 'cameras.txt')
        with open(cameras_path, 'w') as file:
            file.write('# Camera list with one line of data per camera:\n')
            file.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            for camera_id, camera in self.cameras.items():
                params_str = ' '.join(map(str, camera['params']))
                file.write(f'{camera_id} {camera["model"]} {camera["width"]} {camera["height"]} {params_str}\n')

    def _dump_points(self, output_folder):
        points_path = os.path.join(output_folder, 'points3D.txt')
        with open(points_path, 'w') as file:
            file.write('# 3D point list with one line of data per point:\n')
            file.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n')
            for point_id, point in self.points.items():
                xyz_str = ' '.join(map(str, point['xyz']))
                rgb_str = ' '.join(map(str, point['rgb']))
                track_str = ' '.join(map(str, point['track']))
                file.write(f'{point_id} {xyz_str} {rgb_str} {point["error"]} {track_str}\n')

    def _dump_images(self, output_folder):
        images_path = os.path.join(output_folder, 'images.txt')
        with open(images_path, 'w') as file:
            file.write('# Image list with two lines of data per image:\n')
            file.write('#   IMAGE_ID, QVEC (qw, qx, qy, qz), TVEC (tx, ty, tz), CAMERA_ID, NAME\n')
            file.write('#   POINTS2D[] as (x, y, POINT3D_ID)\n')
            for image_id, image in self.images.items():
                qvec_str = ' '.join(map(str, image['qvec']))
                tvec_str = ' '.join(map(str, image['tvec']))
                file.write(f'{image_id} {qvec_str} {tvec_str} {image["camera_id"]} {image["name"]}\n')
                for xy, point3D_id in zip(image['xys'], image['point_ids']):
                    file.write(f'{xy[0]} {xy[1]} {point3D_id} ')
                file.write('\n')

    def reorder_from_db(self, colmap_database_path):
        from nuwa.utils.colmap_utils import get_name2id_from_colmap_db

        name2id = get_name2id_from_colmap_db(colmap_database_path)
        new_images = {}
        for image in self.images.values():
            new_images[name2id[image['name']]] = copy(image)

        nuwa.get_logger().debug(f"Image order: {new_images}")

        self.images = new_images
