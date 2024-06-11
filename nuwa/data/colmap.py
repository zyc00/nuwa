import os
import numpy as np
import plyfile

from nuwa.utils.colmap_utils import colmap_convert_model


class Reconstruction:
    def __init__(self, folder_path):
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

        return images

    def dump(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        self._dump_cameras(output_folder)
        self._dump_points(output_folder)
        self._dump_images(output_folder)

        colmap_convert_model(output_folder, out_type="BIN")

    def export_point_cloud(self, path, write_text=False):
        if path.endswith('.ply'):
            points = np.array([point['xyz'] for point in self.points.values()])
            colors = np.array([point['rgb'] for point in self.points.values()])
            points = np.hstack([points, colors / 255.0])

            vertices = []
            for i, point in enumerate(points):
                vertices.append((point[0], point[1], point[2], point[3], point[4], point[5]))

            vertices = np.array(vertices, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            ply = plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')], text=write_text)
            ply.write(path)

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
            image['tvec'] += translation

        for point in self.points.values():
            point['xyz'] += translation

    def world_scale(self, scale):
        for image in self.images.values():
            image['tvec'] *= scale

        for point in self.points.values():
            point['xyz'] *= scale

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

#
# if __name__ == '__main__':
#     import IPython
#     IPython.embed()
