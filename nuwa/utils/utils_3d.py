from typing import Iterable

import numpy as np


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    else:
        import torch
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=1)
    return pts_hom


def hom_to_cart(pts):
    """
    :param pts: (N, 4 or 3)
    :return pts_hom: (N, 3 or 2)
    """
    return pts[:, :-1] / pts[:, -1:]


def canonical_to_camera(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :return:
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


transform_points = canonical_to_camera


def rotx_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([ones, zeros, zeros,
                    zeros, c, -s,
                    zeros, s, c])
    return rot.reshape((-1, 3, 3))


def roty_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, zeros, s,
                    zeros, ones, zeros,
                    -s, zeros, c])
    return rot.reshape((-1, 3, 3))


def rotz_np(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = np.array([a])
    a = a.astype(float).reshape((-1, 1))
    ones = np.ones_like(a)
    zeros = np.zeros_like(a)
    c = np.cos(a)
    s = np.sin(a)
    rot = np.stack([c, -s, zeros,
                    s, c, zeros,
                    zeros, zeros, ones])
    return rot.reshape((-1, 3, 3))


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def rect_to_img(K, pts_rect):
    pts_2d_hom = pts_rect @ K.T
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


def read_ply(filename):
    import plyfile
    plydata = plyfile.PlyData.read(filename)
    points = np.stack([plydata['vertex'][n] for n in ['x', 'y', 'z']], axis=1)

    if 'red' in plydata['vertex']:
        colors = np.stack([plydata['vertex'][n] for n in ['red', 'green', 'blue']], axis=1)
    else:
        colors = None

    return points, colors


def save_ply(data, path, write_text=False):
    import plyfile
    if isinstance(data, (list, tuple)):
        data = list(filter(lambda x: x is not None, data))
        data = np.concatenate(data, axis=1)

    if len(data[0]) == 3:
        write_color = False
    elif len(data[0]) == 6:
        write_color = True
    else:
        raise ValueError(f"Data shape {data.shape} not supported")

    vertices = []

    if write_color:
        for p in data:
            vertices.append((p[0], p[1], p[2], p[3], p[4], p[5]))
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        for p in data:
            vertices.append((p[0], p[1], p[2]))
        vertices = np.array(vertices, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertices, 'vertex')], text=write_text)
    ply.write(path)


def farthest_point_sampling(xyz, npoint):
    """
    :param xyz: (N, 3)
    :param npoint: int
    :return: (npoint)
    """
    N = xyz.shape[0]
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    return centroids


def voxelized_sampling(xyz, grid_size=0.1, voxelize_n=3):
    min_bounds = np.min(xyz, axis=0)
    max_bounds = np.max(xyz, axis=0)

    voxel_dims = np.ceil((max_bounds - min_bounds) / grid_size).astype(int)

    voxel_grid = np.zeros(voxel_dims, dtype=np.uint16)
    voxel_grid_center = np.zeros((*voxel_dims, 3), dtype=np.half)
    voxel_indices = ((xyz - min_bounds) / grid_size).astype(int)

    for i, idx in enumerate(voxel_indices):
        voxel_grid[tuple(idx)] += 1
        voxel_grid_center[tuple(idx)] += xyz[i]

    voxel_grid_center[voxel_grid > 0] /= voxel_grid[voxel_grid > 0][..., None]

    return voxel_grid_center[voxel_grid >= voxelize_n]
