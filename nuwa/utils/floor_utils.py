import pyransac3d as pyrsc
import numpy as np

import nuwa
from nuwa.utils.utils_3d import read_ply, voxelized_sampling, save_ply
from nuwa.utils.utils_3d import rotx_np, Rt_to_pose


def find_plane(points, thresh=0.05):
    if isinstance(points, str):
        points, _ = read_ply(points)
    return pyrsc.Plane().fit(points, thresh=thresh)


def find_and_colorize(points, thresh=0.05):
    if isinstance(points, str):
        points, _ = read_ply(points)
    plane, inliers = find_plane(points, thresh=thresh)
    outliers = np.delete(points, inliers, axis=0)
    inliers = points[inliers]

    inliner_colors = np.zeros((inliers.shape[0], 3), dtype=np.uint8)
    inliner_colors[:, 1] = 255
    outlier_colors = np.zeros((outliers.shape[0], 3), dtype=np.uint8)
    outlier_colors[:, 0] = 255

    colors = np.concatenate([inliner_colors, outlier_colors])
    points = np.concatenate([inliers, outliers])

    return plane, points, colors


def find_up(xyz, grid_size=0.1, thresh=0.09):
    """
    Return the world up location of the given point cloud. This function assumes the scene is almost upright (+z).

    :param xyz: [n, 3] array of 3D points
    :param grid_size: size of voxels for sampling
    :param thresh: threshold for plane fitting
    :return: the up direction (normal of the floor), the center of the floor
    """
    points = xyz[xyz[:, 2] < 0.5]  # Filter points below 0.5 in z-axis

    points = voxelized_sampling(points, grid_size=grid_size).astype(float)
    floor, inlier_idx = find_plane(points, thresh=thresh)  # floor: Ax+By+Cy+D=0
    normal = np.array(floor[:3])
    up_direction = normal / np.linalg.norm(normal)
    if up_direction[2] < 0:
        up_direction = -up_direction

    center = np.mean(points[inlier_idx], axis=0)

    nuwa.get_logger().info(f"find_up: {up_direction=}, {center=}")

    return up_direction, center


def get_upright_transformation(xyz, grid_size=0.06, thresh=0.06):
    """
    Computes the rotation and translation to align the input points to
    an upright world coordinate system, where floor is z=0.

    :param xyz: [n, 3] array of 3D points
    :param grid_size: size of voxels for sampling
    :param thresh: threshold for plane fitting
    :return: R (rotation matrix), t (translation vector)
    """
    # Find the up direction
    up_direction, center = find_up(xyz, grid_size, thresh)

    # Compute the rotation matrix
    z_axis = np.array([0, 0, 1])

    # Ensure the vectors are unit vectors
    up_direction = up_direction / np.linalg.norm(up_direction)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Compute the rotation axis using the cross product
    rotation_axis = np.cross(up_direction, z_axis)

    # Compute the sine and cosine of the rotation angle using the dot product
    cos_angle = np.dot(up_direction, z_axis)
    # Clip the cosine to the range [-1, 1] to avoid numerical issues with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    rotation_angle = np.arccos(cos_angle)

    # If the vectors are parallel or anti-parallel, special handling is needed
    if np.linalg.norm(rotation_axis) < 1e-8:
        if cos_angle > 0:
            # Vectors are parallel, no rotation needed
            R = np.eye(3)
        else:
            # Vectors are anti-parallel, rotate by 180 degrees around any orthogonal axis
            # Find an orthogonal axis (can use any arbitrary vector that is not collinear)
            orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(up_direction, [1, 0, 0]) else np.array([0, 1, 0])
            rotation_axis = np.cross(up_direction, orthogonal_axis)
            rotation_axis /= np.linalg.norm(rotation_axis)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + 2 * np.dot(K, K)  # 180 degree rotation
    else:
        # Normalize the rotation axis
        rotation_axis /= np.linalg.norm(rotation_axis)
        # Skew-symmetric cross-product matrix of rotation_axis
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        # Rodrigues' rotation formula
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

    t = -center @ R.T

    return R, t


def main():
    import sys
    points, colors = read_ply(sys.argv[1])
    R, t = get_upright_transformation(points, grid_size=0.1, thresh=0.08)
    points = points @ R.T + t
    save_ply((points, colors), sys.argv[2])

    # import sys
    # points, colors = read_ply(sys.argv[1])
    # plane, points, colors = find_and_colorize(voxelized_sampling(points[:], grid_size=0.1), thresh=0.08)
    # save_ply((points, colors), sys.argv[2])


if __name__ == '__main__':
    main()
