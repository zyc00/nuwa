import numpy as np
import torch


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    else:
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


def camera_to_canonical(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :return:
    """
    if isinstance(pts, np.ndarray) and isinstance(pose, np.ndarray):
        pts = pts.T  # 3xN
        pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
        p = np.linalg.inv(pose) @ pts  # 4xN
        p[0:3] /= p[3:]
        p = p[0:3]
        p = p.T
        return p
    else:
        pts = cart_to_hom(pts)
        pts = pts @ torch.inverse(pose).t()
        pts = hom_to_cart(pts)
        return pts


def xyzr_to_pose4x4(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def xyzr_to_pose4x4_torch(x, y, z, r):
    if isinstance(x, torch.Tensor):
        pose = torch.eye(4, device=x.device, dtype=torch.float)
        pose[0, 0] = torch.cos(r)
        pose[0, 2] = torch.sin(r)
        pose[2, 0] = -torch.sin(r)
        pose[2, 2] = torch.cos(r)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        return pose
    else:
        return torch.from_numpy(xyzr_to_pose4x4_np(x, y, z, r)).float()


def pose4x4_to_xyzr(pose):
    x = pose[0, 3]
    y = pose[1, 3]
    z = pose[2, 3]
    cos = pose[0, 0]
    sin = pose[0, 2]
    angle = np.arctan2(sin, cos)
    return x, y, z, angle


def camera_coordinate_to_world_coordinate(pts_in_camera, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_camera: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_camera, np.ndarray):
        pts_hom = np.hstack((pts_in_camera, np.ones((pts_in_camera.shape[0], 1), dtype=np.float32)))
        pts_world = pts_hom @ cam_pose.T
    else:
        ones = torch.ones((pts_in_camera.shape[0], 1), dtype=torch.float32, device=pts_in_camera.device)
        pts_hom = torch.cat((pts_in_camera, ones), dim=1)
        cam_pose = torch.tensor(cam_pose).float().to(device=pts_in_camera.device)
        pts_world = pts_hom @ cam_pose.t()
    pts_world = pts_world[:, :3] / pts_world[:, 3:]
    return pts_world


def world_coordinate_to_camera_coordinate(pts_in_world, cam_pose):
    """
    transform points in camera coordinate to points in world coordinate
    :param pts_in_world: n,3
    :param cam_pose: 4,4
    :return:
    """
    if isinstance(pts_in_world, np.ndarray):
        cam_pose_inv = np.linalg.inv(cam_pose)
        pts_hom = np.hstack((pts_in_world, np.ones((pts_in_world.shape[0], 1), dtype=np.float32)))
        pts_cam = pts_hom @ cam_pose_inv.T
    else:
        cam_pose = cam_pose.float().to(device=pts_in_world.device)
        cam_pose_inv = torch.inverse(cam_pose)
        ones = torch.ones((pts_in_world.shape[0], 1), dtype=torch.float32, device=pts_in_world.device)
        pts_hom = torch.cat((pts_in_world, ones), dim=1)
        pts_cam = pts_hom @ cam_pose_inv.t()
    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]
    return pts_cam


def xyzr_to_pose4x4_np(x, y, z, r):
    pose = np.eye(4)
    pose[0, 0] = np.cos(r)
    pose[0, 2] = np.sin(r)
    pose[2, 0] = -np.sin(r)
    pose[2, 2] = np.cos(r)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def canonical_to_camera_np(pts, pose, calib=None):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib: KITTICalib
    :return:
    """
    pts = pts.T  # 3xN
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # 4XN
    p = pose @ pts  # 4xN
    if calib is None:
        p[0:3] /= p[3:]
        p = p[0:3]
    else:
        p = calib.P2 @ p
        p[0:2] /= p[2:]
        p = p[0:2]
    p = p.T
    return p


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


def roty_torch(a):
    """
    :param a: np.ndarray of (N, 1) or (N), or float, or int
              angle
    :return: np.ndarray of (N, 3, 3)
             rotation matrix
    """
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, zeros, s,
                       zeros, ones, zeros,
                       -s, zeros, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


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


def rotz_torch(a):
    if isinstance(a, (int, float)):
        a = torch.tensor([a])
    if a.shape[-1] != 1:
        a = a[..., None]
    a = a.float()
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(a)
    c = torch.cos(a)
    s = torch.sin(a)
    rot = torch.stack([c, -s, zeros,
                       s, c, zeros,
                       zeros, zeros, ones], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def rotx(t):
    """
    Rotation along the x-axis.
    :param t: tensor of (N, 1) or (N), or float, or int
              angle
    :return: tensor of (N, 3, 3)
             rotation matrix
    """
    if isinstance(t, (int, float)):
        t = torch.tensor([t])
    if t.shape[-1] != 1:
        t = t[..., None]
    t = t.type(torch.float)
    ones = torch.ones_like(t)
    zeros = torch.zeros_like(t)
    c = torch.cos(t)
    s = torch.sin(t)
    rot = torch.stack([ones, zeros, zeros,
                       zeros, c, -s,
                       zeros, s, c], dim=-1)
    return rot.reshape(*rot.shape[:-1], 3, 3)


def matrix_3x4_to_4x4(a):
    if len(a.shape) == 2:
        assert a.shape == (3, 4)
    else:
        assert len(a.shape) == 3
        assert a.shape[1:] == (3, 4)
    if isinstance(a, np.ndarray):
        if a.ndim == 2:
            ones = np.array([[0, 0, 0, 1]])
            return np.vstack((a, ones))
        else:
            ones = np.array([[0, 0, 0, 1]])[None].repeat(a.shape[0], axis=0)
            return np.concatenate((a, ones), axis=1)
    else:
        ones = torch.tensor([[0, 0, 0, 1]]).float().to(device=a.device)
        if a.ndim == 3:
            ones = ones[None].repeat(a.shape[0], 1, 1)
            ret = torch.cat((a, ones), dim=1)
        else:
            ret = torch.cat((a, ones), dim=0)
        return ret


def matrix_3x3_to_4x4(a):
    assert a.shape == (3, 3)
    if isinstance(a, np.ndarray):
        ret = np.eye(4)
    else:
        ret = torch.eye(4).float().to(a.device)
    ret[:3, :3] = a
    return ret


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def dummy_sdf_volume():
    sphere_radius = 0.2  # Sphere radius (in meters)
    sdf_volume_size = 1.0  # SDF volume size (in meters)
    resolution = 128  # Resolution of the SDF volume

    # Calculate the grid spacing
    grid_spacing = sdf_volume_size / resolution

    # Create an empty SDF volume grid
    sdf_volume = np.zeros((resolution, resolution, resolution), dtype=float)

    # Loop through each point in the SDF volume grid
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                # Calculate the world coordinates of the current point
                world_x = x * grid_spacing - sdf_volume_size / 2.0
                world_y = y * grid_spacing - sdf_volume_size / 2.0
                world_z = z * grid_spacing - sdf_volume_size / 2.0

                # Calculate the distance from the current point to the sphere's center
                distance_to_center = np.sqrt(
                    (world_x ** 2) + (world_y ** 2) + (world_z ** 2)
                )

                # Calculate the SDF value for the current point
                sdf_volume[x, y, z] = distance_to_center - sphere_radius
    return sdf_volume


def calc_pose(phis, thetas, size, radius=1.2):
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    device = torch.device('cpu')
    thetas = torch.FloatTensor(thetas).to(device)
    phis = torch.FloatTensor(phis).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses


def voxel_iou(pred_, gt_):
    pred = pred_.clone()
    gt = gt_.clone()

    pred[pred > 0] = 1
    pred[pred < 0] = 0
    gt[gt > 0] = 1
    gt[gt < 0] = 0
    intersection = torch.sum((pred == 1) & (gt == 1), dim=(1, 2, 3, 4))
    union = torch.sum((pred == 1) | (gt == 1), dim=(1, 2, 3, 4))
    return intersection / union


def rect_to_img(K, pts_rect):
    pts_2d_hom = pts_rect @ K.T
    pts_img = hom_to_cart(pts_2d_hom)
    return pts_img


def K_to_opengl_projection(K, width, height, near=0.01, far=100):
    P = np.array([[2 * K[0, 0] / width, 0, 1 - 2 * K[0, 2] / width, 0],
                  [0, -2 * K[1, 1] / height, 1 - 2 * K[1, 2] / height, 0],
                  [0, 0, -far / (far - near), - far * near / (far - near)],
                  [0, 0, -1, 0]])
    return P.astype(np.float32)
