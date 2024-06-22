import numpy as np


def convert_camera_pose(pose, in_type, out_type):
    accepted_types = ["cv", "gl", "blender"]

    assert in_type in accepted_types, f"Input type {in_type} not in {accepted_types}"
    assert out_type in accepted_types, f"Output type {out_type} not in {accepted_types}"

    if in_type == "blender":
        in_type = "gl"
    if out_type == "blender":
        out_type = "gl"

    if in_type == out_type:
        return pose

    return pose @ np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )


def qvec2rotmat(qvec):
    qvec /= np.linalg.norm(qvec)
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


def rotmat2qvec(R):
    qvec = np.empty(4)
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qvec[0] = 0.25 / s
        qvec[1] = (R[2, 1] - R[1, 2]) * s
        qvec[2] = (R[0, 2] - R[2, 0]) * s
        qvec[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qvec[0] = (R[2, 1] - R[1, 2]) / s
            qvec[1] = 0.25 * s
            qvec[2] = (R[0, 1] + R[1, 0]) / s
            qvec[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qvec[0] = (R[0, 2] - R[2, 0]) / s
            qvec[1] = (R[0, 1] + R[1, 0]) / s
            qvec[2] = 0.25 * s
            qvec[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qvec[0] = (R[1, 0] - R[0, 1]) / s
            qvec[1] = (R[0, 2] + R[2, 0]) / s
            qvec[2] = (R[1, 2] + R[2, 1]) / s
            qvec[3] = 0.25 * s

    qvec /= np.linalg.norm(qvec)
    return qvec


def get_rot90_camera_matrices(pose, fx, fy, cx, cy, w, h):
    """
    Get camera matrices for rotating image 90 degrees clockwise

    :param pose: camera pose matrix
    :param fx
    :param fy
    :param cx
    :param cy
    :param w: original image width
    :param h: original image height
    :return:
    """
    new_pose_matrix = pose @ np.array([[0, 1, 0, 0],
                                       [-1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
    nfx = fy
    nfy = fx
    ncx = h - cy
    ncy = cx

    return new_pose_matrix, nfx, nfy, ncx, ncy, h, w


def qt2pose(q, t):
    q = np.array(q).reshape(-1)
    t = np.array(t).reshape(-1)
    return np.linalg.inv(rt2pose(qvec2rotmat(-q), t))


def pose2qt(pose):
    rt = np.linalg.inv(pose)
    return -rotmat2qvec(rt[:3, :3]), rt[:3, 3]


def rt2pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


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


if __name__ == '__main__':
    import unittest

    class TestQuaternionConversion(unittest.TestCase):
        def test_rotmat2qvec(self):
            # Test with known quaternion and its rotation matrix
            qvec = np.array([0.7071068, 0.0, 0.7071068, 0.0])
            R = qvec2rotmat(qvec)
            qvec_converted = rotmat2qvec(R)

            np.testing.assert_almost_equal(qvec, qvec_converted, decimal=6)

        def test_identity_matrix(self):
            # Test with identity matrix
            R = np.eye(3)
            qvec = rotmat2qvec(R)
            expected_qvec = np.array([1.0, 0.0, 0.0, 0.0])

            np.testing.assert_almost_equal(qvec, expected_qvec, decimal=6)

        def test_random_quaternions(self):
            # Test with random quaternions
            for _ in range(30000):
                qvec = np.random.rand(4)
                qvec /= np.linalg.norm(qvec)  # Normalize the quaternion
                R = qvec2rotmat(qvec)
                qvec_converted = rotmat2qvec(R)

                np.testing.assert_almost_equal(qvec, qvec_converted, decimal=6)

    unittest.main()
