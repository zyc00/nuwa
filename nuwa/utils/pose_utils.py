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


def get_rot90_camera_matrices(pose, fx, fy, cx, cy, h):
    """
    Get camera matrices for rotating image 90 degrees clockwise

    :param pose: camera pose matrix
    :param fx
    :param fy
    :param cx
    :param cy
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

    return new_pose_matrix, nfx, nfy, ncx, ncy


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
