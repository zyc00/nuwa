import numpy as np

from nuwa.data.camera import _Camera, OpenCvCamera, PinholeCamera
from nuwa.utils.image_utils import sharpness
from nuwa.utils.pose_utils import convert_camera_pose


class Frame:
    camera: _Camera
    image_path: str   # abs path
    pose: np.ndarray  # OpenCV Convention
    sharpness: float
    seq_id: int
    mask_path: str    # abs path
    org_path: str     # abs path

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
        self.org_path = org_path if org_path else image_path

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
                "c2w": "OPENCV_c2w",
                "w2c": "OPENCV_w2c",
                "transform_matrix": "BLENDER_c2w"
            }
        }
        ret.update(self.camera.to_dict())
        return ret

    @classmethod
    def from_dict(cls, data: dict):
        if data["camera_param_model"] == "OPENCV":
            camera = OpenCvCamera(
                data["w"], data["h"],
                data["fx"], data["fy"],
                data["cx"], data["cy"],
                data["k1"], data["k2"],
                data["p1"], data["p2"]
            )
        elif data["camera_param_model"] == "PINHOLE":
            camera = PinholeCamera(
                data["w"], data["h"],
                data["fx"], data["fy"],
                data["cx"], data["cy"]
            )
        else:
            raise ValueError(f"Unknown camera model: {data['camera_model']}")

        pose = np.array(data["c2w"])
        image_path = data["file_path"]
        seq_id = data["seq_id"]
        sharpness_score = data["sharpness"]
        mask_path = data["mask_path"]
        org_path = data["org_path"]

        return cls(
            camera,
            image_path,
            pose,
            seq_id,
            sharpness_score,
            mask_path,
            org_path
        )

    def __repr__(self):
        return self.to_dict().__repr__()
