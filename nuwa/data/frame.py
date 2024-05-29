import numpy as np

from nuwa.data.camera import _Camera
from nuwa.utils.image_utils import sharpness
from nuwa.utils.pose_utils import convert_camera_pose


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
                "c2w": "OPENCV_c2w",
                "w2c": "OPENCV_w2c",
                "transform_matrix": "BLENDER_c2w"
            }
        }
        ret.update(self.camera.to_dict())
        return ret
