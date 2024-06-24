import plyfile
import numpy as np


def load_gs_simple(path):
    """
    loads gs xyz and color only
    """
    data = plyfile.PlyData.read(path)

    if "red" in data["vertex"]:

        return {
            "xyz": np.array([data["vertex"][k] for k in ["x", "y", "z"]], dtype=np.float32).T,
            "color": np.array([data["vertex"][k] for k in ["red", "green", "blue"]], dtype=np.float32).T,
        }

    else:
        return {
            "xyz": np.array([data["vertex"][k] for k in ["x", "y", "z"]], dtype=np.float32).T,
            "color": np.array([data["vertex"][k] for k in ["f_dc_0", "f_dc_1", "f_dc_2"]], dtype=np.float32).T,
        }
