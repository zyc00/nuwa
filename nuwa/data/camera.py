import abc
import math


class _Camera:
    w: float
    h: float
    fx: float
    fy: float
    cx: float
    cy: float
    is_fisheye: bool = False

    k1: float
    k2: float
    p1: float
    p2: float

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def to_dict(self):
        raise NotImplementedError

    @property
    def intrinsic_matrix(self):
        return [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]

    def __repr__(self):
        return self.to_dict().__repr__()


class OpenCvCamera(_Camera):
    def __init__(self, w, h, fx, fy, cx, cy, k1=0., k2=0., p1=0., p2=0.):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2

    def to_dict(self):
        return {
            "w": self.w,
            "h": self.h,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,

            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,

            "fl_x": self.fx,
            "fl_y": self.fy,
            "camera_angle_x": math.atan(self.w / (self.fx * 2)) * 2,
            "camera_angle_y": math.atan(self.h / (self.fy * 2)) * 2,
            "fovx": math.atan(self.w / (self.fx * 2)) * 2 * 180 / math.pi,
            "foxy": math.atan(self.h / (self.fy * 2)) * 2 * 180 / math.pi,

            "intrinsic_matrix": self.intrinsic_matrix,

            "camera_param_model": "OPENCV"
        }


class PinholeCamera(_Camera):
    def __init__(self, w, h, fx, fy, cx, cy):
        self.w = w
        self.h = h
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def to_dict(self):
        return {
            "w": self.w,
            "h": self.h,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "is_fisheye": self.is_fisheye,

            "fl_x": self.fx,
            "fl_y": self.fy,
            "camera_angle_x": math.atan(self.w / (self.fx * 2)) * 2,
            "camera_angle_y": math.atan(self.h / (self.fy * 2)) * 2,
            "fovx": math.atan(self.w / (self.fx * 2)) * 2 * 180 / math.pi,
            "foxy": math.atan(self.h / (self.fy * 2)) * 2 * 180 / math.pi,

            "intrinsic_matrix": self.intrinsic_matrix,

            "camera_param_model": "PINHOLE"
        }
