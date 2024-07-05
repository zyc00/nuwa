from nuwa.data.db_loader import (
    from_colmap,
    from_polycam,
    from_3dscannerapp,
    from_dear,
    from_haoyang,
    from_nuwadb,
    from_image_folder,
    from_video
)

import sys
import logging


def is_seg_available():
    try:
        import torch
        import torchvision
        import nuwa.utils.seg_utils
        return True
    except ImportError as e:
        get_logger().error(f"Error: {e}")
        return False


def is_ingp_available():
    try:
        import pyngp
        return True
    except ImportError as e:
        get_logger().error(f"Error: {e}")
        return False


def get_logger():
    return logging.getLogger("nuwa")


def set_log_level(level):
    get_logger().setLevel(level)
    for handler in get_logger().handlers:
        handler.setLevel(level)
        if level == logging.DEBUG:
            handler.setFormatter(_NuwaDebugFormatter())
        else:
            handler.setFormatter(_NuwaFormatter())


def get_log_level():
    return get_logger().getEffectiveLevel()


class _NuwaFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class _NuwaDebugFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s - %(asctime)s - %(message)s (%(filename)s:%(lineno)s)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def _init_logger():
    logger = logging.getLogger("nuwa")
    c_handler = logging.StreamHandler(stream=sys.stdout)
    c_handler.setFormatter(_NuwaFormatter())
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    logger.setLevel(logging.INFO)


_init_logger()

__all__ = [
    "from_colmap",
    "from_polycam",
    "from_3dscannerapp",
    "from_dear",
    "from_haoyang",
    "from_nuwadb",
    "from_image_folder",
    "from_video",

    "is_seg_available",

    "get_logger",
    "set_log_level",
    "get_log_level"
]
