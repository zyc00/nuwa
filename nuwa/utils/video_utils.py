import os
from .os_utils import do_system


def run_ffmpeg(video_path, out_image_dir, fps, ext="png"):
    ffmpeg_binary = "ffmpeg"
    fps = float(fps)
    print(f"running ffmpeg with input video file={video_path}, "
          f"output folder={out_image_dir}, "
          f"fps={fps}.")

    os.makedirs(out_image_dir, exist_ok=True)
    do_system(f"{ffmpeg_binary} "
              f"-i {video_path} "
              f"-qscale:v 1 "
              f"-qmin 1 "
              f"-vf \"fps={fps}\" "
              f"{out_image_dir}/%04d.{ext}")
