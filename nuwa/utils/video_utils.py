import os
from .os_utils import do_system


def run_ffmpeg(video_path, out_image_dir, fps, verbose=False):
    ffmpeg_binary = "/usr/bin/ffmpeg"
    fps = float(fps)

    os.makedirs(out_image_dir, exist_ok=True)

    print(f"INFO: ffmpeg - extracting frames from {video_path} to {out_image_dir}")
    do_system((f"{ffmpeg_binary}",
               "-i", f"{video_path}",
               "-qscale:v", "1",
               "-qmin", "1",
               "-vf", f"fps={fps}",
               f"{out_image_dir}/%04d.png"), verbose)
