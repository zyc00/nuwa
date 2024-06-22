import subprocess

import nuwa
import os


def do_system(arg):
    nuwa.get_logger().debug(f"System running: {arg}")

    if nuwa.get_log_level() == nuwa.logging.DEBUG:
        subprocess.check_call(arg)
    else:
        subprocess.check_call(
            arg,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


def run_ffmpeg(video_path, out_image_dir, fps):
    ffmpeg_binary = "/usr/bin/ffmpeg"
    fps = float(fps)

    os.makedirs(out_image_dir, exist_ok=True)

    nuwa.get_logger().info(f"ffmpeg - extracting frames from {video_path} to {out_image_dir}")
    do_system((f"{ffmpeg_binary}",
               "-i", f"{video_path}",
               "-qscale:v", "1",
               "-qmin", "1",
               "-vf", f"fps={fps}",
               f"{out_image_dir}/%04d.png"))
