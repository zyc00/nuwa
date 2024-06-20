import subprocess

import nuwa


def do_system(arg):
    nuwa.get_logger().debug("System running: {arg}")

    if nuwa.get_log_level() == nuwa.logging.DEBUG:
        subprocess.check_call(arg)
    else:
        subprocess.check_call(
            arg,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
