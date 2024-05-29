import os


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        raise RuntimeError(f"command failed with {err=}: {arg}")
