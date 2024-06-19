import subprocess


def do_system(arg, verbose=False):
    if verbose:
        print(f"-> Running: {arg}")
        subprocess.check_call(arg)
    else:
        subprocess.check_call(
            arg,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
