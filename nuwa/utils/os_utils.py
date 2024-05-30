import subprocess


def do_system(arg, verbose=False):
    print(f"-> Running: {arg}")
    if verbose:
        subprocess.check_call(arg)
    else:
        subprocess.check_call(
            arg,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
