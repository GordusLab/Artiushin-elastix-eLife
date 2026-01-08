
"""
    SETUP:

    Ensure the drive is mapped on the linux worker system
    sudo mount -t drvfs '\\10.99.66.32\Team Spider' /mnt/z

    Start a scheduler:
    dask-scheduler --port 9999 --dashboard-address 9998

    Start worker(s) (on linux) *in the directory containing this repository code*
    dask-worker 10.99.66.244:9999 --nprocs 1 --nthreads 1 --memory-limit="200 GiB" --no-nanny
"""

class CONFIG:
    DIRECTORY_ROOT = 'Z:\\StandardBrain\\'
    DASK_SCHEDULER = '10.99.66.244:9999'
    DEBUG_ONLY = False
    USE_CLEARMAP = False

    @staticmethod
    def CACHE_ROOT():
        import os
        if os.path.exists('D:\\StandardBrain\\'):
            return 'D:\\StandardBrain\\'
        else:
            return 'C:\\StandardBrain\\'


