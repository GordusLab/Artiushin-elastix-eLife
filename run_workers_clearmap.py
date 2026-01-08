import os
import shutil


def importElastix():
    # Import Elastix
    import sys, os
    sys.path.insert(0, '/home/acorver/ClearMap2/')
    _cwd = os.getcwd()
    os.chdir('/home/acorver/ClearMap2/')
    from ClearMap.Environment import elx, res, p3d
    os.chdir(_cwd)
    # Return modules
    return elx, res, p3d

def getDaskClient():
    import dask
    from distributed import Client

    # Dask settings
    dask.config.set({'distributed.worker.memory.rebalance.measure': 'managed_in_memory'})
    dask.config.set({'distributed.worker.memory.spill': False})
    dask.config.set({'distributed.worker.memory.pause': False})
    dask.config.set({'distributed.worker.memory.terminate': False})
    dask.config.set({'admin.tick.limit': '1h'})

    # Get cluster
    from config import CONFIG
    client = Client(CONFIG.DASK_SCHEDULER)

    # Return
    numWorkers = len(client.scheduler_info()['workers'])
    return client, numWorkers

# Convert windows filename to linux filename
def filenameToLinux(fn):
    fnL = fn.replace('\\', '/').replace('Z:/', '/mnt/z/').replace('D:/', '/mnt/d/')
    return fnL

# Submit task to Linux workers
def runResampleOnWorker(kwargs):
    # Ensure Elastix is loaded
    elx, res, p3d = importElastix()

    fnameInput = filenameToLinux(kwargs['fnameInput'])
    fnameOutput = filenameToLinux(kwargs['fnameOutput'])

    # Resample
    print('Received command: \nIN: {}\nOUT: {}\nIN RES: {}\nOUT RES: {}'.format(
        fnameInput, fnameOutput, kwargs['resInput'], kwargs['resOutput']
    ))

    resInput = tuple(kwargs['resInput'])
    r = kwargs['resOutput']

    # Write test output
    with open(fnameOutput + '.txt', 'w') as fTest:
        fTest.write('Testing output.')

    res.resample(
        '{}'.format(fnameInput),
        sink='{}'.format(fnameOutput), **{
        "source_resolution": resInput,
        "sink_resolution": (r, r, r),
        "processes": None,
        "verbose": True,
    })

def quote(fn):
    return '\"' + fn + '\"'

def runAlignmentOnWorker(kwargs):
    # Ensure Elastix is loaded
    elx, res, p3d = importElastix()

    nameRefCache = filenameToLinux(kwargs['nameRefCache'])
    nameMovCache = filenameToLinux(kwargs['nameMovCache'])
    fnameRigidCache = filenameToLinux(kwargs['fnameRigidCache']) if kwargs['fnameRigidCache'] is not None else None
    fnameNonrigidCache = filenameToLinux(kwargs['fnameNonrigidCache']) if kwargs['fnameNonrigidCache'] is not None else None

    if fnameRigidCache is None:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_nonrigid')
    elif fnameNonrigidCache is None:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid')
    else:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix')

    # Start alignment (RIGID ONLY)
    elx.align(**{
        "moving_image": quote(nameMovCache),
        "fixed_image": quote(nameRefCache),
        "affine_parameter_file": quote(fnameRigidCache) if fnameRigidCache is not None else None,
        "bspline_parameter_file": quote(fnameNonrigidCache) if fnameNonrigidCache is not None else None,
        "result_directory": dirout
    })

def runAlignment(nameRefCache, nameMovCache, fnameRigidCache, fnameNonrigidCache):
    """

    :param nameRefCache:
    :param nameMovCache:
    :param fnameRigidCache:
    :param fnameNonrigidCache:
    :return:
    """
    from config import CONFIG

    # Connect to linux dask and submit task
    client, numWorkers = getDaskClient()

    def _runAlignmentOnWorker(kwargs):
        import sys, os
        sys.path.insert(0, os.getcwd())
        import run_workers_clearmap
        run_workers_clearmap.runAlignmentOnWorker(kwargs)

    # Run rigid alignment
    client.run(_runAlignmentOnWorker, {
        'nameRefCache': nameRefCache,
        'nameMovCache': nameMovCache,
        'fnameRigidCache': fnameRigidCache,
        'fnameNonrigidCache': None
    })

    # Read rigid transformation file
    fnameRigid = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid\\result.0.tif')
    fnameRigidResultParams = fnameRigid.replace('result.0.tif', 'TransformParameters.0.txt')
    txtRigidResultParams = ''
    with open(fnameRigidResultParams, 'r') as f:
        txtRigidResultParams = f.read()

    # Reconstruct transform matrix
    import re, numpy as np
    params = np.array([float(x) for x in re.search('TransformParameters' + ''.join(
        ['\\s*([\\-\\.0-9]*)' for i in range(12)]), txtRigidResultParams).groups()], dtype=np.float64)
    assert len(params) == 12

    # For ordering see: https://elastix.lumc.nl/doxygen/classelastix_1_1AdvancedAffineTransformElastix.html
    mtx = np.identity(4, dtype=np.float64)
    mtx[0, :3] = params[0:3]
    mtx[1, :3] = params[3:6]
    mtx[2, :3] = params[6:9]
    mtx[:3, 3] = params[9:12]

    # Parse the intended perturbation matrix from the filename
    paramsPerturb = np.array([float(x) for x in re.search('.*_' + '='.join([
        '([\\.e\\-0-9]*)' for i in range(12)]), nameRefCache).groups()], dtype=np.float64)
    assert len(paramsPerturb) == 12

    # Reorder into perturbation matrix
    mtxP = np.identity(4, dtype=np.float64)
    mtxP[0, 0:4] = paramsPerturb[0:4]
    mtxP[1, 0:4] = paramsPerturb[4:8]
    mtxP[2, 0:4] = paramsPerturb[8:12]

    # Apply perturbation matrix
    mtxPerturbed = mtxP @ mtx

    # Re-order into 12-size linear array (https://elastix.lumc.nl/doxygen/classelastix_1_1AdvancedAffineTransformElastix.html)
    paramsNew = np.hstack((mtxPerturbed[0, 0:3], mtxPerturbed[1, 0:3],
        mtxPerturbed[2, 0:3], mtxPerturbed[:3, 3])).tolist()

    # Create a perturbed transformation parameter file
    txtRigidResultParamsPerturbed, subn = re.subn('\\(TransformParameters.*\\)',
        '(TransformParameters {})'.format(' '.join(['{:.6f}'.format(x) for x in paramsNew])), txtRigidResultParams)
    assert (subn == 1)

    # Write to new output file
    fnameTransformPerturbed = fnameRigidResultParams.replace('/','\\').replace(
        '\\elastix_rigid\\', '\\elastix_rigid_perturbed\\')
    assert fnameRigidResultParams != fnameTransformPerturbed
    os.makedirs(os.path.dirname(fnameTransformPerturbed), exist_ok=True)
    with open(fnameTransformPerturbed, 'w') as fp:
        fp.write(txtRigidResultParamsPerturbed)

    # Apply perturbed transform to original image
    def _runTransformOnWorker(kwargs):
        import sys, os
        sys.path.insert(0, os.getcwd())
        import run_workers_clearmap
        run_workers_clearmap.runTransformOnWorker(kwargs)

    fnameRigidPerturb = os.path.join(os.path.dirname(fnameTransformPerturbed), 'result.tif')
    client.run(_runTransformOnWorker, {
        'fileIn': nameMovCache,
        'fileOut': fnameRigidPerturb,
        'fileTransform': fnameTransformPerturbed,
        'dirResults': os.path.dirname(fnameTransformPerturbed)
    })

    # Run nonrigid alignment
    client.run(_runAlignmentOnWorker, {
        'nameRefCache': nameRefCache,
        'nameMovCache': fnameRigidPerturb,
        'fnameRigidCache': None,
        'fnameNonrigidCache': fnameNonrigidCache
    })

def runTransformOnWorker(kwargs):
    # Ensure Elastix is loaded
    elx, res, p3d = importElastix()

    fileIn = filenameToLinux(kwargs['fileIn'])
    fileOut = filenameToLinux(kwargs['fileOut'])
    fileTransform = filenameToLinux(kwargs['fileTransform'])
    dirResults = filenameToLinux(kwargs['dirResults'])

    # Start transformation
    try:
        elx.transform(
            fileIn,
            transform_parameter_file=fileTransform,
            result_directory=dirResults)
    except Exception as e:
        if 'Cannot find a valid result data' in str(e):
            # ClearMap2 appears to erroneously require MHD outputs and ignores the TIF result file?
            if not os.path.exists(os.path.join(dirResults, 'result.tif')):
                # If the TIF is not found, re-raise this error
                raise e
        else:
            raise e

def runTransform(fnameInput, fnameTransform):
    # Apply perturbed transform to original image
    def _runTransformOnWorker(kwargs):
        import sys, os
        sys.path.insert(0, os.getcwd())
        import run_workers_clearmap
        run_workers_clearmap.runTransformOnWorker(kwargs)

    # Connect to linux dask and submit task
    client, numWorkers = getDaskClient()

    fnameOutput = os.path.join(os.path.dirname(fnameTransform), 'result.tif')
    client.run(_runTransformOnWorker, {
        'fileIn': fnameInput,
        'fileOut': fnameOutput,
        'fileTransform': fnameTransform,
        'dirResults': os.path.dirname(fnameTransform)
    })

def runResample(fnameInput, fnameOutput, resInput, resOutput, cacheSuffix, ref, mov):
    """
    :param fnameInput:
    :param fnameOutput:
    :param resInput:
    :param resOutput:
    :param cacheSuffix: Random cache suffix to ensure a unique cache directory is used on each run
    :return:
    """
    from config import CONFIG

    # Move data to cache directory
    fnameOutputCache = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', os.path.basename(
        os.path.dirname(fnameOutput)) + '_{}'.format(cacheSuffix), os.path.basename(fnameOutput))
    fnameInputCache = os.path.join(os.path.dirname(fnameOutputCache), os.path.basename(fnameInput))

    # Copy to cache
    os.makedirs(os.path.dirname(fnameInputCache), exist_ok=True)
    shutil.copy(fnameInput, fnameInputCache)

    # Connect to linux dask and submit task
    client, numWorkers = getDaskClient()

    def _runResampleOnWorker(kwargs):
        import sys, os
        sys.path.insert(0, os.getcwd())
        import run_workers_clearmap
        run_workers_clearmap.runResampleOnWorker(kwargs)

    client.run(_runResampleOnWorker, {
        'fnameInput': fnameInputCache,
        'fnameOutput': fnameOutputCache,
        'resInput': resInput,
        'resOutput': resOutput,
        'ref': ref,
        'mov': mov
    })

    # Copy back
    shutil.copy(fnameOutputCache, fnameOutput)
