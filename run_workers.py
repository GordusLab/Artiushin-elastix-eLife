import glob, os, re, shutil, subprocess, pgzip, time, traceback, tifffile, numpy as np, numba as nb, \
    scipy.stats, joblib as jl
from itertools import chain
from skimage.measure import label, regionprops
import skimage.filters
from tqdm import tqdm

# Monitor errors on resampling task
def runResampleOnWorker(kwargs):
    #try:
    runResampleOnWorkerUnsafe(kwargs)
    #except Exception as e:
    #    print('Error during resampling: {}'.format(kwargs['fnameInput']))
    #    traceback.print_exc()

def getSupervisedTransform(fnameInput, fnameOutput, ref, mov):
    # Only apply to moving image
    if 'mov_' not in fnameOutput:
        return None

    # Find files
    fnameRefLbl = os.path.join(os.path.dirname(fnameOutput), 'ref_pointlabels.iso.tif')
    fnameMovLbl = mov.replace('mov_', 'ref_').replace('synapsin.tif', 'pointlabels.tif')

    # Transform already known?
    fnameTransform = os.path.join(os.path.dirname(fnameOutput), 'mov_transform.npy')
    if os.path.exists(fnameTransform):
        return np.load(fnameTransform)

    ss = 1 # No subsampling (but useful for quicker debugging)

    # Exit if labeled files not found
    if not os.path.exists(fnameRefLbl):
        return None
    if not os.path.exists(fnameMovLbl):
        return None

    labels = []
    for fnameLabels in [fnameRefLbl, fnameMovLbl]:
        # Load labels
        imgLabels = tifffile.imread(fnameLabels)[::ss, ::ss, ::ss]
        # Find regions
        props = regionprops(imgLabels)
        labelPos = {}
        labelSize = {}
        for prop in tqdm(props, desc='Finding annotation centroids...'):
            if prop.label in labelPos:
                if prop.coords.shape[0] < labelSize[prop.label]:
                    continue
            _x, _y, _z = np.mean(prop.coords, axis=0).round().astype(int)
            # _x = float(_x) / imgLabels.shape[0]
            # _y = float(_y) / imgLabels.shape[1]
            # _z = float(_z) / imgLabels.shape[2]
            labelPos[prop.label] = (_x, _y, _z)  # (_z, _x, _y)
            labelSize[prop.label] = prop.label
        labels.append(labelPos)
    # Only use indices that exist in both reference and moving
    labelIDs = np.intersect1d(list(labels[0].keys()), list(labels[1].keys()))

    # Create point correspondences
    pts0, pts1 = [], []
    for i in labelIDs:
        pts0.append(labels[0][i])
        pts1.append(labels[1][i])
    pts0 = np.array(pts0, dtype=np.float64)
    pts1 = np.array(pts1, dtype=np.float64)

    # Pad 1's
    pts0 = np.hstack((pts0, np.ones((pts0.shape[0], 1))))
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))

    def mtxFromCoef(A):
        # Create translation matrix
        mtxTranslation = np.identity(4, dtype=np.float64)
        mtxTranslation[0, 3] = A[0]
        mtxTranslation[1, 3] = A[1]
        mtxTranslation[2, 3] = A[2]

        # Create scaling matrix
        mtxScaling = np.identity(4, dtype=np.float64)
        mtxScaling[0, 0] = A[3]
        mtxScaling[1, 1] = A[4]
        mtxScaling[2, 2] = A[5]

        # Create rotation matrix (X)
        mtxRotation = np.identity(4, dtype=np.float64)
        mtxRotation[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', [
            A[6], A[7], A[8],
        ]).as_matrix()

        # Return affine matrix
        mtx = mtxRotation @ (mtxScaling @ mtxTranslation)
        return mtx

    def _lstsqFit(A):
        mtx = mtxFromCoef(A)
        pts0_ = pts1 @ mtx
        return np.percentile(np.linalg.norm((pts0_ - pts0)[:, 0:3], axis=1), 100)

    _x0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

    r = None
    for k in tqdm(range(500)):
        r_ = scipy.optimize.minimize(_lstsqFit,
            x0=_x0 + 2 * np.random.random(size=9), method='Nelder-Mead',
            options={'maxiter': 1000})
        if r is None:
            r = r_
        else:
            if r_.fun < r.fun:
                r = r_
                print(r.fun)

    A = mtxFromCoef(r.x)

    # Verify transform
    print('Error from least-squares affine fit is: {}'.format(np.sum(np.abs((pts1 @ A - pts0)[:, 0:3]))))

    # SciPy's affine_transform requires an inverse transform
    Ainv = np.linalg.inv(A)

    # Cache transform to ensure other channels use the exact same transformation
    np.save(fnameTransform, Ainv)

    # Done!
    return Ainv

# Resampling
def runSupervisedTransform(img, fnameInput, fnameOutput, ref, mov, interpolationOrder):
    # Only apply to moving image
    if 'mov_' not in fnameOutput:
        return None

    # Find files
    fnameRefLbl = os.path.join(os.path.dirname(fnameOutput), 'ref_pointlabels.iso.tif')
    fnameMovLbl = mov.replace('mov_', 'ref_').replace('synapsin.tif', 'pointlabels.tif')

    # Load labels
    ss = 1 # No subsampling (but useful for quicker debugging)
    imgLabelsRef = tifffile.imread(fnameRefLbl)[::ss, ::ss, ::ss]

    # Transform already known?
    fnameTransform = os.path.join(os.path.dirname(fnameOutput), 'mov_transform.npy')
    xyzRefToMov = None
    if os.path.exists(fnameTransform):
        xyzRefToMov = np.load(fnameTransform)
    else:
        # Exit if labeled files not found
        if not os.path.exists(fnameRefLbl):
            return None
        if not os.path.exists(fnameMovLbl):
            return None

        # Load labels
        imgLabelsMov = tifffile.imread(fnameMovLbl)[::ss, ::ss, ::ss]

        labels = []
        for fnameLabels in [fnameRefLbl, fnameMovLbl]:
            # Find regions
            imgLabels = imgLabelsRef if fnameLabels == fnameRefLbl else imgLabelsMov
            props = regionprops(imgLabels)
            labelPos = {}
            labelSize = {}
            for prop in tqdm(props, desc='Finding annotation centroids...'):
                if prop.label in labelPos:
                    if prop.coords.shape[0] < labelSize[prop.label]:
                        continue
                _x, _y, _z = np.mean(prop.coords, axis=0).round().astype(int)
                labelPos[prop.label] = (_x, _y, _z)  # (_z, _x, _y)
                labelSize[prop.label] = prop.label
            labels.append(labelPos)
        # Only use indices that exist in both reference and moving
        labelIDs = np.intersect1d(list(labels[0].keys()), list(labels[1].keys()))

        # Extract point labels (x0, y0, z0, x1, y1, z1) xyz0 = Ref, xyz1 = Mov
        # -- Will map reference image coords to moving
        pts = np.array([labels[0][i] + labels[1][i] for i in labelIDs], dtype=np.float32)

        # Create data matrix: 1 (intercept), x, y, z, xy, xz, yz, x^2, y^2, z^2   [10 columns]
        data = np.zeros((pts.shape[0], 10), dtype=np.float32)

        data[:, 0] = 1.0

        data[:, 1] = pts[:, 0]
        data[:, 2] = pts[:, 1]
        data[:, 3] = pts[:, 2]

        data[:, 4] = pts[:, 0] * pts[:, 1]
        data[:, 5] = pts[:, 0] * pts[:, 2]
        data[:, 6] = pts[:, 1] * pts[:, 2]

        # data[:, 7] = pts[:, 0]**2
        # data[:, 8] = pts[:, 1]**2
        # data[:, 9] = pts[:, 2]**2

        # Perform linear regression
        coefs, _, _, _ = np.linalg.lstsq(data, pts[:, 3:6], rcond=None)

        @nb.njit(boundscheck=True, nogil=True)
        def getXYZcoords(nx, ny, nz, limx, limy, limz):
            xyz = np.zeros((nx * ny * nz, 3), dtype=np.int32)
            c = 0
            for x in np.linspace(0, limx - 1, nx):
                for y in np.linspace(0, limy - 1, ny):
                    for z in np.linspace(0, limz - 1, nz):
                        xyz[c, 0] = x
                        xyz[c, 1] = y
                        xyz[c, 2] = z
                        c += 1
            return xyz

        # Now transform entire image
        xyz = getXYZcoords(imgLabelsRef.shape[0], imgLabelsRef.shape[1], imgLabelsRef.shape[2],
                           imgLabelsRef.shape[0], imgLabelsRef.shape[1], imgLabelsRef.shape[2])

        data = np.zeros((xyz.shape[0], 10), dtype=np.float32)

        data[:, 0] = 1.0

        data[:, 1] = xyz[:, 0]
        data[:, 2] = xyz[:, 1]
        data[:, 3] = xyz[:, 2]

        data[:, 4] = xyz[:, 0] * xyz[:, 1]
        data[:, 5] = xyz[:, 0] * xyz[:, 2]
        data[:, 6] = xyz[:, 1] * xyz[:, 2]

        data[:, 7] = xyz[:, 0] ** 2
        data[:, 8] = xyz[:, 1] ** 2
        data[:, 9] = xyz[:, 2] ** 2

        xyzRefToMov = data @ coefs

        # Clip
        xyzRefToMov[:, 0] = np.clip(xyzRefToMov[:, 0], 0, img.shape[0] - 1)
        xyzRefToMov[:, 1] = np.clip(xyzRefToMov[:, 1], 0, img.shape[1] - 1)
        xyzRefToMov[:, 2] = np.clip(xyzRefToMov[:, 2], 0, img.shape[2] - 1)

    # Sample new image
    movAligned = scipy.ndimage.map_coordinates(img, xyzRefToMov.T, order=interpolationOrder)
    movAligned_ = movAligned.reshape(imgLabelsRef.shape)

    # Cache transform to ensure other channels use the exact same transformation
    if not os.path.exists(fnameTransform):
        np.save(fnameTransform, xyzRefToMov)

    # Done!
    return movAligned_

# Submit task to Linux workers
def runResampleOnWorkerUnsafe(kwargs):
    # Get filenames
    fnameInput = kwargs['fnameInput']
    fnameOutput = kwargs['fnameOutput']

    # Imports
    import tifffile, scipy.ndimage, numpy as np

    # Load image
    img = tifffile.imread(fnameInput)

    # Use resolution info for rescaling if:
    # 1. This is the reference, OR
    # 2. This is the moving, but no point supervision is specified
    supervisedTransform = kwargs['supervisedTransform'] and ('mov_' in fnameOutput)

    #if 'INITIAL_AFFINE_MOV_TO_REF' not in kwargs or kwargs['INITIAL_AFFINE_MOV_TO_REF']:
    #    supervisedTransform = getSupervisedTransform(fnameInput, fnameOutput, kwargs['ref'], kwargs['mov'])

    if supervisedTransform:
        # Get output size
        #img2 = tifffile.imread(os.path.join(os.path.dirname(fnameOutput), 'ref_synapsin.iso.tif'))
        # Use nearest-neighbor resampling on label IDs
        interpolationOrder = 2
        if 'pointlabels.tif' in fnameInput.lower():
            interpolationOrder = 0
        # Run supervised transform
        imgT = runSupervisedTransform(img, fnameInput, fnameOutput, kwargs['ref'], kwargs['mov'], interpolationOrder)

        # Transform
        #imgT = scipy.ndimage.affine_transform(
        #    img, supervisedTransform, mode='nearest',
        #    order=interpolationOrder, output=img.dtype, output_shape=img2.shape)
    else:
        # Determine scaling
        resIn, resOut = kwargs['resInput'], kwargs['resOutput']

        resIn = np.array(resIn, dtype=np.float64).squeeze()
        resOut = np.array(resOut, dtype=np.float64).squeeze()

        if resIn.size == 1:
            resIn = np.repeat(resIn, 3)
        if resOut.size == 1:
            resOut = np.repeat(resOut, 3)

        scaleXYZ = resIn / resOut

        scaleZXY = np.hstack((scaleXYZ[2:], scaleXYZ[:2]))

        newshape = tuple((np.array(img.shape, dtype=np.float64) * scaleZXY).round().astype(int))

        # Create scaling transform
        mtx = np.identity(4, dtype=np.float64)
        mtx[0, 0] = 1.0 / scaleZXY[0]
        mtx[1, 1] = 1.0 / scaleZXY[1]
        mtx[2, 2] = 1.0 / scaleZXY[2]

        # Use nearest-neighbor resampling on label IDs
        interpolationOrder = 1
        if 'pointlabels.tif' in fnameInput.lower():
            interpolationOrder = 0

        imgT = scipy.ndimage.affine_transform(img, mtx, mode='nearest',
            order=interpolationOrder, output=img.dtype, output_shape=newshape)

    # Write
    tifffile.imwrite(fnameOutput, imgT, bigtiff=True)

def quote(fn):
    return '\"' + fn + '\"'

def findElastix():
    try:
        v = subprocess.check_output(['elastix', '--version'])
        print('Found Elastix! {}'.format(v.decode('ascii').strip()))
        return 'elastix.exe'
    except FileNotFoundError as e:
        fnameElastix = os.path.join(os.path.dirname(__file__), 'elastix_bin\\elastix.exe')
        if not os.path.exists(fnameElastix):
            raise Exception('Could not find the Elastix executable! Aborting...')
        v = subprocess.check_output([fnameElastix, '--version'])
        print('Found Elastix! {}'.format(v.decode('ascii').strip()))
        return fnameElastix

def findTransformix():
    try:
        v = subprocess.check_output(['transformix', '--version'])
        print('Found Transformix! {}'.format(v.decode('ascii').strip()))
        return 'transformix.exe'
    except FileNotFoundError as e:
        fnameTransformix = os.path.join(os.path.dirname(__file__), 'elastix_bin\\transformix.exe')
        if not os.path.exists(fnameTransformix):
            raise Exception('Could not find the Transformix executable! Aborting...')
        v = subprocess.check_output([fnameTransformix, '--version'])
        print('Found Transformix! {}'.format(v.decode('ascii').strip()))
        return fnameTransformix

#@nb.njit(boundscheck=True)
def plotGaussian(arr, cx, cy, cz, s):
    pdft = scipy.stats.multivariate_t(loc=np.array([cx, cy, cz]), df=1, shape=s)
    xyzs = np.array(np.meshgrid(np.arange(arr.shape[0]), np.arange(
        arr.shape[1]), np.arange(arr.shape[2]))).reshape((3, -1)).T
    order = (2, 0, 1)
    tmp = pdft.pdf(xyzs).reshape((arr.shape[order[0]], arr.shape[order[1]], arr.shape[order[2]]))
    arr[:, :, :] = np.moveaxis(tmp, order[::-1], (0, 1, 2))

def runAlignmentOnWorker(kwargs):
    # Re-try if Elastix fails
    for ntry in range(5):
        if runAlignmentOnWorkerTry(kwargs) is not None:
            break
        else:
            print('Elastix failed, re-trying (Try #{})'.format(ntry))

def runAlignmentOnWorkerTry(kwargs):
    # Ensure Elastix is available
    fnameElastix = findElastix()

    # Get parameters
    nameRefCache = kwargs['nameRefCache']
    nameMovCache = kwargs['nameMovCache']
    fnameRigidCache = kwargs['fnameRigidCache']
    fnameNonrigidCache = kwargs['fnameNonrigidCache']
    isManuallySupervised = False #kwargs['isManuallySupervised'] if 'isManuallySupervised' in kwargs else False

    # Is this a manually-supervised alignment?
    txtConfig = ''
    if fnameRigidCache is not None:
        with open(fnameRigidCache, 'r') as fConfig:
            txtConfig = fConfig.read()

    if '(Metric1Weight' in txtConfig:
        pass #isManuallySupervised = True

    if 'SUPERVISEDGRADIENT' in txtConfig:
        pass #isManuallySupervised = True

    # Load points, if so
    fnameRefLbl, fnameMovLbl = None, None
    if isManuallySupervised:
        fnameRefLbl = nameRefCache.replace('ref_synapsin.isopad.tif', 'ref_pointlabels.isopad.tif')
        fnameMovLbl = nameMovCache.replace('mov_synapsin.isopad.tif', 'mov_pointlabels.isopad.tif')

        if not os.path.exists(fnameRefLbl) or not os.path.exists(fnameMovLbl):
            isManuallySupervised= False

    ss = 1 # No subsampling (but useful for quicker debugging)

    fnamePtsRef, fnamePtsMov = None, None
    isManuallySupervised = False
    if isManuallySupervised:
        # Load labels
        labels = []
        for fnameLabels in [fnameRefLbl, fnameMovLbl]:
            # Load labels
            imgLabels = tifffile.imread(fnameLabels)[::ss, ::ss, ::ss]
            # Find regions
            props = regionprops(imgLabels)
            labelPos = {}
            labelSize = {}
            for prop in props:
                if prop.label in labelPos:
                    if prop.coords.shape[0] < labelSize[prop.label]:
                        continue
                _x, _y, _z = np.mean(prop.coords, axis=0).round().astype(int)
                #_x = float(_x) / imgLabels.shape[0]
                #_y = float(_y) / imgLabels.shape[1]
                #_z = float(_z) / imgLabels.shape[2]
                labelPos[prop.label] = (_x, _y, _z) # (_z, _x, _y)
                labelSize[prop.label] = prop.label
            labels.append(labelPos)
        # Only use indices that exist in both reference and moving
        labelIDs = np.intersect1d(list(labels[0].keys()), list(labels[1].keys()))
        # Write points files
        fnamePtsRef = fnameRefLbl.replace('.tif', '') + '_pts.txt'
        with open(fnamePtsRef, 'w') as f:
            f.write('index\n{}\n{}'.format(len(
                labelIDs), '\n'.join(['{},{},{}'.format(*labels[0][i]) for i in labelIDs])))
        fnamePtsMov = fnameMovLbl.replace('.tif', '') + '_pts.txt'
        with open(fnamePtsMov, 'w') as f:
            f.write('index\n{}\n{}'.format(len(
                labelIDs), '\n'.join(['{},{},{}'.format(*labels[1][i]) for i in labelIDs])))
        # Write gaussian supervised gradient
        for fnameLabels in [fnameRefLbl, fnameMovLbl]:
            fnameSupervisedGradientOut = fnameLabels.replace('.tif', '') + '_supervisedgradient.tif'
            if os.path.exists(fnameSupervisedGradientOut):
                continue
            # Status update
            print('Computing supervised gradient for {}'.format(fnameLabels))
            # Open real image
            imgSynapsin = tifffile.imread(fnameLabels.replace('_pointlabels', '_synapsin'))[::ss, ::ss, ::ss]
            imgSynapsin = imgSynapsin.astype(np.float32)
            imgSynapsin /= np.max(imgSynapsin)
            # Load labels
            imgLabels = tifffile.imread(fnameLabels)[::ss, ::ss, ::ss]
            # Find regions
            props = regionprops(imgLabels)
            labelPos = {}
            labelSize = {}
            for prop in props:
                if prop.label in labelPos:
                    if prop.coords.shape[0] < labelSize[prop.label]:
                        continue
                _x, _y, _z = np.mean(prop.coords, axis=0).round().astype(int)
                labelPos[prop.label] = (_x, _y, _z)  # (_z, _x, _y)
                labelSize[prop.label] = prop.label

            # Find distances
            labelDists = {}
            for label in labelIDs:
                labelDists[label] = []
                for label2 in labelPos:
                    if label2 != label:
                        labelDists[label].append(np.linalg.norm(np.array(
                            labelPos[label]) - np.array(labelPos[label2])))
                labelDists[label] = np.min(labelDists[label])

            # Create gradient image
            gradient = np.zeros(imgLabels.shape, dtype=np.float32)
            #gradient2= np.zeros(imgLabels.shape, dtype=np.float32)
            gradientAll = np.zeros(imgLabels.shape, dtype=np.float64)

            for k in list(labelDists.keys()):
                gradient[:, :, :] = 0
                plotGaussian(gradient, labelPos[k][0], labelPos[k][1], labelPos[k][2], labelDists[k] * 30.0) # 0.3
                gradient /= np.max(gradient)
                gradient = np.clip(gradient, 0.0, 1.0)
                gradientAll[:, :, :] += gradient
                #gradient2[:, :, :] = 0
                #plotGaussian(gradient2, labelPos[k][0], labelPos[k][1], labelPos[k][2], 3)
                #gradient2 /= np.sum(gradient2)
                #gradientAll[:, :, :] += 0.8 * gradient + 0.2 * gradient2

            gradientAll /= np.max(gradientAll)
            gradientAll = (gradientAll * 30000).astype(np.uint16)

            #gradientAll += (imgSynapsin * 5000).astype(np.uint16)
            #gradientAll = skimage.filters.gaussian(imgSynapsin, sigma=4)
            #gradientAll /= gradientAll.max()
            #gradientAll = (gradientAll * 5000).astype(np.uint16)

            tifffile.imwrite(fnameSupervisedGradientOut, gradientAll)

    # Create dir
    if isManuallySupervised:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_supervisedgradient')
    elif fnameRigidCache is None:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_nonrigid')
    elif fnameNonrigidCache is None:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid')
    else:
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix')
    os.makedirs(dirout, exist_ok=True)

    # Invoke elastix
    if isManuallySupervised:
        #popen = subprocess.Popen([fnameElastix,
        #     '-f', nameRefCache, '-m', nameMovCache,] + (
        #    ['-p', fnameRigidCache] if fnameRigidCache is not None else []) + (
        #    ['-p', fnameNonrigidCache] if fnameNonrigidCache is not None else []) + [
        #    '-out', dirout] + (['-fp', fnamePtsRef, '-mp', fnamePtsMov] if isManuallySupervised else  []),
        #    creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)

        fnameSupervisedGradientRef = fnameRefLbl.replace('.tif', '') + '_supervisedgradient.tif'
        fnameSupervisedGradientMov = fnameMovLbl.replace('.tif', '') + '_supervisedgradient.tif'
        popen = subprocess.Popen([fnameElastix,
             '-f', fnameSupervisedGradientRef, '-m', fnameSupervisedGradientMov,] + (
            ['-p', fnameRigidCache] if fnameRigidCache is not None else []) + (
            ['-p', fnameNonrigidCache] if fnameNonrigidCache is not None else []) + [
            '-out', dirout],
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        popen = subprocess.Popen([fnameElastix,
             '-f', nameRefCache, '-m', nameMovCache,] + (
            ['-p', fnameRigidCache] if fnameRigidCache is not None else []) + (
            ['-p', fnameNonrigidCache] if fnameNonrigidCache is not None else []) + [
            '-out', dirout],
            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)

    # Compress parameter files as they arrive
    fnamesTransforms = os.path.join(dirout, 'TransformParameters.*.*.It*.txt')
    fnamesTransformsConcat = os.path.join(dirout, 'TransformParameters.txt.gz')
    written = {}
    noresult = 0
    with pgzip.open(fnamesTransformsConcat, "wt", thread=10, blocksize=10**7) as fOut:
        while True:
            # Periodically add new transform files to zip archive (don't want 100k's of files in directory)
            time.sleep(10)
            found = glob.glob(fnamesTransforms)
            if len(found) == 0:
                # Does result.0.tif exist yet? If yes, wait and exit
                if os.path.exists(os.path.join(dirout, 'result.0.tif')) or os.path.exists(
                        os.path.join(dirout, 'result.tif')):
                    time.sleep(60)
                    # If Elastix already started (log exists) but we haven't seen the result pop up for Kx10 seconds,
                    # Elastix may have mysteriously crashed. Restart...
                    if os.path.exists(os.path.join(dirout, 'elastix.log')):
                        noresult += 1
                        if noresult >= 30:
                            # 5 minutes have elapsed with no new transform files... Re-start
                            return None
                    # Process seems to have been successful
                    break
            else:
                # Wait another 5 seconds to ensure the found files are fully flushed
                time.sleep(5)
                for fn in found:
                    if fn not in written:
                        it = int(re.search('(?<=It)[0-9]*', fn).group(0))
                        if (it % 50) == 0:
                            s = ''
                            with open(fn, 'r') as f:
                                s = ''.join(['=' for i in range(100)]) + '\n' + fn + '\n' + f.read()
                            fOut.write(s)
                            written[fn] = True
                    # Remove file
                    try:
                        os.remove(fn)
                    except:
                        break

    # Transform results if this was a supervised gradient run
    if isManuallySupervised:
        # Find transform
        fnameTransform = os.path.join(os.path.dirname(nameRefCache), 'elastix_supervisedgradient/TransformParameters.0.txt')
        # Apply this transform to the synapsin channel
        dirout = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid')
        runTransformOnWorker({
            'fileIn': nameMovCache,
            'fileOut': None,
            'fileTransform': fnameTransform,
            'dirResults': dirout
        })

    return fnamesTransformsConcat

def runAlignment(nameRefCache, nameMovCache, fnameRigidCache, fnameNonrigidCache):
    """

    :param nameRefCache:
    :param nameMovCache:
    :param fnameRigidCache:
    :param fnameNonrigidCache:
    :return:
    """
    from config import CONFIG

    # No-align specified for rigid or nonrigid?
    if 'NOALIGN' in fnameRigidCache:
        fnameRigidCache = 'NOALIGN'
    if 'NOALIGN' in fnameNonrigidCache:
        fnameNonrigidCache = 'NOALIGN'

    # Run rigid alignment
    # To-Do: Disable this by default if supervised is selected
    fnamesTransformsConcat1 = ''
    if fnameRigidCache != 'NOALIGN':
        fnamesTransformsConcat1 = runAlignmentOnWorker({
            'nameRefCache': nameRefCache,
            'nameMovCache': nameMovCache,
            'fnameRigidCache': fnameRigidCache,
            'fnameNonrigidCache': None,
            'isManuallySupervised': False
        })

    # TODO: We temporarily disable perturbation feature.
    if False:
        # Read rigid transformation file
        fnameRigid = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid\\result.0.tif')
        fnameRigidResultParams = fnameRigid.replace('result.0.tif', 'TransformParameters.0.txt')
        txtRigidResultParams = ''
        with open(fnameRigidResultParams, 'r') as f:
            txtRigidResultParams = f.read()

        # Reconstruct transform matrix
        import re, numpy as np
        params = np.array([float(x) for x in re.search('TransformParameters' + ''.join(
            ['\\s*([\\-\\.0-9]{1,})' for i in range(12)]), txtRigidResultParams).groups()], dtype=np.float64)
        assert len(params) == 12

        # For ordering see: https://elastix.lumc.nl/doxygen/classelastix_1_1AdvancedAffineTransformElastix.html
        mtx = np.identity(4, dtype=np.float64)
        mtx[0, :3] = params[0:3]
        mtx[1, :3] = params[3:6]
        mtx[2, :3] = params[6:9]
        mtx[:3, 3] = params[9:12]

        # Parse the intended perturbation matrix from the filename
        paramsPerturb = np.array([float(x) for x in re.search('.*_' + '='.join([
            '([\\.e\\-0-9]{1,})' for i in range(12)]), nameRefCache).groups()], dtype=np.float64)
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
        fnameRigidPerturb = os.path.join(os.path.dirname(fnameTransformPerturbed), 'result.tif')
        runTransformOnWorker({
            'fileIn': nameMovCache,
            'fileOut': fnameRigidPerturb,
            'fileTransform': fnameTransformPerturbed,
            'dirResults': os.path.dirname(fnameTransformPerturbed)
        })
    elif fnameRigidCache != 'NOALIGN':
        fnameRigidPerturb = os.path.join(os.path.dirname(nameRefCache), 'elastix_rigid\\result.tif')
        if not os.path.exists(fnameRigidPerturb):
            fnameRigidPerturb = fnameRigidPerturb.replace('result.tif', 'result.0.tif')
    else:
        print('SKIPPING RIGID PHASE.')
        fnameRigidPerturb = nameMovCache

    # Run nonrigid alignment
    if os.path.exists(fnameRigidPerturb):
        fnamesTransformsConcat2 = runAlignmentOnWorker({
            'nameRefCache': nameRefCache,
            'nameMovCache': fnameRigidPerturb,
            'fnameRigidCache': None,
            'fnameNonrigidCache': fnameNonrigidCache,
            'isManuallySupervised': False
        })
    else:
        # Run nonrigid alignment only, no rigid
        fnamesTransformsConcat2 = runAlignmentOnWorker({
            'nameRefCache': nameRefCache,
            'nameMovCache': nameMovCache,
            'fnameRigidCache': None,
            'fnameNonrigidCache': fnameNonrigidCache,
            'isManuallySupervised': False
        })

    # Copy files from cache to non-cache
    try:
        cacheSuffix = os.path.basename(os.path.dirname(os.path.dirname(os.path.relpath(
            fnamesTransformsConcat1, CONFIG.CACHE_ROOT())))).split('_')[-1]
        fnamesTransformsConcat1dst = os.path.join(CONFIG.DIRECTORY_ROOT, os.path.relpath(
            fnamesTransformsConcat1, CONFIG.CACHE_ROOT())).replace('_{}'.format(cacheSuffix), '')
        os.makedirs(os.path.dirname(fnamesTransformsConcat1dst), exist_ok=True)
    except:
        pass
    try:
        fnamesTransformsConcat2dst = os.path.join(CONFIG.DIRECTORY_ROOT, os.path.relpath(
            fnamesTransformsConcat2, CONFIG.CACHE_ROOT())).replace('_{}'.format(cacheSuffix), '')
        os.makedirs(os.path.dirname(fnamesTransformsConcat2dst), exist_ok=True)
    except:
        pass
    try:
        shutil.copy(fnamesTransformsConcat1, fnamesTransformsConcat1dst)
    except Exception as e:
        print(e)
    try:
        shutil.copy(fnamesTransformsConcat2, fnamesTransformsConcat2dst)
    except Exception as e:
        print(e)

def runTransformOnWorker(kwargs):
    # Find transformix
    fnameTransformix = findTransformix()

    # Get parameters
    fileIn = kwargs['fileIn']
    fileOut = kwargs['fileOut']
    fileTransform = kwargs['fileTransform']
    dirResults = kwargs['dirResults']

    os.makedirs(dirResults, exist_ok=True)

    # Invoke elastix
    output = ''
    popen = subprocess.Popen([fnameTransformix, '-in', fileIn, '-tp', fileTransform, '-out', dirResults],
                             stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        output += stdout_line
    print(output)

def runTransform(fnameInput, fnameTransform):
    # Apply perturbed transform to original image
    fnameOutput = os.path.join(os.path.dirname(fnameTransform), 'result.tif')
    runTransformOnWorker({
        'fileIn': fnameInput,
        'fileOut': fnameOutput,
        'fileTransform': fnameTransform,
        'dirResults': os.path.dirname(fnameTransform)
    })

def runResample(fnameInput, fnameOutput, resInput, resOutput, cacheSuffix, ref, mov, job):
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
    runResampleOnWorker({
        'fnameInput': fnameInputCache,
        'fnameOutput': fnameOutputCache,
        'resInput': resInput,
        'resOutput': resOutput,
        'ref': ref,
        'mov': mov,
        'INITIAL_AFFINE_MOV_TO_REF': job.INITIAL_AFFINE_MOV_TO_REF,
        'supervisedTransform': job.SUPERVISED_MOV_TO_REF
    })

    # Copy back
    shutil.copy(fnameOutputCache, fnameOutput)

# DEBUG
if __name__ == "__main__":
    img, fnameInput, fnameOutput, ref, mov, interpolationOrder = jl.load('C:/Users/GordusLab/Desktop/debug.pickle')
    runSupervisedTransform(img, fnameInput, fnameOutput, ref, mov, interpolationOrder)
