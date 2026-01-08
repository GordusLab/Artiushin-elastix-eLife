"""
This script looks for new jobs and executes them if not already present.
"""

import os, glob, configparser, logging, json, copy, re, numpy as np, pgzip, joblib as jl, \
    scipy.spatial.transform, string, random, lzma, shutil, tifffile, numba as nb, traceback, gc
from config import CONFIG
from tqdm import tqdm

class Job:
    """
    Class 'Job' contains and validates job-related properties.
    """
    def __init__(self):
        self.REFERENCE = None
        self.MOVING = None
        self.RESOLUTION = None
        self.PADDING = None

        self.RIGID_CONFIG = None
        self.RIGID_RESOLUTIONLEVELS = None
        self.RIGID_ITERATIONS = None
        self.RIGID_SPATIALSAMPLES = None

        self.NONRIGID_CONFIG = None
        self.NONRIGID_RESOLUTIONLEVELS = None
        self.NONRIGID_ITERATIONS = None
        self.NONRIGID_SPATIALSAMPLES = None

        self.RIGID_PERTURBATION_ROTATION_X = None
        self.RIGID_PERTURBATION_ROTATION_Y = None
        self.RIGID_PERTURBATION_ROTATION_Z = None
        self.RIGID_PERTURBATION_TRANSLATION_X = None
        self.RIGID_PERTURBATION_TRANSLATION_Y = None
        self.RIGID_PERTURBATION_TRANSLATION_Z = None
        self.RIGID_PERTURBATION_SCALE_X = None
        self.RIGID_PERTURBATION_SCALE_Y = None
        self.RIGID_PERTURBATION_SCALE_Z = None

        self.INITIAL_AFFINE_MOV_TO_REF = True

        self.SUPERVISED_MOV_TO_REF = True

    @staticmethod
    def tryReadConfig(c, s, p):
        try:
            return c[s][p]
        except:
            return None

    @staticmethod
    def fromFile(filename):
        """
        Read job properties from file.
        :param fn: Filename
        :return: Job properties
        """

        # Attempt to read configuration
        c = configparser.ConfigParser()
        try:
            c.read(filename)
        except Exception as e:
            logging.error('Skipping invalid job specification file: {}'.format(filename))
            return None

        # Create job object
        job = Job()

        # Read each property
        job.REFERENCE = job.tryReadConfig(c, 'GENERAL', 'REFERENCE')
        job.MOVING = job.tryReadConfig(c, 'GENERAL', 'MOVING')
        job.RESOLUTION = job.tryReadConfig(c, 'GENERAL', 'RESOLUTION')
        job.PADDING = job.tryReadConfig(c, 'GENERAL', 'PADDING')

        # Read rigid alignment properties
        job.RIGID_CONFIG = Job.tryReadConfig(c, 'RIGID', 'CONFIG')
        job.RIGID_RESOLUTIONLEVELS = Job.tryReadConfig(c, 'RIGID', 'RESOLUTIONLEVELS')
        job.RIGID_ITERATIONS = Job.tryReadConfig(c, 'RIGID', 'ITERATIONS')
        job.RIGID_SPATIALSAMPLES = Job.tryReadConfig(c, 'RIGID', 'SPATIALSAMPLES')

        # Read nonrigid alignment properties
        job.NONRIGID_CONFIG = Job.tryReadConfig(c, 'NONRIGID', 'CONFIG')
        job.NONRIGID_RESOLUTIONLEVELS = Job.tryReadConfig(c, 'NONRIGID', 'RESOLUTIONLEVELS')
        job.NONRIGID_ITERATIONS = Job.tryReadConfig(c, 'NONRIGID', 'ITERATIONS')
        job.NONRIGID_SPATIALSAMPLES = Job.tryReadConfig(c, 'NONRIGID', 'SPATIALSAMPLES')

        # Read perturbations
        job.RIGID_PERTURBATION_ROTATION_X = Job.tryReadConfig(c, 'PERTURBATIONS', 'ROTATION_X')
        job.RIGID_PERTURBATION_ROTATION_Y = Job.tryReadConfig(c, 'PERTURBATIONS', 'ROTATION_Y')
        job.RIGID_PERTURBATION_ROTATION_Z = Job.tryReadConfig(c, 'PERTURBATIONS', 'ROTATION_Z')
        job.RIGID_PERTURBATION_TRANSLATION_X = Job.tryReadConfig(c, 'PERTURBATIONS', 'TRANSLATION_X')
        job.RIGID_PERTURBATION_TRANSLATION_Y = Job.tryReadConfig(c, 'PERTURBATIONS', 'TRANSLATION_Y')
        job.RIGID_PERTURBATION_TRANSLATION_Z = Job.tryReadConfig(c, 'PERTURBATIONS', 'TRANSLATION_Z')
        job.RIGID_PERTURBATION_SCALE_X = Job.tryReadConfig(c, 'PERTURBATIONS', 'SCALE_X')
        job.RIGID_PERTURBATION_SCALE_Y = Job.tryReadConfig(c, 'PERTURBATIONS', 'SCALE_Y')
        job.RIGID_PERTURBATION_SCALE_Z = Job.tryReadConfig(c, 'PERTURBATIONS', 'SCALE_Z')

        # Misc. settings
        job.INITIAL_AFFINE_MOV_TO_REF = Job.tryReadConfig(c, 'GENERAL', 'INITIAL_AFFINE_MOV_TO_REF')
        job.SUPERVISED_MOV_TO_REF = Job.tryReadConfig(c, 'GENERAL', 'SUPERVISED_MOV_TO_REF')

        # Validate settings
        if job.validate():
            return job
        else:
            return None

    def validate(self):
        """
        Validate settings (convert strings to numeric values where relevant).
        :return: True if job is valid, False otherwise
        """
        # Can we locate the refence image?
        ref = findSample(self.REFERENCE)
        if ref is None:
            logging.warning('Could not find reference image {}'.format(self.REFERENCE))
            return False

        # Can we locate the moving image?
        mov = findSample(self.MOVING)
        if mov is None:
            logging.warning('Could not find reference image {}'.format(self.MOVING))
            return False

        # Locate config files
        configRigid = findAlignmentConfig(self.RIGID_CONFIG)
        if configRigid is None:
            logging.warning('Could not find alignment config {}'.format(configRigid))
            return False

        # Locate config files
        configNonrigid = findAlignmentConfig(self.NONRIGID_CONFIG)
        if configNonrigid is None:
            logging.warning('Could not find alignment config {}'.format(configNonrigid))
            return False

        # Interpret numeric/array values
        self.RESOLUTION = self.parseNumeric('RESOLUTION')
        self.PADDING = self.parseNumeric('PADDING')

        self.RIGID_RESOLUTIONLEVELS = self.parseNumeric('RIGID_RESOLUTIONLEVELS')
        self.RIGID_ITERATIONS = self.parseNumeric('RIGID_ITERATIONS')
        self.RIGID_SPATIALSAMPLES = self.parseNumeric('RIGID_SPATIALSAMPLES')

        self.NONRIGID_RESOLUTIONLEVELS = self.parseNumeric('NONRIGID_RESOLUTIONLEVELS')
        self.NONRIGID_ITERATIONS = self.parseNumeric('NONRIGID_ITERATIONS')
        self.NONRIGID_SPATIALSAMPLES = self.parseNumeric('NONRIGID_SPATIALSAMPLES')

        self.RIGID_PERTURBATION_ROTATION_X = self.parseNumeric('RIGID_PERTURBATION_ROTATION_X')
        self.RIGID_PERTURBATION_ROTATION_Y = self.parseNumeric('RIGID_PERTURBATION_ROTATION_Y')
        self.RIGID_PERTURBATION_ROTATION_Z = self.parseNumeric('RIGID_PERTURBATION_ROTATION_Z')
        self.RIGID_PERTURBATION_TRANSLATION_X = self.parseNumeric('RIGID_PERTURBATION_TRANSLATION_X')
        self.RIGID_PERTURBATION_TRANSLATION_Y = self.parseNumeric('RIGID_PERTURBATION_TRANSLATION_Y')
        self.RIGID_PERTURBATION_TRANSLATION_Z = self.parseNumeric('RIGID_PERTURBATION_TRANSLATION_Z')
        self.RIGID_PERTURBATION_SCALE_X = self.parseNumeric('RIGID_PERTURBATION_SCALE_X')
        self.RIGID_PERTURBATION_SCALE_Y = self.parseNumeric('RIGID_PERTURBATION_SCALE_Y')
        self.RIGID_PERTURBATION_SCALE_Z = self.parseNumeric('RIGID_PERTURBATION_SCALE_Z')

        # Misc.
        self.INITIAL_AFFINE_MOV_TO_REF = self.parseBoolean('INITIAL_AFFINE_MOV_TO_REF', False)
        self.SUPERVISED_MOV_TO_REF = self.parseBoolean('SUPERVISED_MOV_TO_REF', False)

        # Valid!
        return True

    def isSingleJob(self):
        """
        Does this job specify a single alignment? Or a range of perturbations?
        :return: True if this job specifies a single alignment only, False otherwise
        """
        if isinstance(self.RIGID_PERTURBATION_ROTATION_X, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_ROTATION_Y, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_ROTATION_Z, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_TRANSLATION_X, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_TRANSLATION_Y, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_TRANSLATION_Z, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_SCALE_X, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_SCALE_Y, (list, tuple)) or \
                isinstance(self.RIGID_PERTURBATION_SCALE_Z, (list, tuple)):
            return False
        else:
            return True

    def getSubJobs(self):
        """
        Return the sub-jobs that make up this job (iterate over perturbations)
        :return: List of sub-jobs, or this job itself.
        """
        if self.isSingleJob():
            return [self, ]
        else:
            subjobs = []

            rotX = np.array([self.RIGID_PERTURBATION_ROTATION_X, ]).flatten()
            rotY = np.array([self.RIGID_PERTURBATION_ROTATION_Y, ]).flatten()
            rotZ = np.array([self.RIGID_PERTURBATION_ROTATION_Z, ]).flatten()

            trX = np.array([self.RIGID_PERTURBATION_TRANSLATION_X, ]).flatten()
            trY = np.array([self.RIGID_PERTURBATION_TRANSLATION_Y, ]).flatten()
            trZ = np.array([self.RIGID_PERTURBATION_TRANSLATION_Z, ]).flatten()

            scX = np.array([self.RIGID_PERTURBATION_SCALE_X, ]).flatten()
            scY = np.array([self.RIGID_PERTURBATION_SCALE_Y, ]).flatten()
            scZ = np.array([self.RIGID_PERTURBATION_SCALE_Z, ]).flatten()

            for rx in rotX:
                for ry in rotY:
                    for rz in rotZ:
                        for tx in trX:
                            for ty in trY:
                                for tz in trZ:
                                    for sx in scX:
                                        for sy in scY:
                                            for sz in scZ:
                                                job = copy.copy(self)

                                                job.RIGID_PERTURBATION_ROTATION_X = rx
                                                job.RIGID_PERTURBATION_ROTATION_Y = ry
                                                job.RIGID_PERTURBATION_ROTATION_Z = rz

                                                job.RIGID_PERTURBATION_TRANSLATION_X = tx
                                                job.RIGID_PERTURBATION_TRANSLATION_Y = ty
                                                job.RIGID_PERTURBATION_TRANSLATION_Z = tz

                                                job.RIGID_PERTURBATION_SCALE_X = sx
                                                job.RIGID_PERTURBATION_SCALE_Y = sy
                                                job.RIGID_PERTURBATION_SCALE_Z = sz

                                                subjobs.append(job)
            # Done!
            return subjobs

    def getPerturbationMatrixAsString(self):
        """
        Return this job's perturbation matrix as string
        :return: perturbation matrix in string format
        """
        # Obtain matrix
        mtx = self.getPerturbationMatrix()

        # Convert upper 3 rows and all 4 columsn to string (flatten by row)
        coords = mtx[0:3, :].flatten()

        # Convert numbers to most space-efficient string representation
        strs = []
        for c in coords:
            if abs(c) < 1e-5:
                strs.append('0')
            elif abs(c - 1) < 1e-5:
                strs.append('1')
            else:
                # Try representation with 5 decimal places
                dec5 = '{:.5f}'.format(c)
                if dec5.startswith('0.'):
                    dec5 = dec5[1:]

                # Try scientific notation
                scn = '{:.2e}'.format(c)
                for n in range(1, 10):
                    scn = scn.replace('e-0' + str(n), 'e-' + str(n))

                # Which is a better approximation of the number?
                dec5e = abs(float(dec5) - c)
                scne = abs(float(scn) - c)
                if dec5e <= scne:
                    strs.append(dec5)
                else:
                    strs.append(scn)

        # Concatenate
        return '='.join(strs)

    def getPerturbationMatrix(self):
        """
        Return this job's perturbation matrix.
        :return: perturbation matrix
        """
        if not self.isSingleJob():
            logging.error('Cannot return perturbation matrix for job containing sub-jobs.')
            return None

        # Create translation matrix
        mtxTranslation = np.identity(4, dtype=np.float64)
        mtxTranslation[0, 3] = self.RIGID_PERTURBATION_TRANSLATION_X
        mtxTranslation[1, 3] = self.RIGID_PERTURBATION_TRANSLATION_Y
        mtxTranslation[2, 3] = self.RIGID_PERTURBATION_TRANSLATION_Z

        # Create scaling matrix
        mtxScaling = np.identity(4, dtype=np.float64)
        mtxScaling[0, 0] = self.RIGID_PERTURBATION_SCALE_X
        mtxScaling[1, 1] = self.RIGID_PERTURBATION_SCALE_Y
        mtxScaling[2, 2] = self.RIGID_PERTURBATION_SCALE_Z

        # Create rotation matrix (X)
        mtxRotation = np.identity(4, dtype=np.float64)
        mtxRotation[:3, :3] = scipy.spatial.transform.Rotation.from_euler('xyz', [
            self.RIGID_PERTURBATION_ROTATION_X,
            self.RIGID_PERTURBATION_ROTATION_Y,
            self.RIGID_PERTURBATION_ROTATION_Z,
        ]).as_matrix()

        # Return affine matrix
        return mtxRotation @ (mtxScaling @ mtxTranslation)

    def getOutputDirectory(self):
        """
        Return the output directory
        :return: The output directory.
        """
        if not self.isSingleJob():
            logging.error('Cannot get output directory for job with multiple sub-jobs.')
            return None
        else:
            # Build directory path
            directory = os.path.join(CONFIG.DIRECTORY_ROOT, (
                'alignments\\{ref}_{mov}_{res}_{pad}_{rigconfig}_{nrigconfig}_{rigiter}_{nrigiter}_' +
                '{riglevels}_{nriglevels}_{rigsamples}_{nrigsamples}_{perturbation}\\').format(
                    ref = self.REFERENCE,
                    mov = self.MOVING,
                    res = self.RESOLUTION,
                    pad = self.PADDING,
                    rigconfig = self.RIGID_CONFIG,
                    rigiter = self.RIGID_ITERATIONS,
                    riglevels = self.RIGID_RESOLUTIONLEVELS,
                    rigsamples = self.RIGID_SPATIALSAMPLES,
                    nrigconfig = self.NONRIGID_CONFIG,
                    nrigiter = self.NONRIGID_ITERATIONS,
                    nriglevels = self.NONRIGID_RESOLUTIONLEVELS,
                    nrigsamples = self.NONRIGID_SPATIALSAMPLES,
                    perturbation = self.getPerturbationMatrixAsString()
                ))
            return directory

    def parseBoolean(self, attrname, default = False):
        if not hasattr(self, attrname):
            logging.warning('Could not find boolean value {}'.format(attrname))
            return default

        # Extract numeric values
        try:
            v = getattr(self, attrname).lower()
        except:
            return default

        if v == 'true':
            return True
        elif v == 'false':
            return False
        else:
            raise Exception('Could not parse Boolean config parameter {} with value {}'.format(
                attrname, getattr(self, attrname)))

    def parseNumeric(self, attrname):
        if not hasattr(self, attrname):
            logging.warning('Could not find numeric value {}'.format(attrname))
            return None

        # Extract numeric values
        v = getattr(self, attrname)
        vf = ''.join([x for x in v if re.match('[0-9\\}\\{\\,\\.]', x)])

        # Array?
        if '{' in vf and '}' in vf:
            vs = [x.strip() for x in vf.replace('{', '').replace('}', '').split(',')]
        else:
            vs = [vf.strip(), ]

        # Parse numbers
        try:
            vsf = [float(x) if '.' in x else int(x) for x in vs]
            return vsf if len(vsf) > 1 else vsf[0]
        except Exception as e:
            logging.warning('Could not parse numeric value(s) {}'.format(vsf))
            return None

class AlignmentConfig:
    def __init__(self, id):
        self.id = id
        self.filename = None

    @staticmethod
    def fromFile(id, fn):
        if not os.path.exists(fn):
            return None

        c = AlignmentConfig(id)
        c.filename = fn

        return c

    @staticmethod
    def fromNoAlign():
        c = AlignmentConfig('NOALIGN')
        c.filename = 'NOALIGN'

        return c

def findAlignmentConfig(id):
    """
    Find alignment config file by ID.
    :param id: The config file ID.
    :return: A AlignmentConfig object, if found.
    """
    # Don't run (e.g. rigid) alignment
    if id == '9999' or id == 9999:
        return AlignmentConfig.fromNoAlign()

    # Find paths to config
    fns = glob.glob(os.path.join(CONFIG.DIRECTORY_ROOT, 'configs\\{}_*.txt'.format(id)))

    # Found?
    if len(fns) == 0:
        logging.warning('Could not locate alignment config {}'.format(id))
        return None
    elif len(fns) > 1:
        logging.warning('Found more than one alignment config with ID {}'.format(id))
        return None
    else:
        return AlignmentConfig.fromFile(id, fns[0])


class Sample():
    def __init__(self, id, filename):
        self.id = id
        self.filename = filename
        self.metadata = {}

    def pathToSynapsin(self):
        """
        Return file path to synapsin channel
        :return: file path to synapsin channel
        """
        fn = os.path.join(os.path.dirname(self.filename), 'synapsin.tif')
        if not os.path.exists(fn):
            logging.error('Could not locate synapsin.tif for recording {} @ {}'.format(self.id, self.filename))
            return None
        else:
            return fn

    @property
    def resolution(self):
        """
        Obtain sample axis resolution
        :return: axis resolutions (3-tuple)
        """
        return np.array([self.metadata['resolution'][ax] for ax in ['x', 'y', 'z']], dtype=np.float64)

    @staticmethod
    def fromFile(id, filename):
        """
        Read sample from file, if exists.
        :param id: Sample ID
        :param filename: Metadata filename
        :return: Sample if found, otherwise None
        """
        if not os.path.exists(filename):
            return None

        sample = Sample(id, filename=filename)

        # Attempt to read metadata
        sample.metadata = {}
        with open(filename, 'r') as f:
            try:
                sample.metadata = json.load(f)
            except Exception as e:
                logging.warning('Could not read sample {} due to invalid metadata'.format(id))
                return None

        return sample

def findSample(id):
    """
    Find sample and return metadata, if exists.
    :param id: sample ID code
    :return: sample
    """
    # Find paths to sample
    fns = glob.glob(os.path.join(CONFIG.DIRECTORY_ROOT, 'data\\{}_*\\metadata.json'.format(id)))

    # Found?
    if len(fns) == 0:
        logging.warning('Could not locate sample {}'.format(id))
        return None
    elif len(fns) > 1:
        logging.warning('Found more than one sample with ID {}'.format(id))
        return None
    else:
        return Sample.fromFile(id, fns[0])

def findJobs():
    """
    Find and read job list.
    :return: List of jobs
    """

    from config import CONFIG

    # Find job files
    jobFiles = glob.glob(os.path.join(CONFIG.DIRECTORY_ROOT, 'jobs\\*.txt'))

    # Debug?
    if CONFIG.DEBUG_ONLY:
        jobFiles = [x for x in jobFiles if '_debug' in x]
    else:
        jobFiles = [x for x in jobFiles if '_debug' not in x]

    # Read each job
    jobs = [j for j in [Job.fromFile(x) for x in jobFiles] if j is not None]

    return jobs

def getFilesAllChannels(fnameSynapsin, prefix=''):
    # Get the dimensions of the synapsin file
    shape = tifffile.imread(fnameSynapsin).shape

    # Then add any other *.tif files to this list with matching dimensions
    fnamesCandidates = glob.glob(os.path.join(os.path.dirname(fnameSynapsin), '{}*.tif'.format(prefix)))

    fnames = []
    for fn in fnamesCandidates:
        if fn.lower().endswith('synapsin.tif'):
            fnames.append(fn)
        else:
            try:
                shape2 = tifffile.imread(fn).shape
                if shape2 == shape:
                    fnames.append(fn)
            except:
                pass

    return fnames

def resample(job, cacheSuffix):
    """
    Resample samples for job
    :param job: Job specifications
    :return: None
    """
    from config import CONFIG
    from run_workers import runResample
    # Find samples
    ref = findSample(job.REFERENCE)
    mov = findSample(job.MOVING)
    # Get output directory
    directory = job.getOutputDirectory()
    # Resample mov-related files
    fnames = getFilesAllChannels(mov.pathToSynapsin())
    jl.Parallel(n_jobs=1)(jl.delayed(runResample)(x,
        os.path.join(directory, 'mov_{}.iso.tif'.format(os.path.basename(x).replace('.tif', ''))),
        mov.resolution, job.RESOLUTION, cacheSuffix, ref.pathToSynapsin(), mov.pathToSynapsin(), job) for x in [
            z for z in fnames if 'synapsin' in z])
    # Resample ref-related files
    fnames = getFilesAllChannels(ref.pathToSynapsin())
    jl.Parallel(n_jobs=6)(jl.delayed(runResample)(x,
        os.path.join(directory, 'ref_{}.iso.tif'.format(os.path.basename(x).replace('.tif', ''))),
        ref.resolution, job.RESOLUTION, cacheSuffix, ref.pathToSynapsin(), mov.pathToSynapsin(), job) for x in fnames)
    fnames = getFilesAllChannels(mov.pathToSynapsin())
    jl.Parallel(n_jobs=6)(jl.delayed(runResample)(x,
        os.path.join(directory, 'mov_{}.iso.tif'.format(os.path.basename(x).replace('.tif', ''))),
        mov.resolution, job.RESOLUTION, cacheSuffix, ref.pathToSynapsin(), mov.pathToSynapsin(), job) for x in [
        z for z in fnames if 'synapsin' not in z])

def mergeTransformFiles(fnameRefCache, subdir):
    fnamesTransforms = glob.glob(os.path.join(os.path.dirname(
        fnameRefCache), subdir + '\\TransformParameters.*.*.It*.txt'))
    fnamesTransformsConcat = os.path.join(os.path.dirname(fnameRefCache), subdir + '\\TransformParameters.txt.gz')
    if not os.path.exists(fnamesTransformsConcat):
        with pgzip.open(fnamesTransformsConcat, "wt", thread=40, blocksize=10**8) as fOut:
            for fn in tqdm(fnamesTransforms, desc='Merging transform files...'):
                it = int(re.search('(?<=It)[0-9]*', fn).group(0))
                if (it % 50) == 0:
                    s = ''
                    with open(fn, 'r') as f:
                        s = ''.join(['=' for i in range(100)]) + '\n' + fn + '\n' + f.read()
                    fOut.write(s)
        # Remove original transform files
        for fn in fnamesTransforms:
            os.remove(fn)

def align(job, cacheSuffix):
    """
    Run pairwise alignment.
    :param subjob:
    :param cacheSuffix:
    :return:
    """
    from config import CONFIG
    if CONFIG.USE_CLEARMAP:
        from run_workers_clearmap import runAlignment
    else:
        from run_workers import runAlignment
    # Find samples
    ref = findSample(job.REFERENCE)
    mov = findSample(job.MOVING)
    # Get output directory
    directory = job.getOutputDirectory()
    # Get reference and moving paths in cache directory
    dirname = os.path.basename(directory)
    if len(dirname) == 0:
        dirname = os.path.basename(os.path.dirname(directory))
    directoryCache = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', dirname + '_{}'.format(cacheSuffix))
    ref = findSample(job.REFERENCE)
    mov = findSample(job.MOVING)
    basenameRef = 'ref_{}.isopad.tif'.format(os.path.basename(ref.pathToSynapsin()).replace('.tif', ''))
    basenameMov = 'mov_{}.isopad.tif'.format(os.path.basename(mov.pathToSynapsin()).replace('.tif', ''))
    fnameRefCache = os.path.join(directoryCache, basenameRef)
    fnameMovCache = os.path.join(directoryCache, basenameMov)
    # Locate config files
    configRigid = findAlignmentConfig(job.RIGID_CONFIG)
    if configRigid is None:
        logging.warning('Could not find alignment config {}'.format(configRigid))
        return False
    configNonrigid = findAlignmentConfig(job.NONRIGID_CONFIG)
    if configNonrigid is None:
        logging.warning('Could not find alignment config {}'.format(configNonrigid))
        return False
    fnameRigidCache = os.path.join(directoryCache, os.path.basename(configRigid.filename.replace('.txt', '') + '.adj.txt'))
    fnameNonrigidCache = os.path.join(directoryCache, os.path.basename(configNonrigid.filename.replace('.txt', '') + '.adj.txt'))
    # Resample reference
    runAlignment(fnameRefCache, fnameMovCache, fnameRigidCache, fnameNonrigidCache)
    # Merge all transformation outputs into single file
    # Sync cache directory to non-cache directory
    for subdir in ['elastix_rigid', 'elastix_nonrigid']:
        mergeTransformFiles(fnameRefCache, subdir)
    cacheSuffix = '_' + os.path.basename(os.path.dirname(fnameRigidCache)).split('_')[-1]
    for subdir in ['elastix_rigid', 'elastix_nonrigid', 'elastix_rigid_perturbed', 'elastix_supervisedgradient']:
        if os.path.exists(os.path.join(os.path.dirname(fnameRefCache), subdir)):
            for folderName, subfolders, filenames in os.walk(os.path.join(os.path.dirname(fnameRefCache), subdir)):
                for filename in filenames:
                    try:
                        oldpath = os.path.join(folderName, filename)
                        relpath = os.path.relpath(oldpath, CONFIG.CACHE_ROOT())
                        newpath = os.path.join(CONFIG.DIRECTORY_ROOT, relpath).replace(cacheSuffix, '')
                        os.makedirs(os.path.dirname(newpath), exist_ok=True)
                        shutil.copy(oldpath, newpath)
                    except Exception as e:
                        print(e)

def runPadding(fnameInputRef, fnameOutputRef, padding):
    # Load
    import tifffile
    img = tifffile.imread(fnameInputRef)
    # Match mean intensity to a set value and clip at 2^15 - 1
    if 'pointlabels' in fnameInputRef:
        imgRescaled = img.copy().astype(np.uint16)
    else:
        imgRescaled = np.maximum(img.astype(np.float32), 1)
        imgRescaled /= (np.percentile(imgRescaled, 99.9) / 16384)
        imgRescaled = np.clip(imgRescaled, 1, 32767).astype(np.uint16)
    # Pad
    imgPadded = np.zeros((
        img.shape[0] + 2 * padding,
        img.shape[1] + 2 * padding,
        img.shape[2] + 2 * padding), dtype=img.dtype)
    imgPadded[
        padding:imgPadded.shape[0] - padding,
        padding:imgPadded.shape[1] - padding,
        padding:imgPadded.shape[2] - padding] = imgRescaled
    # Store
    tifffile.imsave(fnameOutputRef, imgPadded, bigtiff=True)

def preparePadding(job, cacheSuffix):
    """
    Create padded sample after resampling.
    :param subjob: job specifications
    :return:
    """
    import shutil
    # Get output directory
    directory = job.getOutputDirectory()
    # Find samples
    ref = findSample(job.REFERENCE)
    mov = findSample(job.MOVING)
    # Pad files
    for type in ['mov', 'ref']:
        for fn in getFilesAllChannels((ref if type=='ref' else mov).pathToSynapsin()):
            fnameInputRef = os.path.join(directory, '{}_{}.iso.tif'.format(type,
                os.path.basename(fn).replace('.tif', '')))
            fnameOutputRef = os.path.join(directory, '{}_{}.isopad.tif'.format(type,
                os.path.basename(fn).replace('.tif', '')))
            runPadding(fnameInputRef, fnameOutputRef, job.PADDING)
            # Copy to cache (ref)
            fnameOutputRefCache = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', os.path.basename(
                os.path.dirname(fnameOutputRef)) + '_{}'.format(cacheSuffix), os.path.basename(fnameOutputRef))
            shutil.copy(fnameOutputRef, fnameOutputRefCache)

def prepareAlignmentConfigs(job, cacheSuffix):
    """
    Copy alignment configs (rigid + nonrigid) to
    :param subjob:
    :return:
    """
    # Locate config files
    configRigid = findAlignmentConfig(job.RIGID_CONFIG)
    if configRigid is None:
        logging.warning('Could not find alignment config {}'.format(configRigid))
        return False

    # Locate config files
    configNonrigid = findAlignmentConfig(job.NONRIGID_CONFIG)
    if configNonrigid is None:
        logging.warning('Could not find alignment config {}'.format(configNonrigid))
        return False

    # Copy both configs to the local directory
    fnameRigid = os.path.join(job.getOutputDirectory(), os.path.basename(configRigid.filename))
    fnameNonrigid = os.path.join(job.getOutputDirectory(), os.path.basename(configNonrigid.filename))

    import shutil
    if os.path.exists(configRigid.filename):
        shutil.copy(configRigid.filename, fnameRigid)
    if os.path.exists(configNonrigid.filename):
        shutil.copy(configNonrigid.filename, fnameNonrigid)

    # Load config file (rigid)
    txtRigid = ''
    if 'NOALIGN' not in fnameRigid:
        with open(fnameRigid, 'r') as f:
            txtRigid = f.read()

    # Load config file (rigid)
    txtNonrigid = ''
    if 'NOALIGN' not in fnameNonrigid:
        with open(fnameNonrigid, 'r') as f:
            txtNonrigid = f.read()

    # Overwrite setting: rigid
    for key, value in [
            ('MaximumNumberOfIterations', job.RIGID_ITERATIONS),
            ('NumberOfSpatialSamples', job.RIGID_SPATIALSAMPLES),
            ('NumberOfResolutions', job.RIGID_RESOLUTIONLEVELS),
            ('WriteTransformParametersEachIteration', '"false"'),
            ('WriteTransformParametersEachIterationInterval', '100'),
            ('ResultImageFormat', '"tif"')]:
        nstr, nsubs = re.subn('(?<='+key+'\W)\W*[\\"a-zA-Z0-9]*', str(value), txtRigid)
        if nsubs == 0:
            txtRigid += '\n({} {})\n'.format(key, value)
        elif nsubs > 1:
            logging.error('Error in rigid alignment file: >1 instances of "{}"'.format(key))
            return None
        else:
            txtRigid = nstr

    # Overwrite setting: nonrigid
    for key, value in [
            ('MaximumNumberOfIterations', job.NONRIGID_ITERATIONS),
            ('NumberOfSpatialSamples', job.NONRIGID_SPATIALSAMPLES),
            ('NumberOfResolutions', job.NONRIGID_RESOLUTIONLEVELS),
            ('WriteTransformParametersEachIteration', '"false"'),
            ('WriteTransformParametersEachIterationInterval', '100'),
            ('ResultImageFormat', '"tif"')]:
        nstr, nsubs = re.subn('(?<='+key+'\W)\W*[\\"a-zA-Z0-9]*', str(value), txtNonrigid)
        if nsubs == 0:
            txtNonrigid += '\n({} {})\n'.format(key, value)
        elif nsubs > 1:
            logging.error('Error in rigid alignment file: >1 instances of "{}"'.format(key))
            return None
        else:
            txtNonrigid = nstr

    # Store config file (rigid)
    fnameRigidAlt = fnameRigid.replace('.txt', '') + '.adj.txt'
    with open(fnameRigidAlt, 'w') as f:
        f.write(txtRigid)

    # Store config file (rigid)
    fnameNonrigidAlt = fnameNonrigid.replace('.txt', '') + '.adj.txt'
    with open(fnameNonrigidAlt, 'w') as f:
        f.write(txtNonrigid)

    # Copy to cache
    fnameRigidAltCache = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', os.path.basename(
        os.path.dirname(fnameRigidAlt)) + '_{}'.format(cacheSuffix), os.path.basename(fnameRigidAlt))
    fnameNonrigidAltCache = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', os.path.basename(
        os.path.dirname(fnameNonrigidAlt)) + '_{}'.format(cacheSuffix), os.path.basename(fnameNonrigidAlt))

    os.makedirs(os.path.dirname(fnameRigidAltCache), exist_ok=True)
    shutil.copy(fnameRigidAlt, fnameRigidAltCache)
    shutil.copy(fnameNonrigidAlt, fnameNonrigidAltCache)

def runJob(job):
    """
    Run the specified job.
    :param job: job specification
    :return: None
    """
    # Get sub-jobs
    for subjob in job.getSubJobs():
        # Create job directory
        dirJob = subjob.getOutputDirectory()
        try:
            # Create unique cache suffix
            cacheSuffix = ''.join(random.choice(
                string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(10))
            #cacheSuffix = 'LcIn6eloFR' # DEBUG TMP
            # Ensure this directory exists
            os.makedirs(dirJob, exist_ok=True)
            # Prepare config files
            prepareAlignmentConfigs(subjob, cacheSuffix)
            # Resample both samples
            resample(subjob, cacheSuffix)
            # Pad
            preparePadding(subjob, cacheSuffix)
            # Run alignment
            align(subjob, cacheSuffix)
            # Apply final transform to other channels
            applyFinalTransformToAllChannels(subjob, cacheSuffix)
            # Compute mutual information scores at every 1k iterations
            computeMetrics(dirJob, cacheSuffix)
            # Normalize
            normalizeLocally(dirJob, cacheSuffix)
            # Export to zarr
            runExportToZarr(dirJob)
        except Exception as e:
            print('Error encountered in subjob for directory: {}'.format(dirJob))
            traceback.print_exc()

def applyFinalTransformToAllChannels(job, cacheSuffix):
    mov = findSample(job.MOVING)
    fnameSynapsin = os.path.join(job.getOutputDirectory(), 'mov_{}'.format(
        os.path.basename(mov.pathToSynapsin()).replace('.tif', '.isopad.tif')))
    fnames = getFilesAllChannels(fnameSynapsin, prefix='mov_')
    # Run transforms on each
    from run_workers import runTransform
    for fnameInput in fnames:
        # Create new directory
        basename = '_'.join(os.path.basename(fnameInput).replace('.tif', '').replace('.isopad', '').split('_')[1:])
        dirout = os.path.join(job.getOutputDirectory(), 'elastix_{}'.format(basename))
        os.makedirs(dirout, exist_ok=True)
        # Copy phase-1 transform file to temp directory
        fnameTransform = os.path.join(os.path.dirname(fnameInput), 'elastix_supervisedgradient/TransformParameters.0.txt')
        if not os.path.exists(fnameTransform):
            #fnameTransform = os.path.join(os.path.dirname(fnameInput), 'elastix_rigid/TransformParameters.0.txt')
            fnameTransform = os.path.join(os.path.dirname(fnameInput), 'elastix_nonrigid/TransformParameters.0.txt')
        fnameTransformDst0 = os.path.join(dirout, 'TransformParameters.0.txt')
        shutil.copy(fnameTransform, fnameTransformDst0)
        # Copy phase-2 transform file to temp directory
        #fnameTransform = os.path.join(os.path.dirname(fnameInput), 'elastix_nonrigid/TransformParameters.0.txt')
        fnameTransform = os.path.join(os.path.dirname(fnameInput), 'elastix_rigid/TransformParameters.0.txt')
        fnameTransformDst = os.path.join(dirout, 'TransformParameters.1.txt')
        with open(fnameTransform, 'r') as fIn:
            with open(fnameTransformDst, 'w') as fOut:
                fTxt = fIn.read()
                fTxt = fTxt.replace('(InitialTransformParametersFileName "NoInitialTransform")',
                    '(InitialTransformParametersFileName "{}")'.format(fnameTransformDst0))
                fOut.write(fTxt)
        # Run transform
        runTransform(fnameInput, fnameTransformDst)

@nb.njit(boundscheck=True)
def computeJointIntensityHistogram(imgRef, imgAlg, bins):
    hist2d = np.zeros((bins, bins), dtype=np.int64)
    if imgRef.shape[0] == imgAlg.shape[0] and \
            imgRef.shape[1] == imgAlg.shape[1] and \
            imgRef.shape[2] == imgAlg.shape[2]:
        for x in range(imgRef.shape[0]):
            for y in range(imgRef.shape[1]):
                for z in range(imgRef.shape[2]):
                    v0 = max(0, min(255, int(imgRef[x, y, z] / (65536/bins))))
                    v1 = max(0, min(255, int(imgAlg[x, y, z] / (65536/bins))))
                    hist2d[v0, v1] += 1
    return hist2d

def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def computeMetric(file1, file2):
    imgRef = tifffile.imread(file1)
    imgAlg = tifffile.imread(file2)
    assert imgAlg.shape == imgRef.shape

    scores = []
    for zstartf, zendf in [
            (0.000, 1.000),
            (0.000, 0.125),
            (0.125, 0.250),
            (0.375, 0.500),
            (0.500, 0.625),
            (0.625, 0.750),
            (0.750, 0.875),
            (0.875, 1.000)]:

        zs = int(0.5 + zstartf * imgRef.shape[0])
        ze = int(0.5 + zendf * imgRef.shape[0])

        A = imgRef[zs:ze, :, :].reshape(-1)
        B = imgAlg[zs:ze, :, :].reshape(-1)
        p = np.corrcoef(A, B)
        if np.any(np.isnan(p)):
            scores.append(np.nan)
        else:
            N = A.size
            Nbins = int(np.round((1 / np.sqrt(2)) * np.power(1 + np.sqrt(1 + (24 * N) / (1 - p[0, 1] ** 2)), 0.5)))

            h = computeJointIntensityHistogram(imgRef[zs:ze, :, :], imgAlg[zs:ze, :, :], Nbins)
            score = mutual_information(h)

            scores.append(score)

    return scores

@nb.njit(boundscheck=True, nogil=True)
def gaussianBlurSlice(img, z, std):
    if std >= 25:
        raise Exception('Std >= 25 not allowed, will take excessive time with current implementation.')

    r = std

    blurred = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)

    n0 = int(img.shape[0]/4) + 2
    n1 = int(img.shape[1]/4) + 2
    n2 = int(img.shape[2]/4) + 2
    img4x = np.zeros((n0, n1, n2), dtype=np.float32)
    for bz in range(0, img.shape[0], 4):
        for bx in range(0, img.shape[1], 4):
            for by in range(0, img.shape[2], 4):
                img4x[bz // 4, bx // 4, by // 4] = np.mean(img[bz:(bz + 4), bx:(bx + 4), by:(by + 4)])

    kernel = np.zeros((2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=np.float32)
    for ax in range(-r, r + 1):
        for ay in range(-r, r + 1):
            for az in range(-r, r + 1):
                kernel[r + ax, r + ay, r + az] = np.exp(-(ax ** 2 + ay ** 2 + az ** 2) / (2 * std ** 2))
    kernel = kernel / np.sum(kernel)

    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[2]):
            for z2 in range(max(0, min(img.shape[0] - 1, z - r)), max(0, min(img.shape[0] - 1, z + r)), 4):
                for x2 in range(max(0, min(img.shape[1] - 1, x - r)), max(0, min(img.shape[1] - 1, x + r)), 4):
                    for y2 in range(max(0, min(img.shape[2] - 1, y - r)), max(0, min(img.shape[2] - 1, y + r)), 4):
                        ix, iy, iz = r + (x2 - x), r + (y2 - y), r + (z2 - z)
                        blurred[x, y] += kernel[iz, ix, iy] * float(
                            img4x[int(0.5 + 0.25 * z2), int(0.5 + 0.25 * x2), int(0.5 + 0.25 * y2)])
    return blurred

def gaussianBlurSliceWorker(fname, z, std):
    img = tifffile.imread(fname)
    gc.collect()
    return [gaussianBlurSlice(img, _z, std) for _z in range(z, min(img.shape[0], z + 10))]

def gaussianBlur(fname, std):
    img = tifffile.imread(fname)
    gc.collect()
    slices = jl.Parallel(n_jobs=40)(jl.delayed(gaussianBlurSliceWorker)(
        fname, z, std) for z in tqdm(range(0, img.shape[0], 10)))
    blurred = np.concatenate(slices, axis=0)
    return blurred

def computeMetrics(directory, cacheSuffix):
    """
    Compute alignment quality metrics.

    See:
      - https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information/484724#484724
      - https://stats.stackexchange.com/questions/197499/what-is-the-best-way-to-decide-bin-size-for-computing-entropy-or-mutual-informat

    :param job:
    :return:
    """
    # For every 1k iterations, transform the sample
    for dirtype in ['elastix_rigid', 'elastix_nonrigid']:
        fnameTransformParams = os.path.join(directory, dirtype, 'TransformParameters.txt')
        if not os.path.exists(fnameTransformParams):
            fnameTransformParams = os.path.join(directory, dirtype, 'TransformParameters.txt.lzma')
        if not os.path.exists(fnameTransformParams):
            fnameTransformParams = os.path.join(directory, dirtype, 'TransformParameters.txt.gz')
        txt = ''
        # Determine max iterations
        maxiter = -1
        with tqdm() as pbar:
            with (open(fnameTransformParams, 'r') if fnameTransformParams.endswith(
                    '.txt') else (pgzip.open(fnameTransformParams, "rt", thread=16) if fnameTransformParams.endswith(
                    '.gz') else lzma.open(fnameTransformParams, 'rt'))) as f:
                consecEmpty = 0
                while True:
                    try:
                        newline = f.readline()
                    except:
                        break
                    if len(newline) > 0:
                        consecEmpty = 0
                    else:
                        consecEmpty += 1
                        if consecEmpty >= 100:
                            break
                    # Process
                    if '========================================' not in newline:
                        txt += newline
                    elif len(txt) > 0:
                        # Process text chunk
                        txts = txt.split('\n')
                        fn = txts[0]
                        it = int(re.search('(?<=It)[0-9]*', fn).group(0))
                        maxiter = max(maxiter, it)
                        txt = ''
                        pbar.update(1)
        # Build list of transforms to actually process
        contents = []
        with tqdm() as pbar:
            with (open(fnameTransformParams, 'r') if fnameTransformParams.endswith(
                    '.txt') else (pgzip.open(fnameTransformParams, "rt", thread=16) if fnameTransformParams.endswith(
                    '.gz') else lzma.open(fnameTransformParams, 'rt'))) as f:
                consecEmpty = 0
                while True:
                    try:
                        newline = f.readline()
                    except:
                        break
                    if len(newline) > 0:
                        consecEmpty = 0
                    else:
                        consecEmpty += 1
                        if consecEmpty >= 100:
                            break
                    # Process
                    if '========================================' not in newline:
                        txt += newline
                    elif len(txt) > 0:
                        # Process text chunk
                        txts = txt.split('\n')
                        fn = txts[0]
                        it = int(re.search('(?<=It)[0-9]*', fn).group(0))
                        if (it % 25000) == 0 or it == maxiter:
                            pbar.set_description(os.path.basename(fn))
                            contents.append((fn, '\n'.join(txts[1:])))
                        txt = ''
                        pbar.update(1)
        # Ensure temporary directory exists
        from config import CONFIG
        bn = os.path.basename(directory)
        if len(bn) == 0:
            bn = os.path.basename(os.path.dirname(directory))
        fnameTempTransform = os.path.join(CONFIG.CACHE_ROOT(), 'alignments', bn + '_{}'.format(
            cacheSuffix), 'elastix_temp/TransformParameters.0.txt')
        os.makedirs(os.path.dirname(fnameTempTransform), exist_ok=True)
        # Process each file in turn
        fnameLog = os.path.join(os.path.dirname(os.path.dirname(fnameTempTransform)), dirtype, 'metricslog.txt')
        with open(fnameLog, 'w') as flog:
            for itfn, c in reversed(contents):
                # Write to temporary transform file
                with open(fnameTempTransform, 'w') as ftranstemp:
                    ftranstemp.write(c)
                # Run transform
                from config import CONFIG
                if CONFIG.USE_CLEARMAP:
                    from run_workers_clearmap import runTransform
                else:
                    from run_workers import runTransform
                fnameInput = ''
                if dirtype == 'elastix_rigid':
                    fnameInput = os.path.join(os.path.dirname(os.path.dirname(
                        fnameTempTransform)), 'mov_synapsin.isopad.tif')
                elif dirtype == 'elastix_nonrigid':
                    # Look for previous stage output file
                    fnameInput = os.path.join(os.path.dirname(os.path.dirname(
                        fnameTempTransform)), 'elastix_rigid_perturbed\\result.tif')
                    if not os.path.exists(fnameInput):
                        fnameInput = os.path.join(os.path.dirname(os.path.dirname(
                            fnameTempTransform)), 'elastix_rigid_perturbed\\result.0.tif')
                    if not os.path.exists(fnameInput):
                        fnameInput = os.path.join(os.path.dirname(os.path.dirname(
                            fnameTempTransform)), 'elastix_rigid\\result.tif')
                    if not os.path.exists(fnameInput):
                        fnameInput = os.path.join(os.path.dirname(os.path.dirname(
                            fnameTempTransform)), 'elastix_rigid\\result.0.tif')
                # Run transform
                runTransform(fnameInput, fnameTempTransform)
                # Score alignment
                fnameRef = os.path.join(os.path.dirname(os.path.dirname(
                    fnameTempTransform)), 'ref_synapsin.isopad.tif')
                fnameTempResult = fnameTempTransform.replace('TransformParameters.0.txt', '') + 'result.tif'
                metric = computeMetric(fnameRef, fnameTempResult)
                flog.write('{} = {}\n'.format(itfn, metric))
                flog.flush()
    # Copy to non-cache
    fnameLog2 = os.path.join(directory, dirtype, 'metricslog.txt')
    shutil.copy(fnameLog, fnameLog2)
    pass

def normalizeLocally(directory, cacheSuffix):
    # Process rigid & nonrigid
    for mode in ['ref', 'rigid', 'nonrigid']:
        # Get file to blur
        fname = os.path.join(directory, 'ref_synapsin.isopad.tif')
        if mode == 'rigid':
            fname = os.path.join(directory, 'elastix_rigid\\result.0.tif')
        elif mode == 'nonrigid':
            fname = os.path.join(directory, 'elastix_nonrigid\\result.0.tif')
        # Apply blur
        img = tifffile.imread(fname)
        blurred = gaussianBlur(fname, 24)
        # Mask out edges
        fadeE = np.percentile(img[::2, ::2, ::2], 75)
        fadeS = np.percentile(img[::2, ::2, ::2], 50)
        mask = np.clip((img - fadeS) / (fadeE - fadeS), 0, 1.0)
        # Locally normalize image
        normalized = (img / (blurred + 1))
        normalized /= (np.percentile(normalized, 99.9) / 16384)
        normalized = np.clip(normalized * mask, 1, 32767).astype(np.uint16)
        # Save
        fnameOut = fname.replace('.tif', '') + '.normalized.tif'
        tifffile.imwrite(fnameOut, normalized, bigtiff=True)

def runExportToZarr(directory):
    """
    Export the (a) reference, (b) rigid and (c) nonrigid stacks to a Zarr file for easy inspection of the alignment result.
    :param directory: The (non-cache) directory
    :return:
    """
    import zarr
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image
    import ome_zarr, tifffile

    path = os.path.join(directory, 'zarr')
    os.mkdir(path)

    # Find normalized files
    fnames = (
        os.path.join(directory, 'ref_synapsin.isopad.normalized.tif'),
        os.path.join(directory, 'elastix_rigid\\result.0.normalized.tif'),
        os.path.join(directory, 'elastix_nonrigid\\result.0.normalized.tif')
    )

    # Construct 3-channel image
    fnameRef, fnameRigid, fnameNonrigid = fnames

    img1 = tifffile.imread(fnameRef)
    data = np.repeat(img1[np.newaxis, :, :, :], 3, axis=0)
    data[1, :, :, :] = tifffile.imread(fnameRigid)
    data[2, :, :, :] = tifffile.imread(fnameNonrigid)

    data = np.clip(data / (np.max(data)/255), 0, 255).astype(np.uint8)

    # write the image data
    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    sc = ome_zarr.scale.Scaler(method='nearest')
    write_image(image=data, scaler=sc, group=root, axes="czyx", storage_options=dict(chunks=(1, 4, 128, 128)))
    
def runJobs():
    """
    Execute any non-completed jobs
    :return: None
    """

    # Find new jobs
    jobs = findJobs()

    # Run jobs (TODO: Change n_jobs=1 -->3)
    jl.Parallel(n_jobs=3)(jl.delayed(runJob)(job) for job in jobs)

if __name__ == "__main__":
    """
    Entry Point
    """
    runJobs()
    #runExportToZarr('Z:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_25000_50000_8_8_20000_20000_1=0=0=0=0=1=0=0=0=0=1=0\\')
    #mergeTransformFiles('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_1=0=0=0=0=1=0=0=0=0=1=0_9OSgO3klsW\\ref_synapsin.isopad.tif', 'elastix_rigid')
    #mergeTransformFiles('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_1=0=0=0=0=1=0=0=0=0=1=0_9OSgO3klsW\\ref_synapsin.isopad.tif', 'elastix_nonrigid')
    #computeMetrics('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_1=0=0=0=0=1=0=0=0=0=1=0\\', '9OSgO3klsW')
    #mergeTransformFiles('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_.99875=-0.04998=0=0=.04998=.99875=0=0=0=0=1=0_KQyYVqkWIg\\ref_synapsin.isopad.tif', 'elastix_rigid')
    #mergeTransformFiles('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_.99875=-0.04998=0=0=.04998=.99875=0=0=0=0=1=0_KQyYVqkWIg\\ref_synapsin.isopad.tif', 'elastix_nonrigid')
    #computeMetrics('D:\\StandardBrain\\alignments\\BRAIN010w_BRAIN009w_0.667_30_00001_00002_20000_125000_5_5_20000_40000_.99875=-0.04998=0=0=.04998=.99875=0=0=0=0=1=0\\', 'KQyYVqkWIg')

