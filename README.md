# Spider Anatomy Pipeline

This repository contains the computational pipeline implementing the spider atlas creation process.

### Data organization

Data and configuration files are organized as follows:

#### [ROOT_DIRECTORY]/data/[SAMPLEID]_arbitrary_descriptor_suffix/metadata.json

Contains a metadata description for the unique sample stack contained in this directory. Example from *BRAIN009w_220413-water*:

```
{
    "resolution": {
        "x": 0.395,
        "y": 0.395,
        "z": 1.00
    },
    "brain": "22.4.13 - Ud_Synapsin rabbit 555",
    "ID": "BRAIN009w_220413-water",
    "date": "22_04_13",
    "objective": "20x",
    "antibody": "rabbit",
    "isclipped": "no"
}
```

#### [ROOT_DIRECTORY]/data/[SAMPLEID]_arbitrary_descriptor_suffix/synapsin.tif

'synapsin.tif' files are expected to be present for every sample stack as they are used for sample-to-sample alignment.

#### [ROOT_DIRECTORY]/data/[SAMPLEID]_arbitrary_descriptor_suffix/[arbitrary sample descriptor].tif

Additional *.tif files may be present in each sample directory, and will be interpreted as additional color channels 
corresponding to (i.e. sharing dimensions and coordinate system with) the 'synapsin.tif' stack.

#### [ROOT_DIRECTORY]/configs/[CONFIGID]_arbitrary_descriptor_suffix.txt

The 'configs' subdirectory contains alignment specifications for use by ClearMap2/Elastix. Every file should be preceeded by a unique CONFIGID prefix. 

Example file *00001_AFFINE.txt*:
```
//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "AdaptiveStochasticGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "AffineTransform")

(ErodeMask "true" )

(NumberOfResolutions 6)

(HowToCombineTransforms "Compose")
(AutomaticTransformInitialization "true")
(AutomaticScalesEstimation "true")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "true") 
(ShowExactMetricValue "false")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 100000 ) 

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 32 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 3)
(UseRandomSampleRegion "false")
(NumberOfSpatialSamples 4000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)

//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 20.0 )
```

Example file *00002_BSPLINE.txt*:
```
//ImageTypes
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)

//Components
(Registration "MultiResolutionRegistration")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Interpolator "BSplineInterpolator")
(Metric "AdvancedMattesMutualInformation")
(Optimizer "StandardGradientDescent")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(Transform "BSplineTransform")

(ErodeMask "false" )

(NumberOfResolutions 5)
(FinalGridSpacingInVoxels 25.000000 25.000000 25.000000)

(HowToCombineTransforms "Compose")

(WriteTransformParametersEachIteration "false")
(WriteResultImage "true")
(CompressResultImage "false")
(WriteResultImageAfterEachResolution "true")
(ShowExactMetricValue "false")
(WriteDiffusionFiles "true")

// Option supported in elastix 4.1:
(UseFastAndLowMemoryVersion "true")

//Maximum number of iterations in each resolution level:
(MaximumNumberOfIterations 100000 ) 

//Number of grey level bins in each resolution level:
(NumberOfHistogramBins 128 )
(FixedLimitRangeRatio 0.0)
(MovingLimitRangeRatio 0.0)
(FixedKernelBSplineOrder 3)
(MovingKernelBSplineOrder 3)

//Number of spatial samples used to compute the mutual information in each resolution level:
(ImageSampler "RandomCoordinate")
(FixedImageBSplineInterpolationOrder 1 )
(UseRandomSampleRegion "true")
(SampleRegionSize 50.0 50.0 50.0)
(NumberOfSpatialSamples 10000 )
(NewSamplesEveryIteration "true")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

//Order of B-Spline interpolation used in each resolution level:
(BSplineInterpolationOrder 3)

//Order of B-Spline interpolation used for applying the final deformation:
(FinalBSplineInterpolationOrder 3)

//Default pixel value for pixels that come from outside the picture:
(DefaultPixelValue 0)

//SP: Param_a in each resolution level. a_k = a/(A+k+1)^alpha
(SP_a 10000.0 )

//SP: Param_A in each resolution level. a_k = a/(A+k+1)^alpha
(SP_A 100.0 )

//SP: Param_alpha in each resolution level. a_k = a/(A+k+1)^alpha
(SP_alpha 0.6 )
```

Note that the following settings will be overridden by the pipeline:
- NumberOfSpatialSamples
- MaximumNumberOfIterations
- NumberOfResolutions
- WriteTransformParametersEachIteration (always "true")
- ResultImageFormat (always "tif")

#### [ROOT_DIRECTORY]/jobs/[JOB NAME].txt

The 'jobs' directory specifies alignment jobs to be executed. Adding jobs to this directory will [*eventually, not yet implemented*] 
result in the automatic execution of these jobs if and only if they have not already been executed.

Example job file:

```
[GENERAL]
REFERENCE = BRAIN009w
MOVING = BRAIN010w
RESOLUTION = 0.667 um/px
PADDING = 30 px

[RIGID]
CONFIG = 00001
RESOLUTIONLEVELS = 5
ITERATIONS = 50000
SPATIALSAMPLES = 20000

[NONRIGID]
CONFIG = 00002
RESOLUTIONLEVELS = 5
ITERATIONS = 50000
SPATIALSAMPLES = 20000

[PERTURBATIONS]
ROTATION_X = {0, 0.05, -0.05}
ROTATION_Y = 0
ROTATION_Z = {0, 0.05}
TRANSLATION_X = 0
TRANSLATION_Y = 0
TRANSLATION_Z = 0
SCALE_X = 1.0
SCALE_Y = 1.0
SCALE_Z = 1.0
```

Unit specifications, such as those present in the example file above, are optional and will be ignored. A combinatorial set of 
perturbations may be specified using curly brace notation. The example file above will lead to 3 * 1 * 2 = 6 repetitions of the 
alignment, each with a slight translation/rotation/scaling of the rigid alignment result. Such repeated alignments are thought 
to increase the likelihood of obtaining at least one high-quality alignment result.

#### [ROOT_DIRECTORY]/alignments/[REFID]_[MOVID]_[RES]_[PAD]_[RIGID_CONFIG]_[NONRIGID_CONFIG]_[RIGID_ITER]__[NONRIGID_ITER]_[RIGID_LEVELS]_[NONRIGID_LEVELS]_[RIGID_SAMPLES]_[NONRIGID_SAMPLES]_[PERTURBATION]/

The subdirectories within the 'alignments' folder contain the output of each of the alignment runs, and will be populated automatically by the pipeline.

The *[PERTURBATION]* substring contains 12 floating point numbers which make up the top 3 rows of a 4x4 affine transformation matrix. The output of the rigid phase will be transformed according to this matrix before this perturbed result is passed on to the nonrigid alignment phase.

## Post-alignment verification

For every alignment that completes, an alignment score will be computed between the specified reference volume and the moving image transformed according to the transformation state at the given iteration. This alignment metric will be computed every 5,000 iterations, as well as for the final iteration. This allows inspection of convergence as well as alignment quality rankings between repeated sample pairings. The alignment metric is currently computed according to the mutual information between both image stacks.
