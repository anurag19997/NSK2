import os
import sys
import numpy as np
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from animal_performance.spiketrain import SpatialSpikeTrain2D_signalstore, HaftingRateMap
from animal_performance.utils import compute_resize_ratio, speed2D, gaussian_smooth, peak_search, fit_ellipse, bin_width_to_bin_number, _gkern, disk_mask
import animal_performance.errors as err
from animal_performance.errors import ArgumentError, SpeedBandwidthError
import animal_performance.defaults as default
from animal_performance.spikes import place_field, binary_map
from animal_performance.spiketrain import Position2D, SpatialSpikeTrain2D_signalstore
from animal_performance.shuffle_spikes import shuffle_spikes
from openpyxl.utils.cell import get_column_letter
from openpyxl.worksheet.dimensions import ColumnDimension
from scipy.signal import convolve2d
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from skimage import transform
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import math
import cv2
# import Image
REQUIRED_OVERLAP_PIXELS = 0


def custom_flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    masked_rate_map.data[masked_rate_map.mask] = 0
    return  masked_rate_map.data

def _downsample(img, downsample_factor):
    downsampled = block_reduce(img, downsample_factor) 
    return downsampled

# Taken from https://stackoverflow.com/questions/59144828/opencv-getting-all-blob-pixels
#public
def map_blobs(rate_map, nofilter=False, **kwargs):

    '''
        Segments and labels firing fields in ratemap.

        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            tuple:
                image, n_labels, labels, centroids
            --------
            image (np.ndarray):
                Semi-processed image used for blob detection
            n_labels (np.ndarray):
                Array of blob numbers / ID's
            labels (np.ndarray):
                Segmented ratemap with each blob labelled
            centroids (np.ndarray):
                Array of coordinates for each blobs weighted centroid.
            field_sizes (list):
                List of size of each field as a percentage of map coverage
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        print("No smoothing factor provided, using default of 1.0")     
        smoothing_factor = 1.0

    ratemap = rate_map

    if 'downsample' in kwargs:
        if kwargs['downsample'] == True:
            ratemap = _downsample(ratemap, kwargs['downsample_factor'])

    if 'cylinder' in kwargs:
        cylinder = kwargs['cylinder']

        if len(ratemap[ratemap != ratemap]) > 0:
            # already disk masked just repalce nan with 0
            ratemap[ratemap != ratemap] = 0
        else:
            if cylinder:
                ratemap = custom_flat_disk_mask(ratemap)

    

    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)

    # Create kernel for convolutional smoothing
    # kernel = _gkern(26, 3)
    kernel = _gkern(kernlen,std)

    ratemap_copy = np.copy(ratemap)

    # Compute a 'low_noise' threshold where anything below the 10th percentile activity is removed
    low_noise = np.mean(ratemap_copy[ratemap_copy <= np.percentile(ratemap_copy, 20)])
    ratemap_copy[ratemap_copy <= np.percentile(ratemap_copy, 80)] = low_noise
    # threshold = 0.2 * np.max(ratemap_copy)
    # ratemap_copy[ratemap_copy <= threshold] = 0

    # Initial segmentation into blobs
    image = np.array(ratemap_copy * 255, dtype = np.uint8)
    thresh, blobs = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs, connectivity=4)

    if nofilter == False:
        # Filter through each blob, and remove any blob smaller than some threshold.
        for i in range(1, n_labels):
            num_pix = len(np.where(labels==i)[0])
            if num_pix <= (blobs.size * 0.01):
                blobs[np.where(labels==i)] = 0

        # Once smaller blobs are removed, re-smooth, and re-normalize image
        image[np.where(blobs == 0)] = 0
        image = image / max(image.flatten())
        image = cv2.filter2D(image,-1,kernel)
        image = image / max(image.flatten())
        image_2 = np.array(image * 255, dtype = np.uint8)
    else:
        image_2 = image

    # Second round of segmentation to acquire more clean and accurate blobs from pre-preocessed image
    thresh, blobs = cv2.threshold(image_2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blobs, connectivity=4)

    # Flip centroids to follow (x,y) convention
    if len(centroids) > 0:
        centroids = centroids[1:]
        centroids = np.fliplr(centroids)

    field_sizes = []
    for i in range(1, n_labels):
        field_sizes.append(( len(np.where(labels==i)[0]) / len(image_2.flatten()) ) * 100)

    map_blobs_dict = {'image': image, 'n_labels': n_labels, 'centroids': centroids, 'field_sizes': field_sizes}

    return image, n_labels, labels, centroids, field_sizes



def accumulate_spatial(spatial_object: Position2D | SpatialSpikeTrain2D_signalstore | np.ndarray, **kwargs):
    """
    Given a list of positions, create a histogram of those positions. The
    resulting histogram is typically referred to as a map.
    
    The complexity in this function comes down to selecting where the edges of
    the arena are, and generating the bins within those limits.
    
    The histogram bin edges must be defined in one of 3 different ways
    
        * bin_width: based on the keyword `arena_size`, the number of bins will
          be calculated as
            `opexebo.general.bin_width_to_bin_number`
          The histogram will use `num_bins` between the minimum and maximum of the
          positions (or `limit` if provided)
        * bin_number: the histogram will use bin_number of bins between the
          minimum and maximum of the positions (or `limit` if provided)
        * bin_edges: the histogram will use the provided `bin_edg`e arrays

    Either zero or one of the three bin_* keyword arguments must be defined.
    If none are defined, then a default bin_width is used. If more than 1 is
    defined, an error is raised

    Parameters
    ----------
    pos: np.ndarray
        1D or 2D array of positions  in row-major format, i.e. `x` = pos[0],
        `y` = pos[1]. This matches the simplest input creation pos = np.array( [`x`, `y`] )
    arena_size: float or tuple of floats
        Dimensions of arena (in cm)
            * For a linear track, length
            * For a circular arena, diameter
            * For a rectangular arena, length or (length, length)
    bin_width: float
        Bin size in cm. Default 2.5cm. If bin_width is supplied, `limit` must
        also be supplied. One of `bin_width`, `bin_number`, `bin_edges` must be
        provided
    bin_number: int or tuple of int
        Number of bins. If provided as a tuple, then `(x_bins, y_bins)`. One
        of `bin_width`, `bin_number`, `bin_edges` must be provided
    bin_edges: array-like
        Edges of the bins. Provided either as `edges` or `(x_edges, y_edges)`.
        One of `bin_width`, `bin_number`, `bin_edges` must be provided
    limits: tuple or np.ndarray
        (x_min, x_max) or (x_min, x_max, y_min, y_max)
        Provide concrete limits to the range over which the histogram searches
        Any observations outside these limits are discarded
        If no limits are provided, then use np.nanmin(data), np.nanmax(data)
        to generate default limits.
        As is standard in python, acceptable values include the lower bound
        and exclude the upper bound

    Returns
    -------
    hist: np.ndarray
        1D or 2D histogram of the occurrences of the input observations
        Dimensions given by arena_size/bin_width
        Not normalised - each cell gives the integer number of occurrences of
        the observation in that cell
    edges: list-like
        `x`, or (`x`, `y`), where `x`, `y` are 1d np.ndarrays
        Here `x`, `y` correspond to the output histogram
    """

    if isinstance(spatial_object, SpatialSpikeTrain2D_signalstore) or isinstance(spatial_object, Position2D):
        pos_x, pos_y, arena_size = spatial_object.x, spatial_object.y, spatial_object.arena_size
        pos = np.vstack((pos_x, pos_y)).T
    else:
        pos = spatial_object
        arena_size = kwargs.get("arena_size")

    dims = pos.ndim
    if dims not in (1, 2):
        raise ValueError("pos should have either 1 or 2 dimensions. You have"\
                         " provided %d dimensions." % dims)

    # Get kwargs values
    debug = kwargs.get("debug", False)
    limits = kwargs.get("limits", None)
    if not isinstance(limits, (tuple, list, np.ndarray, type(None))):
        raise ValueError("You must provide an array-like 'limits' value, e.g."\
          " (x_min, x_max, y_min, y_max). You provided type %s" % type(limits))

    arena_size, is_2d = _validatekeyword__arena_size(arena_size, dims)
    
    ###########################################################################
    ####### Handle the decision of bin_edges
    # Logic:
        # If none are provided, use default bin_width
        # If more than 1 is provided, raise Exception
        # If bin_edges are provided, use them
        # Else use bin_width, if provided
        # Else use bin_number, if provided
    bin_number = kwargs.get("bin_number", None)
    bin_width = kwargs.get("bin_width", None)
    bin_edges = kwargs.get("bin_edges", None)
    if bin_edges is None and bin_width is None and bin_number is None:
        # No bin decision was provided, so go with default
        bin_width = default.bin_width
        debug_bin_type = "default - bin_width"
    elif sum( [x is not None for x in (bin_edges, bin_width, bin_number)] ) != 1:
        # Count the number of values where a value other than False is present
        # If there are more than 1 "True" value, then the user has provided too many keywords
        raise KeyError("You have provided more than one method for determining"\
                       " the edges of the histogram. Only zero or one methods"\
                       " can be accepted.")

    if bin_edges is not None:
        # First priority: use predefined bin_edges
        # Remember - user provides (x, y), but histogram needs (y, x)
        if is_2d:
            if not isinstance(bin_edges, (tuple, list, np.ndarray)):
                raise ValueError("keyword 'bin_edges' must be either a tuple or list"\
                                 " (of np.ndarrays), or a 2D array")
        else:
            if not isinstance(bin_edges, np.ndarray):
                raise ValueError("Keyword 'bin_edges' must be a numpy array for a 1D histogram")
        bins = (bin_edges[1], bin_edges[0])
        debug_bin_type = "bin_edges"

    elif bool(bin_width):
        # Calculate the number of bins based on the requested width and arena_size
        # Then calculate the actual bin edges that this would give, based on expanding from top left
        # Given arena size in (x, y), this also returns in (x, y) -> have to convert for Numpy
        num_bins = bin_width_to_bin_number(arena_size, bin_width)
        if limits is None:
            # Handle the case that limits is not provided, i.e. is None
            if is_2d:
                lim = (np.nanmin(pos[0]), np.nanmax(pos[0]), np.nanmin(pos[1]), np.nanmax(pos[1]))
            else:
                lim = (np.nanmin(pos), np.nanmax(pos))
        else:
            lim = limits
        if is_2d:
            # Have to swap to (y, x)
            bins = (np.linspace(0, arena_size[1], num_bins[1]+1) + lim[2],
                    np.linspace(0, arena_size[0], num_bins[0]+1) + lim[0])
        else:
            bins = np.linspace(0, arena_size, num_bins+1) + (lim[0])
        debug_bin_type = "bin_width"

    elif bool(bin_number):
        if isinstance(bin_number, int):
            bins = bin_number
        elif isinstance(bin_number, (tuple, list, np.ndarray)):
            # If an array, we expect to recieve (x, y)
            # Have to convert for the Numpy standard of (y, x)
            bins = (bin_number[1], bin_number[0])
        else:
            raise ValueError("Keyword 'bin_number' must be an integer, or an"\
                             " array-like of integers.")
        debug_bin_type = "bin_number"

    if debug:
        print(f"Limits: {limits}")
        print(f"Binning type: {debug_bin_type}")
        print(f"bins : {bins}")

    ###########################################################################
    ###### Make the histogram
    if is_2d:
        x = pos[0]
        y = pos[1]
        if limits is None:
            limits = np.array([np.nanmin(y), np.nanmax(y)*1.001, np.nanmin(x),
                               np.nanmax(x)*1.001]).reshape(2, 2)
            # numpy convention: (y, x)
            if debug:
                print("No limits found. Calculating based on min/max")
        elif len(limits) != 4:
            raise ValueError("You must provide a 4-element 'limits' value for a"\
                             " 2D map. You provided %d elements" % len(limits))
        else:
            limits = np.flipud(np.array(limits).reshape(2, 2))
            # the flipud swaps x and y - rememebr, numpy convention that (y, x)
        in_range = np.logical_and(np.logical_and(np.greater_equal(y, limits[0, 0]), 
                                                 np.less(y, limits[0, 1])),
                                  np.logical_and(np.greater_equal(x, limits[1, 0]), 
                                                 np.less(x, limits[1, 1]))
                                  )
        # the simple operator ">= doesn't respect Masked Arrays
        # As of 2019, it does actually behave correctly (NaN is invalid and so
        # is removed), but I would prefer to be explicit

        in_range_x = x[in_range]
        in_range_y = y[in_range]
        if debug:
            print(f"data points : {len(in_range_x)}")

        '''
        A brief word on the documentation for np.histogram2d()
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
        
        The documentation is subtlely misleading in the use of `x` and `y`. 
        
        The NumPy standard notation is (almost) invariably to call (y, x), and, e.g.,
        in a 2D array, you would create an array with 5 rows and 2 columns like follows:
            np.zeros((5, 2))
        
        We would normally describe this as an array with a height (corresponding to y)
        of 5, and a width (corresponding to x) of 2.
        
        The documentation for histogram2d gives the signature as:
            numpy.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)
        
            Returns:
                H : ndarray, shape(nx, ny)
                    The bi-dimensional histogram of samples x and y. Values in x are 
                    histogrammed along the first dimension and values in y are histogrammed 
                    along the second dimension.
                xedges : ndarray, shape(nx+1,)
                    The bin edges along the first dimension.
                yedges : ndarray, shape(ny+1,)
                    The bin edges along the second dimension.
        
        Note the order in the Returns section: data given as `x` is histogrammed along the
        _first_ dimension, which in NumPy parlance, would _usually_ be labelled `y`.
        The use of `x`, `y` is self-consistent within this page, but misleading
        in the context of other NumPy functions. 
        
        By invoking `y` as the first axis, and then returning edges as [xedges, yedges],
        we are internally consistent with the mathematical notation that I have used throughout
        opexebo (i.e. to call x, then y)
        '''
        hist, yedges, xedges = np.histogram2d(in_range_y, in_range_x, bins=bins, range=limits)
        edges = [xedges, yedges]

    else: # is not 2d
        x = pos
        if limits is None:
            limits = [np.nanmin(x), np.nanmax(x)*1.0001]
        elif len(limits) != 2: 
            raise ValueError("You must provide a 2-element 'limits' value for a"\
                             " 1D map. You provided %d elements" % len(limits))
        in_range = np.logical_and(np.greater_equal(x, limits[0]),
                                  np.less(x, limits[1]))
        in_range_x = x[in_range]
        if debug:
            print(f"data points : {len(in_range_x)}")

        hist, edges = np.histogram(in_range_x, bins=bins, range=limits)

    return hist, edges


def _validatekeyword__arena_size(kwv, provided_dimensions):
    '''
    Decipher the possible meanings of the keyword "arena_size".
    
    "arena_size" is given to describe the arena in which the animal is moving
    It should be either a float, or an array-like of 2 floats (x, y)
    
    Parameters
    ----------
    kw: float or array-like of floats
        The value given for the keyword `arena_size`
    provided_dimensions : int
        the number of spatial dimensions provided to the original function.
        Acceptable values are 1 or 2
        E.g. if the original function was provided with positions = [t, x, y], then
        provided_dimensions=2 (x and y)

    Returns
    -------
    arena_size : float or np.ndarray of floats
    
    Raises
    ------
    ValueError
    IndexError
    '''
    if provided_dimensions == 1:
        is_2d = False
    elif provided_dimensions == 2:
        is_2d = True
    else:
        raise NotImplementedError("Only 1d and 2d arenas are supported. You"\
                                  " provided %dd" % provided_dimensions)
    if type(kwv) in (float, int, str):
        kwv = float(kwv)
        if kwv <= 0: 
            raise err.ArgumentError("Keyword 'arena_size' value must be greater than"\
                             " zero (value given %f)" % kwv)
        if is_2d:
            arena_size = np.array((kwv, kwv))
        else:
            arena_size = kwv
    elif type(kwv) in (list, tuple, np.ndarray):
        if len(kwv) == 1:
            if is_2d:
                
                arena_size = np.array(kwv[0], kwv[0])
            else:
                arena_size = kwv[0]
        elif len(kwv) == 2 and not is_2d:
            raise err.DimensionMismatchError("Mismatch in dimensions: 1d position data but 2d"\
                             " arena specified")
        elif len(kwv) not in [1, 2]:
            raise err.ArgumentError("Keyword 'arena_size' value is invalid. Provide"\
                             " either a float or a 2-element tuple")
        else:
            arena_size = np.array(kwv)
    else:
        raise err.ArgumentError("Keyword 'arena_size' value not understood. Please"\
                         " provide either a float or a tuple of 2 floats. Value"\
                         " provided: '%s'" % str(kwv))
    return arena_size, is_2d


def _validate_keyword_arena_shape(arena_shape):
    '''
    Ensure that the arena_shape is a meaningful value
    
    Parameters
    ----------
    arena_shape : str
        the value given for the keyword `arena_shape`
    
    Returns
    -------
    arena_shape : str
        A value that is guaranteed to be an acceptable member of one of the
        recognised groups of arena_shapes
    '''
    if not isinstance(arena_shape, str):
        raise err.ArgumentError("Keyword `arena_shape` must be a string, not type `{type(arena_shape)}`")
    else:
        arena_shape = arena_shape.lower()
    
    if arena_shape in default.shapes_square:
        # this is ok
        pass
    elif arena_shape in default.shapes_circle:
        # this is ok
        pass
    elif arena_shape in default.shapes_linear:
        # this is ok
        pass
    else:
        raise NotImplementedError(f"Arena shape '{arena_shape}' not implemented")
    
    return arena_shape
    
    

def _smooth(array: np.ndarray, window: int) -> np.ndarray:

    '''
        Smooths an array using sliding wsindow approach

        Params:
            array (np.ndarray):
                Array to be smoothed
            window (int):
                Number of points to be smoothed at a time as a sliding window

        Returns:
            np.ndarray:
                smoothed_array
    '''

    # Initialize empty array
    smoothed_array = np.zeros((len(array), 1))

    # Iterate over array using sliding window and compute averages
    for i in range(len(array) - window + 1):
        current_average = sum(array[i:i+window]) / window
        smoothed_array[i:i+window] = current_average

    return smoothed_array

def _get_head_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    '''
        Will compute the head direction angle of subject over experiemnt.
        Params:
            x, y (np.ndarray):
                Arrays of x and y coordinates.

        Returns:
            np.ndarray:
                angles
            --------
            angles:
                Array of head direction angles in radians, reflecting what heading the subject
                during the course of the sesison.
    '''

    last_point = [0,0]  # Keep track of the last x,y point
    last_angle = 0      # Keep track of the most previous computed angle
    angles = []         # Will accumulate angles as they are computed

    # Iterate over the x and y points
    for i in range(len(x)):

        # Grab the current point
        current_point = [float(x[i]), float(y[i])]

        # If the last point is the same as the current point
        if (last_point[0] == current_point[0]) and (last_point[1] == current_point[1]):
            # Retain the same angle
            angle = last_angle

        else:
            # Compute the arctan (i.e the angle formed by the line defined by both points
            # and the horizontal axis [range -180 to +180])
            # Uses the formula arctan( (y2-y1) / (x2-x1))
            Y = current_point[1] - last_point[1]
            X = current_point[0] - last_point[0]
            angle = math.atan2(Y,X) * (180/np.pi) # Convert to degrees

        # Append angle value to list
        angles.append(angle)

        # Update new angle and last_point values
        last_point[0] = current_point[0]
        last_point[1] = current_point[1]
        last_angle = angle

    # Scale angles between 0 to 360 rather than -180 to 180
    angles = np.array(angles)
    angles = (angles + 360) % 360

    # Convert to radians
    angles = angles * (np.pi/180)

    return angles

# def spatial_tuning_curve(x: np.ndarray, y: np.ndarray, t: np.ndarray, spike_times: np.ndarray, smoothing: int) -> tuple:
def spatial_tuning_curve(spike_times, pos_x, pos_y, pos_t, smoothing_factor) -> tuple:

    '''
        Compute a polar plot of the average directional firing of a neuron.

        Params:
            x, y, t (np.ndarray):
                Arrays of x and y coordinates, and timestamps
            spike_times (np.ndarray):
                Timestamps of when spike events occured
            smoothing (int):
                Smoothing factor for angle data

        Returns:
            tuple: tuned_data, spike_angles, ang_occ, bin_array
            --------
            tuned_data (np.ndarray):
                Tuning curve
            spike_angles (np.ndarray):
                Angles at which spike occured
            ang_occ (np.ndarray):
                Histogram of occupanices within bins of angles
            bin_array (np.ndarray):
                Bins of angles (360 split into 36 bins of width 10 degrees)
    '''
    spike_times = spike_times
    smoothing = smoothing_factor
    t = np.array(pos_t)
    x = np.array(pos_x)
    y = np.array(pos_y)

    # Compute head direction angles
    hd_angles = _get_head_direction(x, y)

    # Split angle range (0 to 360) into angle bins
    bin_array = np.linspace(0,2*np.pi,36)

    # Compute histogram of occupanices in each bin
    ang_occ = _angular_occupancy(t.flatten(), hd_angles.flatten(), bin_width=10)

    # Extract spike angles (i.e angles at which spikes occured)
    spike_angles = []
    for i in range(len(spike_times)):
        index = np.abs(t - spike_times[i]).argmin()
        spike_angles.append(hd_angles[index])

    spike_angles = np.array(spike_angles)
    spike_angles = spike_angles.flatten()

    # Compute tuning curve and smooth
    tuned_data = _opexebo_tuning_curve(ang_occ[0], spike_angles, bin_width=10)
    tuned_data_masked = np.copy(tuned_data)
    bin_array = bin_array
    tuned_data[tuned_data == np.nan] = 0
    tuned_data = _smooth(tuned_data,smoothing)

    dir_dict = {'tuned_data': tuned_data, 'spike_angles': spike_angles, 'ang_occ': ang_occ, 'bin_array': bin_array}



    return tuned_data, spike_angles, ang_occ, bin_array

""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

def _opexebo_tuning_curve(angular_occupancy, spike_angles, **kwargs):
    """Analogous to a RateMap - i.e. mapping spike activity to spatial position
    map spike rate as a function of angle

    Parameters
    ----------
    angular_occupancy : np.ma.MaskedArray
        unsmoothed histogram of time spent at each angular range
        Nx1 array, covering the range [0, 2pi] radians
        Masked at angles of zero occupancy
    spike_angles : np.ndarray
        Mx1 array, where the m'th value is the angle of the animal (in radians)
        associated with the m'th spike
    kwargs
        bin_width : float
            width of histogram bin in DEGREES
            Must match that used in calculating angular_occupancy
            In the case of a non-exact divisor of 360 deg, the bin size will be 
            shrunk to yield an integer bin number. 


    Returns
    -------
    tuning_curve : np.ma.MaskedArray
        unsmoothed array of firing rate as a function of angle
        Nx1 array

    Notes
    --------
    BNT.+analyses.turningcurve

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    """

    occ_ndim = angular_occupancy.ndim
    spk_ndim = spike_angles.ndim
    if occ_ndim != 1:
        raise ValueError("angular_occupancy must be a 1D array. You provided a"\
                         " %d dimensional array" % occ_ndim)
    if spk_ndim != 1:
        raise ValueError("spike_angles must be a 1D array. You provided a %d"\
                         " dimensional array" % spk_ndim)
    if np.nanmax(spike_angles) > 2*np.pi:
        raise Warning("Angles higher than 2pi detected. Please check that your"\
                      " spike_angle array is in radians. If it is in degrees,"\
                      " you can convert with 'np.radians(array)'")

    bin_width = kwargs.get("bin_width", default.bin_angle) # in degrees
    num_bins = bin_width_to_bin_number(360., bin_width) # This is for validation ONLY, the value num_bins here is not passed onwards
    if num_bins != angular_occupancy.size:
        raise ValueError("Keyword 'bin_width' must match the value used to"\
                         " generate angular_occupancy")
    #UNITS!!!
    # bin_width and arena_size need to be in the same units. 
    # As it happens, I hardcoded arena_size as 2pi -> convert bin_width to radians
    # limits and spike_angles need to be in the same units
    
    bin_width = np.radians(bin_width)

    spike_histogram, bin_edges = accumulate_spatial(spike_angles,
                arena_size=2*np.pi, limits=(0, 2*np.pi), bin_width=bin_width)

    tuning_curve = spike_histogram / (angular_occupancy + np.spacing(1))

    return tuning_curve

def _angular_occupancy(time, angle, **kwargs):
    '''
    Calculate angular occupancy from tracking angle and kwargs over (0,2*pi)

    Parameters
    ----------
    time : numpy.ndarray
        time stamps of angles in seconds
    angle : numpy array
        Head angle in radians
        Nx1 array
    bin_width : float, optional
        Width of histogram bin in degrees

    Returns
    -------
    masked_histogram : numpy masked array
        Angular histogram, masked at angles at which the animal was never 
        observed. A mask value of True means that the animal never occupied
        that angle. 
    coverage : float
        Fraction of the bins that the animal visited. In range [0, 1]
    bin_edges : list-like
        x, or (x, y), where x, y are 1d np.ndarrays
        Here x, y correspond to the output histogram
    
    Notes
    --------
    Copyright (C) 2019 by Simon Ball, Horst Obenhaus

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''
    if time.ndim != 1:
        raise err.ArgumentError("time must be provided as a 1D array. You provided %d"\
                         " dimensions" % time.ndim)
    if angle.ndim != 1:
        raise err.ArgumentError("angle must be provided as a 1D array. You provided %d"\
                         " dimensions" % angle.ndim)
    if time.size != angle.size:
        raise err.ArgumentError("Arrays 'time' and 'angle' must have the same number"\
                         f" of elements. You provided {time.size} and {angle.size}")
    if time.size == 0:
        raise err.ArgumentError("Zero length array provided when data expected")
    if np.nanmax(angle) > 2*np.pi:
        raise Warning("Angles greater than 2pi detected. Please check that your"\
                      " angle array is in radians. If it is in degrees, you can"\
                      " convert with 'np.radians(array)'")

    bin_width = kwargs.get('bin_width', default.bin_angle)
    bin_width = np.radians(bin_width)
    arena_size = 2*np.pi
    limits = (0, arena_size)

    angle_histogram, bin_edges = accumulate_spatial(angle, bin_width=bin_width, 
                                                arena_size=arena_size, limits=limits)
    masked_angle_histogram = np.ma.masked_where(angle_histogram==0, angle_histogram)
    
    # masked_angle_histogram is in units of frames. It needs to be converted to units of seconds
    frame_duration = np.mean(np.diff(time))
    masked_angle_seconds = masked_angle_histogram * frame_duration
    
    # Calculate the fractional coverage based on locations where the histogram
    # is zero. If all locations are  non-zero, then coverage is 1.0
    coverage = np.count_nonzero(angle_histogram) / masked_angle_seconds.size
    
    return masked_angle_seconds, coverage, bin_edges

def _moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]

def _get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = _moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = _moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out

# called by batch_process module only
# def hd_score(angles, window_size=23):
def hd_score(spike_angles, **kwargs):

    if 'window_size' in kwargs:
        window_size = kwargs['window_size']
    else:
        window_size = 23
    angles = spatial_tuning_data = spike_angles

    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis

    # IF THIS IS SLOW TRY NP.HISTOGRAM INSTEAD OF PLT.HIST
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = _get_rolling_sum(binned_hd, window=window_size)
    plt.close()
    return smooth_hd


def rate_map_stats(spatial_spike_train: SpatialSpikeTrain2D_signalstore, debug=False, ratemap=None, occmap=None, override=False):
    '''
    Calculate statistics of a rate map that depend on probability distribution
    function (PDF)
    
    Calculates information, sparsity and selectivity of a rate map. Calculations
    are done according to 1993 Skaggs et al. "An Information-Theoretic Approach
    to Deciphering the Hippocampal Code" paper. Another source of information is
    1996 Skaggs et al. paper called "Theta phase precession in hippocampal
    neuronal populations and the compression of temporal sequences".

    Coherence is calculated based on RU Muller, JL Kubie "The firing of
    hippocampal place cells predicts the future position of freely moving rats",
    Journal of Neuroscience, 1 December 1989, 9(12):4101-4110. The paper doesn't
    provide information about how to deal with border values which do not have
    8 well-defined neighbours. This function uses zero-padding technique.

    Parameters
    ----------
    rate_map: np.ma.MaskedArray
        Smoothed rate map: n x m array where cell value is the firing rate,
        masked at locations with low occupancy

    time_map: np.ma.MaskedArray
        time map: n x m array where the cell value is the time the animal spent
        in each cell, masked at locations with low occupancy
        Already smoothed

    Returns
    -------
    rms: dict
        spatial_information_rate: float
            information rate [bits/sec]
        spatial_information_content: float
            spatial information content [bits/spike]
        sparsity: float
            see relevant literature (above)
        selectivity: float
            see relevant literature (above)
        peak_rate: float
            peak firing rate of smoothed map [Hz]
        mean_rate: float
            mean firing rate of smoothed map [Hz]

    Notes
    -----
    BNT.+analyses.mapStatsPDF(map)
    
    BNT.+analyses.coherence(map)

    Copyright (C) 2019 by Simon Ball
    '''
    if override:
        rate_map = ratemap
        time_map = occmap
    else:
        rate_map, _ = spatial_spike_train.get_map('rate').get_rate_map()
        time_map = spatial_spike_train.get_map('occupancy').raw_map_data


    if type(rate_map) != np.ma.MaskedArray:
        rate_map = np.ma.masked_invalid(rate_map, copy=True)
    if type(time_map) != np.ma.MaskedArray:
        time_map = np.ma.masked_invalid(time_map, copy=True)

    duration = np.ma.sum(time_map)
    position_PDF = time_map / (duration + np.spacing(1)) 
    # Probability distribution of where the animal spent its time. 

    if debug:
        print("Duration = %ds" % duration)
        print("Masked locations: %d" % np.sum(rate_map.mask))
        print("Masked values: %s" % rate_map.data[rate_map.mask])

    sparsity = np.nan
    selectivity = np.nan
    inf_rate = np.nan
    inf_content = np.nan
    if rate_map.mask.all():
        # Currently, there is a bug in numpy that causes nanmean to fail
        # on fully masked arrays. This should be a pretty rare occurrence, though!
        rmap_mean = np.nan
        rmap_peak = np.nan
    else:
        rmap_mean = np.nanmean(rate_map)
        rmap_peak = np.nanmax(rate_map)

    mean_rate = np.ma.sum( rate_map * position_PDF )
    mean_rate_sq = np.ma.sum( np.ma.power(rate_map, 2) * position_PDF )

    max_rate = np.max(rate_map)

    if debug:
        print("mean rate: %.2fHz" % mean_rate)
        print("mean rate squared: %.2fHz^2" % mean_rate_sq)
        print("max rate: %.2fHz" % max_rate)


    if mean_rate_sq != 0:
        sparsity = mean_rate * mean_rate / mean_rate_sq

    if mean_rate != 0:
        selectivity = max_rate / mean_rate

        log_argument = rate_map / mean_rate
        log_argument[log_argument < 1] = 1
        if debug:
            print(log_argument.shape)
            #print("log argument: %.4f" % log_argument)
        inf_rate = np.ma.sum(position_PDF * rate_map * np.ma.log2(log_argument))
        inf_content = inf_rate / mean_rate



    return {"spatial_information_rate":inf_rate, "spatial_information_content":inf_content,
            "sparsity":sparsity, "selectivity":selectivity, "peak_rate":rmap_peak,
            "mean_rate":rmap_mean}



def autocorrelation(spatial_map: SpatialSpikeTrain2D_signalstore | HaftingRateMap, **kwargs):
    '''
        Compute the autocorrelation map from ratemap

        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.
            pos_x, pos_y (np.ndarray):
                Arrays  tracking x and y coordinates of subject movement
            arena_size (tuple):
                Width and length of tracking arena

        Returns:
            np.ndarray:
                autocorr_OPEXEBO
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = spatial_map.smoothing_factor

    if 'use_map_directly' in kwargs:
        if kwargs['use_map_directly']:
            ratemap = spatial_map
            arena_size = kwargs['arena_size']
    else:
        if isinstance(spatial_map, HaftingRateMap):
            ratemap, _ = spatial_map.get_rate_map(smoothing_factor)
        elif isinstance(spatial_map, SpatialSpikeTrain2D_signalstore):
            ratemap, _ = spatial_map.get_map('rate').get_rate_map(smoothing_factor)


        arena_size = spatial_map.arena_size

    x_resize, y_resize = compute_resize_ratio(arena_size)
    autocorr_OPEXEBO = opexebo_autocorrelation(ratemap)
    # autocorr_OPEXEBO = _interpolate_matrix(autocorr_OPEXEBO, cv2_interpolation_method=cv2.INTER_NEAREST) #_resize_numpy2D(autocorr_OPEXEBO, x_resize, y_resize)

    if isinstance(spatial_map, HaftingRateMap):
        spatial_map.spatial_spike_train.add_map_to_stats('autocorr', autocorr_OPEXEBO)
    elif isinstance(spatial_map, SpatialSpikeTrain2D_signalstore):
        spatial_map.add_map_to_stats('autocorr', autocorr_OPEXEBO)

    return autocorr_OPEXEBO


# def _resize_numpy2D(array: np.ndarray, x: int, y: int) -> np.ndarray:

#     '''
#         Resizes a numpy array.

#         Params:
#             array (numpy.ndarray):
#                 Numpy array to be resized
#             x (int):
#                 Resizing row number (length)
#             y (int):
#                 Resizing column number (width)

#         Returns:
#             array (numpy.ndarray): Resized array with new dimensions (array.shape = (x,y))
#     '''

#     array = Image.fromarray(array)
#     array = array.resize((x,y))
#     array = np.array(array)

#     return array





""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""



def opexebo_autocorrelation(firing_map):
    """Calculate 2D spatial autocorrelation of a firing map.

    Parameters
    ----------
    firing_map: np.ndarray
        NxM matrix, smoothed firing map. map is not necessary a numpy array.
        May contain NaNs.

    Returns
    -------
    acorr: np.ndarray
        Resulting correlation matrix, which is a 2D numpy array.

    See Also
    --------
    opexebo.general.normxcorr2_general

    Notes
    -----
    BNT.+analyses.autocorrelation

    Copyright (C) 2018 by Vadim Frolov
    """

    # overlap_amount is a parameter that is intentionally not exposed to
    # the outside world. This is because too many users depend on it and we
    # do not what everyone to use their own overlap value.
    # Should be a value in range [0, 1]
    overlap_amount = 0.8
    slices = []

    if type(firing_map) != np.ndarray:
        firing_map = np.array(firing_map)

    if firing_map.size == 0:
        return firing_map

    # make sure there are no NaNs in the firing_map
    firing_map = np.nan_to_num(firing_map)

    # get full autocorrelgramn
    aCorr = normxcorr2_general(firing_map)

    # we are only interested in a portion of the autocorrelogram. Since the values
    # on edges are too noise (due to the fact that very small amount of elements
    # are correlated).
    for i in range(firing_map.ndim):
        new_size = np.round(firing_map.shape[i] + firing_map.shape[i] * overlap_amount)
        if new_size % 2 == 0:
            new_size = new_size - 1
        offset = aCorr.shape[i] - new_size
        offset = np.round(offset/2 + 1)
        d0 = int(offset-1)
        d1 = int(aCorr.shape[i] - offset + 1)
        slices.append(slice(d0, d1))

    return aCorr[tuple(slices)]

"""
Python version of Matlab's normxcorr2_general

This is a Python adaption of code found at
https://se.mathworks.com/matlabcentral/fileexchange/29005-generalized-normalized-cross-correlation
Since we use it for autocorrelograms exclusively some input arguments of the
original function have been dropped.
"""


def normxcorr2_general(array):
    """Calculate spatial autocorrelation.

    Python implementation of the Matlab `generalized-normalized cross correlation`
    function, adapted by Vadim Frolov. Some generality was abandoned in the adaption
    as unnecessary for autocorrelogram calculation.

    For the original function, see https://se.mathworks.com/matlabcentral/fileexchange/29005-generalized-normalized-cross-correlation

    Parameters
    ----------
    array: NxM matrix
        firing array. array is not necessary a numpy array. Must not contain NaNs!

    Returns
    -------
    np.ndarray
        Resulting correlation matrix
    """

    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not np.sum(np.isfinite(array)) == array.size:
        raise ValueError("Input array contains NaN values.")


    A = _shift_data(array)
    T = _shift_data(array)

    number_of_overlap_pixels = _local_sum(np.ones(A.shape), T.shape[0], T.shape[1])

    local_sum_A = _local_sum(A, T.shape[0], T.shape[1])
    local_sum_A2 = _local_sum(A*A, T.shape[0], T.shape[1])

    # Note: diff_local_sums should be nonnegative, but it may have negative
    # values due to round off errors. Below, we use max to ensure the radicand
    # is nonnegative.
    diff_local_sums_A = (local_sum_A2 - np.power(local_sum_A, 2) / number_of_overlap_pixels)
    del local_sum_A2

    denom_A = np.maximum(diff_local_sums_A, 0)
    del diff_local_sums_A

    # Flip T in both dimensions so that its correlation can be more easily
    # handled.
    rotatedT = np.rot90(T, 2)
    local_sum_T = _local_sum(rotatedT, A.shape[0], A.shape[1])
    local_sum_T2 = _local_sum(rotatedT*rotatedT, A.shape[0], A.shape[1])
    del rotatedT

    diff_local_sums_T = (local_sum_T2 - np.power(local_sum_T, 2) / number_of_overlap_pixels)
    del local_sum_T2
    denom_T = np.maximum(diff_local_sums_T, 0)
    del diff_local_sums_T

    denom = np.sqrt(denom_T * denom_A)
    del denom_T, denom_A

    xcorr_TA = _xcorr2_fast(T, A)
    del A, T
    numerator = xcorr_TA - local_sum_A * local_sum_T / number_of_overlap_pixels
    del xcorr_TA, local_sum_A, local_sum_T

    # denom is the sqrt of the product of positive numbers so it must be
    # positive or zero.  Therefore, the only danger in dividing the numerator
    # by the denominator is when dividing by zero. We know denom_T~=0 from
    # input parsing; so denom is only zero where denom_A is zero, and in these
    # locations, C is also zero.
    C = np.zeros(numerator.shape)
    tol = 1000 * np.spacing(np.max(np.abs(denom)))
    i_nonzero = (denom > tol)
    C[i_nonzero] = numerator[i_nonzero] / denom[i_nonzero]
    del numerator, denom

    # Remove the border values since they result from calculations using very
    # few pixels and are thus statistically unstable.
    # By default, REQUIRED_OVERLAP_PIXELS = 0, so C is not modified.
    if REQUIRED_OVERLAP_PIXELS > np.max(number_of_overlap_pixels):
        raise ValueError("ERROR: REQUIRED_OVERLAP_PIXELS")

    C[number_of_overlap_pixels < REQUIRED_OVERLAP_PIXELS] = 0
    return C


def _xcorr2_fast(T, A):
    T_size = T.shape
    A_size = A.shape
    outsize = np.array(A.shape) + np.array(T.shape) - 1

    # Figure out when to use spatial domain vs. freq domain
    conv_time = _time_conv2(T_size, A_size)  # 1 conv2
    fft_time = 3*_time_fft2(outsize)  # 2 fft2 + 1 ifft2

    if conv_time < fft_time:
        cross_corr = convolve2d(np.rot90(T, 2), A)
    else:
        cross_corr = _freqxcorr(T, A, outsize)

    return cross_corr


def _freqxcorr(a, b, outsize):
    # Find the next largest size that is a multiple of a combination of 2, 3,
    # and/or 5.  This makes the FFT calculation much faster.
    optimalSize = np.zeros((2, 1))
    optimalSize[0] = _find_closest_valid_dimension(outsize[0])
    optimalSize[1] = _find_closest_valid_dimension(outsize[1])
    optimalSize = optimalSize.squeeze()
    optimalSize = optimalSize.astype(np.int32)

    # Calculate correlation in frequency domain
    rot_version = np.rot90(a, 2)
    Fa = np.fft.fft2(rot_version, s=(optimalSize[0], optimalSize[1]))
    Fb = np.fft.fft2(b, s=(optimalSize[0], optimalSize[1]))
    xcorr_ab = np.real(np.fft.ifft2(Fa * Fb))

    xcorr_ab = xcorr_ab[0:outsize[0], 0:outsize[1]]
    return xcorr_ab


def _time_conv2(obssize, refsize):
    # K was empirically calculated by the commented-out code above.
    K = 2.7e-8

    # convolution time = K*prod(obssize)*prod(refsize)
    time = K * np.prod(obssize) * np.prod(refsize)
    return time


def _time_fft2(outsize):
    # time a frequency domain convolution by timing two one-dimensional ffts

    R = outsize[0]
    S = outsize[1]

    # Tr = time_fft(R)
    # K_fft = Tr/(R*log(R))

    # K_fft was empirically calculated by the 2 commented-out lines above.
    K_fft = 3.3e-7
    Tr = K_fft * R * np.log(R)

    if S == R:
        Ts = Tr
    else:
        # Ts = time_fft(S)  % uncomment to estimate explicitly
        Ts = K_fft * S * np.log(S)

    time = S*Tr + R*Ts
    return time


def _local_sum(A, m, n):
    """
    This algorithm depends on precomputing running sums.

    If m, n are equal to the size of A, a faster method can be used for
    calculating the local sum.  Otherwise, the slower but more general method
    can be used.  The faster method is more than twice as fast and is also
    less memory intensive.

    As it is currently called (2021-04-12), the `else` case appears to never
    be invoked
    """
    if m == A.shape[0] and n == A.shape[1]:
        s = np.cumsum(A, axis=0)
        # secondPart = np.matlib.repmat(s[-1, :], m-1, 1) - s[0:-1, :]
        secondPart = np.tile(s[-1, :], (m-1, 1)) - s[0:-1, :]
        c = np.concatenate((s, secondPart), axis=0)
        s = np.cumsum(c, axis=1)
        del c
        lastColumn = s[:, -1].reshape((s.shape[0], 1))
        # secondPart = np.matlib.repmat(lastColumn, 1, n-1) - s[:, 0:-1]
        secondPart = np.tile(lastColumn, (1, n-1)) - s[:, 0:-1]

        local_sum_A = np.concatenate((s, secondPart), axis=1)
    else:
        # breal the padding into parts to save on memory
        B = np.zeros((A.shape[0] + 2*m, A.shape[1]))

#        B(m+1:m+size(A,1),:) = A
#        s = cumsum(B,1)
#        c = s(1+m:end-1,:)-s(1:end-m-1,:)
#        d = zeros(size(c,1),size(c,2)+2*n)
#        d(:,n+1:n+size(c,2)) = c
#        s = cumsum(d,2)
#        local_sum_A = s(:,1+n:end-1)-s(:,1:end-n-1)
        local_sum_A = 0

    return local_sum_A

# we assume that we only deal with float number
def _shift_data(A):
    """
    Convert array to type Float, and shift the data range to be greater than zero
    """
    B = A.astype(np.float64)

    if not np.issubdtype(A.dtype, np.unsignedinteger):
        min_B = np.min(B)
        if min_B < 0:
            B -= min_B
    return B


def _find_closest_valid_dimension(n):

    # Find the closest valid dimension above the desired dimension.  This
    # will be a combination of 2s, 3s, and 5s.

    # Incrementally add 1 to the size until
    # we reach a size that can be properly factored.
    new_number = n
    result = 0
    new_number -= 1
    while not result == 1:
        new_number += 1
        result = _factorize_number(new_number)

    return new_number


def _factorize_number(n):
    for ifac in np.array([2, 3, 5]):
        while np.fmod(n, ifac) == 0:
            n = n / ifac
    return n



def speed_score(spike_times, pos_x, pos_y, pos_t, **kwargs):
    '''
    Calculate Speed score.

    Speed score is a correlation between cell firing *rate* and animal speed.
    The Python version is based on BNT.+scripts.speedScore. At Edvard's request,
    both the 2015 and 2016 scores are calculated. The primary difference is how
    the speed smoothing is implemented

    Speed score originates in the following paper in Nature from Emiliano et al
    doi:10.1038/nature14622

    The original Matlab script implemented - but as far as I can tell, did not
    (by default) use, a Kalman filter for smoothing the animal speed. Since it is
    not the default behaviour, I have not (yet) added that Kalman filter to opexebo.
    Its addition is contingent on the score similarity to BNT.

    Summary:
        * The intention is to correlate (spike firing rate) with (animal speed)
        * Convert an N-length array of spike firing times into an M-length array
          of spike firing rates, where M is the same length as tracking times
        * Optional: gaussian_smooth firing rates
        * Optional: gaussian_smooth speeds
        * Optional: apply a bandpass filter to speeds
        * calculate the Pearson correlation coefficient between (speed), (firing rate)

    Parameters
    ----------
    spike_times: np.ndarray
        N-length array listing the times at which spikes occurred. [s]

    tracking_times: np.ndarray
        M-length array of time stamps of tracking frames

    tracking_speeds: np.ndarray
        M-length array of animal speeds at time stamps given in `tracking_times`

    Other Parameters
    ----------------
    bandpass: str
        Type of bandpass filter applied to animal speeds. Acceptable values
        are
            * `"none"` - No speed based filtering is applied
            * `"fixed"` - a fixed lower and upper speed bound are used, based
              on keywords `"lower_bound_speed"`, `"upper_bound_speed"`
            * `"adaptive"` - a fixed lower speed bound is used, based on
              keyword `"lower_bound_speed"`. An upper speed bound is determined
              based on keywords `"upper_bound_time"` and "`speed_bandwidth"`
        Default `none`
    lower_bound_speed: float
        Speed in [cm/s] used as the lower edge of the speed bandpass filter
        (`"fixed"` and `"adaptive"`). Default 2cm/s
    upper_bound_speed: float
        Speed in [cm/s] used as the upper edge of the speed bandpass filter
        (`"fixed"` only). Default 15cm/s
    upper_bound_time: float
        Duration in [s] used for determining the upper edge of the speed
        bandpass filter (`"adaptive"` only). Default 10s
    speed_bandwidth: float
        Range of speeds in [cm/s] used for determining the upper edge of the
        speed bandpass filter (`"adaptive"` only). Default 2cm/s
    sigma: float
        Standard deviation in [s] of Gaussian smoothing kernel for smoothing
        both speed and firing rate data. Default 0.5s
    debug: bool

    Returns
    -------
    scores: dict
        2015: float
        2016: float
            Variations on the speed score. '2015' is based on the code in the paper
            above, but additionally including an upper speed filter
            '2016' is a modification involving a slightly different approach to
            smoothing the firing rate data
    (lower_speed, upper_speed): list of floats
        Speed thresholds using in the bandpass filter
        Most useful in the case of the adaptive filter, because there is no
        other way to find out what was actually used.

    Notes
    -----
    BNT.+scripts.speedScore

    Copyright (C) 2019 by Simon Ball
    '''
    x, y, t = pos_x, pos_y, pos_t 
    spike_times = np.array(spike_times).flatten()
    tracking_times = np.array(pos_t).flatten()
    tracking_speeds = speed2D(x, y, t).squeeze().flatten()

    # Check that the provided arrays have correct dimensions
    if spike_times.ndim != 1:
        raise ArgumentError("spike_times must be an Nx1 array. You have provided"\
                         f" {spike_times.ndim} dimensions")
    elif tracking_times.ndim != 1:
        raise ArgumentError("tracking_times must be an Nx1 array. You have provided"\
                         f" {tracking_times.ndim} dimensions")
    elif tracking_speeds.ndim != 1:
        raise ArgumentError("tracking_speeds must be an Nx1 array. You have provided"\
                         f" {tracking_speeds.ndim} dimensions")
    if tracking_times.size != tracking_speeds.size:
        raise ArgumentError("tracking_times and tracking_speeds must be the same length")

    # Get kwargs values
    speed_bandwidth = kwargs.get('speed_bandwidth', default.speed_bandwidth)
    sigma_time = kwargs.get('sigma', default.sigma_time)
    upper_bound_time = kwargs.get('upper_bound_time', default.upper_bound_time) # Only used in "adaptive" filter
    lower_bound_speed = kwargs.get('lower_bound_speed', default.lower_bound_speed)
    upper_bound_speed = kwargs.get("upper_bound_speed", default.upper_bound_speed) # Only used in "fixed" filter
    bandpass_type = kwargs.get("bandpass", "none").lower()
    available_filters = ("none", "fixed", "adaptive")
    if bandpass_type not in available_filters:
        raise NotImplementedError(f"Bandpass tpye '{bandpass_type}' is not implemented."\
                                  f" Available types are {available_filters}.")
    debug = kwargs.get('debug', False)

    # Convert spike_times to spike firing rate
    sampling_rate = 1 / np.mean(np.diff(tracking_times))
    firing_rate = _spiketimes_to_spikerate(spike_times, tracking_times, sampling_rate)

    # Apply smoothing
    # Smoothing expects to be given a sigma in units [bins], so convert from real units to bins
    tracking_speeds_smoothed = gaussian_smooth(tracking_speeds, sigma_time * sampling_rate)
    firing_rate_smoothed = gaussian_smooth(firing_rate, sigma_time * sampling_rate)

    # Calculate the bandpass filter
    if debug:
        print(bandpass_type)
    if bandpass_type == "none":
        _filter = _bandpass_none(tracking_speeds_smoothed, **kwargs)
        lower_bound_speed = 0
        upper_bound_speed = np.inf
    elif bandpass_type == "fixed":
        _filter = _bandpass_fixed(tracking_speeds_smoothed, lower_bound_speed,
                                  upper_bound_speed, **kwargs)
    elif bandpass_type == "adaptive":
        _filter, upper_bound_speed = _bandpass_adaptive(tracking_speeds_smoothed, sampling_rate,
                                                        lower_bound_speed, upper_bound_time, speed_bandwidth, **kwargs)

    # Apply the filter to speeds
    speeds = tracking_speeds_smoothed[_filter]
        # The filter will be applied differently to rate based on which score version is wanted

    # Score 2016: apply bandpass filter to already-smoothed rate and then correlate
    rate = firing_rate_smoothed[_filter]
    speed_score_2016 = np.corrcoef(speeds, rate)[0, 1]

    # Score 2015: Filter rates first (by setting to NaN), and then gaussian_smooth and correlate
    # Reuse the same filtered_speeds as for 2016, but redefine filtered_rate
    rate = firing_rate[_filter]
    rate = gaussian_smooth(rate, sigma_time * sampling_rate)
    speed_score_2015 = np.corrcoef(speeds, rate)[0, 1]

    scores = {'2015': speed_score_2015, '2016': speed_score_2016}
    return scores, (lower_bound_speed, upper_bound_speed)



def _bandpass_adaptive(speed, sampling_rate, lower_speed, upper_time, speed_bw, **kwargs):
    '''Create a filter list that allows through values based on a defined lower
    value, and an upper value determined by the highest 2cm/s bandwidth at which
    the animal spent at least X time

    Calculating the upper speed:
        * Histogram the speed array with a resolution 10x higher than the
        desired speed-bandwidth
        * Iterate over the resulting histogram to identify the highest speed range
        at which the animal spends at least upper_time
        * select the centre of this speed range as the upper threshold

    parameters
    ----------
    speed : np.ndarray
        1d M-length array of animal speeds at fixed sample rate, in [cm/s]
    sampling_rate : float
        Sampling rate of the tracking system, in [Hz]
    lower_speed : float
        Lower threshold of bandpass filter, in [cm/s]
    upper_time : float
        Time for calculating upper_speed in [s]. The upper_speed is calculated
        as the highest [2cm/s] speed bandwidth that the animal spends at least
        this long.
    speed_bw : float
        Range of speeds for calculating the upper_speed, in [cm/s]

    returns
    -------
    _filter : np.ndarray
        1d M-length array of booleans for indexing the speed array. True where
        the speed PASSES the filter
    '''
    multiplier = 10
    hist_resolution = speed_bw / multiplier          # this is bin_width
    bins = np.arange(np.min(speed), np.max(speed), hist_resolution)
    hist, _ = np.histogram(speed, bins=bins)

    required_frames = upper_time * sampling_rate
    upper_speed = None

    for i in np.arange(-1, -(bins.size - multiplier), -1):
        # Iterate backwards through the histogram, i.e. from highest speeds
        total_frames = np.sum(hist[i:i+multiplier])
        if total_frames >= required_frames:
            # go to the centre of the bandwidth. minus because of reverse direction
            upper_speed = bins[i-int(multiplier/2)]
            break
    if kwargs.get("debug", False):
        print(f"Upper speed determined as {upper_speed} cm/s")
    if upper_speed is not None:
        _filter = _bandpass_fixed(speed, lower_speed, upper_speed, **kwargs)
    else:
        raise SpeedBandwidthError(f"The animal did not spend {upper_time}s within a speed"\
                         f" bandwidth of {speed_bw} cm/s. Try using a"\
                         " larger speed-bandwidth")
    return _filter, upper_speed

def _bandpass_fixed(speed, lower_speed, upper_speed, **kwargs):
    '''Create a filter list that allows through values between the defined upper
    and lower defined values

    parameters
    ----------
    speed : np.ndarray
        1d M-length array of animal speeds at fixed sample rate, in [cm/s]
    sampling_rate : float
        Sampling rate of the tracking system, in [Hz]
    lower_speed : float
        Lower threshold of bandpass filter, in [cm/s]
    upper_speed : float
        Upper threshold of bandpass filter, in [cm/s]
        Required upper_speed > lower_speed

    returns
    -------
    _filter : np.ndarray
        1d M-length array of booleans for indexing the speed array. True where
        the speed PASSES the filter
    '''
    if lower_speed >= upper_speed:
        raise ValueError(f"Your lower bound ({lower_speed}) is higher than your"\
                         f" upper bound ({upper_speed}). Check your argument order.")
    _filter = (lower_speed <= speed) & (speed <= upper_speed)
    passed = np.sum(_filter)
    if kwargs.get("debug", False):
        print(f"{passed:,} survived filter out of {speed.size:,} ({passed/speed.size:3})")
    if passed <= 5:
        raise ValueError("Your filter has excluded nearly all values, only"\
                         f" {passed} remaining. Check your filter criteria")
    return _filter

def _bandpass_none(speed, **kwargs):
    '''Create a filter list that allows all values through'''
    debug = kwargs.get("debug", False)
    if debug:
        print("No filtering")
    _filter = np.ones(speed.size, dtype=bool)
    return _filter

def _spiketimes_to_spikerate(spike_times, tracking_times, sampling_rate):
    '''Convert a list of spike times to a list of spike rates

    parameters
    ----------
    spike_times : np.ndarray
        Nx1 array of times at which spikes occur in [s]
    tracking times : np.ndarray
        Nx1 array of times at which tracking information is known - e.g. time
        stamp of camera frames. Also in [s]

    returns
    -------
    spike_rate : np.adarray
        Nx1 array of spike rate [Hz], with the i'th value being the spike rate
        during the i'th tracking frame.
    '''
    frame_length = 1/sampling_rate
    bin_edges = np.append(tracking_times, tracking_times[-1]+frame_length)

    spikes_per_frame, _ = np.histogram(spike_times, bins=bin_edges)
    spike_rate = spikes_per_frame *sampling_rate

    return spike_rate

def grid_score(rate_map, autocorr_map, **kwargs):

    '''
        Computes the grid score of neuron given spike data.

        Params:
            occupancy_map (np.ndarray):
                A 2D numpy array enconding subjects position over entire experiment.
            ts (np.ndarray):
                Spike time stamp array
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y  coordinate positions as well as timestamps of movement respectively
            arena_size (tuple):
                Dimensions of arena
            spikex, spikey (np.ndarray):
                x and y coordinates of spike events respectively
            kenrnlen, std (int):
                kernel size and standard deviation of kernel for convolutional smoothing.
    '''

    if 'use_autocorr_direclty' in kwargs:
        if kwargs['use_autocorr_direclty'] == True:
            autocorr = kwargs['autocorr']
    else:
        if 'smoothing_factor' in kwargs:
            smoothing_factor = kwargs['smoothing_factor']
        else:
            print("No smoothing factor provided, using default of 1")
            smoothing_factor = 1

        ratemap = rate_map

        autocorr = autocorr_map

    grid_score_object = opexebo_grid_score(autocorr)
    true_grid_score = grid_score_object[0]

    return true_grid_score

# def grid_score_shuffle(self, occupancy_map: np.ndarray, arena_size: tuple, ts: np.ndarray,
#                        pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray, kernlen: int, std: int, **kwargs) -> list:

#     '''
#         Shuffles position and spike data prior to computing grid scores. Shuffling
#         allows us to determine if the probability of a given score is random, or
#         demonstrates a meaningful association in the data.

#         Params:
#             occupancy_map (np.ndarray):
#                 A 2D numpy array enconding subjects position over entire experiment.
#             arena_size (tuple):
#                 Dimensions of arena
#             ts (np.ndarray):
#                 Array of spike event times
#             pos_x, pos_y, pos_t (np.ndarray):
#                 Arrays of x, y  coordinate positions as well as timestamps of movement respectively
#             spikex, spikey (np.ndarray):
#                 x and y coordinates of spike events respectively
#             kernlen, std (int):
#                 kernel size and standard deviation for convolutional smoothing

#         **kwargs:
#             xsheet: xlwings excel sheet

#         Returns:
#             list: grid_scores
#             --------
#             grid_scores: List of 100 grid scores (1 score per shuffle)
#     '''

#     grid_scores = []
#     shuffled_spike_xy = np.zeros((2,len(ts)))

#     # If an excel sheet is passed, set reference
#     s = kwargs.get('xsheet',None)
#     row_index =  kwargs.get('row_index',None)
#     cell_number = kwargs.get('cell_number',None)
#     column_index = 2

#     # Shuffle spike data
#     shuffled_spikes = shuffle_spikes(self, ts, pos_x, pos_y, pos_t)

#     # For each set of shuffled data, compute grid score
#     for element in shuffled_spikes:
#         rate_map_smooth, _ = rate_map(pos_x, pos_y, pos_t, arena_size, element[0], element[1], kernlen, std)
#         autocorr_map = autocorrelation(rate_map_smooth,pos_x,pos_y,arena_size)
#         grid_score_object = opexebo_grid_score(autocorr_map)

#         # Populate the excel sheet with the scores
#         if s != None and row_index != None and cell_number != None:
#             if grid_score_object != None:
#                 grid_scores.append(float(grid_score_object[0]))
#                 if s[get_column_letter((row_index-1)*5 + 1) + str(1)] != 'C_' + str(cell_number) +'_BorderShuffle_Top':
#                     s[get_column_letter(row_index) + str(1)] = 'C_' + str(cell_number) +'_GridShuffle'
#                     s[get_column_letter(row_index) + str(column_index)] = grid_score_object[0]
#                     column_index += 1
#                     ColumnDimension(s, bestFit=True)
#                 else:
#                     s[get_column_letter(row_index*5) + str(1)] = 'C_' + str(cell_number) +'_GridShuffle'
#                     s[get_column_letter(row_index*5) + str(column_index)] = grid_score_object[0]
#                     column_index += 1
#                     ColumnDimension(s, bestFit=True)

#         else:
#             if grid_score_object != None:
#                 grid_scores.append(float(grid_score_object[0]))

#     return grid_scores


""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

# This is included as the return result of an invalid autocorrelogram
# It replaces an earlier mixture of output types that performed calcaultions 
# on a null-acorr instead
INVALID_OUTPUT = (np.nan, {'grid_spacings': np.array([np.nan, np.nan, np.nan]),
  'grid_spacing': np.nan,
  'grid_orientations': np.array([np.nan, np.nan, np.nan]),
  'grid_orientations_std': np.nan,
  'grid_orientation': np.nan,
  'grid_positions': np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
  'grid_ellipse': np.nan,
  'grid_ellipse_aspect_ratio': np.nan,
  'grid_ellipse_theta': np.nan})


def opexebo_grid_score(aCorr, **kwargs):
    """Calculate gridness score for an autocorrelogram.

    Calculates a gridness score by expanding a circle around the centre field
    and calculating a correlation value of that circle with it's rotated versions.
    The expansion is done up until the smallest side of the autocorrelogram.
    The function may also calculate grid statistics.

    Gridness score value by itself is calculated as a maximum over a sliding mean
    of expanded circle. The width of the sliding window is given by a variable
    numGridnessRadii. This is done in order to keep gridness score the same as
    historical values (i.e. older versions of gridness score).

    Parameters
    ----------
    acorr: np.ndarray
        A 2D autocorrelogram.
    
    Other Parameters
    ----------------
    min_orientation: int
        See function "grid_score_stats"
    search_method: str
        Peak searching method, currently limited to either `default` or `sep`
    bin_width: float
        Size of the bins. Distance units will be returned in the same units
        If not provided, distance units will be retuned in units [bins]

    Returns
    -------
    grid_score: float
        Always returns a gridness score value. It ranges from -2 to 2. 2 is
        more of a theoretical bound for a perfect grid. More practical value for
        a good grid is around 1.3. If function can not calculate a gridness
        score, NaN value is returned.
    grid_stats: dictionary
        grid_spacings: np.array
            Spacing of three adjacent fields closest to center in autocorr
            (in [bins] if keyword `bin_width` is not given)
        grid_spacing: float
            Nanmean of 'spacings' (in [bins] if keyword `bin_width` is not
            given)
        grid_orientations: np.array
            Orientation of three adjacent fields closest to center in autocorr
            (in [degrees])
        grid_orientations_std: float
            Standard deviation of orientations % 60
        grid_orientation: float
            Orientation of grid in [degrees] (mean of fields of 3 main axes)
        grid_positions: np.array
            [y,x] coordinates of six fields closest to center
        grid_ellipse: np.array
            Ellipse fit returning 
            [x coordinate, y coordinate, major radius, minor radius, theta]
        grid_ellipse_aspect_ratio: float
            Ellipse aspect ratio (major radius / minor radius)
        grid_ellipse_thet: float
            Ellipse theta (corrected according to previous BNT standard) in [degrees]

    See Also
    --------
    opexebo.analysis.placefield

    Notes
    -----
    BNT.+analyses.gridnessScore

    Copyright (C) 2018 by Vadim Frolov, (C) 2019 by Simon Ball, Horst Obenhaus
    """
    debug = kwargs.get("debug", False)

    # normalize aCorr in order to find contours
    aCorr = aCorr / aCorr.max()
    centre = -0.5 + np.array(aCorr.shape)/2 # centre : also [y, x]
    cFieldRadius = int(np.floor(_findCentreRadius(aCorr, **kwargs)))

    if cFieldRadius == 0:
        if debug:
            print("Terminating due to invalid cFieldRadius")
        return INVALID_OUTPUT

    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth  = np.ceil(aCorr.shape[1]/2)
    heightIndices = np.arange(aCorr.shape[0])
    widthIndices  = np.arange(aCorr.shape[1])

    # Define radii that will be iterated over for the gridness score
    # outer bound is defined by the minimum of autocorrelogram's dimensions
    # this is need for rectangular autocorrelograms.
    outerBound = int(np.floor(np.min(np.array(aCorr.shape)/2)))
    if outerBound < cFieldRadius:
        if debug:
            print("Terminating due to invalid outerBound"\
                  f" ({outerBound} < {cFieldRadius})")
        return INVALID_OUTPUT
        
    radii = np.linspace(max(3, cFieldRadius+1), outerBound, outerBound-cFieldRadius)
    radii = radii.astype(int)
    numSteps = len(radii)
    if numSteps < 1:
        if debug:
            print("Terminating due to invalud numSteps")
        return INVALID_OUTPUT

    rotAngles_deg = np.arange(30, 151, 30)  # 30, 60, 90, 120, 150
    rotatedACorr = np.zeros(
            shape=(aCorr.shape[0], aCorr.shape[1], len(rotAngles_deg)),
            dtype=float)
    # we get rotated maps here as it is a heavy operation
    for n, angle in enumerate(rotAngles_deg):
        rotatedACorr[:, :, n] = transform.rotate(aCorr, angle,
                                            preserve_range=True, clip=False)
    rr, cc = np.meshgrid(widthIndices, heightIndices, sparse=False)
    # This is needed for compatibility with Matlab's code
    rr += 1
    cc += 1
    mainCircle = np.sqrt(np.power((rr - halfWidth), 2) + np.power(cc-halfHeight, 2))
    innerCircle = mainCircle > cFieldRadius


    GNS = np.zeros(shape=(numSteps, 2), dtype=float)
    for i, radius in enumerate(radii):
        mask = innerCircle & (mainCircle < radius)
        aCorrValues = aCorr[mask]

        rotCorr = np.zeros_like(rotAngles_deg).astype(float)
        for j, angle in enumerate(rotAngles_deg):
            rotatedValues = rotatedACorr[mask, j]
            r, _ = pearsonr(aCorrValues, rotatedValues)
            rotCorr[j] = r
        GNS[i, 0] = np.min(rotCorr[[1, 3]]) - np.max(rotCorr[[0, 2, 4]])
        GNS[i, 1] = radius

    # find the greatest gridness score value and radius
    gscoreInd = np.argmax(GNS[:,0])

    numGridnessRadii = 3
    numStep = max(numSteps - numGridnessRadii, 1) # minimum value 1

    if numStep == 1:
        gscore = np.mean(GNS, axis=0)[0]
    else:
        meanGridness = np.zeros(numStep)
        for ii in range(numStep):
            meanGridness[ii] = np.nanmean(GNS[ii:ii+numGridnessRadii, 0])
        gscore = np.max(meanGridness)

    '''Then calculate stats about the autocorrelogram'''
    # Mask center field and fringes of autocorrelogram > best grid score radius
    mask_outwards = _circular_mask(aCorr, radii[gscoreInd]*1.25, 'outwards', centre)
    mask_center   = _circular_mask(aCorr, cFieldRadius*1.5, 'inwards', centre)
    mask = mask_outwards + mask_center

    grid_stats = grid_score_stats(aCorr, mask, centre, **kwargs)

    return gscore, grid_stats

#########################################################
################        Helper Functions
#########################################################


def grid_score_stats(aCorr, mask, centre, **kwargs):
    '''
    Calculate spatial characteristics of grid based on 2D autocorr

    Parameters
    ----------
    aCorr : np.array
        2D Autocorrelation
    mask : np.array
        Mask (masked=True) of shape aCorr for masking center field and
        fringes of aCorr above best grid score radius
    centre : np.array
        Centre coordinate [y,x]
    **kwargs :
        min_orientation : int
            Minimum difference in degrees that two neighbouring fields
            detected in 2D autocorrelation must have. If difference is
            below this threshold, discard the field that has larger
            distance from center
        search_method : str
            Peak searching method, currently limited to either `default` or `sep`
        bin_width : float
            Size of the bins. Distance units will be returned in the same units
            If not provided, distance units will be retuned in units [bins]

    Returns
    -------
    grid_stats : dictionary
        grid_spacings              : np.array
            Spacing of three adjacent fields closest to center in autocorr
            (in [bins])
        grid_spacing                : float
            Nanmean of 'spacings' in [bins]
        grid_orientations           : np.array
            Orientation of three adjacent fields closest to center in autocorr
            (in [degrees])
        grid_orientations_std       : float
            Standard deviation of orientations % 60
        grid_orientation            : float
            Orientation of grid in [degrees] (mean of fields of 3 main axes)
        grid_positions              : np.array
            [y,x] coordinates of six fields closest to center
        grid_ellipse                : np.array
            Ellipse fit returning 
            [x coordinate, y coordinate, major radius, minor radius, theta]
        grid_ellipse_aspect_ratio   : float
            Ellipse aspect ratio (major radius / minor radius)
        grid_ellipse_theta          : float
            Ellipse theta (corrected according to previous BNT standard) in [degrees]
    '''

    # Get kwargs
    debug = kwargs.get('debug', False)
    bin_width = kwargs.get("bin_width", 1) # if not provided, use a 1:1 match from bins to return units
    min_orientation = kwargs.get('min_orientation', default.min_orientation)
    search_method = kwargs.get("search_method", default.search_method)
    min_orientation = np.radians(min_orientation)
    if debug:
        print(f"Search method: {search_method}")
        print('Min orientation: {} degrees'.format(np.degrees(min_orientation)))
    

    # Initialise default output in case stats are uncalculable
    gs_ellipse_theta    = np.nan
    gs_ellipse          = np.nan
    gs_aspect_ratio     = np.nan
    gs_orientations_std = np.nan
    gs_orientation      = np.nan
    gs_positions        = np.full(6, fill_value = np.nan, dtype=float)
    gs_orientations     = np.full(3, fill_value = np.nan, dtype=float)
    gs_spacings         = np.full(3, fill_value = np.nan, dtype=float)

    # Find fields in autocorrelogram
    all_coords = peak_search(aCorr, mask=mask, search_method=search_method,
                                             null_background=True, threshold=0.1, get_maxima=True)
    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(aCorr)
        plt.scatter(centre[1], centre[0], s=600, marker="x", color="black")
        for field_no, coord in enumerate(all_coords):
            plt.scatter(coord[1], coord[0], s=300, marker='x', color='red')
            plt.text(coord[1]+3, coord[0], field_no, label='Center')
        plt.title("All local maxima in acorr")
        
        

    if all_coords.shape[0] >= 6:
        # Calculate orientation and distance of all local maxima to center
        # np.arctan2 gives angles in radians relative to the horizontal axis in the range [-pi, pi]
        # A vector along the horizontal axis from (0,0) to (0,-1) has angle pi.
        # np.arctan2 - accepts arguments (y, x), and gives the angles in each case from (0,0) to (xi, yi)
        orientation = np.arctan2(all_coords[:,0] - centre[0], all_coords[:,1] - centre[1]) # in radians
        distance = np.sqrt(np.square(all_coords[:,0]-centre[0]) + np.square(all_coords[:,1]-centre[1]))

        # Where two fields have a very similar orientation, discard the more distant one
        orient_distsq = np.abs(_circ_dist2(orientation))
        close_fields = orient_distsq < min_orientation
        close_fields = np.triu(close_fields, 1) # Upper triangle only - set lower triangle to zero
                                                # k=1: +1 offset from diagonal: set diagonal to zero too
        to_del = []
        for row,col in np.argwhere(close_fields):
            if distance[row] > distance[col]:
                to_del.append(row)
            else:
                to_del.append(col)

        distance    = np.delete(distance, to_del)
        orientation = np.delete(orientation, to_del)
        all_coords  = np.delete(all_coords, to_del, axis=0)

        # First sort by distance and take first 6 fields
        sorted_ids_dist = np.argsort(distance)[:6] # 6 closest fields
        positions = all_coords[sorted_ids_dist]
        distance = distance[sorted_ids_dist]
        orientation = orientation[sorted_ids_dist]
        
        # ... then re-sort remaining fields by angle
        sorted_ids_ang = np.argsort(orientation)

        ################# GATHER OUTPUT #################
        gs_positions = positions[sorted_ids_ang]
        # For grid orientation and spacing take only 3 out of 6 neighbouring fields
        # Convert from distance in bins to distance in units
        gs_spacings = distance[sorted_ids_ang][:3] * bin_width
        gs_orientations = np.degrees(orientation[sorted_ids_ang][:3]) % 180
        # Work out mean orientation of grid. Take standard deviation as quality marker
        gs_orientation, gs_orientations_std = _extract_grid_orientation(gs_orientations)


        if debug:
            import matplotlib.pyplot as plt
            aCorr_masked = np.ma.masked_where(mask, aCorr.copy())
            plt.figure()
            plt.imshow(aCorr_masked)
            plt.scatter(centre[1],centre[0], s=600, marker='x', color='black')
            for field_no, coord in enumerate(gs_positions):
                plt.scatter(coord[1], coord[0], s=300, marker='x', color='red')
                plt.text(coord[1]+3, coord[0], field_no, label='Center')
            plt.title('Masked autocorr + 6 remaining fields')

        # Fit an ellipse to those remaining fields:
        if len(gs_positions) > 2:
            try:
                gs_ellipse =  fit_ellipse(gs_positions[:,1], gs_positions[:,0])
                gs_ellipse_theta = np.degrees(gs_ellipse[4]+np.pi)%360
                # The +pi term was included in the original BNT, I have kept it to
                # maintain consistency with past results.
                gs_aspect_ratio = gs_ellipse[2]/gs_ellipse[3] # Major radius / Minor radius
            except np.linalg.LinAlgError as e:
                print(f"LinAlgError: {e}")
                print(f"Retuning NaN ellipse stats")

    else:
        if debug: 
            print('Not enough fields detected ({})'.format(len(all_coords)))
            

    grid_stats = {'grid_spacings': gs_spacings,
                  'grid_spacing': np.nanmean(gs_spacings),
                  'grid_orientations': gs_orientations,
                  'grid_orientations_std': gs_orientations_std,
                  'grid_orientation': gs_orientation,
                  'grid_positions': gs_positions,
                  'grid_ellipse': gs_ellipse,
                  'grid_ellipse_aspect_ratio': gs_aspect_ratio,
                  'grid_ellipse_theta': gs_ellipse_theta}
    return grid_stats


def _circular_mask(image, radius, polarity='outwards', center=None):
        '''
        Given height and width, create circular mask around point with defined radius
        Polarity:
            'inwards' : True inside,  False outside
            'outwards': True outside, False inside
        '''
        h = image.shape[0]
        w = image.shape[1]

        if center is None:
            center = [int(h/2), int(w/2)]

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(np.power(X - center[1],2) + np.power(Y-center[0],2))
        if polarity.lower() == 'inwards':
            mask = dist_from_center <= radius
        elif polarity.lower() == 'outwards':
            mask = dist_from_center >= radius
        else:
            raise err.ArgumentError('Polarity "{}" not defined'.format(polarity))
        return mask

def _draw_ellipse(x, y, rl, rs, theta):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import transforms
    theta = (theta%(2*np.pi))# + np.pi
    ell = Ellipse((0,0), width=rs*2, height=rl*2, facecolor=(1,0,0,0.2),
                  edgecolor=(1,1,1,0.75))
    ax = plt.gca()
    transf = transforms.Affine2D().rotate(theta+np.pi/2).translate(x, y)
    ell.set_transform(transf + ax.transData)
    ax.add_patch(ell)

def _plotContours(img, contours):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots()
    ax.imshow(img, cmap='jet', origin='lower')

    centroids = np.zeros(shape=(len(contours), 2), dtype=float)
    radii = np.zeros(shape=(len(contours), 1), dtype=float)
    for n, contour in enumerate(contours):
        x = contour[:, 1]
        y = contour[:, 0]
        ax.plot(x, y, linewidth=2)
        # mean x and y
        centroid = (sum(x) / len(contour), sum(y) / len(contour))
        radius = [np.mean([np.sqrt(np.square(x-centroid[0]) + np.square(y-centroid[1]))])]

        centroids[n] = centroid
        radii[n] = radius

        ax.text(centroid[0], centroid[1], str(n))
        ax.plot((centroid[0], centroid[0]+radius), (centroid[1], centroid[1]), linewidth=1)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return radii, centroids

def _extract_grid_orientation(orientations):
    '''
    Extract grid orientation based on angular difference
    of autocorrelation (aCorr) field coordinates to 60 degree axes. 

    Parameter
    ---------
    orientations      : np.array
                        (Raw) aCorr field angles in degrees 

    Returns
    -------
    orientation       : float
                        Grid orientation in degrees (average)
    orientation_std   : float
                        Standard deviation over grid field
                        orientations
    '''


    orientations = orientations % 60 
    corr_orientations = []
    for orient in orientations: 
        # For every angle extract min to 60 deg
        diff_60 = orient - 60 
        if np.abs(diff_60) < np.abs(orient): 
            corr_orientations.append(diff_60)
        else:
            corr_orientations.append(orient)

    corr_orientations = np.array(corr_orientations)
    # Check 30 degree flips 
    if np.mean(np.abs(np.abs(corr_orientations) - 30)) < np.mean(np.abs(corr_orientations)):
        # Yes, angles close to 30 degrees (flipping axis)
        # Try to reach consensus
        if np.median(corr_orientations) < 0: # Make everything negative
            corr_orientations = np.negative(corr_orientations, where=corr_orientations>0, out=corr_orientations)
        else: # Make everything positive
            corr_orientations = np.negative(corr_orientations, where=corr_orientations<0, out=corr_orientations)

    # Extract average and standard deviation
    orientation     = np.nanmean(corr_orientations)
    orientation_std = np.nanstd(corr_orientations)

    # Test / correct orientation 
    if np.argmin([np.abs(orientation-60), np.abs(orientation)]) == 0:
        orientation -= 60

    return orientation, orientation_std

def _circ_dist2(X):
    '''Given a 1D array of angles, find the 2D array of pairwise differences
    Based on https://github.com/circstat/circstat-matlab/blob/master/circ_dist2.m'''
    x = np.outer(np.exp(1j*X), np.ones(X.size)) # similar to meshgrid, but simpler to implement
    y = np.transpose(x)
    return np.angle(x/y)

def _polyArea(x, y):
    '''Polygon area, from 
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    Seems to be the same as Matlab's polyarea'''
    return 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _contourArea(contours, i):
    contour = contours[i]
    x = contour[:, 1]
    y = contour[:, 0]
    area = _polyArea(x, y)
    return area

def _findCentreRadius(aCorr, **kwargs):
    debug = kwargs.get("debug", False)
    search_method = kwargs.get("search_method", default.search_method)
    halfHeight = np.ceil(aCorr.shape[0]/2)
    halfWidth = np.ceil(aCorr.shape[1]/2)
    peak_coords = np.ones(shape=(1, 2), dtype=np.int32)
    peak_coords[0, 0] = halfHeight-1
    peak_coords[0, 1] = halfWidth-1
    fields = place_field(aCorr, min_bins=5, min_peak=0, min_mean=0, init_thresh=.95, \
                                         peak_coords=peak_coords, search_method=search_method)[0] # Fix all input args for now
    if fields is None or len(fields) == 0:
        if debug:
            print("Terminating _findCentreRadius due to no fields")
        return 0
    elif debug:
        print(f"Fields found: {len(fields)}")
    else:
        pass

    peak_coords = np.ndarray(shape=(len(fields), 2), dtype=np.int32)
    areas = np.ndarray(shape=(len(fields), 1), dtype=np.int32)
    for i, field in enumerate(fields):
        peak_rc = field['peak_coords']
        peak_coords[i, 0] = peak_rc[0]
        peak_coords[i, 1] = peak_rc[1]
        areas[i] = field['area']

    aCorrCentre = np.zeros(shape=(1, 2), dtype=float)
    aCorrCentre[0] = aCorr.shape[0]
    aCorrCentre[0, 1] = aCorr.shape[1]
    aCorrCentre = np.ceil(aCorrCentre/2)

    # index of the closest field to the centre
    closestFieldInd = 0
    if len(peak_coords) >= 2:
        # get all distances and check two minimum of them
        distancesToCentre = cdist(peak_coords, aCorrCentre)
        sortInd = np.squeeze(np.argsort(distancesToCentre, axis=0))

        closestFieldInd = sortInd[0]
        twoMinDistances = distancesToCentre[sortInd[:2]]

        areFieldsClose = np.abs(twoMinDistances[0] - twoMinDistances[1])[0] < 2
        if areFieldsClose:
            # two fields with close middle point. Let's select one with minimum area
            indices_to_test = sortInd[:2]
            min_ind = np.argmin(areas[indices_to_test])
            closestFieldInd = indices_to_test[min_ind]

    radius = np.floor(np.sqrt(areas[closestFieldInd] / np.pi))
    
    if debug:
        print("radius is {}".format(radius))
        
    return radius


def border_score(rate_map, **kwargs) -> tuple:

    '''
        Computes 4 scores which each reflect selectivity of neurons firing at arena edges,
        namely top, bottom, left, and right border scores for each arena side.
        Score range: -1 to 1, with 1 being the highest for border selectivity.

        More details about this scoring criteria can be found under the following publication at the border score section:
        "Grid and Nongrid Cells in Medial Entorhinal Cortex Represent Spatial
            Location and Environmental Features with Complementary Coding Schemes"

        Params:
            binary_map (np.ndarray):
                Map of binarized firing fields in ratemap
            rate_map (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            tuple: top_bscore, bottom_bscore, left_bscore, right_bscore
    '''

    if 'use_objects_directly' in kwargs:
        if kwargs['use_objects_directly'] == True:
            rate_map = kwargs['rate_map']
            bin_map = kwargs['bin_map']
    else:
        if 'smoothing_factor' in kwargs:
            smoothing_factor = kwargs['smoothing_factor']
        else:
            print("No smoothing factor provided, using default of 1")
            smoothing_factor = 1

    rate_map = rate_map

    bin_map = binary_map(rate_map, smoothing_factor=smoothing_factor)
    
    # If for whatever reason the supplied binary map does not match rate map dimensions, throw error.
    if bin_map.shape != rate_map.shape:
        raise Exception("The binary map and rate map must have the same dimensions")

    shortest_side = min(bin_map.shape[0], bin_map.shape[1]) / 2

    # Initializing top, bottom, left, right proportion coverage of map
    top_coverage = 0
    bottom_coverage = 0
    left_coverage = 0
    right_coverage = 0

    # Initializing distances
    top_distance_sum = 0
    bottom_distance_sum = 0
    left_distance_sum = 0
    right_distance_sum = 0

    # Compute coverage on each side
    top_coverage = sum(bin_map[0]) / len(bin_map[0])
    bottom_coverage = sum(bin_map[len(bin_map)-1]) / len(bin_map[0])
    left_coverage = sum(bin_map[:,0]) / len(bin_map[:,0])
    right_coverage = sum(bin_map[:,bin_map.shape[1]-1]) / len(bin_map[:,0])

    indices = np.argwhere(bin_map > 0)

    # Compute distances of all field pixels to each edge
    for index in indices:
        top_distance_sum += (index[0]) *  rate_map[index[0], index[1]]
        bottom_distance_sum += (len(bin_map[0]) - index[0]) * rate_map[index[0], index[1]]
        left_distance_sum += index[1] * rate_map[index[0], index[1]]
        right_distance_sum += (len(bin_map[:,0]) - index[1]) * rate_map[index[0], index[1]]

    avg_top_dist    = (top_distance_sum / len(indices))     / shortest_side
    avg_bottom_dist = (bottom_distance_sum / len(indices))  / shortest_side
    avg_left_dist   = (left_distance_sum / len(indices))    / shortest_side
    avg_right_dist  = (right_distance_sum / len(indices))   / shortest_side

    # Compute the border score
    top_bscore = (top_coverage - avg_top_dist) / (top_coverage + avg_top_dist)
    bottom_bscore = (bottom_coverage - avg_bottom_dist) / (bottom_coverage + avg_bottom_dist)
    left_bscore = (left_coverage - avg_left_dist) / (left_coverage + avg_left_dist)
    right_bscore = (right_coverage - avg_right_dist) / (right_coverage + avg_right_dist)

    return top_bscore, bottom_bscore, left_bscore, right_bscore

#public
def border_score_shuffle(self, occupancy_map: np.ndarray, arena_size: tuple, ts: np.ndarray,
                         pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray, kernlen: int, std: int, **kwargs) -> list:

    '''
        Shuffles position and spike data prior to computing border scores. Shuffling
        allows us to determine the probability if a given score is random, or
        demonstrates a meaningful association in the data.

        Params:
            occupancy_map (np.ndarray):
                A 2D numpy array enconding subjects position over entire experiment.
            arena_size (tuple):
                Dimensions of arena
            ts (np.ndarray):
                Spike time stamp array
            pos_x, pos_y, pos_t (np.ndarray):
                Arrays of x,y  coordinate positions as well as timestamps of movement respectively
            kenrnlen, std (int):
                kernel size and standard deviation of kernel for convolutional smoothing.

        **kwargs:
            xsheet: xlwings excel sheet

        Returns:
            list: border_scores
            --------
            border_scores: List of 100 border scores (1 set of scores [top,bottom,left,right] per shuffle)
    '''

    # If an excel sheet was passed in, set reference
    s = kwargs.get('xsheet', None)

    row_index = kwargs.get('row_index',None)
    cell_number = kwargs.get('cell_number',None)
    column_index = 2

    border_scores = []

    # Shuffle the spike data
    shuffled_spikes = shuffle_spikes(self, ts, pos_x, pos_y, pos_t)

    # For each shuffled set of spiek data, compute border score
    for element in shuffled_spikes:
            rate_map, _ = rate_map(pos_x, pos_y, pos_t, arena_size, element[0], element[1], kernlen, std)
            binary_map = binary_map(rate_map)
            b_score = border_score(binary_map, rate_map)
            border_scores.append(b_score)

            # Copy shuffled border scores into excel sheet
            if s != None and row_index != None and cell_number != None:
                s[get_column_letter(row_index)   + str(1)] = 'C_' + str(cell_number) + '_BS_Top'
                s[get_column_letter(row_index+1) + str(1)] = 'C_' + str(cell_number) + '_BS_Bottom'
                s[get_column_letter(row_index+2) + str(1)] = 'C_' + str(cell_number) + '_BS_Left'
                s[get_column_letter(row_index+3) + str(1)] = 'C_' + str(cell_number) + '_BS_Right'

                s[get_column_letter(row_index)   + str(column_index)] = b_score[0]
                s[get_column_letter(row_index+1) + str(column_index)] = b_score[1]
                s[get_column_letter(row_index+2) + str(column_index)] = b_score[2]
                s[get_column_letter(row_index+3) + str(column_index)] = b_score[3]

                column_index += 1
                # Resize excel columns per iteration
                #s.autofit(axis="columns")
                ColumnDimension(s, bestFit=True)

    return border_scores

def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r