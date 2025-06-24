import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from animal_performance.spiketrain import SpatialSpikeTrain2D_signalstore, HaftingRateMap
import animal_performance.errors as err
import animal_performance.defaults as default
from animal_performance.utils import peak_search
from skimage import measure, morphology
from scipy import ndimage
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Suppress the ConfigurationMissingWarning that Astropy triggers
    from astropy.convolution import convolve


def MatlabNumSeq(start, stop, step, exclude=True):
    """In Matlab you can type:
    start:step:stop and easily create a numerical sequence
    if exclude is true it will exclude any values greater than the stop value
    """

    '''np.arange(start, stop, step) works good most of the time
    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if exclude:
        if seq[-1] > stop:
            seq = seq[:-1]

    return seq


# Currentl;y 'unit' == cell id is passed in, when refactored to take in animal: Animal(), can use agg.sorted_events to get cell data, or agg labels, + add agg FD
def histogram_ISI(spike_times, maxisi=0.01, isi_histo=[0, 1000, 10], req_burst_spikes=2):
    """Performs analysis on the data for a tetrode and unit (cell)
    ts- spike times of the unit (cell)
    cluster_mat - cluster_labels
    isi_histo - a 1x3 matrix [minimum histogram value in ms, maximum value in ms, width in ms]
    FD = features used for cluster quality
    """

    ts = spike_times

    # Convert to flat NumPy array
    if isinstance(ts, xr.DataArray):
        ts = ts.values.flatten()
    elif isinstance(ts, list):
        ts = np.asarray(ts).flatten()
    elif isinstance(ts, np.ndarray):
        ts = ts.flatten()
    else:
        raise TypeError(f"Unsupported type for spike_times: {type(ts)}")

    if len(ts) < 2:
        # cannot perform this calculation with less than 2 spike times
        return [], []

    n_spikes = len(ts)

    if n_spikes < 2:
        print('There are only %d spikes in this cell, skipping analysis!' % n_spikes)
        
    # --------- Interspike Interval Analysis ---------------------- #

    ISI = np.multiply(1000, np.diff(ts))  # ISI in milliseconds

    ISI_min = np.min(ISI)
    ISI_max = np.amax(ISI)
    ISI_mean = np.mean(ISI)  # average ISI, milliseconds
    ISI_median = np.median(ISI)
    ISI_std = np.std(ISI)  # standard deviation of ISI, milliseconds
    ISI_cv = ISI_mean / ISI_std  # coef. of variation of ISI, unitless

    # we will go from 0 to 1 second by 10 ms

    isi_histo_min, isi_histo_max, isi_histo_width = isi_histo
    binedges = MatlabNumSeq(isi_histo_min, isi_histo_max, isi_histo_width, exclude=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(ISI, binedges, histtype='bar')
    # ax.set_xlim([0, 1000])  # limiting the x-axis between 0 and 1000 ms
    # ax.set_title('T%sC%s: Interspike Interval Histogram' % (tet_num, unit))
    ax.set_xlabel("ISI(ms)")
    ax.set_ylabel("Counts(#)")

    ISI_dict = {'min': ISI_min, 'max': ISI_max, 'median': ISI_median, 'mean': ISI_mean, 'std': ISI_std, 'cv':
        ISI_cv, 'n': n, 'bins': bins, 'fig': fig}

    # spike_class.get_map('spike').set_metric('ISI_dict', ISI_dict)

    return ISI_dict

def find_burst(spike_times, maxisi=0.01, req_burst_spikes=2):
    """This function is used to calculate the percentage of bursting

    inputs:
    ts- the spike times
    maxisi- the maximum inter spike interval"""

    ts = spike_times

    # Convert to flat NumPy array
    if isinstance(ts, xr.DataArray):
        ts = ts.values.flatten()
    elif isinstance(ts, list):
        ts = np.asarray(ts).flatten()
    elif isinstance(ts, np.ndarray):
        ts = ts.flatten()
    else:
        raise TypeError(f"Unsupported type for spike_times: {type(ts)}")

    if len(ts) < 2:
        # cannot perform this calculation with less than 2 spike times
        return [], []

    # ts = ts.flatten()

    bursts = []
    singlespikes = []

    isi = np.diff(ts)  # calculating the inter spike intervals
    n = len(ts)

    if isi[0] <= maxisi:
        bursts = [0]
    else:
        singlespikes = [0]

    indices = np.where(isi > maxisi)[0]

    max_index = len(isi) - 1

    for i in indices:
        if (i + 1) > max_index:
            break

        if isi[i + 1] <= maxisi:
            bursts.append(i + 1)
        elif isi[i + 1] > maxisi:
            singlespikes.append(i + 1)

    if isi[n - 2] > maxisi:
        singlespikes.append(n - 1)

    # get spikes that belong to the potential bursts

    total_n = np.arange(len(ts.flatten()))

    burst_spikes = np.setdiff1d(total_n, singlespikes)  # these are all indices belonging to the bursts

    if len(burst_spikes) != 0:
        burst_spikes = _find_consec(burst_spikes)

        # check if the bursts have the required amount of spikes

        burst_spikes = [burst for burst in burst_spikes if len(burst) >= req_burst_spikes]

        bursts = [burst[0] for burst in burst_spikes]

        # flatten the burst spikes
        burst_spikes = [spike for burst in burst_spikes for spike in burst]

        singlespikes = np.setdiff1d(total_n, np.asarray(burst_spikes)).tolist()
    else:

        bursts = np.array([])
        singlespikes = total_n

    # use the new burst spikes to calculate everything else as single spikes

    # return np.asarray(bursts), np.asarray(singlespikes)
    bursting = 100 * len(bursts) / (len(bursts) + len(singlespikes))

    bursts_n_spikes_avg = _avg_spike_burst(ts, bursts, singlespikes)  # num of spikes on avg per burst
    # spike_class.get_map('spike').set_metric('bursting', bursting)
    # spike_class.get_map('spike').set_metric('bursts_n_spikes_avg', bursts_n_spikes_avg)

    # spike_class.stats_dict['spike']['bursting'] = bursting
    # spike_class.stats_dict['spike']['bursts_n_spikes_avg'] = bursts_n_spikes_avg

    return bursting, bursts_n_spikes_avg



def _find_consec(data):
    '''finds the consecutive numbers and outputs as a list'''
    consecutive_values = []  # a list for the output
    current_consecutive = [data[0]]

    if len(data) == 1:
        return [[data[0]]]

    for index in range(1, len(data)):

        if data[index] == data[index - 1] + 1:
            current_consecutive.append(data[index])

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)

        else:
            consecutive_values.append(current_consecutive)
            current_consecutive = [data[index]]

            if index == len(data) - 1:
                consecutive_values.append(current_consecutive)
    return consecutive_values

def _avg_spike_burst(ts, bursts, singleSpikes):
    """calculates the average spikes per burst"""
    total_n = np.arange(len(ts.flatten()))
    burst_spikes = np.setdiff1d(total_n, singleSpikes)  # these are all indices belonging to the bursts

    # divide by total number of burst events to return the output

    if len(bursts) != 0:
        return len(burst_spikes) / len(bursts)
    else:
        return np.NaN

def binary_map(rate_map, percentile=75, **kwargs):

    '''
        Produces a binary map of place fields from a ratemap based on [1].

        "A place field was defined as an area of nine or more (5 × 5 cm) adjacent bins with firing rates exceeding 20% of the peak firing rate of the rate map." (p 1843)

        [1] C. B. Alme, C. Miao, K. Jezek, A. Treves, E. I. Moser, and M.-B. Moser, “Place cells in the hippocampus: Eleven maps for eleven rooms,” Proceedings of the National Academy of Sciences, vol. 111, no. 52, pp. 18428–18435, Dec. 2014, doi: 10.1073/pnas.1421056111.


        Params:
            ratemap (np.ndarray):
                Array encoding neuron spike events in 2D space based on where
                the subject walked during experiment.

        Returns:
            np.ndarray:
                binary_map
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        print("No smoothing factor provided, using default value of 1")
        smoothing_factor = 1

    ratemap = rate_map
    binary_map = np.zeros(ratemap.shape)
    
    binary_map[  ratemap >= np.percentile(ratemap.flatten(), percentile)  ] = 1

    return binary_map


def rate_map_coherence(spatial_map: SpatialSpikeTrain2D_signalstore | HaftingRateMap, **kwargs):
    '''
    Calculate coherence of a rate map

    Coherence is calculated based on RU Muller, JL Kubie "The firing of hippocampal place
    cells predicts the future position of freely moving rats", Journal of Neuroscience, 1 December 1989,
    9(12):4101-4110. The paper doesn't provide information about how to deal with border values
    which do not have 8 well-defined neighbours. This function uses zero-padding technique.

    Parameters
    ----------
    rate_map_raw: np.ma.MaskedArray
        Non-smoothed rate map: n x m array where cell value is the firing rate,
        masked at locations with low occupancy

    Returns
    -------
    coherence: float
        see relevant literature (above)

    Notes
    -----
    BNT.+analyses.coherence(map)
        
    Copyright (C) 2019 by Simon Ball
    '''

    if 'smoothing_factor' in kwargs:
        smoothing_factor = kwargs['smoothing_factor']
    else:
        smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

    if isinstance(spatial_map, HaftingRateMap):
        _, rate_map_raw = spatial_map.get_rate_map(smoothing_factor)
    elif isinstance(spatial_map, SpatialSpikeTrain2D_signalstore):
        _, rate_map_raw = spatial_map.get_map('rate').get_rate_map(smoothing_factor)
    else:
        rate_map_raw = spatial_map


    kernel = np.array([[0.125, 0.125, 0.125],
                      [0.125, 0,     0.125],
                      [0.125, 0.125, 0.125]])

    avg_map = convolve(rate_map_raw, kernel, 'fill', fill_value=0)
    avg_map = avg_map.ravel()
    avg_map = np.nan_to_num(avg_map)

    rmap = np.copy(rate_map_raw)
    rmap = rmap.ravel()
    rmap = np.nan_to_num(rmap)

    coherence = np.corrcoef(avg_map, rmap)[0,1]

    return coherence


def place_field(spatial_map: SpatialSpikeTrain2D_signalstore | HaftingRateMap | np.ndarray, **kwargs):
    '''
    Locate place fields on a firing map.

    Identifies place fields in 2D firing map. Placefields are identified by
    using an adaptive threshold. The idea is that we start with a peak value as
    the threshold. Then we gradually decrease the threshold until the field
    area doesn't change any more or the area explodes (this means the threshold
    is too low).

    Parameters
    ----------
    firing_map: np.ndarray or np.ma.MaskedArray
        smoothed rate map.
        If supplied as an np.ndarray, it is assumed that the map takes values
        of np.nan at locations of zero occupancy. If supplied as an np.ma.MaskedArray,
        it is assumed that the map is masked at locations of zero occupancy
    
    Other Parameters
    ----------------
    min_bins: int
        Fields containing fewer than this many bins will be discarded. Default 9
    min_peak: float
        Fields with a peak firing rate lower than this absolute value will
        be discarded. Default 1 Hz
    min_mean: float
        Fields with a mean firing rate lower than this absolute value will
        be discarded. Default 0 Hz
    init_thresh: float
        Initial threshold to search for fields from. Must be in the range [0, 1].
        Default 0.96
    search_method: str
        Peak detection finding method. By default, use `skimage.morphology.local_maxima`
        Acceptable values are defined in `opexebo.defaults`. Not required if 
        peak_coords are provided
    peak_coords: array-like
        List of peak co-ordinates to consider instead of auto detection. [y, x].
        Default None

    Returns
    -------
    fields: list of dict
        coords: np.ndarray
            Coordinates of all bins in the firing field
        peak_coords: np.ndarray
            Coordinates peak firing rate [y,x]
        centroid_coords: np.ndarray
            Coordinates of centroid (decimal) [y,x]
        area: int
            Number of bins in firing field. [bins]
        bbox: tuple
            Coordinates of bounding box including the firing field
            (y_min, x_min, y_max, y_max)
        mean_rate: float
            mean firing rate [Hz]
        peak_rate: float
            peak firing rate [Hz]
        map: np.ndarray
            Binary map of arena. Cells inside firing field have value 1, all
            other cells have value 0
    fields_map : np.ndarray
        labelled integer image (i.e. background = 0, field1 = 1, field2 = 2, etc.)

    Raises
    ------
    ValueError
        Invalid input arguments
    NotImplementedError
        non-defined peack searching methods

    Notes
    --------
    BNT.+analyses.placefieldAdaptive

    https://se.mathworks.com/help/images/understanding-morphological-reconstruction.html

    Copyright (C) 2018 by Vadim Frolov, (C) 2019 by Simon Ball, Horst Obenhaus
    '''

    if isinstance(spatial_map, HaftingRateMap) or isinstance(spatial_map, SpatialSpikeTrain2D_signalstore):
        if 'smoothing_factor' in kwargs:
            smoothing_factor = kwargs['smoothing_factor']
        else:
            smoothing_factor = spatial_map.session_metadata.session_object.smoothing_factor

        if isinstance(spatial_map, HaftingRateMap):
            firing_map, _ = spatial_map.get_rate_map(smoothing_factor)
        elif isinstance(spatial_map, SpatialSpikeTrain2D_signalstore):
            rate_obj = spatial_map.get_map('rate')
            if rate_obj == None:
                firing_map, _ = HaftingRateMap(spatial_map).get_rate_map(smoothing_factor)
            else:
                firing_map, _ = rate_obj.get_rate_map(smoothing_factor)
    else:
        firing_map = spatial_map

    ##########################################################################
    #####                   Part 1: Handle inputs
    # Get keyword arguments
    min_bins = kwargs.get("min_bins", default.firing_field_min_bins)
    min_peak = kwargs.get("min_peak", default.firing_field_min_peak)
    min_mean = kwargs.get("min_mean", default.firing_field_min_mean)
    init_thresh = kwargs.get("init_thresh", default.initial_search_threshold)
    search_method = kwargs.get("search_method", default.search_method)
    peak_coords = kwargs.get("peak_coords", None)
    debug = kwargs.get("debug", False)

    if not 0 < init_thresh <= 1:
        raise err.ArgumentError("Keyword 'init_thresh' must be in the range [0, 1]."\
                         f" You provided {init_thresh}")
    try:
        search_method = search_method.lower()
    except AttributeError:
        raise err.ArgumentError("Keyword 'search_method' is expected to be a string"\
                         f" You provided a {type(search_method)} ({search_method})")
    if search_method not in default.all_methods:
        raise err.ArgumentError("Keyword 'search_method' must be left blank or given a"\
                         f" value from the following list: {default.all_methods}."\
                         f" You provided '{search_method}'.")

    global_peak = np.nanmax(firing_map)
    if np.isnan(global_peak) or global_peak == 0:
        if debug:
            print(f"Terminating due to invalid global peak: {global_peak}")
        return [], np.zeros_like(firing_map)

    # Construct a mask of bins that the animal never visited (never visited -> true)
    # This needs to account for multiple input formats.
    # The standard that I want to push is that firing_map is type MaskedArray
        # In this case, the cells that an animal never visited have firing_map.mask[cell]=True
        # while firing_map.data[cell] PROBABLY = 0
    # An alternative is the BNT standard, where firing_map is an ndarray
        # In this case, the cells never visited are firing_map[cell] = np.nan
    # In either case, we need to get out the following:
        # finite_firing_map is an ndarray (float) where unvisted cells have a
        # meaningfully finite value (e.g. zero, or min())
        # mask is an ndarray (bool) where unvisited cells are True, all other cells are False

    if isinstance(firing_map, np.ma.MaskedArray):
        occupancy_mask = firing_map.mask
        finite_firing_map = firing_map.data.copy()
        finite_firing_map[np.isnan(firing_map.data)] = 0

    else:
        occupancy_mask = np.zeros_like(firing_map).astype('bool')
        occupancy_mask[np.isnan(firing_map)] = True
        finite_firing_map = firing_map.copy()
        finite_firing_map[np.isnan(firing_map)] = 0

    structured_element = morphology.disk(1)
    image_eroded = morphology.erosion(finite_firing_map, structured_element)
    fmap = morphology.reconstruction(image_eroded, finite_firing_map)
    
    ##########################################################################
    #####                   Part 2: find local maxima
    # Based on the user-requested search method, find the co-ordinates of local maxima
    if peak_coords is None:
        if search_method == default.search_method:
            peak_coords = peak_search(fmap, **kwargs)
        elif search_method == "sep":
            #fmap = finite_firing_map
            peak_coords = peak_search(fmap, **kwargs)
        else:
            raise NotImplementedError("The search method you have requested (%s) is"\
                                      " not yet implemented" % search_method)

    # obtain value of found peaks
    found_peaks = finite_firing_map[peak_coords[:, 0], peak_coords[:, 1]]

    # leave only peaks that satisfy the threshold
    good_peaks = (found_peaks >= min_peak)
    peak_coords = peak_coords[good_peaks, :]


    ##########################################################################
    #####    Part 3: from local maxima get fields by expanding around maxima
    max_value = np.max(fmap)
    # prevent peaks with small values from being detected
    # SWB - This causes problems where a local peak is next to a cell that the animal never went
    # As that risks the field becoming the entire null region
    # Therefore, adding 2nd criterion to avoid adding information where none was actually known.
    fmap[np.logical_and(fmap < min_peak, fmap > 0.01)] = max_value * 1.5

    # this can be confusing, but this variable is just an index for the vector
    # peak_linear_ind
    peaks_index = np.arange(len(peak_coords))
    fields_map = np.zeros(fmap.shape, dtype=np.int32)
    field_id = 1
    for i, peak_rc in enumerate(peak_coords):
        # peak_rc == [row, col]

        # select all peaks except the current one
        other_fields = peak_coords[peaks_index != i]
        if other_fields.size > 0:
            other_fields_linear = np.ravel_multi_index(
                        multi_index=(other_fields[:, 0], other_fields[:, 1]),
                        dims=fmap.shape, order='F')
        else:
            other_fields_linear = []

        used_th = init_thresh
        res = _area_change(fmap, occupancy_mask, peak_rc, used_th,
                           used_th-0.02, other_fields_linear)
        initial_change = res['acceleration']
        area2 = res['area2']
        first_pixels = np.nan
        if np.isnan(initial_change):
            for j in np.linspace(used_th+0.01, 1., 4):
                # Thresholds get higher, area should tend downwards to 1
                # (i.e. only including the actual peak)
                res = _area_change(fmap, occupancy_mask, peak_rc, j, j-0.01, other_fields_linear)
                initial_change = res['acceleration']
                area1 = res['area1']
                area2 = res['area2']
                # initial_change is the change from area1 to area 2
                # area2>area1 -> initial_change > 1
                # area2<area1 -> initial_change < 1
                # area 2 is calculated with lower threshold - should usually be larger
                first_pixels = res['first_pixels']
                if not np.isnan(initial_change) and initial_change > 0:
                    # True is both area1 and area2 are valid
                    # Weird conditonal from Vadim - initial change will EITHER:
                    # be greater than zero (can't get a negative area to give negative % change)
                    # OR be NaN (which will always yield false when compared to a number)
                    used_th = j - 0.01
                    break

            if np.isnan(initial_change) and not np.isnan(area1):
                # For the final change
                pixels = np.unravel_index(first_pixels, fmap.shape, 'F')
                fmap[pixels] = max_value * 1.5
                fields_map[pixels] = field_id
                field_id = field_id + 1

            if np.isnan(initial_change):
                # failed to extract the field
                # Do nothing and continue for-loop
                pass

        pixel_list = _expand_field(fmap, occupancy_mask, peak_rc, initial_change, area2,
                                   other_fields_linear, used_th)
        if np.any(np.isnan(pixel_list)):
            _, pixel_list, _ = _area_for_threshold(fmap, occupancy_mask,
                                                   peak_rc, used_th+0.01,
                                                   other_fields_linear)
        if len(pixel_list) > 0:
            pixels = np.unravel_index(pixel_list, fmap.shape, 'F')
        else:
            pixels = []

        fmap[pixels] = max_value * 1.5
        fields_map[pixels] = field_id
        field_id = field_id + 1


    ##########################################################################
    #####     Part 4: Determine which, if any, fields meet filtering criteria
    regions = measure.regionprops(fields_map)

    fields = []
    fields_map = np.zeros(finite_firing_map.shape)  # void it as we can eliminate some fields

    for region in regions:
        field_map = finite_firing_map[region.coords[:, 0], region.coords[:, 1]]
        mean_rate = np.nanmean(field_map)
        num_bins = len(region.coords)

        peak_rate = np.nanmax(field_map)
        peak_relative_index = np.argmax(field_map)
        peak_coords = region.coords[peak_relative_index, :]

        if num_bins >= min_bins and mean_rate >= min_mean:
            field = {}
            field['coords'] = region.coords
            field['peak_coords'] = peak_coords
            field['area'] = region.area
            field['bbox'] = region.bbox
            field['centroid_coords'] = region.centroid
            field['mean_rate'] = mean_rate
            field['peak_rate'] = peak_rate
            mask = np.zeros(finite_firing_map.shape)
            mask[region.coords[:, 0], region.coords[:, 1]] = 1
            field['map'] = mask

            fields.append(field)

            fields_map[region.coords[:, 0], region.coords[:, 1]] = len(fields)
        elif debug:
            # Print out some information about *why* the field failed
            if num_bins < min_bins:
                print("Field size too small (%d)" % num_bins)
            if mean_rate < min_mean:
                print("Field mean rate too low (%.2f Hz)" % mean_rate)
        else:
            # Field too small and debugging information not needed
            # Do nothing
            pass
    #fields_map = np.ma.masked_where(occupancy_mask, fields_map)
    return (fields, fields_map)


#########################################################
################        Helper Functions
#########################################################


def _expand_field(image, occupancy_mask, peak_rc, initial_change,
                  initial_area, other_fields_linear, initial_th):
    '''
    Adaptive placefield detection:
        Start with a threshold around 80%, step down in ~0.02
        Measure field at threshold
        If the field is invalid, take the previous iteration
        If field size has decreased, keep count
        If field size has increased by less than 300% of initial_change AND the
          field has decreased in size fewer than 3 times in a row
            If the field size hasn't changed in 10 steps, return the current field size
        Else
    '''
    pixel_list = np.nan
    last_area = initial_area
    last_pixels = []
    num_not_changing = 0
    num_decrease = 0

    num_steps = int((initial_th - 0.2) / 0.02) + 1
    for threshold in np.linspace(initial_th, 0.2, num_steps):
        threshold = np.round(threshold, 2)

        area, pixels, is_bad = _area_for_threshold(image, occupancy_mask, peak_rc,
                                                   threshold, other_fields_linear)
        if np.isnan(area) or is_bad:
            pixel_list = last_pixels
            break
        current_change = area / last_area
        if current_change < 1:
            num_decrease = num_decrease + 1
        else:
            num_decrease = 0

        if np.floor(current_change / initial_change) <= 2 and num_decrease < 3:
            if current_change == 1:
                num_not_changing = num_not_changing + 1
            else:
                num_not_changing = 0

            if num_not_changing < 10:
                last_area = area
                last_pixels = pixels
                continue

        pixel_list = last_pixels
        break

    peak_linear = np.ravel_multi_index(multi_index=(peak_rc[0], peak_rc[1]),
                                       dims=image.shape, order='F')
    # last_pixels is a vector, peak_linear is a single value
    if np.any(last_pixels == peak_linear):
        # good field
        pixel_list = last_pixels

    return pixel_list


def _area_change(image, occupancy_mask, peak_rc, first, second, other_fields_linear):
    ''' Compare the change in field area based on two threshold values

    If either threshold results in an invalid field (based on criteria in
    _area_for_threshold() ), then return NaNs to signal this fact

    Otherwise return information about the results from the two thresholds, and
    the % change in area

    Params
    ------
    image : np.ndarray
        Image, i.e. rate map being considered
    occupancy_mask: np.ndarray
        binary mask. True where the animal spent zero time
    peak_rc : tuple
        [row, col] of local maxima
    first : float
        first threshold
    second : float
        second threshold
    other_fields_linear : np.ndarray
        All other local maxima except the ony under consideration
    '''
    results = {'acceleration': np.nan, 'area1': np.nan, 'area2': np.nan,
               'first_pixels': np.nan,
               'second_pixels': np.nan}

    area1, first_pixels, is_bad1 = _area_for_threshold(image, occupancy_mask, peak_rc, first,
                                                       other_fields_linear)
    if np.isnan(area1) or is_bad1:
        return results

    area2, second_pixels, is_bad2 = _area_for_threshold(image, occupancy_mask, peak_rc, second,
                                                        other_fields_linear)
    if np.isnan(area2) or is_bad2:
        return results

    acceleration = area2 / area1 #* 100
    results['acceleration'] = acceleration
    results['area1'] = area1
    results['area2'] = area2
    results['first_pixels'] = first_pixels
    results['second_pixels'] = second_pixels

    return results


def _area_for_threshold(image, occupancy_mask, peak_rc, threshold, other_fields_linear):
    '''Calculate the area of the field defined by the local maxima 'peak_rc' and
    the relative thresholding value 'threshold'

    In addition, determine:
        * is the field contiguous (good), or does it contain voids? (bad)
        * Does the field extend to include other local maxima?

    1 - Based on the threshold, find the binary map of the field
    2 - Determine if there are any voids
        If there are any voids, then return an area of np.nan
    3 - Determine the co-ordinates of all cells inside the field. The area is
        given by the number of cells
        If any included cell ALSO appears in the list of 'other_fields_linear'
        then another local maxima has been included. In that case, return is_bad=True

    returns
    -------
    area : int
        Number of cells in field OR np.nan if field contains holes
    area_linear_indicies : np.ndarray
        indicies of all cells within field if field is valid
    is_bad : bool
        True IF field includes a second local maxima or IF field contains holes
    '''
    area = np.nan
    # Field is bad if it contains any other peak
    is_bad = False

    peak_value = image[peak_rc[0], peak_rc[1]]
    threshold_value = peak_value * threshold
    mask = (image >= threshold_value)

    # Mask includes all pixels above the desired threshold, including other disconnected fields
    # use morphology.label to exclude disconnected fields
    # connectivity=1 means only consider vertical/horizontal connections
    labeled_img = morphology.label(mask, connectivity=1)

    # we need to leave only one label that corresponds to the peak
    target_label = labeled_img[peak_rc[0], peak_rc[1]]
    labeled_img[labeled_img != target_label] = 0
    labeled_img[labeled_img == target_label] = 1

    #labelled_img only includes cells that are:
    #   Above the current threshold
    #   Connected (V/H) to the currently considered local maxima

    # calclate euler_number by hand rather than by regionprops
    # This yields results that are more similar to Matlab's regionprops
    # NOTE - this uses scipy.ndimage.morphology, while most else uses skimage.morphology
    filled_image = ndimage.binary_fill_holes(labeled_img)
    euler_array = (filled_image != labeled_img)  # True where holes were filled in

    euler_array = np.maximum((euler_array*1) - (occupancy_mask*1), 0)
    # Ignore filled-in holes if it is due to the animal never visiting that location
    # Convert both arrays to integer, subtract one from the other, and replace resulting -1 values with 0
    # NOTE! np.maximum is element-wise, i.e. it returns an array. This is DIFFERENT to np.max, which returns a float.

    euler_objects = morphology.label(euler_array, connectivity=2) # connectivity=2 : vertical, horizontal, and diagonal
    num = np.max(euler_objects) # How many holes were filled in
    euler_number = -num + 1

    if euler_number <= 0:
        # If any holes existed, then return this
        is_bad = True
        return (area, [], is_bad)

    regions = measure.regionprops(labeled_img)
    area = np.sum(labeled_img == 1)
    area_linear_indices = np.ravel_multi_index(multi_index=(regions[0].coords[:, 0],
            regions[0].coords[:, 1]), dims=image.shape, order='F') # co-ordinates of members of field
    if len(other_fields_linear) > 0:
        is_bad = len(np.intersect1d(area_linear_indices, other_fields_linear)) > 0 # True if any other local maxima occur within this field

    return (area, area_linear_indices, is_bad)

def firing_rate_vs_time(spike_times, pos_t, window: int) -> tuple:
    '''
        Computes firing rate as a function of time

        Params:
            times (np.ndarray):
                Array of spike_times of when the neuron fired
            pos_t (np.ndarray):
                Time array of entire experiment
            window (int):
                Defines a time window in milliseconds.

            *Example*
            window: 400 means we will attempt to collect firing data in 'bins' of
            400 millisecods before computing the firing rates.

        Returns:
            tuple: firing_rate, firing_time
            --------
            rate_vector (np.ndarray):
                Array containing firing rate data across entire experiment
            firing_time: (np.ndarray):
                spike_times of when firing occured
        '''

    # spike_times = spike_class.event_times

    # pos_t = np.array(spike_class.time_index)

    # if type(spike_times) == list:
    #     spike_times = np.array(spike_times)

    # if type(spike_times) == list:
    #     spike_times = np.asarray(spike_times)

    # Initialize zero time elapsed, zero spike events, and zero bin times.
    time_elapsed = 0
    number_of_elements = 1
    bin_time = [0,0]

    # Initialzie empty firing rate and firing time arrays
    firing_rate = [0]
    firing_time = [0]

    # Collect firing data in bins of 400ms
    for i in range(1, len(spike_times)):
        if time_elapsed == 0:
            # Set a bin start time
            bin_time_start = spike_times[i-1]

        # Increment time elapsed and spike number as spike event times are iterated over
        time_elapsed += (spike_times[i] - spike_times[i-1])
        number_of_elements += 1

        # If the elapsed time exceeds 400ms
        if time_elapsed > (window/1000):
            # Set bin end time
            bin_time_end = bin_time_start + time_elapsed
            # Compute rate, and add element to firing_rate array
            firing_rate.append(number_of_elements/time_elapsed)
            firing_time.append( (bin_time_start + bin_time_end)/2 )
            # Reset elapsed time and spiek events number
            time_elapsed = 0
            number_of_elements = 0

    rate_vector = np.zeros((len(pos_t), 1))
    index_values = []
    for i in range(len(firing_time)):
        index_values.append(  (np.abs(pos_t - firing_time[i])).argmin()  )

    firing_rate = np.array(firing_rate).reshape((len(firing_rate), 1))
    rate_vector[index_values] = firing_rate

    # spike_class.stats_dict['rate_vector'] = rate_vector

    return rate_vector, firing_time



   
