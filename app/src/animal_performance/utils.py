"""
Utility functions for animal performance analysis.
"""
import os, sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.measure import find_contours
import cv2
from scipy import signal
from scipy.ndimage import distance_transform_cdt
from PIL import Image
from skimage import measure, morphology
import warnings
import animal_performance.errors as err
import animal_performance.defaults as default
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Suppress the ConfigurationMissingWarning that Astropy triggers
    from astropy.convolution import convolve, Gaussian2DKernel, Gaussian1DKernel

def insert_to_df_summary(df_summary, index, update_dict):
    if len(df_summary) == 0:
        # If df_summary is empty, initialize it with the keys of update_dict
        df_summary = pd.DataFrame(columns=update_dict.keys())
    # return pd.concat([df_summary, pd.DataFrame([update_dict])], ignore_index=True)
    else:
        for key, value in update_dict.items():
            df_summary.loc[index, key] = value
    return df_summary

# def insert_to_df_summary(df_summary, index, value_dict):
#     """
#     Replaces Excel-style assignments with direct DataFrame updates.
#     Args:
#         df_summary: the DataFrame to update
#         index: row index where data should be inserted
#         value_dict: a dictionary of {column_name: value}
#     """
#     if index not in df_summary.index:
#         df_summary.loc[index] = pd.Series()
#     for key, val in value_dict.items():
#         df_summary.at[index, key] = val
#     return df_summary

class Position2D():
    def __init__(self, input_dict, **kwargs):
        # self.subject = subject
        # self.limb = limb # e.g. head
        self._input_dict = input_dict

        self.t, self.x, self.y, self.arena_height, self.arena_width, self.session_metadata = self._read_input_dict()
        self.time_index = self.t

        if 'session_metadata' in kwargs:
            if self.session_metadata != None: 
                print('Ses metadata is in the input dict and init fxn, init fnx will override')
            self.session_metadata = kwargs['session_metadata']

        self._input_dict = input_dict
        self.stats_dict = {}

        self.arena_size = (self.arena_height, self.arena_width)

    def _read_input_dict(self):
        t, x, y, arena_height, arena_width, session_metadata = None, None, None, None, None, None

        for key in self._input_dict:
            if key == 't' or key == 'time':
                t = self._input_dict['t']
            elif key == 'rate' and 'x' in self._input_dict:
                t = np.arange(0, len(self._input_dict['x']) / self._input_dict['rate'], 1 / self._input_dict['rate'])
            elif key == 'x':
                x = self._input_dict['x']
            elif key == 'y':
                y = self._input_dict['y']
            elif key == 'session_metadata':
                session_metadata = self._input_dict['session_metadata']
            elif key == 'arena_height':
                arena_height = self._input_dict['arena_height']
            elif key == 'arena_width':
                arena_width = self._input_dict['arena_width']

        return t, x, y, arena_height, arena_width, session_metadata

def _interpolate_matrix(matrix, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST):
    '''
        Interpolate a matrix using cv2.INTER_LANCZOS4.
    '''
    return cv2.resize(matrix, dsize=new_size,
                      interpolation=cv2_interpolation_method)

def flat_disk_mask(rate_map):
    masked_rate_map = disk_mask(rate_map)
    # masked_rate_map.data[masked_rate_map.mask] = 0
    # masked_rate_map.data[masked_rate_map.mask] = np.nan
    # print(np.unique(masked_rate_map.data))
    # return  masked_rate_map.data
    copy = np.copy(rate_map).astype(np.float32)
    copy[masked_rate_map.mask] = np.nan
    return copy

def disk_mask(matrix):
        y_segments, x_segments = matrix.shape

        y_center, x_center = (y_segments-1)/2, (x_segments-1)/2

        mask_r = min(x_segments, y_segments)/2

        mask_y,mask_x = np.ogrid[-y_center:y_segments-y_center, -x_center:x_segments-x_center]
        mask = mask_x**2 + mask_y**2 > mask_r**2

        masked_matrix = np.ma.array(matrix, mask=mask)

        return masked_matrix

def _compute_resize_ratio(arena_size: tuple) -> tuple:

    '''
        Computes resize ratio which is later used to shape all ratemaps
        and ocupancy maps to have the same shape.

        Params:
            arena_size (tuple): (arena_height, arena_width)
                Dimensions of arena. Arena is assumed to be square/rectangular

        Returns:
            Tuple: (row_resize, column_resize)
            --------
            row_resize (int):
                Number of rows to resize to
            column_resize (int):
                Number of columns to resize to
    '''

    # Each maps largest dimension is always set to 64
    # base_resolution = 16
    base_resolution = 64
    resize_ratio = arena_size[0] / arena_size[1] # height/width

    # If width is smaller than height, set height resize to 64 and row to less
    if resize_ratio > 1:
      row_resize = int(np.ceil(base_resolution*(1/resize_ratio)))
      column_resize = base_resolution

    # If length is smaller than width, set width resize to 64 and height to less
    elif resize_ratio < 1:
        row_resize = base_resolution
        column_resize = int(np.ceil(base_resolution*(resize_ratio)))

    # If the arena is perfectly square, set both side resizes to 64
    else:
        row_resize = base_resolution
        column_resize = base_resolution

    return row_resize, column_resize

def _resize_numpy2D(array: np.ndarray, x: int, y: int) -> np.ndarray:

    '''
        Resizes a numpy array.

        Params:
            array (numpy.ndarray):
                Numpy array to be resized
            x (int):
                Resizing row number (length)
            y (int):
                Resizing column number (width)

        Returns:
            array (numpy.ndarray): Resized array with new dimensions (array.shape = (x,y))
    '''

    array = Image.fromarray(array)
    array = array.resize((x,y))
    array = np.array(array)

    return array

def _gkern(kernlen: int, std: int) -> np.ndarray:

    '''
        Returns a 2D Gaussian kernel array.

        Params:
            kernlen, std (int):
                Kernel length and standard deviation

        Returns:
            np.ndarray:
                gkern2d
    '''

    gkern1d = signal.windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


 
def deserialize_dataarray(data_object):
        """Deserializes a data object.
        Arguments:
            data_object {dict} -- The data object to deserialize.
        Returns:
            dict -- The deserialized data object.
        """
        attrs = data_object.attrs.copy()
        for key, value in attrs.items():
            if isinstance(value, str):
                value = value.replace("'", '"')
                if value.lower() == 'true':
                    attrs[key] = True
                elif value.lower() == 'false':
                    attrs[key] = False
                elif value.lower() == 'none':
                    attrs[key] = None
                elif value.startswith('{'):
                    attrs[key] = json.loads(value)
            if isinstance(value, np.ndarray):
                attrs[key] = value.tolist()
        data_object.attrs = attrs
        return data_object


def speed2D(x, y, t):
    """
    Calculate an averaged/smoothed speed from 2D position data.
    
    Parameters
    ----------
    x, y : array
        Position coordinates
    t : array
        Time points
        
    Returns
    -------
    array
        Speed values
    """
    N = len(x)
    v = np.zeros((N, 1))

    for index in range(1, N-1):
        v[index] = np.sqrt((x[index + 1] - x[index - 1]) ** 2 + 
                          (y[index + 1] - y[index - 1]) ** 2) / \
                   (t[index + 1] - t[index - 1])

    v[0] = v[1]
    v[-1] = v[-2]

    return v

def speed_bins(lower_speed: float, higher_speed: float, pos_v: np.ndarray,
               pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray) -> tuple:

    '''
        Selectively filters position values of subject travelling between
        specific speed limits.

        Params:
            lower_speed (float):
                Lower speed bound (cm/s)
            higher_speed (float):
                Higher speed bound (cm/s)
            pos_v (np.ndarray):
                Array holding speed values of subject
            pos_x, pos_y, pos_t (np.ndarray):
                X, Y coordinate tracking of subject and timestamps

        Returns:
            Tuple: new_pos_x, new_pos_y, new_pos_t
            --------
            new_pos_x, new_pos_y, new_pos_t (np.ndarray):
                speed filtered x,y coordinates and timestamps
    '''

    # Initialize empty array that will only be populated with speed values within
    # specified bounds
    choose_array = []

    # Iterate and select speeds
    for index, element in enumerate(pos_v):
        if element > lower_speed and element < higher_speed:
            choose_array.append(index)

    # construct new x,y and t arrays
    new_pos_x = np.asarray([ float(pos_x[i]) for i in choose_array])
    new_pos_y = np.asarray([ float(pos_y[i]) for i in choose_array])
    new_pos_t = np.asarray([ float(pos_t[i]) for i in choose_array])

    return new_pos_x, new_pos_y, new_pos_t

def circle_vals(x, y, d, ang):
    """
    Calculate points on a circle.
    
    Parameters
    ----------
    x, y : float
        Center coordinates
    d : float
        Diameter
    ang : array
        Angles in radians
        
    Returns
    -------
    tuple
        (x coordinates, y coordinates)
    """
    r = d / 2
    xp = np.multiply(r, np.cos(ang))
    yp = np.multiply(r, np.sin(ang))
    xp = np.add(xp, x)
    yp = np.add(yp, y)
    return xp.reshape((len(xp), 1)), yp.reshape((len(yp), 1))

def extract_arena_size_from_shape(arena_record):
    """
    Custom helper function to extract arena size specifically as a tuple of
    (height, width). Also returns boolean tag to denote arena as cylinder
    or not. Both outputs are required as inputs to the ratemap functions.
    """
    shape = arena_record['arena_shape']
    unit_of_measure = arena_record['unit_of_measure']
    if shape == 'cylinder':
        diameter = float(arena_record['diameter'])
        if unit_of_measure == 'm':
            diameter *= 100
        arena_size = (diameter, diameter)
        isCylinder = True
    elif shape == 'rectangle':
        arena_height = float(arena_record['arena_height'])
        arena_width = float(arena_record['arena_width'])
        if unit_of_measure == 'm':
            arena_height *= 100
            arena_width *= 100
        arena_size = (arena_height, arena_width)
        isCylinder = False

    return arena_size, isCylinder

def rgb2gray(rgb):
    """
    Convert RGB image to grayscale.
    
    Parameters
    ----------
    rgb : array
        RGB image array
        
    Returns
    -------
    array
        Grayscale image
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_hd_score_for_cluster(hd_hist):
    """
    Calculate head direction score for a cluster.
    
    Parameters
    ----------
    hd_hist : array
        Head direction histogram
        
    Returns
    -------
    float
        Head direction score
    """
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles * np.pi / 180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist) / sum(hd_hist)
    toty = sum(dy * hd_hist) / sum(hd_hist)
    r = np.sqrt(totx * totx + toty * toty)
    return r

def centreBox(posx, posy):
    """
    Computes the centre of the box defined by posx and posy.

    Parameters:
      posx: array-like, x-coordinates of the points.
      posy: array-like, y-coordinates of the points.

    Returns:
      list: [x, y] coordinates of the centre.
    """
    maxX = np.max(posx)
    minX = np.min(posx)
    maxY = np.max(posy)
    minY = np.min(posy)
    
    # Define the corners of the box
    NE = [maxX, maxY]
    NW = [minX, maxY]
    SW = [minX, minY]
    SE = [maxX, minY]
    
    centre = findCentre(NE, NW, SW, SE)
    return centre

def findCentre(NE, NW, SW, SE):
    """
    Calculates the centre of the box from the corner coordinates.
    The centre is the intersection of the diagonals.

    Parameters:
      NE, NW, SW, SE: list or array containing the [x, y] coordinates of the corners.
      
    Returns:
      list: [x, y] coordinates of the centre.
    """
    # Compute slopes for the diagonals
    a = (NE[1] - SW[1]) / (NE[0] - SW[0])  # slope for NE-SW diagonal
    b = (SE[1] - NW[1]) / (SE[0] - NW[0])  # slope for SE-NW diagonal
    c = SW[1]
    d = NW[1]
    x = (d - c + a * SW[0] - b * NW[0]) / (a - b)
    y = a * (x - SW[0]) + c
    return [x, y]

def bwarea(binary_image: np.ndarray) -> float:
    """
    MATLAB's bwarea computes the area as:
        area = N + 0.5*P + 0.25*V
    where:
      - N is the number of object pixels,
      - P is the perimeter (computed along the actual object boundary),
      - V is the number of vertices (assumed 4 for a simple closed shape).
    
    Parameters:
      binary_image: np.ndarray
          Input binary image (0/1 or bool).
    
    Returns:
      float: The estimated area.
    """
    # Ensure the image is boolean.
    bw = binary_image.astype(bool)
    
    # N: number of object pixels.
    N = np.sum(bw)
    
    # Compute perimeter by extracting the contour at level 0.5,
    # which gives the actual continuous boundary length.
    contours = find_contours(bw.astype(float), level=0.5)
    if contours:
        # Use the longest contour if there are several.
        longest = max(contours, key=lambda c: c.shape[0])
        # Compute the Euclidean length along the contour.
        P = np.sum(np.sqrt(np.sum(np.diff(longest, axis=0)**2, axis=1)))
    else:
        P = 0
    
    # Assume V = 4 for a simple shape (e.g. a square).
    V = 4 if N > 0 else 0
    
    area = N + 0.5 * P + 0.25 * V
    return area



def remBadTrack(x, y, t, threshold):
    """
    Remove bad tracking points based on speed threshold.
    
    Parameters
    ----------
    x, y : array
        Position coordinates
    t : array
        Time points
    threshold : float
        Speed threshold
        
    Returns
    -------
    tuple
        (filtered x, filtered y, filtered t)
    """
    diffX = np.diff(x, axis=0)
    diffY = np.diff(y, axis=0)
    diffT = np.diff(t, axis=0)
    
    speed = np.sqrt(diffX ** 2 + diffY ** 2) / diffT
    
    bad_indices = np.where(speed > threshold)[0] + 1
    x[bad_indices] = np.nan
    y[bad_indices] = np.nan
    t[bad_indices] = np.nan
    
    return x, y, t

def is_circle(points, corners):
    """
    Determines if the arena is circular based on the points and corners.

    Parameters:
      points: np.ndarray
          Array of points (x, y) coordinates.
      corners: np.ndarray
          Array of corners (x, y) coordinates.

    Returns:
      bool: True if the arena is circular, False otherwise.
    """
    circle_bool = []
    for corner in range(4):
        if corner == 0:  # NE corner
            bool_val = (points[:, 0] >= 0) * (points[:, 1] >= 0)
        elif corner == 1:  # NW Corner
            bool_val = (points[:, 0] < 0) * (points[:, 1] >= 0)
        elif corner == 2:  # SW Corner
            bool_val = (points[:, 0] < 0) * (points[:, 1] < 0)
        else:  # SE corner
            bool_val = (points[:, 0] > 0) * (points[:, 1] < 0)
        
        current_points = points[bool_val, :]
        circle_bool.append(np.sum((current_points[:, 0] > corners[corner, 0]) * 
                                (current_points[:, 1] > corners[corner, 1])))

    return sum(circle_bool) < 1

def plot_bin_metrics_heatmaps(bin_coverage_png, bin_distance_png, n_bins, bin_coverage, bin_distances):
    """    
    Parameters:
      dimensions: array-like, [xmin, xmax, ymin, ymax] of the arena.
      n_bins: int, number of bins per dimension (e.g. 4 for a 4x4 grid).
      bin_coverage: (n_bins x n_bins) array of percentage coverage values.
      bin_distances: (n_bins x n_bins) array of total distance values.
      output_path_prefix: str, prefix for the output file names.
    
    The function saves two PNG files:
      - {output_path_prefix}_coverage_heatmap.png
      - {output_path_prefix}_distance_heatmap.png
    """

    # --- Coverage Heatmap ---
    fig_cov, ax_cov = plt.subplots(figsize=(n_bins*2, n_bins*2))  # Adjust size based on number of bins
    # origin='lower' to have bin[0,0] at the bottom-left corner
    im_cov = ax_cov.imshow(bin_coverage, cmap='Greys', origin='lower')
    ax_cov.set_title("Bin Coverage Heatmap (%)")
    # Set tick labels to indicate bin positions (optional)
    ax_cov.set_xticks(np.arange(n_bins))
    ax_cov.set_yticks(np.arange(n_bins))
    ax_cov.set_xticklabels([f"{x+1:.1f}" for x in range(n_bins)])
    ax_cov.set_yticklabels([f"{y+1:.1f}" for y in range(n_bins)])
    divider = make_axes_locatable(ax_cov)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig_cov.colorbar(im_cov, cax=cax, label="Coverage")
    plt.tight_layout()
    fig_cov.savefig(bin_coverage_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig_cov)

    # --- Distance Heatmap ---
    fig_dist, ax_dist = plt.subplots(figsize=(n_bins*2, n_bins*2))  # Adjust size based on number of bins
    im_dist = ax_dist.imshow(bin_distances, cmap='Greys', origin='lower')
    ax_dist.set_title("Bin Distance Heatmap (meters)")
    ax_dist.set_xticks(np.arange(n_bins))
    ax_dist.set_yticks(np.arange(n_bins))
    ax_dist.set_xticklabels([f"{x+1:.1f}" for x in range(n_bins)])
    ax_dist.set_yticklabels([f"{y+1:.1f}" for y in range(n_bins)])
    divider = make_axes_locatable(ax_dist)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig_dist.colorbar(im_dist, cax=cax, label="Distance")
    plt.tight_layout()
    fig_dist.savefig(bin_distance_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig_dist)

    print(f"Heatmap images saved as:\n  Coverage: {bin_coverage_png}\n  Distance: {bin_distance_png}")



def plot_bin_metrics(dimensions, n_bins, bin_coverage, bin_distances, bin_metrics_png):
    """
    Create and save a plot of an arena divided into bins (grid) with text annotations
    in each bin showing the bin's coverage and distance metrics.
    
    Parameters:
      dimensions: array-like, [xmin, xmax, ymin, ymax] defining the arena.
      n_bins: int, number of bins along each axis (e.g., 4 for a 4x4 grid).
      bin_coverage: numpy array of shape (n_bins, n_bins) with percentage coverage per bin.
      bin_distances: numpy array of shape (n_bins, n_bins) with total distance per bin.
      output_path: str, the file path to save the plot.
    """
    xmin, xmax, ymin, ymax = dimensions
    # Define bin boundaries in coordinate space.
    x_bins = np.linspace(xmin, xmax, n_bins + 1)
    y_bins = np.linspace(ymin, ymax, n_bins + 1)
    
    # Create a figure and axes.
    fig, ax = plt.subplots(figsize=(20, 10.6066667), dpi=150)
    
    # Draw vertical grid lines.
    for x in x_bins:
        ax.axvline(x=x, color='k', lw=1)
    # Draw horizontal grid lines.
    for y in y_bins:
        ax.axhline(y=y, color='k', lw=1)
    
    # Annotate each bin with its coverage and distance.
    for i in range(n_bins):
        for j in range(n_bins):
            # Compute the center of the bin.
            cx = (x_bins[i] + x_bins[i+1]) / 2
            cy = (y_bins[j] + y_bins[j+1]) / 2
            # Format the text: coverage (in %) and distance.
            text = f"Coverage: {bin_coverage[j, i]:.1f}%\nDistance: {bin_distances[j, i]:.2f}"
            ax.text(cx, cy, text, ha='center', va='center', fontsize=12, color='black')
    
    # Set the axis limits to the arena dimensions.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    
    # Optionally hide the axis.
    ax.axis('off')
    
    # Adjust layout and save the figure.
    plt.tight_layout()
    plt.savefig(bin_metrics_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Plot saved as {bin_metrics_png}")


def attribute_segment_to_bins(x1, y1, x2, y2, x_bins, y_bins):
    """
    Given a segment from (x1, y1) to (x2, y2) and grid boundaries (x_bins, y_bins),
    return a dictionary where keys are (bin_i, bin_j) and values are the length of
    the segment attributed to that bin.
    
    Parameters:
      x1, y1, x2, y2: float
          Endpoints of the segment.
      x_bins: 1D array of bin boundaries in x.
      y_bins: 1D array of bin boundaries in y.
    
    Returns:
      contributions: dict with keys (i,j) and values = subsegment length.
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_length = np.sqrt(dx**2 + dy**2)
    if seg_length == 0:
        return {}
    
    # Start with t=0 and t=1
    t_vals = [0.0, 1.0]
    
    # Find t where the line crosses vertical boundaries (ignore the extreme boundaries)
    for X in x_bins[1:-1]:
        if dx != 0:
            t = (X - x1) / dx
            if 0 < t < 1:
                t_vals.append(t)
    
    # Find t where the line crosses horizontal boundaries
    for Y in y_bins[1:-1]:
        if dy != 0:
            t = (Y - y1) / dy
            if 0 < t < 1:
                t_vals.append(t)
    
    # Remove duplicates and sort
    t_vals = np.unique(t_vals)
    t_vals.sort()
    
    contributions = {}
    # For each subsegment, compute its midpoint and attribute its length to the appropriate bin.
    for i in range(len(t_vals) - 1):
        t_start = t_vals[i]
        t_end = t_vals[i+1]
        t_mid = (t_start + t_end) / 2.0
        xm = x1 + dx * t_mid
        ym = y1 + dy * t_mid
        # Determine which bin this midpoint falls into.
        # np.digitize returns indices starting at 1, so subtract 1.
        bin_i = np.digitize(xm, x_bins) - 1
        bin_j = np.digitize(ym, y_bins) - 1
        subseg_length = seg_length * (t_end - t_start)
        contributions[(bin_i, bin_j)] = contributions.get((bin_i, bin_j), 0) + subseg_length
    return contributions

def gkern(kernlen: int, std: int) -> np.ndarray:

    '''
        Returns a 2D Gaussian kernel array.

        Params:
            kernlen, std (int):
                Kernel length and standard deviation

        Returns:
            np.ndarray:
                gkern2d
    '''

    gkern1d = signal.windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def compute_bin_metrics(posx, posy, BWdfill_total, BWdfill_filled, dimensions, image_resolution, n_bins=4):
    """
    Divide the arena (square) into an n_bins x n_bins grid and compute for each bin:
      - percentage coverage: fraction (0-100%) of the bin area in the binary mask that is "filled"
      - total distance: sum of Euclidean distances for segments, proportionally attributed to bins if they cross boundaries.
    
    Parameters:
      posx, posy: 1D numpy arrays (in arena coordinate space) for animal positions.
      BWdfill_total: 2D binary numpy array representing total area (e.g., arena border mask).
      BWdfill_filled: 2D binary numpy array representing the filled area (e.g., coverage mask).
      dimensions: array-like, [xmin, xmax, ymin, ymax] of the arena.
      image_resolution: tuple (height, width) in pixels of the BWdfill image.
      n_bins: number of bins per dimension (default 4 for a 4x4 grid).
    
    Returns:
      coverage: a (n_bins x n_bins) numpy array with percentage coverage per bin.
      distance: a (n_bins x n_bins) numpy array with total distance traveled in each bin.
    """
    H, W = image_resolution
    xmin, xmax, ymin, ymax = dimensions
    dx = xmax - xmin
    dy = ymax - ymin
    
    # Create bin boundaries in coordinate space:
    x_bins = np.linspace(xmin, xmax, n_bins + 1)
    y_bins = np.linspace(ymin, ymax, n_bins + 1)
    
    # Initialize output arrays.
    coverage = np.zeros((n_bins, n_bins))
    distance = np.zeros((n_bins, n_bins))
    
    # --- Compute coverage per bin ---
    for i in range(n_bins):
        for j in range(n_bins):
            # Bin boundaries in coordinate space:
            x0, x1 = x_bins[i], x_bins[i+1]
            y0, y1 = y_bins[j], y_bins[j+1]
            
            # Map these boundaries to pixel coordinates:
            col0 = int(np.floor(((x0 - xmin) / dx) * (W - 1)))
            col1 = int(np.ceil(((x1 - xmin) / dx) * (W - 1)))
            row0 = int(np.floor(((y0 - ymin) / dy) * (H - 1)))
            row1 = int(np.ceil(((y1 - ymin) / dy) * (H - 1)))
            
            bin_total_area = bwarea(BWdfill_total[row0:row1, col0:col1])
            bin_total_filled = bwarea(BWdfill_filled[row0:row1, col0:col1])
            if bin_total_area > 0:
                bin_percent = (bin_total_filled / bin_total_area) * 100
            else:
                bin_percent = 0
            coverage[j, i] = bin_percent  # note: j for row, i for column

    # --- Compute total distance per bin with proportional attribution ---
    # Ensure posx, posy are 1D arrays.
    posx = posx.flatten()
    posy = posy.flatten()
    for k in range(len(posx) - 1):
        x1_pt, y1_pt = posx[k], posy[k]
        x2_pt, y2_pt = posx[k+1], posy[k+1]
        # Get contributions for the segment.
        contrib = attribute_segment_to_bins(x1_pt, y1_pt, x2_pt, y2_pt, x_bins, y_bins)
        for (i, j), d in contrib.items():
            # Check index bounds
            if 0 <= i < n_bins and 0 <= j < n_bins:
                # Convert to meters if positions are in centimeters (adjust divisor as needed)
                distance[j, i] += d / 100.0
    
    return coverage, distance

def temp_occupancy_map(pos_t, pos_x, pos_y, arena_size, smoothing_factor, interp_size=(64,64), useMinMaxPos=False) -> np.ndarray:

    '''
        Computes the position, or occupancy map, which is a 2D numpy array
        enconding subjects position over entire experiment.

        Params:
            pos_x, pos_y and pos_t (np.ndarray):
                Arrays of the subjects x and y coordinates, as
                well as timestamp array.
            arena_size (tuple):
                Arena dimensions (width (x), height (y) in meters)
            resolution (float):
                Resolution of occupancy map (in meters)
            kernlen, std : kernel size and standard deviation (i.e 'spread') for convolutional smoothing
                of 2D map

            Returns:
                np.ndarray: occ_map_smoothed, occ_map_raw, coverage_map
    '''
    # pos_x, pos_y, pos_t = position.x, position.y, position.t

    # if isinstance(position, Position2D):
    #     arena_size = (position.arena_height, position.arena_width)
    # else:
    #     arena_size = position.arena_size

    if useMinMaxPos:
        min_x = min(pos_x)
        max_x = max(pos_x)
        min_y = min(pos_y)
        max_y = max(pos_y)
    else:
        min_x = [-arena_size[1]/2]
        max_x = [arena_size[1]/2] # width 
        min_y = [-arena_size[0]/2]
        max_y = [arena_size[0]/2] # height
    row_resize, column_resize = interp_size

    # Initialize empty map
    occ_map_raw = np.zeros((row_resize,column_resize))
    coverage_map = np.zeros((row_resize,column_resize))
    row_values = np.linspace(max_y,min_y,row_resize)
    column_values = np.linspace(min_x,max_x,column_resize)

  
    # Generate the raw occupancy map
    for i in range(1, len(pos_t)):

        row_index = np.abs(row_values - pos_y[i]).argmin()
        column_index = np.abs(column_values - pos_x[i]).argmin()
        occ_map_raw[row_index][column_index] += pos_t[i] - pos_t[i-1]
        coverage_map[row_index][column_index] = 1

    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)
    # Normalize and smooth with scaling facotr
    occ_map_normalized = occ_map_raw / pos_t[-1]
    occ_map_smoothed = cv2.filter2D(occ_map_normalized,-1, gkern(kernlen,std))
    # dilate coverage map
    kernel = np.ones((2,2))
    coverage_map = cv2.dilate(coverage_map, kernel, iterations=1)
    occ_map_smoothed = occ_map_smoothed/max(occ_map_smoothed.flatten())
    return occ_map_smoothed, occ_map_raw, coverage_map

def temp_spike_map(pos_x: np.ndarray, pos_y: np.ndarray, pos_t: np.ndarray,
                arena_size: tuple, spike_x: np.ndarray, spike_y: np.ndarray,
                smoothing_factor: int, interp_size=(64,64)) -> np.ndarray:

    # Kernel size
    kernlen = int(smoothing_factor*8)
    # Standard deviation size
    std = int(0.2*kernlen)
    min_x = 0
    max_x = arena_size[1] # width 
    min_y = 0
    max_y = arena_size[0] # height

    # Resize ratio
    row_resize, column_resize = interp_size

    # Instantiate empty maps
    spike_map_raw = np.zeros((row_resize,column_resize))
    
    # Load spike data and set up arrays to map spike timestamps to subject position
    row_values = np.linspace(max_x, min_x, row_resize)
    column_values = np.linspace(min_y,max_y, column_resize)

    # Generate raw spike map
    for i in range(len(spike_x)):
        row_index = np.abs(row_values - spike_y[i]).argmin()
        column_index = np.abs(column_values - spike_x[i]).argmin()
        spike_map_raw[row_index][column_index] += 1

    # Remove low spike counts from spike map (20th percentile)
    # Smooth spike map (must happen before resizing)
    spike_map_smooth = cv2.filter2D(spike_map_raw,-1, gkern(kernlen, std))

    # Resize maps
    spike_map_smooth = spike_map_smooth/max(spike_map_smooth.flatten())

    return spike_map_smooth, spike_map_raw
   
def temp_spike_map_new(pos_x, pos_y, arena_size, spike_x, spike_y, smoothing_factor, interp_size=(64, 64), useMinMaxPos=False):
    kernlen = int(smoothing_factor * 8)
    std = int(0.2 * kernlen)

    if useMinMaxPos:
        min_x, max_x = np.min(pos_x), np.max(pos_x)
        min_y, max_y = np.min(pos_y), np.max(pos_y)
    else:
        min_x = [-arena_size[1]/2]
        max_x = [arena_size[1]/2] # width 
        min_y = [-arena_size[0]/2]
        max_y = [arena_size[0]/2] # height


    row_resize, column_resize = interp_size
    spike_map_raw = np.zeros((row_resize, column_resize))
    row_values = np.linspace(max_x, min_x, row_resize)
    column_values = np.linspace(min_y, max_y, column_resize)
    row_index = np.abs(row_values[:, np.newaxis] - spike_y).argmin(axis=0)
    column_index = np.abs(column_values[:, np.newaxis] - spike_x).argmin(axis=0)
    np.add.at(spike_map_raw, (row_index, column_index), np.ones_like(row_index))
    spike_map_smooth = cv2.filter2D(spike_map_raw, -1, gkern(kernlen, std))
    spike_map_smooth = spike_map_smooth / np.max(spike_map_smooth)
    return spike_map_smooth, spike_map_raw


def compute_resize_ratio(arena_size: tuple) -> tuple:

    '''
        Computes resize ratio which is later used to shape all ratemaps
        and ocupancy maps to have the same shape.

        Params:
            arena_size (tuple): (arena_height, arena_width)
                Dimensions of arena. Arena is assumed to be square/rectangular

        Returns:
            Tuple: (row_resize, column_resize)
            --------
            row_resize (int):
                Number of rows to resize to
            column_resize (int):
                Number of columns to resize to
    '''

    # Each maps largest dimension is always set to 64
    # base_resolution = 16
    base_resolution = 64
    resize_ratio = arena_size[0] / arena_size[1] # height/width

    # If width is smaller than height, set height resize to 64 and row to less
    if resize_ratio > 1:
      row_resize = int(np.ceil(base_resolution*(1/resize_ratio)))
      column_resize = base_resolution

    # If length is smaller than width, set width resize to 64 and height to less
    elif resize_ratio < 1:
        row_resize = base_resolution
        column_resize = int(np.ceil(base_resolution*(resize_ratio)))

    # If the arena is perfectly square, set both side resizes to 64
    else:
        row_resize = base_resolution
        column_resize = base_resolution

    return row_resize, column_resize

def gaussian_smooth(data: np.ndarray | np.ma.MaskedArray, sigma, **kwargs):
    '''Smooth provided data with a Gaussian kernel

    The smoothing is done with a routine from the astronomical package astropy
    Like scipy.ndimage.gaussian_filter, this does not handle MaskedArrays - but
    it handles NaNs much better. Specifically, astropy.convolution.convolve
    replaces NaN values with an interpolation across the void region.

    Therefore, to handle masked arrays, the data at masked positions *is
    replaced by np.nan* prior to smoothing, and thus avoids influencing nearby,
    unmasked cells. The masked cells are then returned to their original values
    prior to return.

    The package is discussed at
    http://docs.astropy.org/en/stable/convolution/index.html

    Parameters
    ----------
    data: np.ndarray or np.ma.MaskedArray
        Data that will be smoothed
    sigma: float
        Standard deviations for Gaussian kernel in units of pixels/bins
    mask_fill: float, optional
        The value that masked locations should be treated as
        This can either be provided as an absolute number (e.g. 0), or nan
        If nan, then each masked location will get a value by interpolating
        from nearby cells. This will only apply if the input is a
        MaskedArray to start with. If the input is a standard np.ndarray,
        then no values will be substituted, even if there are nans present.
    circular: bool, optional
        If True, then smoothing at the edge of the array will be handled in
        a circular manner, i.e. the value to the left of data[0] will be
        data[-1]. If False, the edge will be handled by padding with values
        equal to the boundary value. Default False

    Returns
    -------
    smoothed_data: np.ndarray or np.ma.MaskedArray
        Smoothed data

    Notes
    --------
    BNT.+general.smooth
    http://docs.astropy.org/en/stable/convolution/index.html
    https://github.com/astropy/astropy/issues/6511

    Copyright (C) 2019 by Simon Ball
    '''
    d = data.ndim
    if  d == 2:
        kernel = Gaussian2DKernel(x_stddev=sigma)
    elif d == 1:
        kernel = Gaussian1DKernel(stddev=sigma)
    else:
        raise NotImplementedError("This function currently supports smoothing"\
                f" 1D, 2D data. You have provided {d} dimensional data")

    mask_fill = kwargs.get('mask_fill', default.mask_fill)
    circular = kwargs.get("circular", False)
    if not isinstance(circular, bool):
        raise ValueError("You must provide a boolean (True/False) value for"\
                         f" keyword 'circular'. You provided {circular}, which"\
                         f" is type {type(circular)}")

    working_data = data.copy()
    if type(data) == np.ma.MaskedArray:
        working_data[data.mask] = mask_fill

    width = int(4*sigma)

    if width == 0:
        width = 1

    if circular:
        smoothed_data = convolve(working_data, kernel, boundary="wrap")
        # Don't bother with padding. Use the values from the other end of the 
        # array, i.e. imagine the array wrapped around a cylinder
    elif not circular:
        working_data = np.pad(working_data, pad_width=width, mode='symmetric')
        # pad the outer boundary to depth "width
        # The padding values are based on reflecting at the border
        # mode='symmetrical' results in
        # [0, 1, 2, 3, 4] -> [1,0  ,0,1,2,3,4,  4,3]
        # mode='reflect' results in
        # [0, 1, 2, 3, 4] -> [2,1  ,0,1,2,3,4,  3,2]
        # i.e. changing whether the reflection axis is outside the original data
        # or overlaid on the outermost row
    
        smoothed_data = convolve(working_data, kernel, boundary='extend')
        # Because of the padding, the boundary mode isn't really relevant
        # By choosing a large width, the edge effects arising from this additional
        # padding (boundary='extend') is minimised

        if d == 2:
            smoothed_data = smoothed_data[width:-width, width:-width]
        elif d == 1:
            smoothed_data = smoothed_data[width:-width]
        else: # This condition should never happen, due to checking above
            raise NotImplementedError("This function currently supports smoothing"\
                    f" 1D, 2D data. You have provided {d} dimensional data")
        # We have to get rid of the padding that we previously added, and the only
        # way to do that is slicing, which is NOT dimensional-agnostic
        # There may be a more elegant solution than if/else, but this will do now

    if type(data) == np.ma.MaskedArray:
        smoothed_data = np.ma.masked_where(data.mask, smoothed_data)
        smoothed_data.data[data.mask] = data.data[data.mask]

    assert smoothed_data.shape == data.shape, "Output array is a different shape to input array"

    return smoothed_data
def bin_width_to_bin_number(length, bin_width):
    '''This conversion is done in several separate places, and must be done the
    same in every case. Therefore, refactor into a quick helper function.
    
    In cases where the bin_width is not a perfect divisor of length, the actual
    bins will be slightly smaller
    
    Parameters
    ----------
    length: float or np.ndarray
        Length of a side to be divided into equally spaced bins
    bin_width: float
        Dimension of a square bin or pixel
    
    Returns
    -------
    num_bins: int or np.ndarray
        Same type as `length`. Integer number of bins.
    '''
    if type(length) in (list, tuple):
        length = np.array(length)
    num_bins = np.ceil(length / bin_width).astype(int)
    return num_bins

################################################################################

""""""""""""""""""""""""""" From Opexebo https://pypi.org/project/opexebo/ """""""""""""""""""""""""""

def peak_search(image, **kwargs):
    """Given a 1D or 2D array, return a list of co-ordinates of the local 
    maxima or minima
    
    Multiple searching techniques are provided:
        
        * `default`: uses `skimage.morphology.get_maxima`
        * `sep`: uses the Python wrapper to the Source Extractor astronomy tool
          to identify peaks
    
    Parameters
    ----------
    image: np.ndarray
        1D or 2D array of data
    search_method : str, optional, {"default", "sep"}
    mask : np.ndarray, optional
        Array of masked locations in the image with the same dimensions.
        Locations where the mask value is True are ignored for the purpose of
        searching.
    maxima: bool, optional
        [`default` search method only] Define whether to search for maxima or
        minima in the provided array
    null_background: bool
        [`sep` search method only] Set the image background to zero for
        calculation purposes rather than attempt to calculate a background
        gradient. This should generally be True, as our images are not directly
        comparable to standard telescope output
    threshold : float, optional
        [`sep` search method only] Threshold for identifiying maxima area
    
    Returns
    -------
    peak_coords: tuple
        Co-ordinates of peaks, in the form ((x0, x1, x2...), (y0, y1, y2...))
    
    
    Notes
    --------
    Copyright (C) 2019 by Simon Ball
    """
    
    search_method = kwargs.get("search_method", default.search_method)
    get_maxima = kwargs.get("maxima", True)
    
    if search_method not in default.all_methods:
        raise ValueError("Keyword 'search_method' must be left blank or given a"\
                    " value from the following list: %s. You provided '%s'."\
                    % (default.all_methods, search_method) )
    if search_method != "default" and not get_maxima:
        raise NotImplementedError("Local minima detection is currently only"\
                            " implemented for the 'default' search method")
        
    if search_method == default.search_method:
        peak_coords = peak_search_skimage(image, **kwargs)
    elif search_method == "sep":
        peak_coords = peak_search_sep_wrapper(image, **kwargs)
    else:
        raise NotImplementedError("The search method you have requested (%s) is"\
                                  " not yet implemented" % search_method)
        
    return peak_coords
         

def peak_search_skimage(image, **kwargs):
    '''Default peak detection method:
        skimage.morphology.get_m**ima (either minima or maxima)
    Since skimage doesn't handle masked arrays, the masking is a bit of a bodge
    job here. The basic plan is as follows:
        * Set the area of the image covered by the mask to a value that cannot include a maxima (or minima, as appropriate)
        * Search for peak coordinates
        * Check if, after rounding, any of the peaks are outside the image dimensions
        * Check if, after rounding, any of the peaks are extremely close to the mask
    
    Since the mask is a ahrd-edged area, if there is even a slight rise just
    outside it, spurious peaks can be detected. Therefore, we automatically reject
    any peaks for a short distance outside the actual mask
    
    '''
    connectivity = 2
    get_maxima = kwargs.get("maxima", True)
    mask = kwargs.get("mask", np.zeros(image.shape, dtype=bool))
    image_copy = image.copy()
    if get_maxima:
        image_copy[mask] = np.nanmin(image_copy)
        regionalMaxMap = morphology.local_maxima(image_copy, connectivity=connectivity, allow_borders=True)
    else:
        image_copy[mask] = np.nanmax(image_copy)
        regionalMaxMap = morphology.local_minima(image_copy, connectivity=connectivity, allow_borders=True)
    labelled_max = measure.label(regionalMaxMap, connectivity=connectivity)
    regions = measure.regionprops(labelled_max)
    peak_coords = np.zeros(shape=(len(regions), 2), dtype=np.int32)
    
    distance_from_mask = distance_transform_cdt(image_copy * (1-mask))

    for i, props in enumerate(regions):
        y0, x0 = props.centroid
        peak = np.array([y0, x0])

        # ensure that there are no peaks off the map (due to rounding)
        peak[peak < 0] = 0
        for j in range(image_copy.ndim):
            if peak[j] > image_copy.shape[j]:
                peak[j] = image_copy.shape[j] - 1
        
        peak_index = tuple(np.round(peak, 0).astype(int)) # indexing with a floating point array sucks, so convert to a more convenient form
        if distance_from_mask[peak_index] > 2*connectivity:
            peak_coords[i, :] = peak
    return peak_coords


def peak_search_sep_wrapper(firing_map, **kwargs):
    ''' Wrapper around the 'sep' Peak Search method
    Because the 'sep' package is a nightmare to install, and not used for most
    analysis routines, this wrapper allows most users to ignore it
    
    If the user tries to invoke the 'sep' routines, this will try to do so
    If it fails due to ModuleNotFound, it will use the default algorithm instead
    with a warning to the user
    '''
    try:
        import sep
        return peak_search_sep(firing_map, **kwargs)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("The package 'sep' is missing from your system."\
                " You can invoke an alternative algorithm that does not depend on"\
                " 'sep' by assigning a different value to 'search_method'."\
                " Alternatively, install 'sep' on your system:"\
                " 'pip install sep'")


def peak_search_sep(firing_map, **kwargs):
    '''Peak search using sep, a Python wrapper for a standard astronomy library.
    sep is typically used to identify astonomical objects in telescope images
    Sep requires copies of the arrays that are C-ordered (Python default is 
    Fortran-ordered, the difference is row vs column-major).
    
    TODO: The behaviour of sep with masks needs to be investigated - where the 
    edge of the image is masked, objects are still sometimes found. Example: 
        {'path': 'N:\\davidcr\\84932\\19032019',
         'basename': '19032019s1',
         'tetrode': 6,
         'cell': 23}
    '''
    import sep
    
    mask = kwargs.get("mask", np.zeros(firing_map.shape, dtype=bool))
    null_background = kwargs.get("null_background", True)
    threshold = kwargs.get("threshold", 0.2)
    
    tmp_firing_map = firing_map.copy('C')
    tmp_mask = mask.copy(order='C')
    
    if null_background:
        bkg = sep.Background(np.zeros_like(tmp_firing_map))
    else:
        bkg = sep.Background(tmp_firing_map, mask=tmp_mask, fw=2, fh=2, \
                     bw=int(tmp_firing_map.shape[0]), bh=int(tmp_firing_map.shape[1]))

    init_fields = sep.extract(tmp_firing_map-bkg, mask=tmp_mask, thresh=threshold, \
                          err=bkg.globalrms)

    peak_coords = np.zeros(shape=(len(init_fields), 2), dtype=np.int32)

    for i, props in enumerate(init_fields):
        peak = np.array([props['y'], props['x']])
        peak = np.round(peak)
        # ensure that there are no peaks off the map (due to rounding)
        peak[peak < 0] = 0
        for j in range(firing_map.ndim):
            if peak[j] > firing_map.shape[j]:
                peak[j] = firing_map.shape[j] - 1

        peak_coords[i, :] = peak
    return peak_coords


def fit_ellipse(X, Y):
    '''
    Fit an ellipse to the provided set of X, Y co-ordinates

    Based on the approach taken in
    Authors: Andrew Fitzgibbon, Maurizio Pilu, Bob Fisher
    Reference: "Direct Least Squares Fitting of Ellipses", IEEE T-PAMI, 1999
        
    @Article{Fitzgibbon99,
    author = "Fitzgibbon, A.~W.and Pilu, M. and Fisher, R.~B.",
    title = "Direct least-squares fitting of ellipses",
    journal = pami,
    year = 1999,
    volume = 21,
    number = 5,
    month = may,
    pages = "476--480"
    }
    and implemented in MatLab by Vadim Frolov

    Parameters
    ----------
    X - np.ndarray
        x co-ordinates of points
    Y - np.ndarray
        y co-ordinates of points

    Returns
    -------
    x_centre : float
        Centre of ellipse
    y_centre : float
        Centre of ellipse
    Ru : float
        Major radius
    Rv : float
        Minor radius
    theta_rad : float
        Ellipse orientation (in radians)


    Notes
    --------
    BNT.+general.fitEllipse
    opexebo.analysis.gridscore

    Copyright (C) 2019 by Simon Ball

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
    '''

    if X.size != Y.size:
        raise ValueError("X and Y must be the same length. You provided"\
                         " (%d, %d) values respectively" % (X.size, Y.size))
    if not np.isfinite(X).all():
        raise ValueError("X cannot contain values that are nan or inf."\
                         f" You provided {X}")
    if not np.isfinite(Y).all():
        raise ValueError("X cannot contain values that are nan or inf."\
                         f" You provided {Y}")
    # Normalise the data and move it to the origin
    mx = np.mean(X)
    my = np.mean(Y)

    sx = 0.5 * (np.max(X) - np.min(X))
    sy = 0.5 * (np.max(Y) - np.min(Y))

    x = (X - mx) / sx
    y = (Y - my) / sy

    # Construct design matrix and scatter matrix
    D = np.zeros((X.size, 6))
    D[:, 0] = x*x
    D[:, 1] = x*y
    D[:, 2] = y*y
    D[:, 3] = x
    D[:, 4] = y
    D[:, 5] = np.ones(X.size)

    S = D.T @ D

    # Construct contraint matrix
    C = np.zeros((6, 6))
    C[1, 1] = 1
    C[2, 0] = -2
    C[0, 2] = -2
    # Solve eigensystem
    # Break into blocks
    tmpA = S[:3, :3]
    tmpB = S[:3, 3:]
    tmpC = S[3:, 3:]
    tmpD = C[:3, :3]
    tmpE = np.linalg.inv(tmpC) @ tmpB.T

    eval_x, evec_x = np.linalg.eig(np.linalg.inv(tmpD) @ (tmpA - (tmpB@tmpE)))

    # Find the positive eigenvalue (as det(tmpD) < 0)
    idx = np.argmax(np.logical_and(np.real(eval_x) < 1e-8, np.isfinite(eval_x)))
    vec_x = np.real(evec_x[:, idx]) # vector associated with idx
    vec_y = -tmpE @ vec_x
    A = np.concatenate((vec_x, vec_y))

    # Un-normalise
    par = np.zeros(6)
    par[0] = A[0] * sy * sy
    par[1] = A[1] * sx * sy
    par[2] = A[2] * sx * sx
    par[3] = (-2*A[0]*sy*sy*mx) - (A[1]*sx*sy*my) + (A[3]*sx*sy*sy)
    par[4] = (-A[1]*sx*sy*mx) - (2*A[2]*sx*sx*my) + (A[4]*sx*sx*sy)
    par[5] = (A[0]*sy*sy*mx*mx) + (A[1]*sx*sy*mx*my) + (A[2]*sx*sx*my*my) \
                - (A[3]*sx*sy*sy*mx) - (A[4]*sx*sx*sy*my) + (A[5]*sx*sx*sy*sy)

    # Geometric radii and centres 
    theta_rad = 0.5*np.arctan2(par[1], par[0]-par[2])
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    sin2 = sin_t * sin_t
    cos2 = cos_t * cos_t
    Ao = par[5]
    Au = (par[3] * cos_t) + (par[4] * sin_t)
    Av = (-par[3] * sin_t) + (par[4] * cos_t)
    Auu = (par[0] * cos2) + (par[2] * sin2) + (par[1] * cos_t * sin_t)
    Avv = (par[0] * sin2) + (par[2] * cos2) - (par[1] * cos_t * sin_t)

    tu_centre = -Au / (2*Auu)
    tv_centre = -Av / (2*Avv)
    w_centre = Ao - (Auu*tu_centre*tu_centre) - (Avv*tv_centre*tv_centre)

    x_centre = (tu_centre * cos_t) - (tv_centre * sin_t)
    y_centre = (tu_centre * sin_t) + (tv_centre * cos_t)

    Ru = -w_centre / Auu
    Rv = -w_centre / Avv

    Ru = np.sqrt(np.abs(Ru)) * np.sign(Ru)
    Rv = np.sqrt(np.abs(Rv)) * np.sign(Rv)

    return x_centre, y_centre, Ru, Rv, theta_rad

def _prepare_out_argument(out, dtype, expected_shape):
    if out is None:
        return np.empty(expected_shape, dtype=dtype)

    if out.shape != expected_shape:
        raise ValueError("Output array has incorrect shape.")
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous.")
    if out.dtype != np.float64:
        raise ValueError("Output array must be double type.")
    return out

def bin_width_to_bin_number(length, bin_width):
    '''This conversion is done in several separate places, and must be done the
    same in every case. Therefore, refactor into a quick helper function.
    
    In cases where the bin_width is not a perfect divisor of length, the actual
    bins will be slightly smaller
    
    Parameters
    ----------
    length: float or np.ndarray
        Length of a side to be divided into equally spaced bins
    bin_width: float
        Dimension of a square bin or pixel
    
    Returns
    -------
    num_bins: int or np.ndarray
        Same type as `length`. Integer number of bins.
    '''
    if type(length) in (list, tuple):
        length = np.array(length)
    num_bins = np.ceil(length / bin_width).astype(int)
    return num_bins


