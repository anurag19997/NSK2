"""
Core functionality for animal performance analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage import color
from scipy import ndimage
import skimage.measure
import importlib.util
import sys
import imageio
import skimage
import matplotlib
import pandas as pd
from openpyxl.utils.cell import get_column_letter
import math
from pathlib import Path


from utils import speed2D, circle_vals, rgb2gray, is_circle, bwarea, compute_bin_metrics, plot_bin_metrics_heatmaps, plot_bin_metrics


def batch_map_via_queries(uow, session_queries, settings_dict, save_dir, analysis_fn, analysis_name="Results"):
    csv_header = settings_dict['header']
    run_number = 1
    root_path = os.path.join(save_dir, f'{analysis_name}_{run_number}')
    while os.path.isdir(root_path):
        run_number += 1
        root_path = os.path.join(save_dir, f'{analysis_name}_{run_number}')
    os.mkdir(root_path)

    # Save settings
    with open(os.path.join(root_path, analysis_name + "_settings.txt"), 'w') as f:
        for k, v in settings_dict.items():
            f.write(f"{k}: {v}\n")
        f.close()

    # Initialize dataframe
    headers = [k for k, v in csv_header.items() if v]
    df_summary = pd.DataFrame(columns=headers)

    visited = set()
    for query in session_queries:
        session_name = query['data_name']
        if session_name in visited:
            continue
        visited.add(session_name)

        session_data = uow.data.get(query['schema_ref'], session_name)
        row = analysis_fn(session_data, settings_dict, session_name, root_path)
        df_summary.loc[len(df_summary)] = row

    summary_csv_path = os.path.join(root_path, f'{analysis_name.lower()}_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")


def get_animal_performance(session_data, settings_dict, session_name, root_path):
    """
    Analyze animal performance metrics from a query result containing position data.
    
    Parameters
    ----------
    query_result : dict
        Query result containing position data and metadata
    save_figures_directory : str
        Directory to save output figures
    settings_dict : dict
        Dictionary containing analysis settings (e.g., ppm, arena_size, etc.)
        
    Returns
    -------
    None
    Saves (total_area, total_filled, percent, total_distance, speed etc) or None if analysis fails
    """
    try:
        posx = session_data[:,0].to_numpy().squeeze()
        posy = session_data[:,1].to_numpy().squeeze()
        post = session_data[:,2].to_numpy().squeeze()
        print(posx.shape, posy.shape, post.shape)
        save_figures_directory = Path(root_path)
        ## adjust based on calulations of ppm and mice length
        plot_linewidth = 10.5
        # plot_linewidth = 10000 ## avg mice len is 2.25 according to the ppm calcluations
        if len(posx) == 0:
            print('There are no valid positions (all NaNs)')
        posx = posx.reshape(-1, 1)
        posy = posy.reshape(-1, 1)
        print('calculating the total distance for the .pos file: %s ' % session_data.attrs.get('session_data_ref')["data_name"])
        diffX = np.diff(posx, axis=0)
        diffY = np.diff(posy, axis=0)
        dist_sample = np.sqrt((diffX ** 2) + (diffY ** 2))
        total_distance = np.sum(dist_sample)
        total_distance = total_distance/100  # convert to meters
        print('Calculating coverage!')
        points = np.hstack((posx.reshape((len(posx), 1)), posy.reshape((len(posy), 1))))
        dimensions = np.array([np.amin(posx), np.amax(posx), np.amin(posy), np.amax(posy)])

        arena_fig = plt.figure(figsize=(20, 10.6066667), dpi=150)
        ax = arena_fig.add_subplot(111)
        ax.plot(posx, posy, 'r-', lw=plot_linewidth)
        ax.axis('off')

        bin_x = np.linspace(np.amin(posx), np.amax(posx), 11)
        bin_y = np.linspace(np.amin(posy), np.amax(posy), 11)
        corners = np.array([[bin_x[-2], bin_y[-2]], [bin_x[1], bin_y[-2]], 
                            [bin_x[1], bin_y[1]], [bin_x[-2], bin_y[1]]])
        
        plot_corners = False
        if plot_corners:
            # plots to see the test if it is a square or cirlce. I essentially
            # determine if there is data at the corners of the arena
            fig_corners = plt.figure()
            ax_corners = fig_corners.add_subplot(111)
            ax_corners.plot(posx, posy, 'r-')
            ax_corners.plot(corners[0:2, 0], corners[0:2, 1], 'g')
            ax_corners.plot(corners[1:3, 0], corners[1:3, 1], 'g')
            ax_corners.plot(corners[[0, 3], 0], corners[[0, 3], 1], 'g')
            ax_corners.plot(corners[[2, 3], 0], corners[[2, 3], 1], 'g')
            ax_corners.set_title('Corners of the bins')

        circle_bool = is_circle(points, corners)
        coverage_figure = plt.figure(figsize=(20, 10.6066667), dpi=150) ## Params are based on screen resolution and trying to match values from matlab code
        ax_coverage = coverage_figure.add_subplot(111)
        if not circle_bool:
            print('Arena detected as being square')

            # Define bins for x and y
            bins = np.linspace(np.amin(posx), np.amax(posx), 20)
            bin_edges = np.hstack((bins[:-1].reshape((-1, 1)), bins[1:].reshape((-1, 1))))

            # Define rectangle for square arena
            rectangle_points = np.array([
                [np.amin(posx), np.amax(posy)],  # NW
                [np.amax(posx), np.amax(posy)],  # NE
                [np.amax(posx), np.amin(posy)],  # SE
                [np.amin(posx), np.amin(posy)],  # SW
                [np.amin(posx), np.amax(posy)]   # NW (closing the loop)
            ])

            border = ax_coverage.plot(rectangle_points[:, 0], rectangle_points[:, 1], 'b', lw=plot_linewidth / 10)
            # Plot arena boundaries
            ax_coverage.plot(rectangle_points[:, 0], rectangle_points[:, 1], 'b', lw=plot_linewidth / 10)
            ax_coverage.plot(posx, posy, 'r-', lw=plot_linewidth)

            # Set axis limits
            ax_coverage.set_xlim([min([dimensions[0], np.amin(rectangle_points[:, 0])]) - 0.5,
                                    max([dimensions[1], np.amax(rectangle_points[:, 0])]) + 0.5])
            ax_coverage.set_ylim([min([dimensions[2], np.amin(rectangle_points[:, 1])]) - 0.5,
                                    max([dimensions[3], np.amax(rectangle_points[:, 1])]) + 0.5])
            ax.set_xlim([min([dimensions[0], np.amin(rectangle_points[:, 0])]) - 0.5,
                            max([dimensions[1], np.amax(rectangle_points[:, 0])]) + 0.5])
            ax.set_ylim([min([dimensions[2], np.amin(rectangle_points[:, 1])]) - 0.5,
                                    max([dimensions[3], np.amax(rectangle_points[:, 1])]) + 0.5])
                    

        else:
            print('Arena detected as being circular')

            bins = np.linspace(np.amin(posx), np.amax(posx), 50)
            bin_edges = np.hstack((bins[:-1].reshape((-1, 1)), bins[1:].reshape((-1, 1))))

            # Compute radii
            radii = np.array([np.abs(np.amin(posx)), np.amax(posx), np.abs(np.amin(posy)), np.amax(posy)])
            for bin_value in range(len(bin_edges)):
                bin_bool = (posx >= bin_edges[bin_value, 0]) * (posx < bin_edges[bin_value, 1])
                if sum(bin_bool) == 0:
                    continue

                posx_bin = posx[bin_bool]
                posy_bin = posy[bin_bool]
                max_val = np.amax(posy_bin)
                max_i = np.where(posy_bin == max_val)[0][0]
                min_val = np.amin(posy_bin)
                min_i = np.where(posy_bin == min_val)[0][0]
                append_radii = np.array([np.sqrt(max_val ** 2 + posx_bin[max_i] ** 2),
                                        np.sqrt(min_val ** 2 + posx_bin[min_i] ** 2)])
                radii = np.concatenate((radii, append_radii))

            # Generate circle for circular arena
            step = 0.001
            ang = np.arange(np.round((4 * np.pi + step) / step)) / (1 / step)
            xp, yp = circle_vals(0, 0, 2 * np.amax(radii), ang)
            border = ax_coverage.plot(xp, yp, 'b', lw=plot_linewidth / 10)
            ax_coverage.plot(xp, yp, 'b', lw=plot_linewidth / 10)
            ax_coverage.plot(posx, posy, 'r-', lw=plot_linewidth)
            

            ax_coverage.set_xlim([min([dimensions[0], np.amin(xp)]) - 0.5,
                                    max([dimensions[1], np.amax(xp)]) + 0.5])
            ax_coverage.set_ylim([min([dimensions[2], np.amin(yp)]) - 0.5,
                                    max([dimensions[3], np.amax(yp)]) + 0.5])
            ax.set_xlim([min([dimensions[0], np.amin(xp)]) - 0.5,
                                    max([dimensions[1], np.amax(xp)]) + 0.5])
            ax.set_ylim([min([dimensions[2], np.amin(yp)]) - 0.5,
                                    max([dimensions[3], np.amax(yp)]) + 0.5])
        
        # Save coverage figure
        cover_png_total = os.path.join(save_figures_directory, '%s_total.png' % session_data.attrs.get('session_data_ref')["data_name"])
        ax_coverage.axis('off')
        coverage_figure.patch.set_facecolor('white')
        coverage_figure.savefig(cover_png_total, dpi=150, pad_inches=0)
        print(f"Figure saved as {cover_png_total}")
        RGBA = imageio.imread(cover_png_total)
        RGB = color.rgba2rgb(RGBA)
        I = rgb2gray(RGB)
        I = np.round(I).astype('int32')

        BWs_x = ndimage.sobel(I, 0)  # horizontal derivative
        BWs_y = ndimage.sobel(I, 1)  # vertical derivative
        BWs = np.hypot(BWs_x, BWs_y)  # magnitude
        BWs *= 255.0 / np.amax(BWs) 
        BWsdil = ndimage.morphology.binary_dilation(BWs)
        BWdfill = ndimage.morphology.binary_fill_holes(BWsdil)
        total_area = bwarea(BWdfill)
        BWdfill_total = BWdfill.copy()
        border[0].remove()  # remove border 
        cover_png = os.path.join(save_figures_directory, '%s_coverage.png' % session_data.attrs.get('session_data_ref')["data_name"])
        ax.axis('off')
        arena_fig.savefig(cover_png, dpi=150, pad_inches=0, transparent=False)

        # reading in the positions without the arena trace
        RGBA = imageio.imread(cover_png)
        try:
            RGB = color.rgba2rgb(RGBA)
        except ValueError:
            RGB = RGBA
        I = rgb2gray(RGB)
        I = np.round(I).astype('int32')
        if np.amax(I) <= 1:
            # then the image was saved from numpy
            BWdfill = I < 1
        else:
            BWdfill = I < 255
        # finding the contours of the path so we can find the area

        coverage_fill_figure = plt.figure()
        ax_coverage_fill = coverage_fill_figure.add_subplot(111)
        ax_coverage_fill.imshow(BWdfill, cmap=plt.cm.gray)
        plt.title("BWdfill2")
        filled_png = os.path.join(save_figures_directory, '%s_filled.png' % session_data.attrs.get('session_data_ref')["data_name"])
        coverage_fill_figure.savefig(filled_png, bbox_inches='tight')
        total_filled = bwarea(BWdfill)
        BWdfill_filled = BWdfill.copy()
        image_resolution = (1591, 3000)
        n_bins = math.sqrt(settings_dict['num_bins'])
        bin_coverage, bin_distances = compute_bin_metrics(posx, posy, BWdfill_total,BWdfill_filled, dimensions, image_resolution, n_bins)
        bin_coverage_png = os.path.join(save_figures_directory, '%s_bin_coverage_heatmap.png' % session_data.attrs.get('session_data_ref')["data_name"])
        bin_distance_png = os.path.join(save_figures_directory, '%s_bin_distance_heatmap.png' % session_data.attrs.get('session_data_ref')["data_name"])
        bin_metrics_png = os.path.join(save_figures_directory, '%s_bin_metrics.png' % session_data.attrs.get('session_data_ref')["data_name"])
        plot_bin_metrics_heatmaps(bin_coverage_png, bin_distance_png, n_bins, bin_coverage, bin_distances)
        plot_bin_metrics(dimensions, n_bins, bin_coverage, bin_distances, bin_metrics_png)
        percent = (total_filled / total_area) * 100
        speed = speed2D(posx, posy, post)
        row = {
            "signature": session_name,
            "total_area": total_area,
            "total_filled": total_filled,
            "coverage": percent,
            "total_distance": total_distance,
            "min_speed": np.min(speed),
            "max_speed": np.max(speed),
            "mean_speed": np.mean(speed),
            "median_speed": np.median(speed)
        }
        centre_distance = 0.0
        centre_coverage = 0.0
        boundaries_distance = 0.0
        boundaries_coverage = 0.0
        num_bins = bin_distances.shape[0]  # should be 4
        for i in range(num_bins):
            for j in range(num_bins):
                row[f"bin_distance_r{i+1}_c{j+1}"] = bin_distances[i, j]
                row[f"bin_coverage_r{i+1}_c{j+1}"] = bin_coverage[i, j]
                if i in [1, 2] and j in [1, 2]:
                    centre_distance += bin_distances[i, j]
                    centre_coverage += bin_coverage[i, j]
                else:
                    boundaries_distance += bin_distances[i, j]
                    boundaries_coverage += bin_coverage[i, j]

        centre_coverage /= 4.0 ## average over the 4 centre bins
        boundaries_coverage /= 12.0 ## average over the 12 boundary bins

        row["centre_distance"] = centre_distance
        row["centre_coverage"] = centre_coverage
        row["boundaries_distance"] = boundaries_distance
        row["boundaries_coverage"] = boundaries_coverage

        return row

    except Exception as e:
        print(f'Error processing query result: {str(e)}')
        return None
