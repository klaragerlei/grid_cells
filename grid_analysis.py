import numpy as np
import pandas as pd
import array_utility
from skimage import measure
from scipy import misc
from scipy.ndimage import rotate
import matplotlib.pylab as plt


# shifts array by x and y
def get_shifted_map(firing_rate_map, x, y):
    shifted_map = array_utility.shift_2d(firing_rate_map, x, 0)
    shifted_map = array_utility.shift_2d(shifted_map, y, 1)
    return shifted_map


# remove from both where either of them is 0
def remove_zeros(array1, array2):
    array2 = np.nan_to_num(array2).flatten()
    array1 = np.nan_to_num(array1).flatten()
    array2_tmp = np.take(array2, np.where(array1 != 0))
    array1_tmp = np.take(array1, np.where(array2 != 0))
    array2 = np.take(array2_tmp, np.where(array2_tmp[0] != 0))
    array1 = np.take(array1_tmp, np.where(array1_tmp[0] != 0))
    return array1.flatten(), array2.flatten()


# remove from both where either of them is not a number (nan) - I am not proud of this, but nothing worked with np.nan
def remove_nans(array1, array2):
    array2 = array2.flatten()
    array2[np.isnan(array2)] = 666
    array1 = array1.flatten()
    array1[np.isnan(array1)] = 666
    array2_tmp = np.take(array2, np.where(array1 != 666))
    array1_tmp = np.take(array1, np.where(array2 != 666))
    array2 = np.take(array2_tmp, np.where(array2_tmp[0] != 666))
    array1 = np.take(array1_tmp, np.where(array1_tmp[0] != 666))
    return array1.flatten(), array2.flatten()



'''
The array is shifted along the x and y axes into every possible position where it overlaps with itself starting from
the position where the shifted array's bottom right element overlaps with the top left of the map. Correlation is
calculated for all positions and returned as a correlation_vector. TThe correlation vector is 2x * 2y.
'''


def get_rate_map_autocorrelogram(firing_rate_map):
    length_y = firing_rate_map.shape[0] -1
    length_x = firing_rate_map.shape[1] - 1
    correlation_vector = np.empty((length_x * 2 + 1, length_x * 2 + 1)) * 0
    for shift_x in range(-length_x, length_x + 1):
        for shift_y in range(-length_y, length_y + 1):
            # shift map by x and y and remove extra bits
            shifted_map = get_shifted_map(firing_rate_map, shift_x, -shift_y)
            firing_rate_map_to_correlate, shifted_map = remove_zeros(firing_rate_map, shifted_map)

            correlation_y = shift_x + length_x
            correlation_x = shift_y + length_y

            if len(shifted_map) > 20:
                # np.corrcoef(x,y)[0][1] gives the same result for 1d vectors as matlab's corr(x,y) (Pearson)
                # https://stackoverflow.com/questions/16698811/what-is-the-difference-between-matlab-octave-corr-and-python-numpy-correlate
                correlation_vector[correlation_x, correlation_y] = np.corrcoef(firing_rate_map_to_correlate, shifted_map)[0][1]
            else:
                correlation_vector[correlation_x, correlation_y] = np.nan
    return correlation_vector


# make autocorr map binary based on threshold
def threshold_autocorrelation_map(autocorrelation_map):
    autocorrelation_map[autocorrelation_map > 0.2] = 1
    autocorrelation_map[autocorrelation_map <= 0.2] = 0
    return autocorrelation_map


# find peaks of autocorrelogram
def find_autocorrelogram_peaks(autocorrelation_map):
    autocorrelation_map_thresholded = threshold_autocorrelation_map(autocorrelation_map)
    autocorr_map_labels = measure.label(autocorrelation_map_thresholded)  # each field is labelled with a single digit
    field_properties = measure.regionprops(autocorr_map_labels)
    return field_properties



def calculate_grid_metrics(autocorr_map, field_properties):
    bin_size = 2.5  # cm
    field_distances_from_mid_point = find_field_distances_from_mid_point(autocorr_map, field_properties)
    # the field with the shortest distance is the middle and the next 6 closest are the middle 6
    ring_distances = get_ring_distances(field_distances_from_mid_point)
    grid_spacing = calculate_grid_spacing(ring_distances, bin_size)
    field_size = calculate_field_size(field_properties, field_distances_from_mid_point, bin_size)
    grid_score = calculate_grid_score(autocorr_map, field_properties, field_distances_from_mid_point)
    return grid_spacing, field_size, grid_score



# spatial firing is a pandas data frame that contains firing fields for each cluster. Firing fields are 2d arrays
def process_grid_data(spatial_firing):
    rate_map_correlograms = []
    grid_spacings = []
    field_sizes = []
    grid_scores = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        rate_map_correlogram = get_rate_map_autocorrelogram(firing_rate_map)
        rate_map_correlograms.append(rate_map_correlogram)
        field_properties = find_autocorrelogram_peaks(rate_map_correlogram)
        if len(field_properties) > 7:
            grid_spacing, field_size, grid_score = calculate_grid_metrics(rate_map_correlogram, field_properties)
            grid_spacings.append(grid_spacing)
            field_sizes.append(field_size)
            grid_scores.append(grid_score)
        else:
            print('Not enough fields to calculate grid metrics.')
            rate_map_correlograms.append(np.nan)
            grid_spacings.append(np.nan)
            field_sizes.append(np.nan)
            grid_scores.append(np.nan)
    spatial_firing['rate_map_autocorrelogram'] = rate_map_correlograms
    spatial_firing['grid_spacing'] = grid_spacings
    spatial_firing['field_size'] = field_sizes
    spatial_firing['grid_score'] = grid_scores
    return spatial_firing
