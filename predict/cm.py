import numpy as np
import pdb

def cm_lengths(starts, stops, recombination_data):
    np_starts = np.array(starts, dtype = np.uint32, copy = False)
    np_stops = np.array(stops, dtype = np.uint32, copy = False)
    base_ends = recombination_data.bases
    start_index = np.searchsorted(base_ends, np_starts, side = "right") - 1
    stop_index = np.searchsorted(base_ends, np_stops, side = "left")
    cm_ends = recombination_data.cm
    unadjusted_differences = cm_ends[stop_index] - cm_ends[start_index]

    # TODO: Ensure the rates is in the correct units
    rates = recombination_data.rates
    # TODO: Handle edge cases at beginning and end of chrom
    start_adjustment = (np_starts - base_ends[start_index]) * rates[start_index + 1]
    stop_adjustment = (base_ends[stop_index] - np_stops) * rates[stop_index]
    adjusted_lengths = unadjusted_differences - start_adjustment - stop_adjustment
    return adjusted_lengths

def length_with_cm_cutoff(ibd_segments, recombination_data, cutoff):
    starts, stops = zip(*ibd_segments)
    np_starts = np.array(starts, dtype = np.uint32)
    np_stops = np.array(stops, dtype = np.uint32)
    
    lengths = cm_lengths(starts, stops, recombination_data)
    detectible = lengths > cutoff

    np.sum(np_stops[detectible] - np_starts[detectible])
