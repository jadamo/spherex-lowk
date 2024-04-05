import glob
import os
import numpy as np
import h5py
import pickle 
import re

# Custom key function to extract <x> from the filename
def extract_core_number(filename):
    match = re.search(r'core_(\d+)', filename)
    return int(match.group(1)) if match else 0


def load_and_clean_single_catalog(fileIn):
    print('Catalog: ' + fileIn)
    with h5py.File(fileIn, 'r') as f:
        items = list(f.keys())
        # print(items)
        raw_data = {}
        for item in items:
            if isinstance(f[item], h5py.Dataset):
                raw_data[item] = f[item][()]
            elif isinstance(f[item], h5py.Group):
                group_data = {}
                for sub_item in f[item].keys():
                    group_data[sub_item] = f[item][sub_item][()]
                raw_data[item] = group_data
        
        f.close()
        print('Total number of original galaxies: %d'%len(raw_data[items[3]]))  # Displaying the length of mag_i_sdss as an example


    # Identify invalid entries across all datasets starting with "mag_"
    mag_invalid_indices = set()
    for key, value in raw_data.items():
        if key.startswith("mag_") and isinstance(value, np.ndarray):
            invalid_indices = np.where(np.isinf(value) | np.isnan(value))[0]
            mag_invalid_indices.update(invalid_indices)

    mag_invalid_indices = np.array(list(mag_invalid_indices))

    cleaned_data = {}
    removed_data = {}  # New dictionary for storing removed items
    for key, value in raw_data.items():
        if isinstance(value, np.ndarray):
            valid_indices = mag_invalid_indices[mag_invalid_indices < len(value)]
            if key not in ['SED_wavelength', 'time_bins_SFH']:
                cleaned_value = np.delete(value, valid_indices, axis=0)
                removed_value = value[valid_indices]  # Extracting removed items
            else:
                cleaned_value = value
                removed_value = np.array([])  # No items removed for exceptions
            cleaned_data[key] = cleaned_value
            removed_data[key] = removed_value  # Storing removed items

    print('Total number of cleaned galaxies: %d'%len(cleaned_data[items[3]]))
    print('Total number of removed galaxies: %d'%len(removed_data[items[3]]))
    print(10*'=--=')

    return cleaned_data, removed_data, items  # Now also returning removed_data




def load_all_available_catalogs(dirIn=None, 
                                exclude_core_files_numbers=[]):
    all_data = {}
    all_items = []
    special_items = ['SED_wavelength', 'time_bins_SFH']
    special_items_included = {key: False for key in special_items}
    
    files = sorted(glob.glob(os.path.join(dirIn, '*.hdf5')), key=extract_core_number)
    
    for file_path in files:
        # Extract the number following 'core_' using regular expressions
        match = re.search(r'core_(\d+)', file_path)
        
        if match:
            x_number = int(match.group(1))  # Convert the extracted part to an integer
            
            if x_number in exclude_core_files_numbers:
                continue  # Skip this file and move to the next one

        
        cleaned_data, _, items = load_and_clean_single_catalog(file_path)

        if not all_data:
            all_data = {key: [] for key in cleaned_data if key not in special_items}
            all_items = items

        for key, value in cleaned_data.items():
            if key in special_items and not special_items_included[key]:
                all_data[key] = value
                special_items_included[key] = True
            elif key not in special_items:
                all_data[key].append(value)

    for key in all_data:
        if key not in special_items:
            all_data[key] = np.concatenate(all_data[key], axis=0)

    print('Grand total number of cleaned galaxies: %d'%len(all_data[items[3]]))
    return all_data, all_items



def load_survey_pickle(survey, dirIn_bands='Bands/'):
        
    if (survey=='LSST'):
        FILTER_NAME = dirIn_bands + 'LSST.pickle'
    elif (survey=='SPHEREx'):
        FILTER_NAME = dirIn_bands + 'SPHEREx.pickle'
    elif (survey=='COSMOS'):
        FILTER_NAME = dirIn_bands + 'COSMOS.pickle'      
    elif (survey=='WISE'):
        FILTER_NAME = dirIn_bands + 'WISE.pickle'      
    elif (survey=='LEGACYSURVEY'):
        FILTER_NAME = dirIn_bands + 'LEGACYSURVEY.pickle'       
    elif (survey=='2MASS'):
        FILTER_NAME = dirIn_bands + '2MASS.pickle'
    elif (survey=='F784'):
        FILTER_NAME = dirIn_bands + 'F784.pickle'
    else: 
        raise NotImplementedError("Filter specifications not included")
        
    with open(FILTER_NAME, 'rb') as f:
     central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names = pickle.load(f)
    
    return central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names