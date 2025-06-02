import os
from AxonaDataReader import AxonaDataReader, extract_MECO1_cut_file_meta_data
import xarray as xr
import numpy as np
import tkinter as tk
from tkinter import filedialog
import shutil

def select_directories():
    """
    Uses a GUI file dialog (via tkinter) to select the data directory and the save directory.
    
    Returns:
      data_directory, save_dir: both as strings.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window.
    data_directory = filedialog.askdirectory(title="Select Read Data Directory")
    if not data_directory:
        raise Exception("No data directory selected.")
    
    save_dir = filedialog.askdirectory(title="Select Save Data Directory")
    if not save_dir:
        raise Exception("No save directory selected.")
    
    return data_directory, save_dir



def save_xarrays(session_xarrays, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    for i, session in enumerate(session_xarrays):
        # session_path = os.path.join(save_dir, f"session_{i+1}")
        # os.makedirs(session_path, exist_ok=True)  # Create a subdirectory for each session

        for key, xarr in session.items():
            if isinstance(xarr, xr.Dataset) or isinstance(xarr, xr.DataArray):  # If it's a single Xarray, save it
                data_name = xarr.attrs.get("data_name")
                save_path = os.path.join(save_dir, f"{data_name}_{key}.nc")
                if key == "pos_array" and isinstance(xarr, xr.Dataset):
                    xarr = xarr.to_dataarray(name="animal_position")
                xarr.to_netcdf(save_path)
                print(f"Saved {key} to {save_path}")
            elif isinstance(xarr, list) and key!="meta_data":  # If it's a list of Xarrays, save each separately
                for j, xarr_item in enumerate(xarr):
                    data_name = xarr_item.attrs.get("data_name")
                    save_path = os.path.join(save_dir, f"{data_name}_{key}.nc")
                    xarr_item.to_netcdf(save_path)
                    print(f"Saved {key}_{j+1} to {save_path}")



def main():
    # Define the directory containing Axona data files
    data_directory, save_dir = select_directories()
    print(f"Data directory: {data_directory}")
    print(f"Save directory: {save_dir}")
    
    # Check if the directory exists
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Data directory '{data_directory}' does not exist.")
    subdirs = np.sort([ f.path for f in os.scandir(data_directory) if f.is_dir() ])

    # Initialize AxonaDataReader with a metadata extractor function
    reader = AxonaDataReader(meta_data_extractor=extract_MECO1_cut_file_meta_data)
    
    for subdir in subdirs:
        # Read all sessions and get Xarray objects
        session_xarrays = reader._read_batch_session(subdir)

        # Print the structure of the first session for verification
        if session_xarrays:
            print("Successfully read sessions! Structure of the first session:")
            for key, xarr in session_xarrays[0].items():
                print(f"{key}: {type(xarr)}")
                if isinstance(xarr, list):
                    print(f"  Contains {len(xarr)} elements")

        
        save_xarrays(session_xarrays, save_dir)

    for file in os.listdir(data_directory):
        if file.endswith(".xlsx") or file.endswith(".csv"):
            shutil.copy(os.path.join(data_directory, file), save_dir)
            print(f"Copied {file} to {save_dir}")

    # Return the processed Xarray data
    return session_xarrays

if __name__ == "__main__":
    session_data = main()
