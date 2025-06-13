import os
import pandas as pd
import json
import pathlib
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from pathlib import Path
from signalstore import UnitOfWorkProvider
from pymongo import MongoClient
import shutil
from utils import deserialize_dataarray


import tkinter as tk
from tkinter import filedialog


def choose_path(dir_type):
    root = tk.Tk()
    root.title(f"Please select {dir_type}")
    root.geometry("400x100")  # Minimal window

    selected_path = tk.StringVar()
    label_text = f"Click to select {dir_type} directory" if ".yml" not in dir_type else "Click to select a YAML file"
    label = tk.Label(root, text=label_text)
    label.pack(pady=20)

    def choose():
        if ".yml" in dir_type:
            path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yml *.yaml")])
        elif "property models" in dir_type:
            path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")]) 
        else:
            path = filedialog.askdirectory()
        selected_path.set(path)
        root.quit()
        root.destroy()  # This will close the window immediately


    button = tk.Button(root, text="Browse", command=choose)
    button.pack()

    root.mainloop()
    return selected_path.get()



def add_data(mongo_uri: str, input_dir: str, tmpdir: str, property_models_dir: str, metamodels_dir: str, data_models_dir: str):
    """
    Add data to Signalstore 

    Parameters
    ----------
    mongo_uri : str
        MongoDB URI.
    data_dir : str
        Directory containing the data files.
    """
    # Mock DB client
    mongo_uri = os.environ.get("MONGO_URI", mongo_uri)
    mongo_client = MongoClient(mongo_uri)

    #filesystem
    tmpdir = Path(tmpdir)
    if not tmpdir.exists():
        tmpdir.mkdir(parents=True, exist_ok=True)

    tmpdir = pathlib.Path(tmpdir)
    filesystem = LocalFileSystem(root=str(tmpdir))

    # memory store to save analysis data
    memory_store = dict()

    # Get uow to act on database
    uow_provider = UnitOfWorkProvider(mongo_client, filesystem, memory_store)
    unit_of_work = uow_provider(str(tmpdir.name))
    shutil.rmtree("./internal", ignore_errors=True)
    with unit_of_work as uow:
        uow.data._data._directory = str(tmpdir)


    # Get raw_property_models
    property_models_path = Path(property_models_dir)
    with open(property_models_path, 'r') as file:
        raw_property_models = json.load(file)

    # Get raw_metamodels
    metamodels_dir = Path(metamodels_dir)
    metamodel_filepaths = list(metamodels_dir.glob("*.json"))
    raw_metamodels = []
    for filepath in metamodel_filepaths:
        with open(filepath, 'r') as file:
            raw_metamodels.append(json.load(file))

    # Get raw_data_models
    data_models_dir = Path(data_models_dir)
    data_model_filepaths = list(data_models_dir.glob("*.json"))
    raw_data_models = []
    for filepath in data_model_filepaths:
        with open(filepath, 'r') as file:
            raw_data_models.append(json.load(file))

    # Get raw_records
    netcdf_dir = Path(input_dir)
    records_files = list(netcdf_dir.glob("*.xlsx"))
    raw_records = []
    for filepath in records_files:
            file = pd.read_excel(filepath, engine='openpyxl', dtype=str)
            file_json = file.to_json(orient="records")
            records = json.loads(file_json)
            raw_records.extend(records)

    # Get dataarrays
    netcdf_files = list(netcdf_dir.glob("*.nc"))
    dataarrays = []
    for filepath in netcdf_files:
        dataarray = xr.open_dataarray(filepath)
        dataarray = deserialize_dataarray(dataarray)
        dataarrays.append(dataarray)

    # Add property models, metamodels, data models, and records
    with unit_of_work as uow:
        for property_model in raw_property_models:
            if not uow.domain_models.exists(property_model['schema_name']):
                print(f"Adding {property_model['schema_name']} to Mongo.")
                uow.domain_models.add(property_model)
            else:
                print(f"Model {property_model['schema_name']} already exists; skipping.")
        for metamodel in raw_metamodels:
            if not uow.domain_models.exists(metamodel['schema_name']):
                print(f"Adding {metamodel['schema_name']} to Mongo.")
                uow.domain_models.add(metamodel)
            else:
                print(f"Model {metamodel['schema_name']} already exists; skipping.")
        for data_model in raw_data_models:
            if not uow.domain_models.exists(data_model['schema_name']):
                print(f"Adding {data_model['schema_name']} to Mongo.")
                uow.domain_models.add(data_model)
            else:
                print(f"Model {data_model['schema_name']} already exists; skipping.")
        for record in raw_records:
            if not record.get("has_file"):
                schema_ref = record.get("schema_ref")
                data_name = record.get("data_name")
                version_timestamp = record.get("version_timestamp")  # Could be None if not defined
        
                if not uow.data.exists(schema_ref, data_name, version_timestamp):
                    uow.data.add(record)
                    print(f"Adding {schema_ref}/{data_name} to Mongo.")
                else:
                    print(f"Record with schema_ref {schema_ref} and data_name {data_name} already exists.")
        for dataarray in dataarrays:
            # Skip dataarrays with schema_ref of "test"
            if dataarray.attrs.get("schema_ref") == "test":
                continue    
            # Extract the required identifying attributes from the dataarray's attrs
            schema_ref = dataarray.attrs.get("schema_ref")
            data_name = dataarray.attrs.get("data_name")
        
            # Check whether the data array already exists in the repository.
            # Note: if you're using a specific data adapter, pass it in; otherwise, it will use the default.
            if not uow.data.has_file(schema_ref, data_name):
                print(f"Adding {schema_ref}/{data_name} to Mongo.")
                uow.data.add(dataarray)
            else:
                print(f"{schema_ref}/{data_name} already in Mongo _and_ file exists; skipping.")

        uow.commit()

    return unit_of_work