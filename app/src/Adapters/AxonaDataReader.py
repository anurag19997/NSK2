import os, sys
import numpy as np
import re
import xarray as xr

if os.path.exists('/.dockerenv'):
    PROJECT_PATH = '/app/src/NSK-isolated-main'
else:
    PROJECT_PATH = 'SignalStore/app/src/NSK-isolated-main'
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)

from x_io.rw.axona.read_pos import grab_position_data
from x_io.rw.axona.read_set import read_set_file
from x_io.rw.axona import read_tetrode_and_cut 




## We will need to seperate the sessions so that it will only contain 1 pos and set file

class AxonaDataReader:
    def __init__(self, meta_data_extractor):
        self.extract_meta_data = meta_data_extractor

    def _get_all_filepaths(self, base_name, dir):
        ## modify the re
        # query = rf"^{base_name}(\_\d+\.cut|\d+\_matched\.cut|\.d+|\.pos|\.set)"
        query = rf"^{base_name}(\_\d+\.cut|\d+\_matched\.cut|\.d+|\.pos|\.set|\.\d+)"

        all_files = os.listdir(dir)
        matched_files = {"tet_files":[], "cut_files":[], "set_files":[], "pos_files":[]}
        for file in all_files:
            if re.match(query, file):
                ext = self._get_file_type(file)
                matched_files[f"{ext}_files"].append(os.path.join(dir, file))
        matched_files['tet_files'] = sorted(matched_files['tet_files'])
        # print(matched_files['tet_files'])
        matched_files['cut_files'] = sorted(matched_files['cut_files'])
        # matched_files['paired_tet_cut_files'] = self._check_pair_tet_cut_files(matched_files) ## we might not need this func
        if len(matched_files['set_files'])>1:
            raise ValueError(f"too many set files: {matched_files['set_files']}")
        if len(matched_files['pos_files'])!=1:
            raise ValueError(f"wrong number of pos files: {len(matched_files['pos_files'])} \n {matched_files['pos_files']}")

        tet_len = len(matched_files['tet_files'])
        cut_len = len(matched_files['cut_files'])

        if tet_len == 0:
            raise ValueError(f'No tet files found in session {base_name} in {dir}')
        
        if cut_len == 0:
            raise ValueError(f'No cut files found in session {base_name} in {dir}')

        if tet_len != cut_len:
            raise ValueError(f"unequal cut({cut_len}) and tet({tet_len}) files: \n {matched_files['cut_files']} \n {matched_files['tet_files']}")


        return matched_files
    
    def _check_pair_tet_cut_files(self, matched_files):
        tet_files = matched_files['tet_files']
        cut_files = matched_files['cut_files']
        pairs = zip(tet_files, cut_files)

        for tet_file, cut_file in pairs:
           tet_num = tet_file.split('.')[-1] 
           cut_num = cut_file.split('.')[-2].split('_')[-1] ## This will not work for matched_cut_files
           if tet_num!=cut_num:
               raise ValueError(f"tet num:{tet_num} is not equal to cut num: {cut_num}\n tet_files: {tet_files}\n cut_files:{cut_files}") 

        return pairs
    


    def _xarray_read_files(self, file_paths):
        ## try to read cut files before anything else to get non noisy indices
        x_arrays = {'pos_array':self._read_pos_file(file_paths['pos_files'][0]), 
                    'set_array':self._read_set_file(file_paths['set_files'][0]),
                    'cut_arrays':[self._read_cut_file(f) for f in file_paths['cut_files']]} ## we could save noise/non-noisy idx and query it out later
        tet_spike_time_meta_combined = [self._read_data_from_tet_file(f) for f in file_paths['tet_files']]
        x_arrays['tet_waveform_arrays'] = [row[0] for row in tet_spike_time_meta_combined]
        x_arrays['event_time_arrays'] = [row[1] for row in tet_spike_time_meta_combined]
        x_arrays['meta_data'] = [row[2] for row in tet_spike_time_meta_combined]
        
        return x_arrays


    def _read_session(self, base_path, dir):
        file_paths = self._get_all_filepaths(base_path, dir) ## return matched filepaths(checking dataset constraints)
        # meta_data_dict = self.extract_meta_data(file_paths) ## return meta_dict_for_all_files_respectively

        x_arrays = self._xarray_read_files(file_paths) 
        meta_data_dict = x_arrays['meta_data']
        ## check for the update func and make sure we're not duplicating the references
        # print(x_arrays['pos_array'])
        # print(type(x_arrays['pos_array']))
        ## have to decide on this one
        x_arrays['pos_array'].attrs.update(meta_data_dict[0]['pos_meta'])
        # x_arrays['pos_array']['schema_ref'] = 'animal_position'
        # x_arrays['pos_array']['data_name'] = ses_pos_name
        # x_arrays['pos_array']["has_file"] = "true"
        # x_arrays['set_array'].attrs.update(meta_data_dict)

        for i in range(len(x_arrays['tet_waveform_arrays'])):
            # print(meta_data_dict)
            # print(meta_data_dict[i]['tet_meta'])
            x_arrays['tet_waveform_arrays'][i].attrs.update(meta_data_dict[i]['tet_meta'])
            x_arrays['cut_arrays'][i].attrs.update(meta_data_dict[i]['cut_meta'])
            x_arrays['event_time_arrays'][i].attrs.update((meta_data_dict[i]['event_time_meta']))
            # print(meta_data_dict['tet_meta'][file_paths['tet_files'][i]])
            # x_arrays['tet_waveform_arrays'][i].attrs.update(meta_data_dict['tet_meta'][file_paths['tet_files'][i]])
            # x_arrays['cut_arrays'][i].attrs.update(meta_data_dict['cut_meta'][file_paths['cut_files'][i]])

        return x_arrays

    def _get_base_paths(self, dir):
        '''Since each session has one pos file
        return the list of all pos files.
        base_paths == pos files'''

        all_files = os.listdir(dir)
        base_paths = [f for f in all_files if '.pos' in f]
        return base_paths 
    
    def _read_batch_session(self, dir):
        base_paths = self._get_base_paths(dir)
        session_xarrays = []
        for base_path in base_paths:
            session_xarrays.append(self._read_session(base_path.split('.')[0], dir))

        return session_xarrays ## can be used to save as netcdf
        

    def _get_file_type(self, path):
        ext = path.split('.')[-1]
        if ext.isnumeric():
            ext = 'tet'
            return ext
        if ext in ('pos', 'set', 'cut'):
            return ext

        else:
            raise ValueError(f"Incorrect file extension {ext} for file: {path}")


    def _read_cut_file(self, cut_path):
    
        with open(cut_path, 'r') as open_cut_file:
            cut_data = read_tetrode_and_cut._read_cut(open_cut_file)

        cut_values = np.array(cut_data, dtype=np.float32).squeeze()

        cut_values_dataarray = xr.DataArray(
                data=np.asarray(cut_values).reshape((-1,1)),
                dims=("spike_idx", "1"),
                coords={"spike_idx": np.arange(len(cut_values))},
            )

        return cut_values_dataarray
    
    ## the returned xarray is not sorted. It doesn't make sense to keep this private and extract meta data from it through a public function. Instead make this function public
    def _read_data_from_tet_file(self, tet_path):

        with open(tet_path, 'rb+') as open_tet_file:
            tet_data = read_tetrode_and_cut._read_tetrode(open_tet_file)

        tet_file = tet_data[0]

        ## you can move it to where you extract meta data
        spike_params = tet_data[1]
        dateandtime = spike_params["datetime"]
        duration = float(spike_params["duration"])
        trial_time = str(dateandtime.strftime("%H:%M:%S")) # STR
        trial_time_without_colon = re.sub(":", "", trial_time)
        ch1 = tet_file['ch1']
        ch2 = tet_file['ch2']
        ch3 = tet_file['ch3']
        ch4 = tet_file['ch4']

        tet_waveforms = np.array([ch1, ch2, ch3, ch4], dtype=np.float32)
        tet_waveforms = tet_waveforms.transpose(1,0,2)
        spike_times = np.array(tet_file['t'], dtype=np.float32).squeeze()

        tet_waveforms_dataarray = xr.DataArray(tet_waveforms,
                                       dims=("spike_idx", "channel", "sample"),
                                       coords={"spike_idx": np.array(np.arange(tet_waveforms.shape[0]),dtype=np.float32).squeeze(),
                                                "channel": np.array(np.arange(4),dtype=np.float32).squeeze(),
                                                "sample": np.array(np.arange(len(ch1[0])),dtype=np.float32).squeeze()})

        spike_times_dataarray = xr.DataArray(spike_times.reshape((-1,1)),
                dims=("spike_idx", "1"),
                coords={"spike_idx": np.array(np.arange(len(spike_times)),dtype=np.float32).squeeze()})

        
        # meta_data_dict = self.extract_meta_data(tet_path)
        ses_name, ses_pos, aid, tetrode_num, duration = self.extract_meta_data(tet_path.split('/')[-1],trial_time_without_colon)
        # spike_count = tet_file[1]['num_spikes']

        # tet_waveforms_dataarray.attrs["schema_ref"] = "tet_waveforms"
        # spike_times_dataarray.attrs["schema_ref"] = "spike_times"
        # spike_times_dataarray.attrs["data_name"] = ses_name
        session_data_ref_dict = {"schema_ref": "session", "data_name": ses_pos}
        animal_data_ref_dict = {"schema_ref": "animal", "data_name": aid}
        probe_data_ref_dict = {"schema_ref": "probe", "data_name": str(tetrode_num)}
        session_data_ref = str(session_data_ref_dict)
        animal_data_ref = str(animal_data_ref_dict)
        probe_data_ref = str(probe_data_ref_dict)

        ## change the name to ref_dict instead of meta_data_dict
        common_meta_data_dict = {"session_data_ref":session_data_ref,
                         "animal_data_ref":animal_data_ref,
                         "probe_data_ref":probe_data_ref,
                         "has_file":"true"}
        spike_times_meta_dict = common_meta_data_dict.copy()
        set_meta_dict = common_meta_data_dict.copy()
        pos_meta_dict = common_meta_data_dict.copy()
        cut_meta_dict = common_meta_data_dict.copy()
        tet_meta_dict = common_meta_data_dict.copy()
        del pos_meta_dict["probe_data_ref"]
        cut_meta_dict["has_file"] = pos_meta_dict["has_file"] = "true"
        cut_meta_dict['schema_ref'] = 'spike_labels'
        set_meta_dict['schema_ref'] = 'set_file'
        cut_meta_dict['data_name'] = ses_name
        pos_meta_dict['schema_ref'] = 'animal_position'
        pos_meta_dict['data_name'] = ses_pos
        pos_meta_dict["recording_length"] = str(duration)
        tet_meta_dict["schema_ref"] = "tet_waveforms"
        tet_meta_dict["data_name"] = ses_name
        tet_meta_dict["data_dimensions"] = ["spike_idx", "channel", "sample"]
        tet_meta_dict["dimension_of_measure"] = "[charge]"
        tet_meta_dict["sampling_rate"] = str(spike_params["sample_rate"])
        tet_meta_dict["duration"] = str(spike_params["duration"])
        spike_times_meta_dict["schema_ref"] = "spike_times"
        spike_times_meta_dict["data_name"] = ses_name
        # spike_times_meta_dict["spike_count"] = str(len(cluster_event_times))

        # ses_name = str(aid) + "_" + str(tetrode) + '_' + str(date) + "_" + str(trial_time_without_colon) # with tet id
        # ses_pos_name = str(aid) + '_' + str(date) + "_" + str(trial_time_without_colon) # without tet id

        cut_meta_dict["data_dimensions"] = pos_meta_dict["data_dimensions"] = ["sample", "xyt"]
        spike_times_meta_dict["data_dimensions"] = ["spike_idx", "1"]
        spike_times_meta_dict["dimension_of_measure"] = "[time]"
        pos_meta_dict["dimension_of_measure"] = "[space]"
        pos_meta_dict["unit_of_measure"] = "cm" 
        cut_meta_dict["dimension_of_measure"] = "[nominal]"


        meta_data_dict = {"tet_meta":tet_meta_dict,
                          "cut_meta":cut_meta_dict,
                          "set_meta":set_meta_dict,
                          "pos_meta":pos_meta_dict,
                          'event_time_meta': spike_times_meta_dict
                          }

        return [tet_waveforms_dataarray, spike_times_dataarray, meta_data_dict]
    
    def _read_pos_file(self, file_path):
        pos_obj = grab_position_data(file_path)
        t, x, y, arena_height, arena_width = pos_obj["t"], pos_obj["x"], pos_obj["y"], pos_obj["arena_height"], pos_obj["arena_width"]
        position_pairs = np.array([x, y, t]).T.squeeze()
        arena_size = [float(arena_height), float(arena_width)]
        position_features_dataarray = xr.DataArray(data=position_pairs,
                                                    dims=("sample", "xyt"),
                                                    coords={"sample": np.array(np.arange(len(position_pairs)),dtype=np.float32).squeeze(),
                                                            "xyt": np.array(np.arange(len(position_pairs[0])),dtype=np.float32).squeeze()})
        position_features_dataarray.attrs["sample_rate"] = str(pos_obj["sample_rate"])
        
        return position_features_dataarray



    def _read_set_file(self, file_path):
        return read_set_file()


    # def _read_cut_tet_pair(cut_path, tet_path): ## 
    #     tet_xarr = self._read_tet_file(tet_path)
    #     cut_xarr = self._read_cut_file(cut_path)
    #     return cut_xarr, tet_xarr
        
    

## can add other metat data as well! like unit of measure, spike_count and dimension_of_measure
def extract_MECO1_cut_file_meta_data(path, trial_time_without_colon):

    def extract_animal_id(tet_file_name):
        return tet_file_name.split('_')[0]
    
    def extract_date(tet_file_name):
        return tet_file_name.split('_')[1].split('-')[0]
    
    def extract_tetrode_num(tet_file_name):
        return tet_file_name.split('.')[-1] 
    
    ## need to clarify this one
    def extract_trial_time(tet_file_name):
        return tet_file_name.split('_')[1].split('-')[0]
    
    ## need to clarify this one
    def extract_duration(tet_file_name):
        return tet_file_name.split('_')[1].split('-')[0]

    def extract_ses_pos_name(aid, date, trial_time_without_colon):
        return str(aid) + '_' + str(date) + "_" + str(trial_time_without_colon) 
    
    def extract_ses_name(aid, date, tetrode_num, trial_time_without_colon):
        return str(aid) + "_" + str(tetrode_num) + '_' + str(date) + "_" + str(trial_time_without_colon) # with tet id


    tet_file_name = path
    aid = extract_animal_id(tet_file_name)
    date = extract_date(tet_file_name)
    tetrode_num = extract_tetrode_num(tet_file_name)
    # trial_time = extract_trial_time(tet_file_name)
    duration = extract_duration(tet_file_name) ## can return None or "NO"
    ses_name = extract_ses_name(aid, date, tetrode_num, trial_time_without_colon) ## aid_date_trial_time_tetrodenum
    ses_pos = extract_ses_pos_name(aid, date, trial_time_without_colon)

    return ses_name, ses_pos, aid, tetrode_num, duration
    



    
