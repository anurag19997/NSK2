import os, sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
import numpy as np
import xarray as xr
from animal_performance.utils import speed2D, speed_bins, temp_occupancy_map, temp_spike_map_new, Position2D
from animal_performance.shuffle_spikes import shuffle_spikes_new
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

  
class WaveformTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(12, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 4, left=0.05, right=0.95, bottom=0.1, top=0.85, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :1]),
            '2': self.f.add_subplot(self.gs['all'][:, 1:2]),
            '3': self.f.add_subplot(self.gs['all'][:, 2:3]),
            '4': self.f.add_subplot(self.gs['all'][:, 3:]),
        }

    def waveform_channel_plot(self, waveforms, avg_waveform, channel, ax):

        ax.plot(waveforms.T, color='grey')

        ax.plot(avg_waveform, c='k', lw=2)

        ax.set_title('Channel ' + str(int(channel)))

class SpatialSpikeTrain2D_signalstore:
    def __init__(self, input_dict: dict, **kwargs):
        self._input_dict = input_dict
        self.spike_obj = self._read_input_dict()
        self.spike_times = self.spike_obj['spike_times']

        # Use xarray position directly
        self.position = self.spike_obj['position']
        assert isinstance(self.position, xr.DataArray)

        self.x = self.position.isel(xyt=0).values.squeeze()
        self.y = self.position.isel(xyt=1).values.squeeze()
        self.t = self.position.isel(xyt=2).values.squeeze()


        assert len(self.t) == len(self.x) == len(self.y)

        self.session_metadata = kwargs.get('session_metadata', None)
        self.arena_size = kwargs.get('arena_size', None)
        self.smoothing_factor = kwargs.get('smoothing_factor', None)
        self.is_Cylinder = kwargs.get('is_Cylinder', False)

        self.speed_bounds = input_dict.get('speed_bounds', (0, 100))

        self.spike_x, self.spike_y, self.new_spike_times = self.get_spike_positions()

        assert len(self.spike_x) == len(self.spike_y) == len(self.new_spike_times)

        self.stats_dict = self._init_stats_dict()

    def _read_input_dict(self):
        spike_obj= {}
        if 'spike_times' in self._input_dict:
            spike_obj['spike_times'] = self._input_dict['spike_times']
        if 'cell' in self._input_dict:
            spike_obj['cell_id'] = self._input_dict['cell']
        if 'position' in self._input_dict:
            spike_obj['position'] = self._input_dict['position']
        return spike_obj

    def _init_stats_dict(self):
        map_names = ['autocorr', 'binary', 'spatial_tuning', 'pos_vs_speed', 'rate_vs_time', 
                     'hafting', 'occupancy', 'rate', 'spike', 'map_blobs']
        return {k: None for k in map_names}

    def add_map_to_stats(self, map_name, map_class):
        assert map_name in self.stats_dict
        self.stats_dict[map_name] = map_class

    def get_map(self, map_name):
        assert map_name in self.stats_dict
        map_obj = self.stats_dict[map_name]
        hafting_maps = {'rate': HaftingRateMap, 'spike': HaftingSpikeMap, 'occupancy': HaftingOccupancyMap}
        if map_obj is None and map_name in hafting_maps:
            map_obj = hafting_maps[map_name](self, smoothing_factor=self.smoothing_factor)
            self.stats_dict[map_name] = map_obj
        return map_obj

    def get_spike_positions(self, shuffled_spikes=None):
        v = speed2D(self.x, self.y, self.t)
        x, y, t = speed_bins(self.speed_bounds[0], self.speed_bounds[1], v, self.x, self.y, self.t)
        cPost = np.copy(t)

        spike_times = shuffled_spikes if shuffled_spikes is not None else self.spike_times
        spike_times = np.asarray(spike_times, dtype=np.float64)
        N = len(spike_times)
        spike_positions_x = np.zeros((N, 1))
        spike_positions_y = np.zeros_like(spike_positions_x)
        new_spike_times = np.zeros_like(spike_positions_x)
        count = -1

        for index in range(N):
            tdiff = (t - spike_times[index])**2
            tdiff2 = (cPost - spike_times[index])**2
            m = np.amin(tdiff)
            ind = np.where(tdiff == m)[0]
            m2 = np.amin(tdiff2)
            if m == m2:
                count += 1
                spike_positions_x[count] = x[ind[0]]
                spike_positions_y[count] = y[ind[0]]
                new_spike_times[count] = spike_times[index]

        spike_positions_x = spike_positions_x[:count + 1]
        spike_positions_y = spike_positions_y[:count + 1]
        new_spike_times = new_spike_times[:count + 1]

        return spike_positions_x.flatten(), spike_positions_y.flatten(), new_spike_times.flatten()

class HaftingOccupancyMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D_signalstore | Position2D, **kwargs):
        self.x = spatial_spike_train.x
        self.y = spatial_spike_train.y
        self.t = spatial_spike_train.t
        self.spatial_spike_train = spatial_spike_train
        self.arena_size = spatial_spike_train.arena_size

        self.map_data = None

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata

        if 'smoothing_factor' in kwargs:
            print('overriding session smoothing factor for input smoothing facator')
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
            print('overriding session smoothing factor for input smoothing facator')
        else:
            self.smoothing_factor = self.session_metadata.session_object.smoothing_factor

    def get_occupancy_map(self, smoothing_factor=None, new_size=None, useMinMaxPos=False):
        if self.map_data is None or (self.map_data is not None and new_size != self.map_data.shape[0]) == True:
            if self.smoothing_factor != None:
                smoothing_factor = self.smoothing_factor
                assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'
            else:
                self.smoothing_factor = smoothing_factor
            if new_size is not None:
                self.map_data, self.raw_map_data, self.coverage = self.compute_occupancy_map(self.t, self.x, self.y, self.arena_size, smoothing_factor, new_size=new_size, useMinMaxPos=useMinMaxPos)
            else:
                self.map_data, self.raw_map_data, self.coverage = self.compute_occupancy_map(self.t, self.x, self.y, self.arena_size, smoothing_factor, useMinMaxPos=useMinMaxPos)

            if isinstance(self.spatial_spike_train, SpatialSpikeTrain2D_signalstore):
                self.spatial_spike_train.add_map_to_stats('occupancy', self)

        return self.map_data, self.raw_map_data, self.coverage

    def compute_occupancy_map(self, pos_t, pos_x, pos_y, arena_size, smoothing_factor, new_size=64, mask_threshold=1, useMinMaxPos=False):
        # position_2d = Position2D(pos_x, pos_y, pos_t, arena_size=arena_size, smoothing_factor=smoothing_factor)
        valid_occupancy_map, raw_occ, coverage = temp_occupancy_map(pos_t, pos_x, pos_y, arena_size,
                                                                    self.smoothing_factor, interp_size=(new_size, new_size),
                                                                    useMinMaxPos=useMinMaxPos)

        return valid_occupancy_map, raw_occ, coverage


class HaftingSpikeMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D_signalstore, **kwargs):
        self.spatial_spike_train = spatial_spike_train
        self.spike_x, self.spike_y, self.new_spike_times = self.spatial_spike_train.spike_x, self.spatial_spike_train.spike_y, self.spatial_spike_train.new_spike_times
        self.arena_size = self.spatial_spike_train.arena_size
        self.map_data = None
        self.raw_map_data = None
        self.shuffled_map_data = None
        self.shuffled_raw_map_data = None
        self.metrics = {}

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata

        if 'smoothing_factor' in kwargs:
            print('overriding session smoothing factor for input smoothing facator')
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
            print('overriding session smoothing factor for input smoothing facator')
        else:
            self.smoothing_factor = self.session_metadata.session_object.smoothing_factor

    def set_metric(self, name, value):
        self.metrics[name] = value

    def get_metric(self, name):
        return self.metrics.get(name, None)

    def get_spike_map(self, smoothing_factor=None, new_size=None, shuffle=False, n_repeats=None, useMinMaxPos=False):
        if self.map_data is None or (self.map_data is not None and new_size != self.map_data.shape[0]) == True or shuffle == True:
            if self.smoothing_factor != None:
                smoothing_factor = self.smoothing_factor
                assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'
            else:
                self.smoothing_factor = smoothing_factor

            if shuffle:
                spike_x, spike_y, new_spike_times = shuffle_spikes_new(self.spatial_spike_train.spike_times,self.spike_x, self.spike_y, self.new_spike_times, iters=n_repeats)
            else:
                spike_x, spike_y, new_spike_times = self.spike_x, self.spike_y, self.new_spike_times
            if not shuffle:
                if new_size is not None:
                    self.map_data, self.map_data_raw = self.compute_spike_map(spike_x, spike_y, smoothing_factor, self.arena_size, new_size=new_size, useMinMaxPos=useMinMaxPos)
                else:
                    self.map_data, self.map_data_raw = self.compute_spike_map(spike_x, spike_y, smoothing_factor, self.arena_size, useMinMaxPos=useMinMaxPos)
            else:
                assert spike_x.shape[0] == n_repeats, 'Number of repeats {} does not match number of shuffled spike trains {}'.format(n_repeats, spike_x.shape[0])
                map_data = np.empty((n_repeats, new_size, new_size))
                map_data_raw = np.empty((n_repeats, new_size, new_size))
                for i in range(n_repeats):
                    spike_map, spike_map_raw = self.compute_spike_map(
                        spike_x[i], spike_y[i], smoothing_factor, self.arena_size, new_size=new_size,  useMinMaxPos=useMinMaxPos
                    )
                    map_data[i,:,:] = spike_map
                    map_data_raw[i,:,:] = spike_map_raw

                self.shuffled_map_data = map_data
                self.shuffled_map_data_raw = map_data_raw

            self.spatial_spike_train.add_map_to_stats('spike', self)

        if shuffle:
            return self.shuffled_map_data, self.shuffled_map_data_raw
        else:
            return self.map_data, self.map_data_raw

    def compute_spike_map(self, spike_x, spike_y, smoothing_factor, arena_size, new_size=64, useMinMaxPos=False):
        spike_map, spike_map_raw = temp_spike_map_new(self.spatial_spike_train.x, self.spatial_spike_train.y, arena_size, spike_x, spike_y, smoothing_factor, interp_size=(new_size,new_size), useMinMaxPos=useMinMaxPos)

        return spike_map, spike_map_raw

class HaftingRateMap():
    def __init__(self, spatial_spike_train: SpatialSpikeTrain2D_signalstore, **kwargs):

        self.occ_map = spatial_spike_train.get_map('occupancy')
        if self.occ_map == None:
            self.occ_map = HaftingOccupancyMap(spatial_spike_train)
        self.spike_map = spatial_spike_train.get_map('spike')
        if self.spike_map == None:
            self.spike_map = HaftingSpikeMap(spatial_spike_train)
        self.spatial_spike_train = spatial_spike_train
        self.arena_size = spatial_spike_train.arena_size

        assert isinstance(self.occ_map, HaftingOccupancyMap)
        assert isinstance(self.spike_map, HaftingSpikeMap)

        self.map_data = None
        self.raw_map_data = None
        self.map_data_shuffled = None
        self.raw_map_data_shuffled = None

        if 'session_metadata' in kwargs:
            self.session_metadata = kwargs['session_metadata']
        else:
            self.session_metadata = spatial_spike_train.session_metadata

        

        if 'smoothing_factor' in kwargs:
            
            self.smoothing_factor = kwargs['smoothing_factor']
        elif 'settings' in kwargs and 'smoothing_factor' in kwargs['settings']:
            self.smoothing_factor = kwargs['settings']['smoothing_factor']
        else:
            self.smoothing_factor = self.session_metadata.session_object.smoothing_factor
            

    def get_rate_map(self, smoothing_factor=None, new_size=None, shuffle=False, n_repeats=None, settings_arena_size=None):
        if self.map_data is None or (self.map_data is not None and new_size != self.map_data.shape[0]) == True or shuffle == True:
            if self.map_data is None:
                pass
            elif new_size != self.map_data.shape[0]:
                pass
            elif shuffle == True:
                pass
            elif new_size != len(self.map_data[0]):
                pass
            if self.smoothing_factor == None:
                self.smoothing_factor = smoothing_factor
                assert smoothing_factor != None, 'Need to add smoothing factor to function inputs'

            if new_size is not None:
                map_data, raw_map_data = self.compute_rate_map(self.occ_map, self.spike_map, new_size=new_size, shuffle=shuffle, n_repeats=n_repeats,settings_arena_size=settings_arena_size)
            else:
                map_data, raw_map_data = self.compute_rate_map(self.occ_map, self.spike_map, shuffle=shuffle, n_repeats=n_repeats,settings_arena_size=settings_arena_size)

            if shuffle == False:
                self.map_data = map_data
                self.raw_map_data = raw_map_data
            else:
                self.map_data_shuffled = map_data
                self.raw_map_data_shuffled = raw_map_data

            self.spatial_spike_train.add_map_to_stats('rate', self)
        else:
            print('using existing rate map')
        
        if shuffle == False:
            return self.map_data, self.raw_map_data
        else:
            return self.map_data_shuffled, self.raw_map_data_shuffled
    def compute_rate_map(self, occupancy_map, spike_map, new_size=None, shuffle=False, n_repeats=None, settings_arena_size=None):
        '''
        Parameters:
            spike_x: the x-coordinates of the spike events
            spike_y: the y-coordinates of the spike events
            pos_x: the x-coordinates of the position events
            pos_y: the y-coordinates of the position events
            pos_time: the time of the position events
            smoothing_factor: the shared smoothing factor of the occupancy map and the spike map
            arena_size: the size of the arena
        Returns:
            rate_map: spike density divided by occupancy density
        '''
        if settings_arena_size is not None:
            self.arena_size = settings_arena_size
            useMinMaxPos = False
        else:
            useMinMaxPos = True

        if new_size is None:
            if self.smoothing_factor != None:
                occ_map_data, raw_occ, coverage = occupancy_map.get_occupancy_map(self.smoothing_factor, useMinMaxPos=useMinMaxPos)
                spike_map_data, spike_map_data_raw = spike_map.get_spike_map(self.smoothing_factor, shuffle=shuffle, n_repeats=n_repeats, useMinMaxPos=useMinMaxPos)
            else:
                print('No smoothing factor provided, proceeding with value of 3')
                occ_map_data, raw_occ, coverage = occupancy_map.get_occupancy_map(3, useMinMaxPos=useMinMaxPos)
                spike_map_data, spike_map_data_raw = spike_map.get_spike_map(3, shuffle=shuffle, n_repeats=n_repeats, useMinMaxPos=useMinMaxPos)
        else:
            if self.smoothing_factor != None:
                # print('getting occ map')
                occ_map_data, raw_occ, coverage = occupancy_map.get_occupancy_map(self.smoothing_factor, new_size=new_size, useMinMaxPos=useMinMaxPos)
                # print('getting spike map')
                spike_map_data, spike_map_data_raw = spike_map.get_spike_map(self.smoothing_factor, new_size=new_size, shuffle=shuffle, n_repeats=n_repeats, useMinMaxPos=useMinMaxPos)
            else:
                print('No smoothing factor provided, proceeding with value of 3')
                occ_map_data, raw_occ, coverage = occupancy_map.get_occupancy_map(3, new_size=new_size, useMinMaxPos=useMinMaxPos)
                spike_map_data, spike_map_data_raw = spike_map.get_spike_map(3, new_size=new_size, shuffle=shuffle, n_repeats=n_repeats, useMinMaxPos=useMinMaxPos)

        if not shuffle:
            rate_map_raw = np.where(raw_occ<0.0001, 0, spike_map_data_raw/raw_occ)
            rate_map = np.where(occ_map_data<0.0001, 0, spike_map_data/occ_map_data)
            rate_map = rate_map/max(rate_map.flatten())
        else:
            rate_map = list(map(lambda i: np.where(occ_map_data < 0.0001, 0, spike_map_data[i] / occ_map_data), range(len(spike_map_data))))
            rate_map_raw = list(map(lambda i: np.where(raw_occ < 0.0001, 0, spike_map_data_raw[i] / raw_occ) / max(spike_map_data_raw[i].flatten()), range(len(spike_map_data_raw))))
            rate_map_raw = np.array(rate_map_raw)

            rate_map_copy = []
            for rmp in rate_map:
                rate_map_copy.append(rmp/np.max(rmp))
            rate_map = np.array(rate_map_copy)
        return rate_map, rate_map_raw
