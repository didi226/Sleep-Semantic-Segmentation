"""
Sleep Semantic Segmentation: fast and robust detection of 'Spindle',
'Background', 'Arousal','K-complex', 'Slow wave', 'Vertex Sharp', 'Sawtooth' from one channel sleep EEG recordings.
- Author: Xiaoyu Bao
- GitHub:
- License:
"""
import logging
import numpy as np
import pandas as pd
from scipy import signal
from mne.filter import filter_data

from scipy.interpolate import interp1d
from scipy.fftpack import next_fast_len

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import ipywidgets as ipy
from wrap_sssm.utils.io import set_log_level
from sssm_core.model import Model as ssm
from yasa.numba import _detrend, _rms
from yasa.spectral import stft_power
from yasa.others import (
    _zerocrossings,
)

logger = logging.getLogger("sssm")

__all__ = [
    "sleep_event_detect",
    "SleepEventDetect",
]

def  sleep_event_detect(
    data,
    sf=None,
    wave_name= ['Spindle', 'Background', 'Arousal','K-complex', 'Slow wave', 'Vertex Sharp', 'Sawtooth'],

    device='cuda',
    model_name='model.pt',
    model_path=None,
    step = 50,
    event_threshold = {
                'Spindle': 0.95,
                'Background': 0.9,
                'Arousal': 0.9,
                'K-complex': 0.6,
                'Slow wave': 0.6,
                'Vertex Sharp': 0.6,
                'Sawtooth': 0.6},
overall_threshold=0.5,
    verbose=False,
):
    """Detects sleep events in the provided data.

    Args:
        data (array-like): Input data for sleep event detection.
        sf (float, optional): Sampling frequency. Defaults to None.
        wave_name (list, optional): List of wave names. Defaults to
            ['Spindle', 'Background', 'Arousal', 'K-complex', 'Slow wave',
            'Vertex Sharp', 'Sawtooth'].
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
        model_name (str, optional): Name of the model file. Defaults to 'model.pt'.
        model_path (str, optional): Path to the model file. Defaults to None.
        step (int, optional): Step size for detection. Defaults to 50.
        event_threshold (dict, optional): Thresholds for each event. Defaults to
            {'Spindle': 0.95, 'Background': 0.9, 'Arousal': 0.9, 'K-complex': 0.6,
            'Slow wave': 0.6, 'Vertex Sharp': 0.6, 'Sawtooth': 0.6}.
        overall_threshold (float, optional): Overall detection threshold.
            Defaults to 0.5.
        verbose (bool, optional): Verbose output. Defaults to False.

    Returns:
        results: The detected sleep events.
    """
    set_log_level(verbose)

    thresh_standard = {'Spindle': 0.95,'Background': 0.9,'Arousal': 0.9,'K-complex': 0.6,
    'Slow wave': 0.6,'Vertex Sharp': 0.6,'Sawtooth': 0.6}
    for i_thresh in thresh_standard.keys():
        if i_thresh not in event_threshold.keys():
            event_threshold[i_thresh] = thresh_standard[i_thresh]
    model = ssm.SSM(device=device, model_name=model_name, model_path=model_path)
    return SleepEventDetect(model, wave_name, data, sf, event_threshold,step,overall_threshold)

class SleepEventDetect():
    def __init__(self, model, wave_name, data, sf, thresh,step,overall_threshold):
        """Initializes the SleepEventDetect class.

        Args:
           model: The SSM class used for detection.
           wave_name (list): List of wave names.
           data (ndarray): Input data for detection.
           sf (float): Sampling frequency.
           thresh (dict): Thresholds for each event.
           step (int): Step size for detection.
           overall_threshold (float): Overall detection threshold.
        """
        self._model = model
        self._wave_name = wave_name
        self._data = data
        self._sf = sf
        self._event_threshold = thresh
        self._step = step
        ret = self._model.predict(data.astype(np.float16), step=step)
        self.event_df = self._model.to_pandas(overall_threshold=overall_threshold, event_threshold=self._event_threshold)
        self._times = np.arange(  self._data .shape[-1]) / self._sf

    def summary(self,event=None):
        """Provides a summary of detected events.

        Args:
            event (str, optional): The specific event name to summarize. Defaults to None.

        Returns:
            dict or pd.DataFrame: If event is None, returns a dictionary with event details for all wave_names.
                                  If event is provided, returns the details for the specified event.

        Raises:
            ValueError: If the specified event is not in the wave names.
        """

        filtered_df = self.calculate_feature()
        if event is None:
            for i_event in filtered_df.keys():
                print(i_event)
                print(filtered_df[i_event])
            return filtered_df
        elif  event in self._wave_name:
            return filtered_df[event]
        else:
            raise ValueError(f'sleep event {event} is not corrected')


    def _get_event_df(self, wave_names):
        """Filters the event dataframe by wave names.

        Args:
            wave_names (list): List of wave names to filter by.

        Returns:
            DataFrame: Filtered event dataframe for wave_names.
        """
        filtered_df = self.event_df[self.event_df['label'].isin(wave_names)]
        return filtered_df


    def plot_average(self, event_type = None,  figsize=(6, 4.5),**kwargs):
        """Plots the average waveform of specified event types.

        Args:
            event_type (list, optional): List of event types to plot. Defaults to None.
            figsize (tuple, optional): Figure size for the subplot. Defaults to (6, 4.5).

        Returns:
            list: Axes of the average sleep event plot.
        """
        if event_type is None:
            event_type = self._wave_name
        ###图位置参数的设置与波的特征有关（这部分后面再处理）
        if set(event_type).issubset(set( self._wave_name)):
            event_data = []
            for i_event in event_type:
                i_event_df = self.event_df[self.event_df['label'] == i_event]
                segments = []
                for index, row in i_event_df.iterrows():
                    center = (row['Start']+row['End'])/2
                    segment = self._data[:,int(center-1.5*self._sf): int(center+1.5*self._sf)]
                    segments.append(segment)
                segment_ave = np.mean(np.array(segments),axis=0)
                event_data.append(segment_ave)
            figsize = (len(event_type) * figsize[0], figsize[1])
            if len(event_type) == 1:
                fig, axs = plt.subplots(1, 1, figsize=figsize)
                axs = [axs]
            else:
                fig, axs = plt.subplots(1, len(event_type), figsize=figsize)
            for index, i_event in enumerate(event_type):
                if isinstance(event_data[index], float) and np.isnan(event_data[index]):
                    axs[index].plot()
                elif isinstance(event_data[index], np.ndarray):
                    times = np.arange(event_data[index].shape[-1]) / self._sf
                    data = np.squeeze(event_data[index])
                    axs[index].plot(times, data, **kwargs)
                axs[index].set_title(f"Average {i_event}")
                axs[index].set_xlabel("Time (sec)")
                axs[index].set_ylabel("Amplitude (uV)")
        else:
            raise ValueError(f'event_type {event_type} is not corrected')
        return axs
    def _get_mask(self):
        """Generates a mask for every event type.

        Returns:
          dict: Mask dictionary for every event type.
        """
        mask_dict = {}
        for event in self._wave_name:
            i_event_df = self._get_event_df([event])
            mask_dict[event] =[]
            for index, row in i_event_df.iterrows():
                mask_dict[event].append(range(row['Start'], row['End']))
        return mask_dict

    def _plot_events(self, ax, event_type, cmap, norm, xrng=None):
        """Plots the detected events on the given axis.

        Args:
            ax (Axes): Matplotlib axis to plot on.
            event_type (list): List of event types to plot.
            cmap: Colormap to use for plotting.
            norm: Normalization for colormap.
            xrng (range, optional): X-axis range for plotting. Defaults to None.
        """
        mask = self._get_mask()
        data = np.squeeze(self._data)
        for index, i_event in enumerate(event_type):
            for i_list in mask[i_event]:
                if xrng is not None:
                    ax.plot(self._times[i_list][xrng], data[i_list][xrng],
                            color=cmap(norm(index)), label=f'{i_event}')
                else:
                    ax.plot(self._times[i_list], data[i_list],
                            color=cmap(norm(index)), label=f'{i_event}')
    def plot_detection(self, event_type=None,  figsize=(12, 4), cmap='Spectral'):
        """Plots the detection of specified event types.

        Args:
          event_type (list, optional): List of event types to plot. Defaults to None.
          figsize (tuple, optional): Figure size for the plot. Defaults to (12, 4).
          cmap (str, optional): Colormap for plotting. Defaults to 'Spectral'.

        Returns:
          interactive: Interactive plot object.
        """
        if event_type is None:
            event_type = self._wave_name
        if not set(event_type).issubset(set(self._wave_name)):
            raise ValueError(f'event_type {event_type} is not correct')
        cmap = get_cmap(cmap)
        win_size = 10
        n_epochs = int((self._data.shape[-1] / self._sf) / win_size)
        data = np.squeeze(self._data)
        norm = Normalize(vmin=0, vmax=len(event_type) - 1)
        fig, ax = plt.subplots(figsize=figsize)
        initial_line, = ax.plot(self._times, data, "k", lw=1, label='Original Data')
        mask = self._get_mask()

        handles = []
        labels = []
        for index, i_event in enumerate(event_type):
            dummy_line, = ax.plot([], [], color=cmap(norm(index)), label=f'{i_event}')  # 创建一个虚拟的line
            handles.append(dummy_line)
            labels.append(f'{i_event}')
            for i_list in mask[i_event]:
                ax.plot(self._times[i_list], data[i_list], color=cmap(norm(index)))

        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(event_type))
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (uV)")
        fig.canvas.header_visible = False
        fig.tight_layout()

        layout = ipy.Layout(width="50%", justify_content="center", align_items="center")

        sl_ep = ipy.IntSlider(
            min=0,
            max=n_epochs,
            step=1,
            value=0,
            layout=layout,
            description="Epoch:",
        )

        sl_amp = ipy.IntSlider(
            min=25,
            max=500,
            step=25,
            value=150,
            layout=layout,
            orientation="horizontal",
            description="Amplitude:",
        )

        dd_win = ipy.Dropdown(
            options=[1, 5, 10, 30, 60],
            value=win_size,
            description="Window size:",
        )

        def update(epoch, amplitude, win_size):
            """Update plot."""
            n_epochs = int((self._data.shape[-1] / self._sf) / win_size)
            sl_ep.max = n_epochs
            xlim = [epoch * win_size, (epoch + 1) * win_size]
            xrng = np.arange(xlim[0] * self._sf, min(xlim[1] * self._sf, self._data.shape[-1]), dtype=int)

            try:
                initial_line.set_data(self._times[xrng], data[xrng])
            except IndexError:
                pass
            for idx, i_event in enumerate(event_type):
                for i_list in mask[i_event]:
                    try:
                        ax.plot(self._times[i_list][xrng], data[i_list][xrng], color=cmap(norm(idx)))
                    except IndexError:
                        pass

            ax.set_xlim(xlim)
            ax.set_ylim([-amplitude, amplitude])
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=len(event_type))

        interact_obj = ipy.interact(update, epoch=sl_ep, amplitude=sl_amp, win_size=dd_win)
        return interact_obj

    def calculate_feature(self, event_type = None,**kwargs):
        """Calculates features for the specified event types.

        Args:
            event_type (list, optional): List of event types to calculate features for. Defaults to None.

        Returns:
            dict: Calculated features for each event type.
        """
        feature_dict ={}
        if event_type is None:
            event_type = self._wave_name
        for i_event_type in event_type:
            if i_event_type == "Spindle":
                feature_dict[i_event_type] = self.calculate_feature_spindle(**kwargs)
            elif i_event_type == "Slow wave":
                feature_dict[i_event_type] = self.calculate_feature_slow_wave(**kwargs)
            else:
                feature_dict[i_event_type] = self.calculate_feature_other(i_event_type,**kwargs)
        return feature_dict


    def calculate_feature_spindle(self,**kwargs):
        """Calculate features of sleep spindles from EEG data.

        Args:
            freq_sp (tuple): Frequency range for spindle detection (default: (12, 15)).
            freq_broad (tuple): Broad frequency range for filtering (default: (1, 30)).

        Returns:
            pandas.DataFrame: DataFrame containing spindle features including Peak, Duration,
                              Amplitude, RMS, AbsPower, RelPower, Frequency, Oscillations, and Symmetry.
        Notes:
            This function is based on the original implementation provided by https://github.com/raphaelvallat/yasa

        """
        freq_sp = kwargs.get('freq_sp', (12, 15))
        freq_broad = kwargs.get('freq_broad', (1, 30))
        data = self._data
        n_samples = data.shape[-1]
        data_df = self._get_event_df(['Spindle'])
        new_columns = ["Peak",  "Duration", "Amplitude", "RMS", "AbsPower", "RelPower", "Frequency", "Oscillations",
                       "Symmetry"]
        for col in new_columns:
            data_df[col] = np.nan
        nfast = next_fast_len(n_samples)
        data_broad = filter_data(data, self._sf, freq_broad[0], freq_broad[1], method="fir", verbose=0)
        f, t, Sxx = stft_power(data_broad[0, :],  self._sf, window=2, step=0.2, band=freq_broad, interp=False, norm=True)
        idx_sigma = np.logical_and(f >= freq_sp[0], f <= freq_sp[1])
        rel_pow = Sxx[idx_sigma].sum(0)
        distance = 60 * self._sf / 1000
        func = interp1d(t, rel_pow, kind="cubic", bounds_error=False, fill_value=0)
        t = np.arange(n_samples) / self._sf
        rel_pow = func(t)
        data_sigma = filter_data(
            data,
            self._sf,
            freq_sp[0],
            freq_sp[1],
            l_trans_bandwidth=1.5,
            h_trans_bandwidth=1.5,
            method="fir",
            verbose=0,
        )
        analytic = signal.hilbert(data_sigma, N=nfast)[:, :n_samples]
        inst_phase = np.angle(analytic)
        inst_pow = np.square(np.abs(analytic))
        inst_freq = self._sf / (2 * np.pi) * np.diff(inst_phase, axis=-1)
        for index, row in self.event_df.iterrows():
            start_value = row['Start']
            end_value = row['End']
            segment_broad = np.squeeze(data_broad[:, int(start_value):int(end_value)])
            # segment = data[:, int(start_value):int(end_value)]
            sp_x = np.arange(segment_broad.shape[-1], dtype=np.float64)
            sp_det = _detrend(sp_x, segment_broad)
            sp_amp= np.ptp(sp_det)  # Peak-to-peak amplitude
            sp_rms= _rms(sp_det)  # Root mean square
            sp_rel = np.median(rel_pow[start_value:end_value])  # Median relative power
            # Hilbert-based instantaneous properties
            sp_inst_freq = inst_freq[0, start_value:end_value]
            sp_inst_pow = inst_pow[0, start_value:end_value]
            sp_abs = np.median(np.log10(sp_inst_pow[sp_inst_pow > 0]))
            sp_freq = np.median(sp_inst_freq[sp_inst_freq > 0])

            # Number of oscillations
            peaks, peaks_params = signal.find_peaks(
                sp_det, distance=distance, prominence=(None, None)
            )
            sp_osc = len(peaks)
            pk = peaks[peaks_params["prominences"].argmax()]
            sp_pro = start_value/self._sf + pk / self._sf
            sp_sym = pk / sp_det.size
            data_df.loc[index, "Peak"] = sp_pro
            data_df.loc[index, "Amplitude"] = sp_amp
            data_df.loc[index, "RMS"] = sp_rms
            data_df.loc[index, "RMS"] = sp_rms
            data_df.loc[index, "AbsPower"] = sp_abs
            data_df.loc[index, "RelPower"] = sp_rel
            data_df.loc[index, "Frequency"] = sp_freq
            data_df.loc[index, "Oscillations"] = sp_osc
            data_df.loc[index, "Symmetry"] =  sp_sym,
        return data_df

    def calculate_feature_slow_wave(self,**kwargs):
        """Calculate features of slow waves from EEG data.

        Args:
            freq_sw (tuple): Frequency range for slow wave detection (default: (0.3, 1.5)).
            amp_neg (tuple): Amplitude range for negative peaks (default: (40, 200)).
            amp_pos (tuple): Amplitude range for positive peaks (default: (10, 150)).

        Returns:
            pandas.DataFrame: DataFrame containing slow wave features including NegPeak,
                              MidCrossing, PosPeak, ValNegPeak, ValPosPeak, PTP, Slope,
                              and Frequency.
        Notes:
            This function is based on the original implementation provided by https://github.com/raphaelvallat/yasa

        """
        freq_sw = kwargs.get('freq_sw', (0.3, 1.5))
        amp_neg = kwargs.get('amp_neg', (40, 200))
        amp_pos = kwargs.get('amp_pos', (10, 150))
        #sw
        data = self._data
        times = np.arange(data.size) / self._sf
        data_filt = filter_data(
            data,
            self._sf,
            freq_sw[0],
            freq_sw[1],
            method="fir",
            verbose=0,
            l_trans_bandwidth=0.2,
            h_trans_bandwidth=0.2,
        )
        Slow_wave_data_df = self._get_event_df( ['Slow wave'])
        new_columns = ["NegPeak", "MidCrossing", "PosPeak", "ValNegPeak", "ValPosPeak", "PTP", "Slope", "Frequency"]

        data_df = pd.DataFrame(columns=new_columns, dtype=object)
        for index, idx_mask in enumerate(self._get_mask()['Slow wave']):
            idx_neg_peaks, _ = signal.find_peaks(-1 * data_filt[0, idx_mask],height=amp_neg)#
            idx_pos_peaks, _ = signal.find_peaks(data_filt[0, idx_mask],height=amp_pos)#
            idx_neg_peaks = np.intersect1d(idx_neg_peaks, idx_mask, assume_unique=True)
            idx_pos_peaks = np.intersect1d(idx_pos_peaks, idx_mask, assume_unique=True)

            # If no peaks are detected, return None
            if len(idx_neg_peaks) == 0 or len(idx_pos_peaks) == 0:
                logger.warning("no neg_peaks or pos_peaks")
                continue

            # Make sure that the last detected peak is a positive one
            if idx_pos_peaks[-1] < idx_neg_peaks[-1]:
                # If not, append a fake positive peak one sample after the last neg
                idx_pos_peaks = np.append(idx_pos_peaks, idx_neg_peaks[-1] + 1)

            # For each negative peak, we find the closest following positive peak
            pk_sorted = np.searchsorted(idx_pos_peaks, idx_neg_peaks)
            closest_pos_peaks = idx_pos_peaks[pk_sorted] - idx_neg_peaks
            closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
            idx_pos_peaks = idx_neg_peaks + closest_pos_peaks
            sw_ptp = np.abs(data_filt[0, idx_neg_peaks]) + data_filt[0, idx_pos_peaks]




            zero_crossings = _zerocrossings(data_filt[0, :])
            # Make sure that there is a zero-crossing after the last detected peak
            if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
                # If not, append the index of the last peak
                zero_crossings = np.append(zero_crossings, max(idx_pos_peaks[-1], idx_neg_peaks[-1]))

            # Find distance to previous and following zc
            neg_sorted = np.searchsorted(zero_crossings, idx_neg_peaks)
            previous_neg_zc = zero_crossings[neg_sorted - 1] - idx_neg_peaks
            following_neg_zc = zero_crossings[neg_sorted] - idx_neg_peaks

            # Distance between the positive peaks and the previous and
            # following zero-crossings
            pos_sorted = np.searchsorted(zero_crossings, idx_pos_peaks)
            previous_pos_zc = zero_crossings[pos_sorted - 1] - idx_pos_peaks
            following_pos_zc = zero_crossings[pos_sorted] - idx_pos_peaks

            # Duration of the negative and positive phases, in seconds
            neg_phase_dur = (np.abs(previous_neg_zc) + following_neg_zc) / self._sf
            pos_phase_dur = (np.abs(previous_pos_zc) + following_pos_zc) / self._sf

            # We now compute a set of metrics
            sw_start = times[idx_neg_peaks + previous_neg_zc]
            sw_end = times[idx_pos_peaks + following_pos_zc]
            # This should be the same as `sw_dur = pos_phase_dur + neg_phase_dur`
            # We round to avoid floating point errr (e.g. 1.9000000002)
            sw_dur = (sw_end - sw_start).round(4)
            sw_dur_both_phase = (pos_phase_dur + neg_phase_dur).round(4)
            sw_midcrossing = times[idx_neg_peaks + following_neg_zc]
            sw_idx_neg = times[idx_neg_peaks]  # Location of negative peak
            sw_idx_pos = times[idx_pos_peaks]  # Location of positive peak
            # Slope between peak trough and midcrossing
            sw_slope = sw_ptp / (sw_midcrossing - sw_idx_neg)


            data_df.loc[index, "NegPeak"] = sw_idx_neg.tolist()
            data_df.loc[index, "MidCrossing"] = sw_midcrossing.tolist()
            data_df.loc[index, "PosPeak"] = sw_idx_pos.tolist()
            data_df.loc[index, "ValNegPeak"] = data_filt[0,idx_neg_peaks].tolist()
            data_df.loc[index, "ValPosPeak"] = data_filt[0,idx_pos_peaks].tolist()
            data_df.loc[index, "PTP"] = sw_ptp.tolist()
            data_df.loc[index, "Slope"] = sw_slope.tolist()
            data_df.loc[index, "Frequency"] = 1 / sw_dur
        combined_df = pd.concat([Slow_wave_data_df, data_df], ignore_index=True)
        return combined_df
    def calculate_feature_other(self,i_event_type):
        """Retrieve features of other specified event types from EEG data.

         Args:
             i_event_type (str): The event type for which features are to be retrieved.

         Returns:
             pandas.DataFrame: DataFrame containing features of the specified event type.
         """
        data_df = self._get_event_df([i_event_type])
        return data_df
if __name__ == '__main__':
    import mne
    # load data
    raw = mne.io.read_raw_edf('./SC4001E0-PSG.edf', preload=True)
    raw.filter(0.1, 40)
    data = raw.get_data(['EEG Fpz-Cz'], units="uV")
    print(data.shape)
    sf = 100
    # test sleep_event_detect
    sp = sleep_event_detect(data[:, :50000], sf)
    # test calculate_feature
    sp.calculate_feature()
    # test summary
    sp.summary()
    print(sp.summary())
    # test plot_average
    figure = sp.plot_average()
    plt.show()
    sp.plot_detection()













