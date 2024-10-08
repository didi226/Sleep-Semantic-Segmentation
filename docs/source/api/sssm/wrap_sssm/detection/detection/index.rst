sssm.wrap_sssm.detection.detection
==================================

.. py:module:: sssm.wrap_sssm.detection.detection

.. autoapi-nested-parse::

   Sleep Semantic Segmentation: fast and robust detection of 'Spindle',
   'Background', 'Arousal','K-complex', 'Slow wave', 'Vertex Sharp', 'Sawtooth' from one channel sleep EEG recordings.
   - Author: Xiaoyu Bao
   - GitHub:
   - License:

   ..
       !! processed by numpydoc !!


Classes
-------

.. autoapisummary::

   sssm.wrap_sssm.detection.detection.SleepEventDetect


Functions
---------

.. autoapisummary::

   sssm.wrap_sssm.detection.detection.sleep_event_detect


Module Contents
---------------

.. py:class:: SleepEventDetect(model, wave_name, data, sf, thresh, step, overall_threshold)

   .. py:method:: _get_event_df(wave_names)

      
      Filters the event dataframe by wave names.

      :param wave_names: List of wave names to filter by.
      :type wave_names: list

      :returns: Filtered event dataframe for wave_names.
      :rtype: DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: _get_mask()

      
      Generates a mask for every event type.

      :returns: Mask dictionary for every event type.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: _plot_events(ax, event_type, cmap, norm, xrng=None)

      
      Plots the detected events on the given axis.

      :param ax: Matplotlib axis to plot on.
      :type ax: Axes
      :param event_type: List of event types to plot.
      :type event_type: list
      :param cmap: Colormap to use for plotting.
      :param norm: Normalization for colormap.
      :param xrng: X-axis range for plotting. Defaults to None.
      :type xrng: range, optional















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_feature(event_type=None, **kwargs)

      
      Calculates features for the specified event types.

      :param event_type: List of event types to calculate features for. Defaults to None.
      :type event_type: list, optional

      :returns: Calculated features for each event type.
      :rtype: dict















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_feature_other(i_event_type)

      
      Retrieve features of other specified event types from EEG data.

      :param i_event_type: The event type for which features are to be retrieved.
      :type i_event_type: str

      :returns: DataFrame containing features of the specified event type.
      :rtype: pandas.DataFrame















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_feature_slow_wave(**kwargs)

      
      Calculate features of slow waves from EEG data.

      :param freq_sw: Frequency range for slow wave detection (default: (0.3, 1.5)).
      :type freq_sw: tuple
      :param amp_neg: Amplitude range for negative peaks (default: (40, 200)).
      :type amp_neg: tuple
      :param amp_pos: Amplitude range for positive peaks (default: (10, 150)).
      :type amp_pos: tuple

      :returns:

                DataFrame containing slow wave features including NegPeak,
                                  MidCrossing, PosPeak, ValNegPeak, ValPosPeak, PTP, Slope,
                                  and Frequency.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      This function is based on the original implementation provided by https://github.com/raphaelvallat/yasa















      ..
          !! processed by numpydoc !!


   .. py:method:: calculate_feature_spindle(**kwargs)

      
      Calculate features of sleep spindles from EEG data.

      :param freq_sp: Frequency range for spindle detection (default: (12, 15)).
      :type freq_sp: tuple
      :param freq_broad: Broad frequency range for filtering (default: (1, 30)).
      :type freq_broad: tuple

      :returns:

                DataFrame containing spindle features including Peak, Duration,
                                  Amplitude, RMS, AbsPower, RelPower, Frequency, Oscillations, and Symmetry.
      :rtype: pandas.DataFrame

      .. rubric:: Notes

      This function is based on the original implementation provided by https://github.com/raphaelvallat/yasa















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_average(event_type=None, figsize=(6, 4.5), **kwargs)

      
      Plots the average waveform of specified event types.

      :param event_type: List of event types to plot. Defaults to None.
      :type event_type: list, optional
      :param figsize: Figure size for the subplot. Defaults to (6, 4.5).
      :type figsize: tuple, optional

      :returns: Axes of the average sleep event plot.
      :rtype: list















      ..
          !! processed by numpydoc !!


   .. py:method:: plot_detection(event_type=None, figsize=(12, 4), cmap='Spectral')

      
      Plots the detection of specified event types.

      :param event_type: List of event types to plot. Defaults to None.
      :type event_type: list, optional
      :param figsize: Figure size for the plot. Defaults to (12, 4).
      :type figsize: tuple, optional
      :param cmap: Colormap for plotting. Defaults to 'Spectral'.
      :type cmap: str, optional

      :returns: Interactive plot object.
      :rtype: interactive















      ..
          !! processed by numpydoc !!


   .. py:method:: summary(event=None)

      
      Provides a summary of detected events.

      :param event: The specific event name to summarize. Defaults to None.
      :type event: str, optional

      :returns:

                If event is None, returns a dictionary with event details for all wave_names.
                                      If event is provided, returns the details for the specified event.
      :rtype: dict or pd.DataFrame

      :raises ValueError: If the specified event is not in the wave names.















      ..
          !! processed by numpydoc !!


.. py:function:: sleep_event_detect(data, sf=None, wave_name=['Spindle', 'Background', 'Arousal', 'K-complex', 'Slow wave', 'Vertex Sharp', 'Sawtooth'], device='cuda', model_name='model.pt', model_path=None, step=50, event_threshold={'Spindle': 0.95, 'Background': 0.9, 'Arousal': 0.9, 'K-complex': 0.6, 'Slow wave': 0.6, 'Vertex Sharp': 0.6, 'Sawtooth': 0.6}, overall_threshold=0.5, verbose=False)

   
   Detects sleep events in the provided data.

   :param data: Input data for sleep event detection.
   :type data: array-like
   :param sf: Sampling frequency. Defaults to None.
   :type sf: float, optional
   :param wave_name: List of wave names. Defaults to
                     ['Spindle', 'Background', 'Arousal', 'K-complex', 'Slow wave',
                     'Vertex Sharp', 'Sawtooth'].
   :type wave_name: list, optional
   :param device: Device to run the model on. Defaults to 'cuda'.
   :type device: str, optional
   :param model_name: Name of the model file. Defaults to 'model.pt'.
   :type model_name: str, optional
   :param model_path: Path to the model file. Defaults to None.
   :type model_path: str, optional
   :param step: Step size for detection. Defaults to 50.
   :type step: int, optional
   :param event_threshold: Thresholds for each event. Defaults to
                           {'Spindle': 0.95, 'Background': 0.9, 'Arousal': 0.9, 'K-complex': 0.6,
                           'Slow wave': 0.6, 'Vertex Sharp': 0.6, 'Sawtooth': 0.6}.
   :type event_threshold: dict, optional
   :param overall_threshold: Overall detection threshold.
                             Defaults to 0.5.
   :type overall_threshold: float, optional
   :param verbose: Verbose output. Defaults to False.
   :type verbose: bool, optional

   :returns: The detected sleep events.
   :rtype: results















   ..
       !! processed by numpydoc !!

