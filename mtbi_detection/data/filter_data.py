
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

def filter_raw(raw, l_freq=0.3, h_freq=250, notches=[], notch_width=2, chan_axis=0, fs_baseline=500, order=6):
    fs = raw.info["sfreq"]
    if fs != fs_baseline:
        # resample the data to 500 Hz
        raw.resample(fs_baseline)
    assert raw.info["sfreq"] == fs_baseline, "Sampling frequency is not 500 Hz"
    
    # filter the data
    eeg_arry = raw.copy().get_data()
    print("filtering data")
    print("shape of eeg array before filtering: {}".format(eeg_arry.shape))
    filtered_eeg = filter_eeg(eeg_arry, l_freq=l_freq, h_freq=h_freq, fs=fs_baseline, notches=notches, notch_width=notch_width, chan_axis=chan_axis, order=order).T
    new_raw = mne.io.RawArray(filtered_eeg, raw.info)
    return new_raw

def filter_raw_mne(raw, l_freq=0.3, h_freq=250, notches=[], notch_width=2, picks='all', fs_baseline=500, order=6):

    iir_params= {
        'ftype': 'butter',
        'order': order,
        'verbose' : 50
    }
    # resample the data to the baseline frequency
    fs = raw.info["sfreq"]
    if fs != fs_baseline:
        # resample the data to 500 Hz
        print(f"Resampling data from {fs} to {fs_baseline}")
        raw.resample(fs_baseline)
    filtered_raw = raw.copy().load_data().filter(l_freq, h_freq, method='iir', picks=picks, iir_params=iir_params, verbose=50)

    if len(notches) > 0:
        for idx, notch in enumerate(notches):
            if hasattr(notch_width, '__iter__'):
                filtered_raw = filtered_raw.load_data().notch_filter(notch, notch_widths=notch_width[idx], method='iir', iir_params=iir_params, verbose=50)
            else:
                filtered_raw = filtered_raw.load_data().notch_filter(notch, notch_widths=notch_width, method='iir', iir_params=iir_params, verbose=50)
    return filtered_raw

def butter_bandpass(lowcut, highcut, fs, filter_type, order):
    # note: highcut < fs/2, lowcut > 0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if (filter_type == "lowpass") or (filter_type == "highpass"):
        sos = butter(order, high, analog=False, btype=filter_type, output='sos')
    else:
        sos = butter(order, [low, high], analog=False, btype=filter_type, output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, filter_type, order):
    sos = butter_bandpass(lowcut, highcut, fs, filter_type, order=order)
    y = sosfiltfilt(sos, data)
    return y

def filter_eeg(eeg_data, l_freq=0.1, h_freq=500, fs=500, notch_width=2, order=6, chan_axis=0, notches=[60, 120]):
    """
    filter eeg data using butterworth filter and notch filter
    """
    # filter data
    if chan_axis == 0:
        eeg_data = eeg_data.T
    filtered_eeg = np.zeros((eeg_data.shape[0], eeg_data.shape[1]))
    for i in range(eeg_data.shape[1]):
        filtered_eeg[:, i] = butter_bandpass_filter(eeg_data[:, i], l_freq, h_freq, fs, "bandpass", order)
        for idx, notch in enumerate(notches):
#             print("notch filter at {} for channel index {}".format(notch, i))
            if hasattr(notch_width, '__iter__'):
                filtered_eeg[:, i] = butter_bandpass_filter(filtered_eeg[:, i], notch - notch_width[idx]/2, notch + notch_width[idx]/2, fs, "bandstop", order)
            else:
                filtered_eeg[:, i] = butter_bandpass_filter(filtered_eeg[:, i], notch - notch_width/2, notch + notch_width/2, fs, "bandstop", order)
    return filtered_eeg

def remove_ecg_artifact(raw, ecg_channel='X1', method='correlation', l_freq=None, h_freq=None, thresh='auto'):
    """
    Given a raw mne object contaminated by ecg artifacts
    remove the ecg artifacts
    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#sphx-glr-auto-tutorials-preprocessing-40-artifact-correction-ica-py
    """
    raw_copy = raw.load_data().copy()
    filt_raw = raw_copy.filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=None, max_iter='auto', random_state=97)
    ica.fit(filt_raw.copy())
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_copy, method=method, threshold=thresh, l_freq=l_freq, h_freq=h_freq, ch_name=ecg_channel)
    print("ECG INDICES", ecg_indices)
    ica.exclude = ecg_indices
    ecg_removed_raw = raw_copy.copy()
    ica.apply(ecg_removed_raw)
    return ecg_removed_raw