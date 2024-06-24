import mne
import numpy as np
import itertools

def rereference_raw(raw, channels=None, reference_channels=['A1', 'A2'], method='ipsilateral', keep_refs=False, include_ecg=False, ecg_channel='X1', verbose=0):
    """
    Rereference the raw.pick_channels(channels) data to the method given.
    Inputs:
        raw: mne.io.Raw object
        channels: list of channels to rereference (must include reference channels)
        reference_channels: list of reference channels
        method: 'nva' takes the average of all channels except the current channel where n is the number of channels to average over
            other methods: 'avg', 'Cz', 'linked', 'ipsilateral', 'contralateral', 'A1', 'A2', 'None'
    Returns:
        new_raw: referenced mne raw object
    """
    if channels is not None:
        raw_copy = raw.copy().pick_channels(channels, ordered=True)
        new_raw = raw_copy.copy()
    else:
        raw_copy = raw.copy()
        new_raw = raw_copy.copy()
        channels = raw_copy.ch_names
    channel_set = set(channels)
    assert all([ch in channels for ch in reference_channels]), "Reference channels must be in channels"
    if method[1:] == 'va':
        n = int(method[0])
        if n == 1:
            for ch in channels:
                # get the other channels
                other_channels = list(channel_set - set([ch]))
                raw_copy[ch] = raw_copy[ch][0] - raw_copy.copy().pick_channels(other_channels, ordered=True).get_data().mean(axis=0)
            new_raw = raw_copy
        elif n > 1:
            # len(channels) choose n combinations of channels
            channel_combos = list(itertools.combinations(channels, n))
            new_channels = ['-'.join(combo) for combo in channel_combos]
            new_info = mne.create_info(new_channels, raw_copy.info['sfreq'], ch_types='eeg', verbose=verbose)
            new_raw_data = np.zeros((len(channel_combos), raw_copy.get_data().shape[1]))
            for i, combo in enumerate(channel_combos):
                other_channels = list(channel_set - set(combo))
                new_raw_data[i, :] = raw_copy.copy().pick_channels(combo, ordered=True).get_data().mean(axis=0) - raw_copy.copy().pick_channels(other_channels, ordered=True).get_data().mean(axis=0)
            new_raw = mne.io.RawArray(new_raw_data, new_info, verbose=False)
    elif method == 'avg':
        # first remove the reference channels
        new_raw_copy = raw_copy.copy().drop_channels(reference_channels)
        new_raw = new_raw_copy.set_eeg_reference(ref_channels='average', projection=False)
    elif method == 'Cz':
        new_raw = raw_copy.set_eeg_reference(ref_channels=['Cz'], projection=False)
    elif method == 'A1':
        new_raw = raw_copy.set_eeg_reference(ref_channels=['A1'], projection=False)
    elif method == 'linked':
        new_raw = raw_copy.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False)
    elif method == 'ipsilateral':
        new_raw_copy = raw_copy.copy()
        for ch in channels:
            if ch[-1].isdigit() and 'A' not in ch:
                if int(ch[-1]) % 2 == 0:
                    new_raw_copy[ch] = new_raw_copy[ch][0] - new_raw_copy['A2'][0]
                else:
                    new_raw_copy[ch] = new_raw_copy[ch][0] - new_raw_copy['A1'][0]
            elif 'A' not in ch:
                new_raw_copy[ch] = new_raw_copy[ch][0] - (new_raw_copy['A1'][0] + new_raw_copy['A2'][0]) / 2
            else:
                # do nothing
                pass
        new_raw = new_raw_copy
    elif method == 'contralateral':
        new_raw_copy = raw_copy.copy()
        for ch in channels:
            if ch[-1].isdigit() and 'A' not in ch:
                if int(ch[-1]) % 2 == 0:
                    new_raw_copy[ch] = new_raw_copy[ch][0] - new_raw_copy['A1'][0]
                else:
                    new_raw_copy[ch] = new_raw_copy[ch][0] - new_raw_copy['A2'][0]
            elif 'A' not in ch:
                # do the average
                new_raw_copy[ch] = new_raw_copy[ch][0] - (new_raw_copy['A1'][0] + new_raw_copy['A2'][0]) / 2
            else:
                # do nothing
                pass
        new_raw = new_raw_copy
    elif method == 'A1':
        new_raw = raw_copy.set_eeg_reference(ref_channels=['A1'], projection=False)
    elif method == 'A2':
        new_raw = raw_copy.set_eeg_reference(ref_channels=['A2'], projection=False)
    elif method.upper() == 'REST':
        ten_twenty = mne.channels.make_standard_montage('standard_1020')
        # throw away any eeg channels not in the montage
        # if T1 and T2 are in the channels, rename them to FT9 and FT10
        if 'T1' in raw_copy.ch_names:
            raw_copy = raw_copy.rename_channels({'T1': 'FT9'})  # https://www.acns.org/UserFiles/file/EEGGuideline2Electrodenomenclature_final_v1.pdf
        if 'T2' in raw_copy.ch_names:
            raw_copy = raw_copy.rename_channels({'T2': 'FT10'})
        raw_copy = raw_copy.copy().pick_channels([ch for ch in raw_copy.ch_names if ch in ten_twenty.ch_names and ch not in reference_channels])
        raw_copy.set_montage(ten_twenty)
        sphere = mne.make_sphere_model("auto", "auto", raw_copy.info)
        src = mne.setup_volume_source_space(sphere=sphere, exclude=30.0, pos=15.0)
        forward = mne.make_forward_solution(raw_copy.info, trans=None, src=src, bem=sphere)
        new_raw = raw_copy.copy().set_eeg_reference("REST", forward=forward)
    elif method.upper() == 'CSD':
        ten_twenty = mne.channels.make_standard_montage('standard_1020')
        # throw away any eeg channels not in the montage
        if 'T1' in raw_copy.ch_names:
            raw_copy = raw_copy.rename_channels({'T1': 'FT9'})  # https://www.acns.org/UserFiles/file/EEGGuideline2Electrodenomenclature_final_v1.pdf
        if 'T2' in raw_copy.ch_names:
            raw_copy = raw_copy.rename_channels({'T2': 'FT10'})
        raw_copy = raw_copy.copy().pick_channels([ch for ch in raw_copy.ch_names if ch in ten_twenty.ch_names and ch not in reference_channels], verbose=verbose)
        raw_copy.set_montage(ten_twenty)
        new_raw = mne.preprocessing.compute_current_source_density(raw_copy.copy(), verbose=verbose)
    elif method.lower() == 'none':
        new_raw = raw_copy
    else:
        print("Unsupported method: {}".format(method))
        return
    
    # remove the reference channels
    if not keep_refs:
        # first make sure the reference channels are in the new raw
        if all([ch in new_raw.ch_names for ch in reference_channels]):
            new_raw = new_raw.copy().drop_channels(reference_channels)
            channel_names_without_ref = [ch for ch in channels if ch not in reference_channels]
            assert new_raw.ch_names == channel_names_without_ref

    if include_ecg:
        # add the ECG channel back in
        if ecg_channel in raw.ch_names:
            ecg_chanel_raw  = raw.copy().pick_channels([ecg_channel])
            new_raw = new_raw.add_channels([ecg_chanel_raw], force_update_info=True)
        else:
            raise ValueError(f"ECG channel {ecg_channel} not in raw channels")
    return new_raw

def find_common_channels_from_list(raw_list, sort=True):
    """
    Given a list of raw objects, find the common channels
    Input:
        raw_list: list of mne raw objects
    Returns:
        common_channels: list of common channels
    """
    common_channels = set(raw_list[0].ch_names)
    for raw in raw_list:
        common_channels = common_channels.intersection(set(raw.ch_names))
    if sort:
        common_channels = sorted(common_channels)
    return list(common_channels)

def find_common_channels_from_dict(data_dict, sort=True, ignore=['E']):
    """
    Given a dictionary of raw objects, find the common channels
    Input:
        data_dict: dictionary of mne raw objects of structure {subject: [raw1, raw2, ...], ...}
    Returns:
        common_channels: list of common channels
    """
    raw_list = [raw_item for value in data_dict.values() for raw_item in value]
    common_channels = find_common_channels_from_list(raw_list, sort=sort)
    common_channels = [channel for channel in common_channels if channel[0] not in ignore]
    return common_channels