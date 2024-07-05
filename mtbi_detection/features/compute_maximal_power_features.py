import numpy as np
import pandas as pd
import time
import os
import json

import mtbi_detection.data.transform_data as td
import mtbi_detection.data.data_utils as du

ROI_DICT = {
    'Temporal': ['F7', 'F8', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',  'Fz'],
    'Anterior': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz', 'T1', 'T2'],
    'Temporocentral': ['C3', 'C4', 'Cz', 'T3', 'T4'],
    'Occiparietal': ['P3', 'P4', 'Pz', 'O1', 'O2'],
    'Parietal': ['P3', 'P4', 'Pz'],
    'Posterior': ['P3', 'P4', 'Pz', 'O1', 'O2', 'T5', 'T6'],
    'Centroposterior': ['P3', 'P4', 'Pz', 'O1', 'O2', 'T5', 'T6', 'C3', 'C4', 'Cz'],
    'Central': ['C3', 'C4', 'Cz'],
    'Midline': ['Fz', 'Cz', 'Pz'],
    'Left': ['Fp1', 'F3', 'F7', 'T1', 'T3', 'C3', 'T5', 'P3', 'O1'],
    'Right': ['Fp2', 'F4', 'F8', 'T2', 'T4', 'T6', 'C4', 'T6', 'P4', 'O2'],
    'LTemporocentral': ['F7', 'T1', 'T3', 'T5'],
    'RTemporocentral': ['F8', 'T2', 'T4', 'T6'],
    'LAnterior': ['Fp1', 'F3', 'F7', 'T1'],
    'RAnterior': ['Fp2', 'F4', 'F8', 'T2'],
    'LCentroposterior': ['T3', 'T5', 'C3', 'P3', 'O1'],
    'RCentroposterior': ['T4', 'T6', 'C4', 'P4', 'O2'],
    'LPosterior': ['T5', 'P3', 'O1'],
    'RPosterior': ['T6', 'P4', 'O2'],
}

LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(LOCD_DATAPATH[:-1]), 'features')

def main(transform_data_dict=None, featurepath=FEATUREPATH, power_increment=None, num_powers=20, percentile_edge_method='automated', save=True):
    """ 
    Given the transform_data_dict, compute the maximal power features
    Inputs:
        transform_data_dict: dictionary of the transform data
        channels: list of channel names
        featurepath: path to save the features
        power_increment: increment for the spectral edge
        num_powers: number of edges to compute
        save: whether to save the dataframe
    Outputs:
        combined_power_df: dataframe of the maximal power
    
    """
    savepath = os.path.join(featurepath, 'maximal_power')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if percentile_edge_method == 'custom':
        percentile_edges = [0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 1]
    else:
        percentile_edges = None
    power_params = {'power_increment': power_increment, 'num_powers': num_powers} if percentile_edge_method == 'automated' else {'percentile_edges': percentile_edges}
    du.clean_params_path(savepath)
    savepath, found_match = du.check_and_make_params_folder(savepath, power_params, save_early=False)

    if found_match:
        combined_power_df = pd.read_csv(os.path.join(savepath, 'maximal_power_df.csv'), index_col=0)
    else:
        # load the power spectral density data
        if transform_data_dict is None:
            loadtime = time.time()
            transform_data_dict = td.main()
            print('Time taken to load data: ', time.time() - loadtime)
        channels = list(transform_data_dict['channels'])
        ROI_DICT['Full'] = channels

        unraveled_mtd = td.unravel_multitaper_dataset(transform_data_dict['subj_data'])
        open_subjs = unraveled_mtd['avg']['open_subjs']
        closed_subjs =unraveled_mtd['avg']['closed_subjs']
        assert open_subjs == closed_subjs
        subjs = open_subjs


        open_power = unraveled_mtd['avg']['open_power']
        closed_power = unraveled_mtd['avg']['closed_power']
        stack_open_power = np.stack(open_power)
        stack_closed_power = np.stack(closed_power)
        assert stack_open_power.shape == stack_closed_power.shape
        assert stack_open_power.shape[0] == len(subjs)

        roi_dict = trim_roi_dict(ROI_DICT, channels)
        print("Computing log maximal power features")
        if percentile_edge_method == 'custom':
            print("Computing open maximal power features")
            ot = time.time()
            open_maximal_power_out = get_maximal_power(np.log10(stack_open_power), percentile_edge_names=percentile_edges, roi_dict=roi_dict, channels=channels, chan_axis=1)
            open_power_edges = open_maximal_power_out['edge_powers']
            open_roi_names = open_maximal_power_out['roi_names']
            open_percentile_edge_names = open_maximal_power_out['percentile_edge_names']
            
            open_feat_names = [f'{roi}_{edge:.4f}' for roi in open_roi_names for edge in open_percentile_edge_names]
            open_power_df = pd.DataFrame(open_power_edges.reshape((open_power_edges.shape[0], open_power_edges.shape[1]*open_power_edges.shape[2])), columns=open_feat_names)
            open_power_df.index = subjs
            print(f"Time taken to compute open maximal power features: {time.time() - ot} seconds")
            ct = time.time()
            print("Computing closed maximal power features")
            closed_maximal_power_out = get_maximal_power(np.log10(stack_closed_power), percentile_edge_names=percentile_edges, roi_dict=roi_dict, channels=channels, chan_axis=1)

            closed_power_edges = closed_maximal_power_out['edge_powers']
            closed_roi_names = closed_maximal_power_out['roi_names']
            closed_percentile_edge_names = closed_maximal_power_out['percentile_edge_names']
            # make a dataframe
            closed_feat_names = [f'{roi}_{edge:.4f}' for roi in closed_roi_names for edge in closed_percentile_edge_names]
            closed_power_df = pd.DataFrame(closed_power_edges.reshape((closed_power_edges.shape[0], closed_power_edges.shape[1]*closed_power_edges.shape[2])), columns=closed_feat_names)
            closed_power_df.index = subjs
            print(f"Time taken to compute closed maximal power features: {time.time() - ct} seconds")

            power_df = pd.concat([open_power_df, closed_power_df], axis=1)
            power_df.columns = [f'maximal_open_{col}' for col in open_power_df.columns] + [f'maximal_closed_{col}' for col in closed_power_df.columns]
            combined_power_df = power_df

        else:
            # does forward and backward edge searching
            log_max_time = time.time()
            open_log_maximal_power_out = get_maximal_power(np.log10(stack_open_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=True, reverse_log=False, chan_axis=1)

            open_log_power_edges = open_log_maximal_power_out['edge_powers']
            open_log_roi_names = open_log_maximal_power_out['roi_names']
            open_log_percentile_edge_names = open_log_maximal_power_out['percentile_edge_names']
            # make a dataframe
            open_log_feat_names = [f'{roi}_{edge:.4f}' for roi in open_log_roi_names for edge in open_log_percentile_edge_names]
            open_log_power_df = pd.DataFrame(open_log_power_edges.reshape((open_log_power_edges.shape[0], open_log_power_edges.shape[1]*open_log_power_edges.shape[2])), columns=open_log_feat_names)
            open_log_power_df.index = subjs

            closed_log_maximal_power_out = get_maximal_power(np.log10(stack_closed_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=True, reverse_log=False, chan_axis=1)

            closed_log_power_edges = closed_log_maximal_power_out['edge_powers']
            closed_log_roi_names = closed_log_maximal_power_out['roi_names']
            closed_log_percentile_edge_names = closed_log_maximal_power_out['percentile_edge_names']
            # make a dataframe
            closed_log_feat_names = [f'{roi}_{edge:.4f}' for roi in closed_log_roi_names for edge in closed_log_percentile_edge_names]
            closed_log_power_df = pd.DataFrame(closed_log_power_edges.reshape((closed_log_power_edges.shape[0], closed_log_power_edges.shape[1]*closed_log_power_edges.shape[2])), columns=closed_log_feat_names)
            closed_log_power_df.index = subjs

            log_power_df = pd.concat([open_log_power_df, closed_log_power_df], axis=1)
            log_power_df.columns = [f'maximal_open_{col}' for col in open_log_power_df.columns] + [f'maximal_closed_{col}' for col in closed_log_power_df.columns]

            print(f"Time taken to compute log maximal power features: {time.time() - log_max_time} seconds")
            print("Computing reverse log maximal power features")
            reverse_log_max_time = time.time()
            open_reverse_log_maximal_power_out = get_maximal_power(np.log10(stack_open_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=True, reverse_log=True, chan_axis=1)
            
            open_reverse_log_power_edges = open_reverse_log_maximal_power_out['edge_powers']
            open_reverse_log_roi_names = open_reverse_log_maximal_power_out['roi_names']
            open_reverse_log_percentile_edge_names = open_reverse_log_maximal_power_out['percentile_edge_names']
            # make a dataframe
            open_reverse_log_feat_names = [f'{roi}_{edge:.4f}' for roi in open_reverse_log_roi_names for edge in open_reverse_log_percentile_edge_names]
            open_reverse_log_power_df = pd.DataFrame(open_reverse_log_power_edges.reshape((open_reverse_log_power_edges.shape[0], open_reverse_log_power_edges.shape[1]*open_reverse_log_power_edges.shape[2])), columns=open_reverse_log_feat_names)
            open_reverse_log_power_df.index = subjs

            closed_reverse_log_maximal_power_out = get_maximal_power(np.log10(stack_closed_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=True, reverse_log=True, chan_axis=1)

            closed_reverse_log_power_edges = closed_reverse_log_maximal_power_out['edge_powers']
            closed_reverse_log_roi_names = closed_reverse_log_maximal_power_out['roi_names']
            closed_reverse_log_percentile_edge_names = closed_reverse_log_maximal_power_out['percentile_edge_names']
            # make a dataframe
            closed_reverse_log_feat_names = [f'{roi}_{edge:.4f}' for roi in closed_reverse_log_roi_names for edge in closed_reverse_log_percentile_edge_names]

            closed_reverse_log_power_df = pd.DataFrame(closed_reverse_log_power_edges.reshape((closed_reverse_log_power_edges.shape[0], closed_reverse_log_power_edges.shape[1]*closed_reverse_log_power_edges.shape[2])), columns=closed_reverse_log_feat_names)

            closed_reverse_log_power_df.index = subjs

            reverse_log_power_df = pd.concat([open_reverse_log_power_df, closed_reverse_log_power_df], axis=1)
            reverse_log_power_df.columns = [f'open_{col}' for col in open_reverse_log_power_df.columns] + [f'closed_{col}' for col in closed_reverse_log_power_df.columns]



            print(f"Time taken to compute reverse log maximal power features: {time.time() - reverse_log_max_time} seconds")
            print("Computing maximal power features")
            max_time = time.time()
            open_maximal_power_out =  get_maximal_power(np.log10(stack_open_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=False, reverse_log=False, chan_axis=1)

            open_power_edges = open_maximal_power_out['edge_powers']
            open_roi_names = open_maximal_power_out['roi_names']
            open_percentile_edge_names = open_maximal_power_out['percentile_edge_names']
            # make a dataframe
            open_feat_names = [f'{roi}_{edge:.4f}' for roi in open_roi_names for edge in open_percentile_edge_names]
            open_power_df = pd.DataFrame(open_power_edges.reshape((open_power_edges.shape[0], open_power_edges.shape[1]*open_power_edges.shape[2])), columns=open_feat_names)
            open_power_df.index = subjs

            closed_maximal_power_out =  get_maximal_power(np.log10(stack_closed_power), channels=channels, roi_dict=roi_dict, num_powers=num_powers, power_increment=power_increment, log_edges=False, reverse_log=False, chan_axis=1)

            closed_power_edges = closed_maximal_power_out['edge_powers']
            closed_roi_names = closed_maximal_power_out['roi_names']
            closed_percentile_edge_names = closed_maximal_power_out['percentile_edge_names']
            # make a dataframe
            closed_feat_names = [f'{roi}_{edge:.4f}' for roi in closed_roi_names for edge in closed_percentile_edge_names]
            closed_power_df = pd.DataFrame(closed_power_edges.reshape((closed_power_edges.shape[0], closed_power_edges.shape[1]*closed_power_edges.shape[2])), columns=closed_feat_names)
            closed_power_df.index = subjs

            power_df = pd.concat([open_power_df, closed_power_df], axis=1)
            power_df.columns = [f'open_{col}' for col in open_power_df.columns] + [f'closed_{col}' for col in closed_power_df.columns]


            print(f"Time taken to compute maximal power features: {time.time() - max_time} seconds")


            # concatenate the dataframes
            print("Concatenating dataframes")
            combined_power_df = pd.concat([log_power_df, reverse_log_power_df, power_df], axis=1)
            # drop any identical columns
            combined_power_df = combined_power_df.loc[:,~combined_power_df.columns.duplicated()]
        # save the dataframe
        if save:
            print("Saving dataframe")
            savetime = time.time()
            combined_power_df.to_csv(os.path.join(savepath, f'maximal_power_df.csv'))
            print(f"Saved to {os.path.join(savepath, 'maximal_power_df.csv')} in {time.time() - savetime} seconds")
            with open(os.path.join(savepath, 'params.json'), 'w') as f:
                json.dump(power_params, f)
    return combined_power_df

def get_maximal_power(psd, channels=None, percentile_edge_names=None, roi_dict=None,power_increment=None, num_powers=None, log_edges=True, reverse_log=False, chan_axis=0):
    """
    Compute the spectral edge at edge_increment increments
    Inputs:
        data: numpy array of shape (channels, freq_samples) if chan_axis=0, else (n_samples, channels, freq_samples) if chan_axis=1
        freqs: numpy array of shape (freq_samples,)
        channels: list of channel names
        fs: sampling frequency
        power_increment: increment for the spectral edge
        num_powers: number of edges to compute
        log_edges: whether to compute log edges
        reverse_log: whether to reverse the log edges
        chan_axis: axis of channels
        return_power_names: whether to return the power names
    Outputs:
        dictionary of:
            power_edges: numpy array of shape (channels, num_powers) if chan_axis=0, else (n_samples, channels, num_powers) if chan_axis=1
            roi_names: list of roi names
            power_names: list of power names
    """
    assert hasattr(channels, '__iter__')
    assert chan_axis in [0, 1]
    assert psd.ndim==2 if chan_axis==0 else psd.ndim==3
    if percentile_edge_names is not None:
        num_powers = len(percentile_edge_names)
    else:
        if power_increment is not None and num_powers is None:
            num_powers = int(100 / power_increment)-1
            percentile_edge_names = [(edge+1)*power_increment / 100 for edge in range(num_powers)]
        elif num_powers is not None and power_increment is None:
            if log_edges:
                if reverse_log:
                    percentile_edge_names = 1-np.logspace(-2, 0, num_powers, base=10)
                    percentile_edge_names = np.flip(percentile_edge_names)
                    # add a 1 to the end
                    percentile_edge_names = np.append(percentile_edge_names, 1)
                    num_powers += 1
                else:
                    percentile_edge_names = np.logspace(-2, 0, num_powers, base=10)
            else:
                percentile_edge_names = np.linspace(0, 1, num_powers)
        else:
            raise ValueError("Must specify either edge_increment or num_edges, but not both.")
    n_channels = psd.shape[chan_axis]
    if chan_axis == 0:
        power_edges = np.zeros((n_channels, num_powers))
    elif chan_axis == 1:
        power_edges = np.zeros((psd.shape[0], n_channels, num_powers))
    else:
        raise ValueError("chan_axis must be 0 or 1")

    if chan_axis == 0:
        for cdx, channel in enumerate(channels):
            for edx in range(num_powers):
                edge_power = np.percentile(psd[cdx, :], percentile_edge_names[edx] * 100)
                power_edges[cdx, edx] = edge_power
                time.sleep(0)
    elif chan_axis == 1:
        for ndx in range(psd.shape[0]):
            for cdx, channel in enumerate(channels):
                for edx in range(num_powers):
                    edge_power = np.percentile(psd[ndx, cdx, :], percentile_edge_names[edx] * 100)
                    power_edges[ndx, cdx, edx] = edge_power
                    time.sleep(0)
        

    # now let's get the roi_dict data
    roi_edges, rois = get_roi_maximal_power(psd, percentile_edge_names=percentile_edge_names, roi_dict=roi_dict, channels=channels, chan_axis=chan_axis)

    if len(rois)>0:
        out_edges = np.concatenate((power_edges, roi_edges), axis=chan_axis)
        out_roi_names = channels + rois
    else:
        out_edges = power_edges
        out_roi_names = channels

    out_dict = {
        'edge_powers': out_edges,
        'roi_names': out_roi_names,
        'percentile_edge_names': percentile_edge_names
    }    

    return out_dict

def get_roi_maximal_power(psd, percentile_edge_names=None, roi_dict=None, channels=None, chan_axis=0):
    """
    Given the power spectral density, compute the maximal power for each roi
    Inputs:
        psd: numpy array of shape (channels, freq_samples) if chan_axis=0, else (n_samples, channels, freq_samples) if chan_axis=1
        percentile_edge_names: list of percentile edge names
        roi_dict: dictionary of {roi: [channels]}
        channels: list of channel names
        chan_axis: axis of channels 
    Outputs:
        roi_edges: numpy array of shape (n_rois, num_powers) if chan_axis=0, else (n_samples, n_rois, num_powers) if chan_axis=1
        rois: list of roi names   
    """
    if roi_dict is None:
        roi_dict = trim_roi_dict(ROI_DICT, channels)
    num_powers = len(percentile_edge_names)
    n_rois = len(roi_dict.keys())
    if chan_axis == 0:
        roi_edges = np.zeros((n_rois, num_powers))
    elif chan_axis == 1:
        roi_edges = np.zeros((psd.shape[0], n_rois, num_powers))
    rois = []
    for rdx, (roi, roi_channels) in enumerate(roi_dict.items()):
        channel_indices = np.array([channels.index(channel) for channel in roi_channels])
        if chan_axis == 0:
            sub_psd = psd[channel_indices, :]
        
            for edx in range(num_powers):
                edge_power = np.percentile(sub_psd, percentile_edge_names[edx] * 100)
                roi_edges[rdx, edx] = edge_power
                time.sleep(0)
        elif chan_axis == 1:
            sub_psd = psd[:, channel_indices, :]
            for ndx in range(psd.shape[0]):
                for edx in range(num_powers):
                    edge_power = np.percentile(sub_psd[ndx, :, :], percentile_edge_names[edx] * 100)
                    roi_edges[ndx, rdx, edx] = edge_power
                    time.sleep(0)
        rois.append(roi)
    return roi_edges, rois

def trim_roi_dict(roi_dict, channels):
    """
    Given a dictionary of {roi: [channels]}, return a new dictionary with only the channels that are in the channels list
    Args:
        roi_dict: dictionary of {roi: [channels]}
        channels: list of channels
    Returns:
        new_roi_dict: dictionary of {roi: [channels]}
    """
    new_roi_dict = {}
    for roi, roi_channels in roi_dict.items():
        new_roi_dict[roi] = [channel for channel in roi_channels if channel in channels]
    return new_roi_dict

if __name__ == '__main__':
    print("Computing maximal power features")
    st = time.time()
    combined_maximal_powers_df = main()
    print(f"Done computing maximal power features in {time.time() - st} seconds")