import numpy as np
import pandas as pd
import time
import os
import json
import mtbi_detection.features.gradiompy_integrate as gp_integrate
import mtbi_detection.data.transform_data as td
import mtbi_detection.features.feature_utils as fu
import mtbi_detection.data.data_utils as du


CHANNELS = ['C3','C4','Cz','F3','F4','F7','F8','Fp1','Fp2','Fz','O1','O2','P3','P4','Pz', 'T3','T4','T5','T6']
LOCD_DATAPATH = open('open_closed_path.txt', 'r').read().strip()
FEATUREPATH = os.path.join(os.path.dirname(os.path.dirname(LOCD_DATAPATH[:-1])), 'features')


def main(transform_data_dict=None, channels=CHANNELS, save=False, featurepath=FEATUREPATH, num_edges=None, edge_increment=None, log_edges=True, reverse_log=False, spectral_edge_method='automated', chan_axis=0, choose_subjs='train', skip_ui=False, internal_folder='data/internal/'):
    """
    Compute the spectral edge features for the open and closed eyes data
    Inputs:
        transform_data_dict: dictionary containing the psd transform data
        channels: list of channel names
        save: whether to save the features
        featurepath: path to save the features
        num_edges: number of edges to compute
        edge_increment: increment for the spectral edge
        log_edges: whether to compute log edges
        reverse_log: whether to reverse the log edges
        spectral_edge_method: method to compute spectral edges
        chan_axis: axis of channels
        choose_subjs: which data split to choose
        internal_folder: folder containing the data splits
    Outputs:
        df_concat: dataframe containing the spectral edge frequencies
    """
   
    savepath = os.path.join(featurepath, 'spectral_edge_frequencies')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if spectral_edge_method == 'custom':
        spectral_edges = [0.001, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99, 0.999]
    else:
        spectral_edges = None
    spectral_params = {'edge_increment': edge_increment, 'num_powers': num_edges, 'log_edges':log_edges, 'reverse_log': reverse_log} if spectral_edge_method == 'manual' else {'spectral_edges': spectral_edges}
    if spectral_edge_method == 'automated':
        spectral_params = {'spectral_edge_method': 'automated'}
    spectral_params['choose_subjs'] = choose_subjs
    du.clean_params_path(savepath, skip_ui=skip_ui)
    savepath, found_match = du.check_and_make_params_folder(savepath, spectral_params, save_early=False, skip_ui=skip_ui)

    if found_match:
        sef_df = pd.read_csv(os.path.join(savepath, 'all_spectral_edge_frequencies.csv'), index_col=0)
        assert set(sef_df.index) == set(fu.select_subjects_from_dataframe(sef_df, choose_subjs, internal_folder=internal_folder).index)
    else:
        if transform_data_dict is None:
            # load the psd data
            loadtime = time.time()
            transform_data_dict = td.main()
            print('Time taken to load data: ', time.time() - loadtime)

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

        freqs = transform_data_dict['common_freqs']

        if spectral_edge_method == 'custom':
            edges_open, edge_names = get_spectral_edges(stack_open_power, freqs, channels=channels, spectral_edges=spectral_edges, chan_axis=1, return_edge_names=True)
            edges_closed, edge_names = get_spectral_edges(stack_closed_power, freqs, channels=channels, spectral_edges=spectral_edges, chan_axis=1, return_edge_names=True)
            open_columns = [f"open_{chan}_{feature}" for chan in channels for feature in edge_names]
            closed_columns = [f"closed_{chan}_{feature}" for chan in channels for feature in edge_names]
            open_edge_df = pd.DataFrame(edges_open.reshape(-1, edges_open.shape[1]*edges_open.shape[2]), columns=open_columns, index=subjs)
            closed_edge_df = pd.DataFrame(edges_closed.reshape(-1, edges_closed.shape[1]*edges_closed.shape[2]), columns=closed_columns, index=subjs)
            df_concat = pd.concat([open_edge_df, closed_edge_df], axis=1)
            df_concat = fu.drop_duplicate_columns(df_concat)
            sef_df = df_concat

        elif spectral_edge_method == 'manual':
            edges_open, edge_names = get_spectral_edges(stack_open_power, freqs, channels=channels, edge_increment=edge_increment, chan_axis=1, return_edge_names=True, num_edges=num_edges, log_edges=log_edges, reverse_log=reverse_log)
            edges_closed, edge_names = get_spectral_edges(stack_closed_power, freqs, channels=channels, edge_increment=edge_increment, chan_axis=1, return_edge_names=True, num_edges=num_edges, log_edges=log_edges, reverse_log=reverse_log)

            open_edge_df = pd.DataFrame(edges_open.reshape(-1, edges_open.shape[1]*edges_open.shape[2]), columns=[f"open_{chan}_{feature}" for chan in channels for feature in edge_names], index=subjs)
            closed_edge_df = pd.DataFrame(edges_closed.reshape(-1, edges_closed.shape[1]*edges_closed.shape[2]), columns=[f"closed_{chan}_{feature}" for chan in channels for feature in edge_names], index=subjs)
            df_concat = pd.concat([open_edge_df, closed_edge_df], axis=1)
            df_concat = fu.drop_duplicate_columns(df_concat)
            sef_df = df_concat
        elif spectral_edge_method == 'automated':

            # create the spectral edges using various number of edges and spectral edge increments
            edge_increments = [1, 2, 5, 10]
            num_edges = [10, 20, 50, 100]
            log_edges = [True, False]
            reverse_log = [True, False]

            open_edges = []
            closed_edges = []
            edge_names_open = []
            edge_names_closed = []

            print("Computing spectral edges...")
            edgetimes=  time.time()
            for edge_increment in edge_increments:
                edges_open, edge_names = get_spectral_edges(stack_open_power, freqs, channels=channels, edge_increment=edge_increment, chan_axis=1, return_edge_names=True)
                edges_closed, edge_names = get_spectral_edges(stack_closed_power, freqs, channels=channels, edge_increment=edge_increment, chan_axis=1, return_edge_names=True)
            
                open_edges.append(edges_open)
                closed_edges.append(edges_closed)
                edge_names_open.append(edge_names)
                edge_names_closed.append(edge_names)

            for num_edge in num_edges:
                for log_edge in log_edges:
                    for reverse in reverse_log:
                        if not log_edge and reverse:
                            continue
            
                        edges_open, edge_names = get_spectral_edges(stack_open_power, freqs, channels=channels, num_edges=num_edge, log_edges=log_edge, reverse_log=reverse, chan_axis=1, return_edge_names=True)
                        edges_closed, edge_names = get_spectral_edges(stack_closed_power, freqs, channels=channels, num_edges=num_edge, log_edges=log_edge, reverse_log=reverse, chan_axis=1, return_edge_names=True)
                        open_edges.append(edges_open)
                        closed_edges.append(edges_closed)
                        edge_names_open.append(edge_names)
                        edge_names_closed.append(edge_names)
            print(f"Finished computing spectral edges in {time.time() - edgetimes} seconds")



            dfs_open = []
            dfs_closed = []
            dfs_concat = []
            # save each in dataframes, where index is subject and columns are channels and spectral edges
            for idx, (oes, ces) in enumerate(zip(open_edges, closed_edges)):
                oes = oes.reshape(-1, oes.shape[1]*oes.shape[2])
                ces = ces.reshape(-1, ces.shape[1]*ces.shape[2])
                col_names_open = [f"open_{chan}_{feature}" for chan in channels for feature in edge_names_open[idx]]
                col_names_closed = [f"closed_{chan}_{feature}" for chan in channels for feature in edge_names_closed[idx]]
                # can check that this correct by making sure the first N columns have the same channel and are monotonically increasing in oes and ces
                assert np.all([oes[:, idx] <= oes[:, idx+1] for idx in range(len(edge_names_open[idx])-1)])
                assert np.all([ces[:, idx] <= ces[:, idx+1] for idx in range(len(edge_names_closed[idx])-1)])
                assert len(np.unique([col.split('_')[1] for col in col_names_open[:len(edge_names_open[idx])]])) == 1
                assert len(np.unique([col.split('_')[1] for col in col_names_closed[:len(edge_names_closed[idx])]])) == 1

                df_open = pd.DataFrame(oes, columns=col_names_open, index=subjs)
                df_closed = pd.DataFrame(ces, columns=col_names_closed, index=subjs)
                dfs_open.append(df_open)
                dfs_closed.append(df_closed)

                concat_df = pd.concat([df_open, df_closed], axis=1)
                # concat_df.columns = ['open_' + col for col in df_open.columns] + ['closed_' + col for col in df_closed.columns]
                dfs_concat.append(concat_df)

            df_concat = pd.concat(dfs_concat, axis=1)
            df_concat = fu.drop_duplicate_columns(df_concat)
            sef_df = df_concat
        sef_df = fu.select_subjects_from_dataframe(sef_df, choose_subjs, internal_folder=internal_folder)

        if "sef" not in sef_df.columns[0]:
            sef_df.columns = ['SEF_' + col for col in sef_df.columns]
        if save:
            sef_df.to_csv(os.path.join(savepath, 'all_spectral_edge_frequencies.csv'))
            # save the parameters
            with open(os.path.join(savepath, 'params.json'), 'w') as f:
                json.dump(spectral_params, f)
            print("Saved all spectral edge features to ", os.path.join(savepath, 'all_spectral_edge_frequencies.csv'))
        # remove duplicate columns (same header same values)
    return sef_df
    

def get_spectral_edges(psd, freqs, channels=CHANNELS, spectral_edges=None, edge_increment=None, num_edges=None, log_edges=True, reverse_log=False, chan_axis=0, return_edge_names=True):
    """
    Compute the spectral edge at edge_increment increments
    Inputs:
        data: numpy array of shape (channels, freq_samples) if chan_axis=0, else (n_samples, channels, freq_samples) if chan_axis=1
        freqs: numpy array of shape (freq_samples,)
        channels: list of channel names
        fs: sampling frequency
        edge_increment: increment for the spectral edge
        num_edges: number of edges to compute
        log_edges: whether to compute log edges
        reverse_log: whether to reverse the log edges
        chan_axis: axis of channels
        return_edge_names: whether to return the edge names
    Outputs:
        edges array of shape (channels, num_edges)
    """
    if spectral_edges is not None:
        edge_names = spectral_edges
        num_edges = len(edge_names)
    else:
        if edge_increment is not None and num_edges is None:
            num_edges = int(100 / edge_increment)-1
            edge_names = [(edge+1)*edge_increment / 100 for edge in range(num_edges)]
        elif num_edges is not None and edge_increment is None:
            if log_edges:
                if reverse_log:
                    edge_names = 1-np.logspace(-2, 0, num_edges, base=10)
                    edge_names = np.flip(edge_names)
                else:
                    edge_names = np.logspace(-2, 0, num_edges, base=10)
            else:
                edge_names = np.linspace(0, 1, num_edges)
        else:
            raise ValueError("Must specify either edge_increment or num_edges, but not both.")
    n_channels = psd.shape[chan_axis]
    if chan_axis == 0:
        edges = np.zeros((n_channels, num_edges))
    elif chan_axis == 1:
        edges = np.zeros((psd.shape[0], n_channels, num_edges))
    else:
        raise ValueError("chan_axis must be 0 or 1")

    dx = freqs[1] - freqs[0]


    if chan_axis == 0:
        for cdx, channel in enumerate(channels):
            total_cum_power = gp_integrate.cumulative_simpson(psd[cdx, :], dx=dx, initial=0) # https://stackoverflow.com/questions/18215163/cumulative-simpson-integration-with-scipy
            total_chan_power = total_cum_power[-1]
            # another approximation for integration of signal
            # total_simps_power = scipy.integrate.simps(pxs[channel,:], dx=dx) # close but not exact
            for edx in range(num_edges):
                edge_power = total_chan_power * edge_names[edx]
                edges[cdx, edx] = freqs[np.where(total_cum_power >= edge_power)[0][0]]
    elif chan_axis == 1:
        for ndx in range(psd.shape[0]):
            for cdx, channel in enumerate(channels):
                total_cum_power = gp_integrate.cumulative_simpson(psd[ndx, cdx, :], dx=dx, initial=0)
                total_chan_power = total_cum_power[-1]
                for edx in range(num_edges):
                    edge_power = total_chan_power * edge_names[edx]
                    edges[ndx, cdx, edx] = freqs[np.where(total_cum_power >= edge_power)[0][0]]
        
    if return_edge_names:
        return edges, edge_names
    else:
        return edges


if __name__ == '__main__':
    sef_df = main()
    # big_df.to_csv(os.path.join(savepath, 'all_spectral_edge_frequencies.csv'))
    # print("Saved all spectral edge features to ", os.path.join(savepath, 'all_spectral_edge_frequencies.csv'))