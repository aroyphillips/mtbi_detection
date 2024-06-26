### code to run the whole pipeline

import argparse

import mtbi_detection.data.extract_data as extract_data
import mtbi_detection.data.load_open_closed_data as load_open_closed_data

def main(dlpath, datapath):
    extract_data.main(dlpath, datapath)
    load_open_closed_data.load_open_closed_pathdict(datapath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the whole pipeline')
    parser.add_argument('--dlpath', type=str, help='Path to download the data', default='data/downloads/')
    parser.add_argument('--datapath', type=str, help='Path to save the data', default='data/processed_data/raw_files/')
    main()