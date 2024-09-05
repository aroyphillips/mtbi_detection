### code to run the whole pipeline

import argparse

import mtbi_detection.data.extract_data as extract_data
import mtbi_detection.data.load_open_closed_data as load_open_closed_data
import mtbi_detection.modeling.train_all_baselearners as train_all_baselearners
import mtbi_detection.modeling.evaluate_final_models as evaluate_final_models

def main(dlpath, datapath):
    extract_data.main(dlpath, datapath)
    load_open_closed_data.load_open_closed_pathdict(datapath)
    train_all_baselearners.main()
    final_metamodel.main()
    evaluate_final_models.main()

    # regression models
    extract_symptoms_data.main(dlpath, datapath)
    train_all_baseregressors.main()
    final_metaregressor.main()
    evaluate_final_regressors.main()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run the whole pipeline')
    parser.add_argument('--dlpath', type=str, help='Path to download the data', default='data/downloads/')
    parser.add_argument('--datapath', type=str, help='Path to save the data', default='data/processed_data/raw_files/')
    main()