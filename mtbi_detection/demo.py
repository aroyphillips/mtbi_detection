### code to run the whole pipeline

import mtbi_detection.data.extract_data as extract_data
import mtbi_detection.data.load_open_closed_data as load_open_closed_data

def main():
    extract_data.main(dlpath, datapath)
    load_open_closed_data.oad_open_closed_pathdict(datapath)

if __name__ == "__main__":
    main()