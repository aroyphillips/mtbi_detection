import shutil
import os
import argparse

DATAPATH = open('extracted_path.txt', 'r').read().strip()
def cleanpath(path, foldername='params'):
    """
    given a path that contains directories of the name foldername{0,1,2,...}
    remove all folders that are empty and rename the remaining folders to
    foldername{0,1,2,...}
    """
    # get all folders in path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    # remove all empty folders
    for folder in folders:
        if not os.listdir(os.path.join(path, folder)):
            ui = input(f'Remove empty folder: {os.path.join(path, folder)}? (y/n)')
            if ui == 'y':
                shutil.rmtree(os.path.join(path, folder))


    # remove all folders that do not contain a params.json file
    for folder in folders:
        if 'params.json' not in os.listdir(os.path.join(path, folder)):
            ui = input(f'Remove folder since it has no params.json: {os.path.join(path, folder)}? (y/n)')
            if ui == 'y':
                shutil.rmtree(os.path.join(path, folder))


    # rename the remaining folders
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # sort folders by name
    folders.sort(key=lambda x: int(x[len(foldername):]))
    for i, folder in enumerate(folders):
        if folder == foldername + str(i):
            continue
        ui = input(f'renaming folder: {os.path.join(path, folder)} to {os.path.join(path, foldername + str(i))}? (y/n)')
        if ui == 'y':
            os.rename(os.path.join(path, folder), os.path.join(path, foldername + str(i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean path')
 
    parser.add_argument('--path', type=str, default=os.path.join(os.path.dirname(DATAPATH[:-1]), 'open_closed_segments', 'params'), help='path to clean')
    parser.add_argument('--foldername', type=str, default='params', help='foldername to rename to')
    args = parser.parse_args()

    cleanpath(args.path, args.foldername)