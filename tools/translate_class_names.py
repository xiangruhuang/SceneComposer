import numpy as np
import pickle
import glob, os
import argparse
from tqdm import tqdm
from multiprocessing import Pool

TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

pickle_files = None

def parse_args():
    parser = argparse.ArgumentParser("Add class names into annotations")
    parser.add_argument('split')
    parser.add_argument('--root_path', default='data/Waymo', type=str)
    args = parser.parse_args()

    return args

def convert(idx):
    pickle_file = pickle_files[idx]
    with open(pickle_file, 'rb') as fin:
        annos = pickle.load(fin)
    
    objects = annos['objects']
    for o in objects:
        o['class_name'] = TYPE_LIST[o['label']]
    annos['objects'] = objects

    with open(pickle_file, 'wb') as fout:
        pickle.dump(annos, fout)

def main():
    global pickle_files
    args = parse_args()

    path = os.path.join(
               args.root_path,
               args.split,
               'annos',
               '*.pkl'
           )
    pickle_files = sorted(list(glob.glob(path)))

    with Pool(128) as p: # change according to your cpu
        frames = list(tqdm(p.imap(convert, range(len(pickle_files))), total=len(pickle_files)))

if __name__ == '__main__':
    main()
