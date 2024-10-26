import argparse
import os
import pandas as pd
import json
import pdb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("d", help="directory holding json files")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    files = os.listdir(args.d)
    
    json_files = [os.path.join(args.d, f) for f in files if f.endswith('.json') ]
    
    columns_of_interest = ['id', 'nelements', 'chemical_formula', 'elements', 
        'formula_prototype', '_oqmd_entry_id', '_oqmd_calculation_id', '_oqmd_icsd_id',
        '_oqmd_band_gap', '_oqmd_delta_e', '_oqmd_stability', '_oqmd_prototype', '_oqmd_spacegroup',
        '_oqmd_natoms']
    
    df = {}
    for c in columns_of_interest:
        df[c] = []
        
    for jf in tqdm(json_files):
        j = json.load(open(jf, 'r'))
        for c in columns_of_interest:
            for mof in j:
                df[c].append(mof[c])
            
    df = pd.DataFrame(df)
    
    df.to_csv(os.path.join(args.d, 'joined.csv'), index=False)