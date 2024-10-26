import pandas as pd

import simple_featurizer as sf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--threshold', type=float, default=1.5, help="Threshold for conductivity. Default 1.5eV")
    parser.add_argument('--drop_cols', action='store_true', help="Drop original oqmd data columns")
    parser.add_argument('--drop_duplicates', action='store_true', help="Drop duplicate chemical formulas with the same label")

    return parser.parse_args()

def PCA_vis(train_df, test_df, all_features):
    # build X
    coremof_df = all_features[(all_features['label'] == 'unknown')]

    all_df = pd.concat([train_df, test_df, coremof_df])
    labels = ['train']*train_df.shape[0] + ['test']*test_df.shape[0] + ['coremof']*coremof_df.shape[0]

    all_df['label'] = labels
    X = all_df[[c for c in all_df.columns if c.startswith('f_')]].to_numpy()

    # in this case, it is necessary.
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print('train', train_df.shape[0], 'test', test_df.shape[0])

    sns.scatterplot(x=X_transformed[:,0], y=X_transformed[:,1], hue=all_df['label'].to_numpy())

    plt.show()

def standard_embedder(X):
    X = StandardScaler().fit_transform(X)

    return X
 
def build_data(joined_csv, dataset, cif_csv='../data/cif_chemical_formulas.csv', threshold=1.5, keep_cols=False, drop_duplicates=True):
    # cif_csv this contains chemical formulas of compounds from COREMOF

    # read_csv files
    all_df = pd.read_csv(joined_csv)
    all_df['label'] = all_df['_oqmd_band_gap'].apply(lambda x: 'conducting' if x <= threshold else 'non-conducting')
    if drop_duplicates:
        all_df = remove_duplicates(all_df)

    cif_df = pd.read_csv(cif_csv)
    cif_df = cif_df.rename(columns={'filename':'id'})
    cif_df['label'] = ['unknown']*len(cif_df)

    all_df = pd.concat([all_df, cif_df])
    all_features = sf.from_df_to_df2(dataset, all_df, keep_cols)
    
    return all_features
    
def remove_duplicates(oqmd_df):
    # remove duplicates with the same label
    print('before', oqmd_df.shape[0])
    print('conducting', oqmd_df[oqmd_df['label']=='conducting'].shape[0], 'non-conducting', oqmd_df[oqmd_df['label']=='non-conducting'].shape[0])
    oqmd_df = oqmd_df.drop_duplicates(subset=['chemical_formula', 'label'])
    print('after', oqmd_df.shape[0])
    print('conducting', oqmd_df[oqmd_df['label']=='conducting'].shape[0], 'non-conducting', oqmd_df[oqmd_df['label']=='non-conducting'].shape[0])
    
    return oqmd_df
    
def get_all_features(sld, threshold, drop_duplicates, keep_cols):
    num_feats = len(sld.get_feature_columns())

    if keep_cols:
        col_str = '_withcols'
    else:
        col_str = ''

    if drop_duplicates:
        dup_str = ''
    else:
        dup_str = '_withdups'

    all_compounds_file = f'../data/{num_feats}{col_str}{dup_str}_all_compounds.csv'
    if os.path.exists(all_compounds_file):
        all_features = pd.read_csv(all_compounds_file,
            dtype={'id':str, 'sanity_id':str})
    else:
        all_features = build_data(
            joined_csv = '../data/query_files9/joined.csv',
            dataset = sld,
            threshold=threshold,
            drop_duplicates=drop_duplicates,
            keep_cols=keep_cols)
        
        all_features.to_csv(all_compounds_file, index=False)

    all_features['id'] = all_features['id'].apply(str)
    all_features.set_index('id', drop=False)

    coremof_df = all_features[all_features['label']=='unknown']
    coremof_file = f'../data/{num_feats}{col_str}{dup_str}_coremof.csv'
    coremof_df.to_csv(coremof_file, index=False)

    return all_features