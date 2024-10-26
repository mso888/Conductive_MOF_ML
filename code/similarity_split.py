import pandas as pd
import numpy as np
import os

import simple_featurizer as sf
import split_util as su

from sklearn.neighbors import BallTree


def similarity_split(all_features, embedding_function, k=1, desired_test_frac=.3, random_state=0):
    oqmd_df = all_features[(all_features['label'] == 'conducting') | (all_features['label'] == 'non-conducting')]
    coremof_df = all_features[(all_features['label'] == 'unknown')]
    
    oqmd_X = oqmd_df[[c for c in oqmd_df.columns if c.startswith('f_')]].to_numpy()
    coremof_X = coremof_df[[c for c in coremof_df.columns if c.startswith('f_')]].to_numpy()
    print('before shapes', oqmd_X.shape, coremof_X.shape)

    split_index = oqmd_X.shape[0]
    total_X = np.concatenate([oqmd_X, coremof_X])
    embedded_X = embedding_function(total_X)

    oqmd_X = embedded_X[:split_index]
    coremof_X = embedded_X[split_index:]
    print('after shapes', oqmd_X.shape, coremof_X.shape)

    oqmd_tree = BallTree(oqmd_X, leaf_size=40)

    dists, indexes = oqmd_tree.query(coremof_X, k=k)
    indexes = np.concatenate(indexes)

    closest_df = oqmd_df.iloc[list(set(indexes))]
    closest_df = closest_df.sample(frac=1, random_state=random_state) # shuffle

    test_split = int(desired_test_frac*len(oqmd_X))
    if test_split > len(closest_df):
        print("Warning. k too small to satisfy desired test frac")
    test_df = closest_df.iloc[:test_split]
    valid_df = closest_df.iloc[test_split:]
    train_df = oqmd_df[~oqmd_df.index.isin(closest_df.index)]

    return train_df, valid_df, test_df

def main(dataset, threshold, drop_duplicates, keep_cols):
    data_csv = '../data/property_matrix_6-14.csv'

    # read the property matrix using StudentData
    sd = sf.StudentPropertyMatrix(data_csv)
    print(dataset)
    sld = dataset(sd)

    print(sld.get_feature_columns())
    num_feats = len(sld.get_feature_columns())

    all_features = su.get_all_features(sld, threshold, drop_duplicates, keep_cols)

    train_df, valid_df, test_df = similarity_split(all_features, su.standard_embedder, k=700)

    print('train_df pos/neg split', len(train_df[train_df['label']=='conducting']), len(train_df[train_df['label']=='non-conducting']))
    print('valid_df pos/neg split', len(valid_df[valid_df['label']=='conducting']), len(valid_df[valid_df['label']=='non-conducting']))
    print('test_df pos/neg split', len(test_df[test_df['label']=='conducting']), len(test_df[test_df['label']=='non-conducting']))

    dup_str = '_nodup' if drop_duplicates else ''
    train_df.to_csv(f"../data/query_files9/similarity_train_df_{num_feats}{dup_str}.csv", index=False)
    valid_df.to_csv(f"../data/query_files9/similarity_valid_df_{num_feats}{dup_str}.csv", index=False)
    test_df.to_csv(f"../data/query_files9/similarity_test_df_{num_feats}{dup_str}.csv", index=False)
    
if __name__ == '__main__':

    args = su.parse_args()
    main(sf.SoLab_Dataset, threshold=args.threshold, drop_duplicates=args.drop_duplicates, keep_cols=(not args.drop_cols))
    main(sf.Yuping_Dataset, threshold=args.threshold, drop_duplicates=args.drop_duplicates, keep_cols=(not args.drop_cols))



    