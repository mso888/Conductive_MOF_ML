import split_util as su
import simple_featurizer as sf

def random_split(oqmd_df, desired_test_frac=.3, random_state=0):
    oqmd_df = oqmd_df.copy()
    # shuffle
    oqmd_df = oqmd_df.sample(frac=1, random_state=random_state)
    split_index = int(desired_test_frac*len(oqmd_df))
    train_df = oqmd_df[split_index:]
    test_df = oqmd_df[:split_index]
    return train_df, test_df

def main(dataset, threshold, drop_duplicates, keep_cols):
    data_csv = '../data/property_matrix_6-14.csv'

    # read the property matrix using StudentData
    sd = sf.StudentPropertyMatrix(data_csv)
    print(dataset)
    sld = dataset(sd)

    print(sld.get_feature_columns())
    num_feats = len(sld.get_feature_columns())

    all_features = su.get_all_features(sld, threshold, drop_duplicates, keep_cols)

    oqmd_df = all_features[(all_features['label'] == 'conducting') | (all_features['label'] == 'non-conducting')]

    train_df, test_df = random_split(oqmd_df=oqmd_df)

    print('train_df pos/neg split', len(train_df[train_df['label']=='conducting']), len(train_df[train_df['label']=='non-conducting']))
    print('test_df pos/neg split', len(test_df[test_df['label']=='conducting']), len(test_df[test_df['label']=='non-conducting']))

    dup_str = '_nodup' if drop_duplicates else ''
    train_df.to_csv(f"../data/query_files9/random_train_df_{num_feats}{dup_str}.csv", index=False)
    test_df.to_csv(f"../data/query_files9/random_test_df_{num_feats}{dup_str}.csv", index=False)
    
if __name__ == '__main__':

    args = su.parse_args()
    main(sf.SoLab_Dataset, threshold=args.threshold, drop_duplicates=args.drop_duplicates, keep_cols=(not args.drop_cols))
    main(sf.Yuping_Dataset, threshold=args.threshold, drop_duplicates=args.drop_duplicates, keep_cols=(not args.drop_cols))


    