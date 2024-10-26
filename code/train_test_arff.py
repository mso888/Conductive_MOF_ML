import pandas as pd
import simple_featurizer as sf


def total_arff(train_df_fn, test_df_fn, outname, dataset):
    train_df = pd.read_csv(train_df_fn)
    test_df = pd.read_csv(test_df_fn)
    total_df = pd.concat([train_df, test_df])

    dataset.from_df_to_arff_file(total_df, out_arff=outname)

def train_test_arff(train_df_fn, test_df_fn, dataset):
    train_df = pd.read_csv(train_df_fn)
    test_df = pd.read_csv(test_df_fn)
    
    # change out_arff to your own directories
    dataset.from_df_to_arff_file(train_df, out_arff=train_df_fn.replace('.csv', '.arff'))
    # change out_arff to your own directories
    dataset.from_df_to_arff_file(test_df, out_arff=test_df_fn.replace('.csv', '.arff'))

if __name__ == '__main__':
    
    data_csv = '../data/property_matrix_6-14.csv'
    
    # read the property matrix using StudentData
    sd = sf.StudentPropertyMatrix(data_csv)
    sld = sf.SoLab_Dataset(sd)
    yd = sf.Yuping_Dataset(sd)

    train_test_arff(
        train_df_fn='../data/query_files9/train_df_44_nodup.csv',
        test_df_fn='../data/query_files9/test_df_44_nodup.csv',
        dataset=sld)
        
    train_test_arff(
        train_df_fn='../data/query_files9/train_df_70_nodup.csv',
        test_df_fn='../data/query_files9/test_df_70_nodup.csv',
        dataset=yd)
    
    total_arff(
        train_df_fn='../data/query_files9/train_df_44_nodup.csv',
        test_df_fn='../data/query_files9/test_df_44_nodup.csv',
        outname='../data/query_files9/total_df_44_nodup.arff',
        dataset=sld)
        
    total_arff(
        train_df_fn='../data/query_files9/train_df_70_nodup.csv',
        test_df_fn='../data/query_files9/test_df_70_nodup.csv',
        outname='../data/query_files9/total_df_70_nodup.arff',
        dataset=yd)