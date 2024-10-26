import pandas as pd
import os


def drop_metals_and_outliers(csv_file, label_col='_oqmd_band_gap'):
    df = pd.read_csv(csv_file)
    print(f"Before length {len(df)}")

    df = df.dropna(subset=[label_col])
    df = df[df[label_col]>0]
    df = df[df[label_col]<15]

    print(f"After {len(df)}")    

    new_path = os.path.splitext(csv_file)[0]+'_regression.csv'
    df.to_csv(new_path, index=False)

if __name__ == '__main__':
    drop_metals_and_outliers('../data/query_files9/random_train_df_44_nodup.csv')
    drop_metals_and_outliers('../data/query_files9/random_test_df_44_nodup.csv')

    drop_metals_and_outliers('../data/query_files9/random_train_df_70_nodup.csv')
    drop_metals_and_outliers('../data/query_files9/random_test_df_70_nodup.csv')
