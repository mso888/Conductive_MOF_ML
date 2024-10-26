import ensemble_test as et
import simple_featurizer as sf
import train_models as tm

import argparse
import pandas as pd
import pdb


excluded_elements = set(['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 
                         'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 
                         'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 
                         'Fl', 'Mc', 'Lv', 'Ts', 'Og', #87-188
                         'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
                         'Ho', 'Er', 'Tm', 'Yb', 'Lu']) # 58-71

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_44', nargs='+', help='Models using 44 features.')
    parser.add_argument('--model_70', nargs='+', help='Models using 70 features.')
    parser.add_argument('--vote_method', choices=['gt_marjority', 'gteq_majority', 'consensus'], 
                        default='gteq_majority',
                        help='Does the number of conductive votes need to be greater than or'
                        ' greater than or equal to the number of non-conductive votes to win.')

    parser.add_argument('--data_44', default='../data/44_withcols_coremof.csv',
        help='COREMOF data featurized with 44 features')
    parser.add_argument('--data_70', default='../data/70_withcols_coremof.csv',
        help='COREMOF data featurized with 44 features')

    parser.add_argument('--output_results', default='../data/screening_results.csv',
        help='Where to save screening results.')

    parser.add_argument('--formula_col', default='chemical_formula',
        help='The column in data_44 and data_70 that contains the chemical formula')
    parser.add_argument('--filename_col', default='id',
        help='The column in data_44 and data_70 that contains the cif filename'
        'The filename should have the format <REFCODE>_<status>.cif')
    parser.add_argument('--id_col', default='id',
        help='Column containing the ID.')

    parser.add_argument('--add_doi_information', action='store_false',
        help='Option to add doi information to screening results')
    parser.add_argument('--doi_csv', default='../data/coremof_doi.csv',
        help='CSV file with doi numbers for coremof compounds')
    parser.add_argument('--refcode_col', default='REFCODE',
        help='The column with cif REFCODEs')
    parser.add_argument('--doi_col', default='DOI',
        help='Column containing DOI information. Default - used to indicate no DOI.')
    parser.add_argument('--no_doi_symbol', default='-',
        help='Column containing DOI information. no_doi_symbol used to indicate no DOI.')
    parser.add_argument('--keep_no_doi', action='store_true',
        help='Keep conducdtive mofs that do not have a doi number.')

    return parser.parse_args()

def element_filter(chemical_formula):
    elements = chemical_formula.split(r'/d+')

    for e in elements:
        if e in excluded_elements:
            return False
    return True

def curate_coremof_csv(csv_file, df_info):
    df = pd.read_csv(csv_file, dtype={df_info.id_col:str})

    # we need to filter out compounds with certain elements
    keep = [element_filter(formula) for formula in df[df_info.formula_col]]
    df = df[keep]

    df = df.set_index(df_info.id_col, drop=False)

    return df

def add_doi_information(df, doi_df, df_info, keep_no_doi):
    # create a REFCODE column to check for DOI later
    df[df_info.refcode_col] = df[df_info.filename_col].apply(lambda x: x[:x.rfind('_')])
    df = df.merge(doi_df, on=df_info.refcode_col, how='inner')

    print('After adding DOI numbers, there are', len(df), 'mofs')

    if not keep_no_doi:
        df = df[df[df_info.doi_col]!=df_info.no_doi_symbol]

        print('After dropping cifs with no doi there are', len(df), 'mofs')

    df = df.set_index(df_info.id_col, drop=False)

    return df

def screen(ensemble_model, dataset_df):
    feature_cols = sf.get_feature_columns(dataset_df)
    X = dataset_df[feature_cols].values

    preds = ensemble_model.predict(X)
    is_conductive = [tm.labels[p]=='conducting' for p in preds]

    return dataset_df[is_conductive]

if __name__ == "__main__":
    args = parse_args()

    em_44 = et.EnsembleModel([et.load_model(pkl) for pkl in args.model_44],
                             vote_method=args.vote_method) 
    em_70 = et.EnsembleModel([et.load_model(pkl) for pkl in args.model_70],
                             vote_method=args.vote_method)

    df_44 = curate_coremof_csv(args.data_44, args)
    df_70 = curate_coremof_csv(args.data_70, args)

    assert(len(set(df_44[args.id_col])) == len(df_44))
    assert(len(set(df_70[args.id_col])) == len(df_70))

    conductive_44_df = screen(em_44, df_44)
    conductive_70_df = screen(em_70, df_70)

    print(f'using 44 features found {len(conductive_44_df)}')
    print(f'using 70 features found {len(conductive_70_df)}')
    
    both_ids = set(conductive_44_df.index).intersection(set(conductive_70_df.index))
    both_conductive_df = conductive_70_df.loc[list(both_ids)]

    print('found', len(both_conductive_df), 'conducting mofs')
    
    if args.add_doi_information:
        doi_df = pd.read_csv(args.doi_csv)
        both_conductive_df = add_doi_information(both_conductive_df, doi_df, args, args.keep_no_doi)

        column_order = [args.id_col, args.doi_col, args.formula_col]
    else:
        column_order = [args.id_col, args.formula_col]

    # reorder columns for convenience
    remaining_columns = list(both_conductive_df.columns)
    for c in column_order:
        remaining_columns.remove(c)

    new_order = column_order+remaining_columns
    both_conductive_df = both_conductive_df[new_order]

    # output csv
    both_conductive_df.to_csv(args.output_results, index=False)