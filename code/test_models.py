## Test models created by test_and_save_model.py

import ensemble_test as et
import simple_featurizer as sf
import train_models as tm

import sklearn.metrics as metrics

import argparse
import pandas as pd
import pprint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_pkls', nargs='+', help='PKL of trained model.')
    parser.add_argument('--test_csv', help='Test csv file')
    parser.add_argument('--label_col', default='label', help='Which column contains labels')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for pkl in args.model_pkls:
        print(pkl)
        model = et.load_model(pkl)

        test_X, test_y = tm.load_single_csv(args.test_csv)

        pred_y = model.predict(test_X)

        test_metric = tm.get_objective_result(test_y, pred_y)

        test_metric['weighted_precision'] = metrics.precision_score(test_y, pred_y, average='weighted')

        pprint.pprint(test_metric)

        print(f'{test_metric["precision"]:0.4f}\t{test_metric["inv_precision"]:0.4f}\t{test_metric["weighted_precision"]:0.4f}')

        print('-----------------------------')
