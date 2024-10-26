import argparse
import json
import os
import numpy as np

import train_regression as tr
import train_models as tm



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('params_file', help='JSON file with model parameters.')
    parser.add_argument('--train_csv', help='Train csv file')
    parser.add_argument('--test_csv', help='Test csv file')

    return parser.parse_args()

def perform_test(args):
    with open(args.params_file, 'r') as in_file:
        params = json.load(in_file)

    tr.set_random_state(params['random_state'])

    train_X, train_y, valid_X, valid_y, test_X, test_y, feature_cols = tr.load_data(args.train_csv, args.test_csv)

    model = tr.make_model(
        feature_cols=feature_cols,
        preprocess_pipe=tm.get_preprocess_pipe(),
        params=params)

    model.train(X_train=train_X, y_train=train_y,
                X_valid=valid_X, y_valid=valid_y)

    _ = tr.evaluate_model(model, 
        train_X, train_y, valid_X, valid_y, test_X, test_y)

    return model

def train_production_model(args, max_epochs):
    with open(args.params_file, 'r') as in_file:
        params = json.load(in_file)

    tr.set_random_state(params['random_state'])

    train_X, train_y, valid_X, valid_y, test_X, test_y, feature_cols = tr.load_data(args.train_csv, args.test_csv)

    # production model will combine train, valid, and test datasets
    train_X = np.concatenate([train_X, valid_X, test_X])
    train_y = np.concatenate([train_y, valid_y, test_y])

    model = tr.make_model(
        feature_cols=feature_cols,
        preprocess_pipe=tm.get_preprocess_pipe(),
        params=params,
        early_stopping=False,
        max_epochs=max_epochs)

    model.train(X_train=train_X, y_train=train_y,
                X_valid=valid_X, y_valid=valid_y)

    return model


if __name__ == '__main__':
    args = parse_args()

    print("testing")
    test_model = perform_test(args)

    # create output file name from input filename
    test_output_filename = os.path.splitext(args.params_file)[0]
    test_output_filename = test_output_filename+'_test_model.pt'
    test_model.save(test_output_filename)

    print("Test model saved to", test_output_filename)

    print("Training production model")
    prod_model = train_production_model(args, test_model.best_epoch)

    # create output file name from input filename
    output_filename = os.path.splitext(args.params_file)[0]
    output_filename = output_filename+'_production_model.pt'
    prod_model.save(output_filename)

    print("Production model saved to", output_filename)
