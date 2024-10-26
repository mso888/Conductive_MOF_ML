# After hyperopt training 
# - retrain model using best hyperparamters
# - save the model using sklearn
# - report metrics

import argparse
import json
import os
import pickle

import train_models as tm
import train_rf as t_rf
import train_svm as t_svm
import train_NN as t_NN
import train_lr as t_lr

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', choices=['rf', 'svm', 'NN', 'lr'], help='The type of model the parameters are for.')
    parser.add_argument('params_file', help='JSON file with model parameters.')
    parser.add_argument('--train_csv', help='Train csv file')
    parser.add_argument('--valid_csv', default='', help='Test csv file')
    parser.add_argument('--test_csv', help='Test csv file')

    return parser.parse_args()

def get_model(params, args):
    print(args.model_type)
    if args.model_type == 'rf':
        clf = t_rf.make_model(params)
    elif args.model_type == 'svm':
        clf = t_svm.make_model(params)
    elif args.model_type == 'NN':
        clf = t_NN.make_model(params)
    elif args.model_type == 'lr':
        clf = t_lr.make_model(params)

    return clf

def perform_test(args):
    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    # sometimes there's no validation data when using cross validation
    if valid_X is not None:
        # use both training and validation at this point since validation is no longer necessary
        train_X = np.concatenate([train_X, valid_X], axis=0)
        train_y = np.concatenate([train_y, valid_y], axis=0)

    with open(args.params_file, 'r') as in_file:
        params = json.load(in_file)

    clf = get_model(params, args)
    clf.fit(train_X, train_y)

    pred_y = clf.predict(test_X)

    test_metric = tm.get_objective_result(test_y, pred_y)

    print(f'    test_precision:{test_metric["precision"]:0.4f}')
    print(f'    test_inv_precision:{test_metric["inv_precision"]:0.4f}')

    return clf

def train_production_model(args):
    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    # sometimes there's no validation data when using cross validation
    if valid_X is not None:
        # use both training and validation at this point since validation is no longer necessary
        train_X = np.concatenate([train_X, valid_X, test_X], axis=0)
        train_y = np.concatenate([train_y, valid_y, test_y], axis=0)
    else:
        train_X = np.concatenate([train_X, test_X], axis=0)
        train_y = np.concatenate([train_y, test_y], axis=0)

    with open(args.params_file, 'r') as in_file:
        params = json.load(in_file)

    clf = get_model(params, args)
    clf.fit(train_X, train_y)

    return clf

def save_model(clf, output_filename):
    with open(output_filename, 'wb') as f:
        pickle.dump(clf, f, protocol=5)        

    return output_filename

if __name__ == '__main__':
    args = parse_args()

    print("testing")
    test_clf = perform_test(args)

     # create output file name from input filename
    test_output_filename = os.path.splitext(args.params_file)[0]
    test_output_filename = test_output_filename+'_test_model.pkl'

    save_model(test_clf, test_output_filename)

    print("Test model saved to", test_output_filename)

    print("training production model")
    clf = train_production_model(args)

     # create output file name from input filename
    output_filename = os.path.splitext(args.params_file)[0]
    output_filename = output_filename+'_production_model.pkl'

    save_model(clf, output_filename)

    print("Production model saved to", output_filename)