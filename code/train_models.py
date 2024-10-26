import sklearn
import pandas as pd
import argparse

import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.ensemble
import sklearn.metrics
import simple_featurizer as sf

import json
import pickle
import time
import os
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL

labels = ['non-conducting', 'conducting']
label_map = {labels[i]:i for i in range(len(labels))}
label_inv_map = {i:labels[i] for i in range(len(labels))}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', help='Path to training csv file.')
    parser.add_argument('--valid_csv', default='', help='Path to valid csv file. If blank, use cross validation using training set')
    parser.add_argument('--test_csv', help='Path to testing csv file.')

    parser.add_argument('--output_dir', default='', help='Where trained models are saved.')
    parser.add_argument('--max_evals', type=int, default=100, help='Number of hyperoptimization trials')
    parser.add_argument('--random_state', type=int, default=0, help='Random seed')
    parser.add_argument('--cv_parallel_jobs', type=int, default=1, help='Number of parallel jobs when using cross validation')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross validation folds')
    parser.add_argument('--cv_valid_frac', type=float, default=0.5)

    return parser.parse_args()

def get_X_y(df, feature_cols, label_col):
    X = df[feature_cols].values
    conducting = df[label_col].values

    y = [label_map[c] for c in conducting]

    assert X.shape[1] == 44 or X.shape[1] == 70

    return X, y


def load_single_csv(csv_file):
    df = pd.read_csv(csv_file, dtype={'id':str})
    feature_cols = sf.get_feature_columns(df)

    X, y = get_X_y(df, feature_cols, label_col='label')

    return X, y

def load_data(train_csv, valid_csv, test_csv):
    train_df = pd.read_csv(train_csv, dtype={'id':str})
    test_df = pd.read_csv(test_csv, dtype={'id':str})

    feature_cols = sf.get_feature_columns(test_df)
    train_X, train_y = get_X_y(train_df, 
                            feature_cols=feature_cols,
                            label_col='label')
    test_X, test_y = get_X_y(test_df,
                            feature_cols=feature_cols,
                            label_col='label')

    if valid_csv != '':
        valid_df = pd.read_csv(valid_csv)
        valid_X, valid_y = get_X_y(valid_df,
                            feature_cols=feature_cols,
                            label_col='label')
    else:
        valid_X = None
        valid_y = None

    return train_X, train_y, valid_X, valid_y, test_X, test_y

def get_preprocess_steps():
    return [('StandardScaler', sklearn.preprocessing.StandardScaler())]

def get_preprocess_pipe():
    return sklearn.pipeline.Pipeline(
        get_preprocess_steps()
    )

def make_model_pipeline(clf):
    pipeline = sklearn.pipeline.Pipeline(
        get_preprocess_steps()+[('Classifier', clf)]
    )

    return pipeline

def get_model_dir(default):
    if default == '':
        model_dir = f'model_{time.strftime("%Y%m%d-%H%M%S")}'
    else:
        model_dir = default

    os.makedirs(model_dir, exist_ok=True)

    return model_dir

def get_param_path(param_dir, default_name='model'):
    assert os.path.exists(param_dir)

    param_path = os.path.join(param_dir,
        f'{default_name}_best_params.json')

    return param_path

def get_checkpoint_path(checkpoint_dir, default_name='model'):
    assert os.path.exists(checkpoint_dir)

    param_path = os.path.join(checkpoint_dir,
        f'{default_name}_checkpoint.pth')

    return param_path

def get_trials_path(trials_dir, default_name='model'):
    assert os.path.exists(trials_dir)

    trials_path = os.path.join(trials_dir,
        f'{default_name}_trails.pkl')

    return trials_path

def save_params(params, param_path, args):
    params = params.copy()
    params['random_state'] = args.random_state
    with open(param_path, 'w') as out_file:
        s = json.dumps(params, indent=4, default=str)
        out_file.write(s)

def save_trials(trials, trials_path):
    with open(trials_path, 'wb') as out_file:
        pickle.dump(trials, out_file)

def train(clf, train_X, train_y, valid_X, valid_y, cv_parallel_jobs):
    if valid_X is None and valid_y is None:
        pred_y = sklearn.model_selection.cross_val_predict(clf,
                train_X, train_y, n_jobs=cv_parallel_jobs, cv=4
            )

        result = get_objective_result(train_y, pred_y)
    else:
        clf.fit(train_X, train_y)

        pred_y = clf.predict(valid_X)

        result = get_objective_result(valid_y, pred_y)

    return result

def get_objective_result(true_y, pred_y):
    precision_score = sklearn.metrics.precision_score(true_y, pred_y)
    recall_score = sklearn.metrics.recall_score(true_y, pred_y)
    matthews_score = sklearn.metrics.recall_score(true_y, pred_y)
    cross_entropy = sklearn.metrics.log_loss(true_y, pred_y)
    confusion_matrix = sklearn.metrics.confusion_matrix(true_y, pred_y)
    accuracy = sklearn.metrics.accuracy_score(true_y, pred_y)
    f1_score = sklearn.metrics.f1_score(true_y, pred_y)

    tn, fp, fn, tp = confusion_matrix.ravel()
    inv_precision = tn / (tn+fn)

    if matthews_score==1:
        # for some reason, predicting all 1 means matthews_score==1
        status = STATUS_FAIL
    else:
        status = STATUS_OK

    return {
        'loss': 1-f1_score,
        'status': status,
        # additional metrics
        'precision': precision_score,
        'recall': recall_score,
        'matthew': matthews_score,
        'cross_entropy': cross_entropy,
        'confusion_matrix': confusion_matrix,
        'inv_precision': inv_precision,
        'accuracy': accuracy,
        'f1_score': f1_score,
    }