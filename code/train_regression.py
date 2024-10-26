import sklearn.metrics as skmetrics
import numpy as np
import argparse
import os
import copy
import json
import pandas as pd
import random
import train_models as tm
import simple_featurizer as sf
import test_and_save_model as tasm
import ensemble_test as et

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from hyperopt import STATUS_OK, STATUS_FAIL, fmin, tpe, hp, Trials


class RegressionModel:
    def __init__(self,  lr=0, momentum=0, weight_decay=0,
                 layers=[50,50,50], feature_cols=None, 
                 preprocessing_pipe=None, early_stopping=True, max_epochs=300):
        self.prep_pipe = preprocessing_pipe
        self.feature_cols = feature_cols
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.layers = layers
        self.early_stopping = early_stopping
        self.max_epochs = max_epochs

        if self.feature_cols is None:
            # This means we need to load parameters
            pass
        else:
            self._build()

    def _build(self):
        self.num_features = len(self.feature_cols)
        tuples = list(zip(self.layers, self.layers[1:]))
        seq = [nn.Linear(self.num_features, self.layers[0])]
        dropout_p = 0.30
        for t0,t1 in tuples:
            seq.append(nn.ReLU())
            seq.append(nn.Dropout(p=dropout_p))
            seq.append(nn.Linear(t0, t1))
        seq.append(nn.ReLU())
        seq.append(nn.Dropout(p=dropout_p))
        seq.append(nn.Linear(self.layers[-1], 1))
        self.net = nn.Sequential(*seq)
        print(self.net)

    def train(self, X_train, y_train, X_valid, y_valid):
        self.prep_pipe.fit(X_train)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)

        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(self.prep_pipe.transform(X_train)), 
                torch.Tensor(y_train.reshape((-1,1)))), 
            batch_size=1000,
            shuffle=True)

        validloader = DataLoader(
            TensorDataset(
                torch.Tensor(self.prep_pipe.transform(X_valid)))
            )

        self.best_loss = 1e20
        self.best_epoch = 0
        self.best_net = None
        patience = 10
        interval = 10

        for epoch in range(self.max_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            count = 0
            self.net.train(True)
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0], data[1]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                count += 1

            avg_batch_loss = running_loss/count
            print('[%d] running loss: %.3f' %
                    (epoch + 1, avg_batch_loss))

            if (epoch%interval==0) and (epoch!=0):
                pred_valid = np.array(self._predict(validloader))

                vr2 = skmetrics.r2_score(y_valid, pred_valid)
                valid_loss = skmetrics.mean_squared_error(y_valid, pred_valid)
                print("valid loss %0.2f"%(valid_loss))
                print("valid r2_score %0.2f"%(vr2))

                if valid_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = valid_loss
                    print(f'\tNew best loss: {self.best_loss}', flush=True)
                    self.best_net = copy.deepcopy(self.net.state_dict())

                if epoch > (self.best_epoch+patience) and self.early_stopping:
                    # we've waited long enough
                    break

        # once we have finished training, ratchet back to the best weights
        self.restore_best_net()

    def _predict(self, loader):
        self.net.eval()
        with torch.no_grad():
            all_preds = []
            for data in loader:
                features = data[0]
                outputs = self.net(features)
                all_preds.append(outputs.cpu().numpy()[0][0])

        return all_preds

    def predict(self, X):
        loader = DataLoader(
            TensorDataset(
                torch.Tensor(self.prep_pipe.transform(X)))
        )
        return self._predict(loader)

    def restore_best_net(self):
        self.net.load_state_dict(self.best_net)

    def save(self, location):
        torch.save(self.best_net, location)

        pipe_path = os.path.splitext(location)[0]+'_pipe.pkl'
        tasm.save_model(self.prep_pipe, pipe_path)

        feature_path = os.path.splitext(location)[0]+'_features.json'
        feature_json = {'features':self.feature_cols}
        with open(feature_path, 'w') as out_file:
            s = json.dumps(feature_json, indent=4, default=str)
            out_file.write(s)

    def load(self, location):
        feature_path = os.path.splitext(location)[0]+'_features.json'
        with open(feature_path, 'r') as in_file:
            feature_json = json.load(in_file)

        self.feature_cols = feature_json['features']

        self._build()

        self.net.load_state_dict(torch.load(location))

        pipe_path = os.path.splitext(location)[0]+'_pipe.pkl'
        self.prep_pipe = et.load_model(pipe_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', help='Path to training csv file.')
    parser.add_argument('--test_csv', help='Path to testing csv file.')

    parser.add_argument('--output_dir', default='', help='Where trained models are saved.')
    parser.add_argument('--max_evals', type=int, default=20, help='Number of hyperopt evals')
    parser.add_argument('--random_state', type=int, default=0, help='Random seed')

    return parser.parse_args()

def make_model(feature_cols, preprocess_pipe, params, early_stopping=True, max_epochs=300):
    layers = []
    layer_count = 1
    while True:
        layer_str = f'layer{layer_count}'
        if layer_str in params:
            layers.append(int(params[layer_str]))
            layer_count += 1
        else:
            break

    model = RegressionModel(
        feature_cols=feature_cols,
        preprocessing_pipe=preprocess_pipe,
        lr=params['lr'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay'],
        layers=layers,
        early_stopping=early_stopping,
        max_epochs=max_epochs)

    return model

def evaluate_model(model, train_X, train_y, valid_X, valid_y, test_X, test_y):
    train_pred = model.predict(train_X)
    train_r2 = skmetrics.r2_score(train_y, np.array(train_pred))
    print(f"Train r2_score {train_r2:0.2f}")

    valid_pred = model.predict(valid_X)
    valid_r2 = skmetrics.r2_score(valid_y, np.array(valid_pred))
    print(f"Valid r2_score {valid_r2:0.2f}")

    test_pred = model.predict(test_X)
    test_r2 = skmetrics.r2_score(test_y, np.array(test_pred))
    print(f"Test r2_score {test_r2:0.2f}")

    return {
            'loss': -1*valid_r2,
            'train_r2': train_r2,
            'valid_r2': valid_r2,
            'test_r2': test_r2,
        }

def optimize_NN(train_X, train_y, valid_X, valid_y, test_X, test_y, args, feature_cols):
    def NN_objective(params):
        try:
            model = make_model(
                feature_cols=feature_cols,
                preprocess_pipe=tm.get_preprocess_pipe(),
                params=params)

            model.train(X_train=train_X, y_train=train_y,
                        X_valid=valid_X, y_valid=valid_y)

            result = evaluate_model(model, 
                train_X, train_y, valid_X, valid_y, test_X, test_y)

            result['status'] = STATUS_OK

        except Exception as e:
            print(e)
            print("This trial failed, returning STATUS_FAIL...")
            return {
                'loss': 1e20,
                'train_r2': 0,
                'valid_r2': 0,
                'test_r2': 0,
                'status': STATUS_FAIL
            }

        return result

    space = {
            'layer1': hp.uniformint('layer1', 10, 100),
            'layer2': hp.uniformint('layer2', 10, 100),
            'layer3': hp.uniformint('layer3', 10, 100),
            'layer4': hp.uniformint('layer4', 10, 100),
            'layer5': hp.uniformint('layer5', 5, 50),
            'lr': hp.loguniform('lr', np.log(.00001), np.log(.01)),
            'momentum': hp.uniform('momentum', 0, 2),
            'weight_decay': hp.uniform('weight_decay', 0, 2),
            }

    trials = Trials()
    print(f"Hyperopt for {args.max_evals} trials")
    best = fmin(NN_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_evals,
                trials=trials,
                rstate=np.random.default_rng(seed=args.random_state))

    print(best)
    return best, trials

def load_data(train_csv, test_csv, feature_cols=None):
    # load data
    train_X, train_y, test_X, test_y, feature_cols = load_regression_data(
        train_csv,
        test_csv,
        feature_cols
    )

    # shuffle train_X and train_y then cut out 20% for validation
    num_training = len(train_X)
    print(f'Total training samples {num_training}')
    shuffle_ind = list(range(num_training))
    random.shuffle(shuffle_ind)
    shuffle_ind = np.array(shuffle_ind)
    cutoff = int(num_training*0.8)
    train_ind = shuffle_ind[:cutoff]
    valid_ind = shuffle_ind[cutoff:]

    valid_X = train_X[valid_ind]
    valid_y = train_y[valid_ind]

    train_X = train_X[train_ind]
    train_y = train_y[train_ind]
    print(f'\tTraining samples {len(train_X)}')
    print(f'\tValidation samples {len(valid_X)}')

    return train_X, train_y, valid_X, valid_y, test_X, test_y, feature_cols

def load_regression_data(train_csv, test_csv, feature_cols=None):
    train_df = pd.read_csv(train_csv, dtype={'id':str})
    test_df = pd.read_csv(test_csv, dtype={'id':str})

    if feature_cols is None:
        feature_cols = sf.get_feature_columns(test_df)

    train_X, train_y = get_X_r(train_df, 
                            feature_cols=feature_cols,
                            label_col='_oqmd_band_gap')
    test_X, test_y = get_X_r(test_df,
                            feature_cols=feature_cols,
                            label_col='_oqmd_band_gap')

    return train_X, train_y, test_X, test_y, feature_cols

def get_X_r(df, feature_cols, label_col):
    df = df.dropna(subset=[label_col])
    X = df[feature_cols].values
    y = df[label_col].values

    assert X.shape[1] == 44 or X.shape[1] == 70

    return X, y

def set_random_state(random_state):
    # set random seeds
    random.seed(random_state)
    torch.manual_seed(random_state)

if __name__ == '__main__':
    args = parse_args()
    
    # set random seeds
    set_random_state(args.random_state)

    # prepare output paths
    output_dir = tm.get_model_dir(args.output_dir)
    param_path = tm.get_param_path(output_dir, 'regression')
    trials_path = tm.get_trials_path(output_dir, 'regression')
    checkpoint_path = tm.get_checkpoint_path(output_dir, 'regression')

    train_X, train_y, valid_X, valid_y, test_X, test_y, feature_cols = load_data(args.train_csv, args.test_csv)

    best_params, trials = optimize_NN(train_X, train_y, valid_X, valid_y, test_X, test_y, args, feature_cols)

    tm.save_params(best_params, param_path, args)
    tm.save_trials(trials, trials_path)

