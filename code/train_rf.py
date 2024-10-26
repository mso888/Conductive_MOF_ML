import train_models as tm
from hyperopt import fmin, tpe, hp, Trials
import sklearn.ensemble
import numpy as np

criterion_chioces = ['gini', 'log_loss', 'entropy']
max_features_choices = ['sqrt', 'log2', None]

def make_model(params):
    clf = sklearn.ensemble.RandomForestClassifier(
        n_estimators= int(params['n_estimators']),
        criterion= criterion_chioces[int(params['criterion'])],
        max_depth= int(params['max_depth']),
        max_features= max_features_choices[int(params['max_features'])],
        min_samples_split= int(params['min_samples_split']),
        random_state=params['random_state']
    )

    clf = tm.make_model_pipeline(clf)

    return clf

def optimize_rf(train_X, train_y, valid_X, valid_y, test_X, test_y, args):

    def rf_objective(params):

        clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            max_features=params['max_features'],
            random_state=args.random_state
        )

        clf = tm.make_model_pipeline(clf)

        result = tm.train(clf, train_X, train_y, valid_X, valid_y, args.cv_parallel_jobs)

        return result

    space = {
            'n_estimators': hp.uniformint('n_estimators', 50, 500),
            'criterion': hp.choice('criterion', criterion_chioces),
            'max_depth': hp.uniformint('max_depth', 5, 20),
            'min_samples_split': hp.uniformint('min_samples_split', 5, 40),
            'max_features': hp.choice('max_features', max_features_choices),
            }

    trials = Trials()
    best = fmin(rf_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(seed=args.random_state))

    print(best)

    return best, trials

if __name__ == '__main__':
    args = tm.parse_args()
    output_dir = tm.get_model_dir(args.output_dir)
    param_path = tm.get_param_path(output_dir, 'rf')
    trials_path = tm.get_trials_path(output_dir, 'rf')

    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    best_params, trials = optimize_rf(train_X, train_y, valid_X, valid_y, test_X, test_y, args)

    tm.save_params(best_params, param_path, args)
    tm.save_trials(trials, trials_path)
