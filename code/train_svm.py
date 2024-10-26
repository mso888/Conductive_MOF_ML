import train_models as tm
from hyperopt import fmin, tpe, hp, Trials
import sklearn.svm
import numpy as np

kernel_choices  = [
                {'kernel': 'rbf', 'gamma': hp.loguniform('gamma', np.log(0.01), np.log(10))},
            ]

def make_model(params):
    params = params.copy()

    params['kernel'] = kernel_choices[int(params['ktype'])]['kernel']
    del params['ktype']

    clf = sklearn.svm.SVC(**params)

    clf = tm.make_model_pipeline(clf)

    return clf

def optimize_svm(train_X, train_y, valid_X, valid_y, test_X, test_y, args):

    def svm_objective(params):
        params = params.copy()
        k_params = params['ktype']
        del params['ktype']
        params.update(k_params)
        clf = sklearn.svm.SVC(random_state=args.random_state, **params)

        clf = tm.make_model_pipeline(clf)

        result = tm.train(clf, train_X, train_y, valid_X, valid_y, args.cv_parallel_jobs)

        return result

    space = {
            'C': hp.loguniform('C', np.log(0.1), np.log(100)),
            'ktype': hp.choice('ktype', kernel_choices),}

    trials = Trials()
    best = fmin(svm_objective,
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
    param_path = tm.get_param_path(output_dir, 'svm')
    trials_path = tm.get_trials_path(output_dir, 'svm')

    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    best_params, trials = optimize_svm(train_X, train_y, valid_X, valid_y, test_X, test_y, args)

    tm.save_params(best_params, param_path, args)
    tm.save_trials(trials, trials_path)
