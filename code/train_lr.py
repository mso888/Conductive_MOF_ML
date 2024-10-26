import train_models as tm
from hyperopt import fmin, tpe, hp, Trials
import sklearn.linear_model
import numpy as np

solver_choice = ['newton-cholesky']
penalty_choice = ['l2']

def make_model(params):
    clf = sklearn.linear_model.LogisticRegression(
        random_state=params['random_state'],
        solver=solver_choice[int(params['solver'])],
        penalty=penalty_choice[int(params['penalty'])],
        C=float(params['C']),
        max_iter=200)
    
    clf = tm.make_model_pipeline(clf)

    return clf

def optimize_lr(train_X, train_y, valid_X, valid_y, test_X, test_y, args):

    def lr_objective(params):

        clf = sklearn.linear_model.LogisticRegression(
            random_state=args.random_state,
            solver=params['solver'],
            penalty=params['penalty'],
            C=params['C'],
            max_iter=200)

        clf = tm.make_model_pipeline(clf)

        result = tm.train(clf, train_X, train_y, valid_X, valid_y, args.cv_parallel_jobs)

        return result

    space = {
            'C': hp.loguniform('C', np.log(0.1), np.log(15)),
            'solver': hp.choice('solver', solver_choice),
            'penalty': hp.choice('penalty', penalty_choice),
            }

    trials = Trials()
    best = fmin(lr_objective,
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
    param_path = tm.get_param_path(output_dir, 'lr')
    trials_path = tm.get_trials_path(output_dir, 'lr')

    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    best_params, trials = optimize_lr(train_X, train_y, valid_X, valid_y, test_X, test_y, args)

    tm.save_params(best_params, param_path, args)
    tm.save_trials(trials, trials_path)
