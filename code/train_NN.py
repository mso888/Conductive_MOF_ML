import train_models as tm
from hyperopt import fmin, tpe, hp, Trials
import sklearn.neural_network
import numpy as np

activation_choices = ['logistic', 'tanh', 'relu']

def make_model(params):
    ''' Use saved params to make a model
    This makes a model from the saved params. It is different
    than making a model from params from the hyperopt sampler
    '''
    hidden_layer_sizes = [params['layer1'], params['layer2'], params['layer3'], params['layer4']]
    hidden_layer_sizes = [int(ls) for ls in hidden_layer_sizes]

    clf = sklearn.neural_network.MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation_choices[int(params['activation'])],
        learning_rate_init=float(params['learning_rate_init']),
        alpha=float(params['alpha']),
        random_state=params['random_state']
    )

    clf = tm.make_model_pipeline(clf)

    return clf

def optimize_NN(train_X, train_y, valid_X, valid_y, test_X, test_y, args):

    def NN_objective(params):
        clf = sklearn.neural_network.MLPClassifier(
                hidden_layer_sizes=[params['layer1'], params['layer2'], params['layer3'], params['layer4']],
                activation=params['activation'],
                learning_rate_init=params['learning_rate_init'],
                alpha=params['alpha'],
                random_state=args.random_state
            )

        clf = tm.make_model_pipeline(clf)

        result = tm.train(clf, train_X, train_y, valid_X, valid_y, args.cv_parallel_jobs)

        return result

    space = {
            'layer1': hp.uniformint('layer1', 10, 200),
            'layer2': hp.uniformint('layer2', 10, 200),
            'layer3': hp.uniformint('layer3', 10, 200),
            'layer4': hp.uniformint('layer4', 10, 200),
            'activation': hp.choice('activation', activation_choices),
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(.00001), np.log(.001)),
            'alpha': hp.loguniform('alpha', np.log(0.001), np.log(1))
            }

    trials = Trials()
    best = fmin(NN_objective,
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
    param_path = tm.get_param_path(output_dir, 'NN')
    trials_path = tm.get_trials_path(output_dir, 'NN')

    train_X, train_y, valid_X, valid_y, test_X, test_y = tm.load_data(
        args.train_csv,
        args.valid_csv,
        args.test_csv,
    )

    best_params, trials = optimize_NN(train_X, train_y, valid_X, valid_y, test_X, test_y, args)

    tm.save_params(best_params, param_path, args)
    tm.save_trials(trials, trials_path)
