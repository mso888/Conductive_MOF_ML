import train_models as tm
import argparse
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--models', nargs='+', default='*test_model.pkl', 
        help='Wildcard path of models to include in this ensemble. E.g. *test_model.pkl')
    
    parser.add_argument('--test_csv', help='Test csv file')
    parser.add_argument('--vote_method', choices=['gt_marjority', 'gteq_majority', 'consensus'], 
                        default='gt_majority',
                        help='Does the number of conductive votes need to be greater than or'
                        ' greater than or equal to the number of non-conductive votes to win.')

    return parser.parse_args()

def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        clf = pickle.load(f)

    return clf

class EnsembleModel:
    def __init__(self, models, vote_method='gteq_majority'):
        self.models = models

        vote_method_choices = ['gt_majority', 'gteq_majority', 'consensus']
        if vote_method not in vote_method_choices:
            raise ValueError(f'{vote_method} not in {vote_method_choices}')

        self.vote_method = vote_method

    def vote(self, votes):
        ''' Vote on final prediction

        1: conducting
        0: non-conducting

        '''
        unique, count = np.unique(votes, return_counts=True)

        if len(unique) == 1:
            return unique[0]

        # consensus voting says that all models must agree
        # that a compound conducts. Otherwise it is non-conducting
        if self.vote_method == 'consensus':
            return unique[0]

        assert len(unique) < 3, "EnsembleModel only supports binary classification"

        if count[0] == count[1]:
            if self.vote_method == 'gteq_majority':
                return unique[1]
            else:
                return unique[0]
        else:
            if count[0] < count[1]:
                return unique[1]
            else:
                return unique[0]

    def predict(self, X):
        preds = self.individual_votes(X)

        assert (preds.shape[0]==len(X)) and (preds.shape[1]==len(self.models))

        consensus = [self.vote(preds[i]) for i in range(len(preds))]

        return consensus

    def individual_votes(self, X):
        preds = []
        for m in self.models:
            pred = m.predict(X)
            preds.append(pred)

        preds = np.stack(preds, axis=-1)

        return preds

def test_EnsembleModel():
    em = EnsembleModel([None, None, None, None], vote_method='gteq_majority') 
    assert em.vote([0,0,0,0]) == 0
    assert em.vote([0,1,1,1]) == 1
    assert em.vote([0,0,1,1]) == 1
    assert em.vote([0,0,0,1]) == 0
    assert em.vote([1,1,1,1]) == 1

    em = EnsembleModel([None, None, None, None], vote_method='gt_majority') 
    assert em.vote([0,0,0,0]) == 0
    assert em.vote([0,1,1,1]) == 1
    assert em.vote([0,0,1,1]) == 0
    assert em.vote([0,0,0,1]) == 0
    assert em.vote([1,1,1,1]) == 1

    em = EnsembleModel([None, None, None, None], vote_method='consensus') 
    assert em.vote([0,0,0,0]) == 0
    assert em.vote([0,1,1,1]) == 0
    assert em.vote([0,0,1,1]) == 0
    assert em.vote([0,0,0,1]) == 0
    assert em.vote([1,1,1,1]) == 1

if __name__ == '__main__':
    args = parse_args()

    models = [load_model(f) for f in args.models]

    em = EnsembleModel(models, args.vote_method)

    X, y = tm.load_single_csv(args.test_csv)

    preds = em.predict(X)

    results = tm.get_objective_result(y, preds)
    print(results)