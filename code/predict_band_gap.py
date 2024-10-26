import argparse
import json
import pandas as pd
import sklearn.metrics as skmetrics
import scipy.stats as scistats
import train_regression as tr
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('params_file', help='JSON file with model parameters.')
    parser.add_argument('checkpoint_file', help='Checkpoint file with model parameters.')
    parser.add_argument('--in_csv', help='A csv file containing features')
    parser.add_argument('--out_csv', default='', help='An output file for predictions')
    parser.add_argument('--label_col', default='', help='Optional: Column containing labels.')
    parser.add_argument('--save_cols', nargs='+', default=[], help='Columns to copy over to keep in out_csv')
    parser.add_argument('--id_col', default='id', help='Column with ids.')
    parser.add_argument('--plot', default='', type=str, help='Save a scatter plot of predictions. Only if label_col is set.')

    return parser.parse_args()

def scatter_plot(y, preds, plot_png):

    plot = sns.regplot(x = y, y = preds,
            scatter_kws = {"color": "black", "alpha": 0.5},
            line_kws = {"color": "red"})
    plot.get_figure().savefig(plot_png)


if __name__ == '__main__':
    args = parse_args()

    with open(args.params_file, 'r') as in_file:
        params = json.load(in_file)

    tr.set_random_state(params['random_state'])

    # build model
    model = tr.make_model(
        feature_cols=None,
        preprocess_pipe=None,
        params=params)

    # load state
    model.load(args.checkpoint_file)

    df = pd.read_csv(args.in_csv, dtype={args.id_col:str})
    if len(args.label_col)>0:
        X, y = tr.get_X_r(df, model.feature_cols, args.label_col)
        preds = model.predict(X)
        r2 = skmetrics.r2_score(y, preds)

        print(f"r2 {r2:0.2f}")

        spearman, pvalue = scistats.spearmanr(y, preds)
        print(f"Spearman: {spearman:0.2f}, pvalue: {pvalue:0.2f}")
    
        if len(args.plot)>0:
            scatter_plot(y, preds, args.plot)

    else:
        X = df[model.feature_cols].values
        preds = model.predict(X)

    # add prediction column to df file
    if len(args.out_csv)>0:
        out_cols = [args.id_col]
        if len(args.save_cols)>0:
            out_cols = out_cols + args.save_cols
        out_df = df[out_cols].copy()
        out_df['model_pred'] = preds

        out_df.to_csv(args.out_csv, index=False)