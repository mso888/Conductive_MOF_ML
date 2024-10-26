# Introduction

# Environment

```
qmpy-rester
pandas
tqdm
scipy
scikit-learn
seaborn
```

# Gathering and creating data.

**Re-running this code is not necessary as the output is already 
in the repository. Data was gathered from OQMD in 2020 and we are unsure 
if re-running it now will produce the same results.**

These scripts were run in order.
 1. `download.py` Goes to OQMD and downloads all the data. 
 The results included here are from 2020. Re-running this script may produce
 different results. Produces one json file per query.
 2. `join_json.py ../data/query_files9/` Combines all json files from 
 `download.py` and divides them into compounds with band gap greater than or
 less than 2.2. Creates two csv files, one with compounds greater than the
 2.2 threshold and one with less than 2.2 threshold.
 3. `scripts/generate_random_split.sh` or `scripts/generate_similarity_split.sh` 
 4. `scripts/generate_regression_data.sh` Generates regression data
 by removing all metals. Metal is defined as compounds with band gap <0. It
 also removes outlier band gaps >15.

# Classification steps

Run these scripts in order.
 1. `scripts/random_train_and_test_models.sh` This trains, evaluates, and screens
 using the random split. Screening results can be found in `data/screening_results.csv`
    - `scripts/similarity_train_and_test_models.sh` This trains and evaluates
    models using the similarity split. Since the production model would be the 
    same as the random split, this script does not perform screening.
 2. Call `scripts/screen.sh` to screen CoREMOF data.

# Regression steps
1. We had to remove metals from the regression models because they all
had bandgap zero and such a discontinutiy made regression difficult.
Plus, there should be no metals in CoREMOF. 
This is generated using `scripts/generate_regression_data.py`.
It looks for datafiles from Gathering and creating data step 3
and removes metals, things with bandgap < 0. Some compounds also
did not have a bandgap. Those are moved too.
2. `scripts/regression_train_and_test.sh` This trains and evaluates the
regression models using randomly split data. 
3. `scripts/regression_predict_band_gap.sh` This looks for `data/screening_results.csv` from
the classification scripts and creates `data/screening_results_plus_band_gap_*.csv`
with an additional predicted bandgap column.
    - There are 3 variations on the file: `44_prod`, `44`, `70_prod`, and `70`.
    Files that end with `_prod` are generated using production models.
    Files without `_prod` are generated using the model that produced metrics
    reported in the paper.
    - This also creates the pngs used in the supplimental section of the paper.