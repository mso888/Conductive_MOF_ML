cd ../code

date

echo "Make predictions for test set to make sure load is working correctly"
echo "Compare with test scores in scripts/regression_test_*.out"
python ./predict_band_gap.py \
    ../models/regression_sklearn_models_44_random/regression_best_params.json \
    ../models/regression_sklearn_models_44_random/regression_best_params_test_model.pt \
    --in_csv ../data/query_files9/random_test_df_44_nodup_regression.csv \
    --label_col _oqmd_band_gap \
    --plot ../data/scatterplot_random_test_df_44_nodup_regression.png \
    &> ../scripts/regression_test_load_44.out &

python ./predict_band_gap.py \
    ../models/regression_sklearn_models_70_random/regression_best_params.json \
    ../models/regression_sklearn_models_70_random/regression_best_params_test_model.pt \
    --in_csv ../data/query_files9/random_test_df_70_nodup_regression.csv \
    --label_col _oqmd_band_gap \
    --plot ../data/scatterplot_random_test_df_70_nodup_regression.png \
    &> ../scripts/regression_test_load_70.out &

wait

echo "Make predictions using normally trained model"
python ./predict_band_gap.py \
    ../models/regression_sklearn_models_70_random/regression_best_params.json \
    ../models/regression_sklearn_models_70_random/regression_best_params_test_model.pt \
    --in_csv ../data/screening_results.csv \
    --out_csv ../data/screening_results_plus_band_gap_70.csv \
    --save_cols chemical_formula DOI REFCODE &

python ./predict_band_gap.py \
    ../models/regression_sklearn_models_44_random/regression_best_params.json \
    ../models/regression_sklearn_models_44_random/regression_best_params_test_model.pt \
    --in_csv ../data/screening_results.csv \
    --out_csv ../data/screening_results_plus_band_gap_44.csv \
    --save_cols chemical_formula DOI REFCODE &

wait

echo "Make predictions using production model"
python ./predict_band_gap.py \
    ../models/regression_sklearn_models_70_random/regression_best_params.json \
    ../models/regression_sklearn_models_70_random/regression_best_params_production_model.pt \
    --in_csv ../data/screening_results.csv \
    --out_csv ../data/screening_results_plus_band_gap_70_prod.csv \
    --save_cols chemical_formula DOI REFCODE &

python ./predict_band_gap.py \
    ../models/regression_sklearn_models_44_random/regression_best_params.json \
    ../models/regression_sklearn_models_44_random/regression_best_params_production_model.pt \
    --in_csv ../data/screening_results.csv \
    --out_csv ../data/screening_results_plus_band_gap_44_prod.csv \
    --save_cols chemical_formula DOI REFCODE &

wait

date