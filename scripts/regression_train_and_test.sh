cd ../code

date

echo "44 features"
python ./train_regression.py \
    --train_csv ../data/query_files9/random_train_df_44_nodup_regression.csv \
    --test_csv ../data/query_files9/random_test_df_44_nodup_regression.csv \
    --output_dir ../models/regression_sklearn_models_44_random/ \
    --max_evals 200 &> ../scripts/regression_train_44.out &

sleep 1
echo "70 features"
python ./train_regression.py \
    --train_csv ../data/query_files9/random_train_df_70_nodup_regression.csv \
    --test_csv ../data/query_files9/random_test_df_70_nodup_regression.csv \
    --output_dir ../models/regression_sklearn_models_70_random/ \
    --max_evals 200 &> ../scripts/regression_train_70.out &

wait

echo "44 features"
python ./test_and_save_regression.py \
    ../models/regression_sklearn_models_44_random/regression_best_params.json \
    --train_csv ../data/query_files9/random_train_df_44_nodup_regression.csv \
    --test_csv ../data/query_files9/random_test_df_44_nodup_regression.csv \
    &> ../scripts/regression_test_44.out &

sleep 1

echo "70 features"
python ./test_and_save_regression.py \
    ../models/regression_sklearn_models_70_random/regression_best_params.json \
    --train_csv ../data/query_files9/random_train_df_70_nodup_regression.csv \
    --test_csv ../data/query_files9/random_test_df_70_nodup_regression.csv \
    &> ../scripts/regression_test_70.out &

wait

date