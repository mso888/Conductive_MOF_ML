 cd ../code

 echo "44 features"
 python ./train_NN.py --train_csv ../data/query_files9/random_train_df_44_nodup.csv --test_csv ../data/query_files9/random_test_df_44_nodup.csv --output_dir ../models/classification_sklearn_models_44_cv/  --max_evals 20 --cv_parallel_jobs 4

 echo "70 features"
 python ./train_NN.py --train_csv ../data/query_files9/random_train_df_70_nodup.csv --test_csv ../data/query_files9/random_test_df_70_nodup.csv --output_dir ../models/classification_sklearn_models_70_cv/  --max_evals 20 --cv_parallel_jobs 4

echo "44 features"
python ./test_and_save_model.py NN ../models/classification_sklearn_models_44_cv/NN_best_params.json --train_csv ../data/query_files9/random_train_df_44.csv --test_csv ../data/query_files9/random_test_df_44.csv

echo "70 features"
python ./test_and_save_model.py NN ../models/classification_sklearn_models_70_cv/NN_best_params.json --train_csv ../data/query_files9/random_train_df_70.csv --test_csv ../data/query_files9/random_test_df_70.csv