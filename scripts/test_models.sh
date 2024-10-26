cd ../code

echo "44 features"
python ./test_and_save_model.py rf ../models/classification_sklearn_models_44_cv/rf_best_params.json --train_csv ../data/query_files9/random_train_df_44_nodup.csv --test_csv ../data/query_files9/random_test_df_44_nodup.csv
python ./test_and_save_model.py svm ../models/classification_sklearn_models_44_cv/svm_best_params.json --train_csv ../data/query_files9/random_train_df_44_nodup.csv --test_csv ../data/query_files9/random_test_df_44_nodup.csv
python ./test_and_save_model.py NN ../models/classification_sklearn_models_44_cv/NN_best_params.json --train_csv ../data/query_files9/random_train_df_44_nodup.csv --test_csv ../data/query_files9/random_test_df_44_nodup.csv
python ./test_and_save_model.py lr ../models/classification_sklearn_models_44_cv/lr_best_params.json --train_csv ../data/query_files9/random_train_df_44_nodup.csv --test_csv ../data/query_files9/random_test_df_44_nodup.csv

echo "70 features"
python ./test_and_save_model.py rf ../models/classification_sklearn_models_70_cv/rf_best_params.json --train_csv ../data/query_files9/random_train_df_70_nodup.csv --test_csv ../data/query_files9/random_test_df_70_nodup.csv
python ./test_and_save_model.py svm ../models/classification_sklearn_models_70_cv/svm_best_params.json --train_csv ../data/query_files9/random_train_df_70_nodup.csv --test_csv ../data/query_files9/random_test_df_70_nodup.csv
python ./test_and_save_model.py NN ../models/classification_sklearn_models_70_cv/NN_best_params.json --train_csv ../data/query_files9/random_train_df_70_nodup.csv --test_csv ../data/query_files9/random_test_df_70_nodup.csv
python ./test_and_save_model.py lr ../models/classification_sklearn_models_70_cv/lr_best_params.json --train_csv ../data/query_files9/random_train_df_70_nodup.csv --test_csv ../data/query_files9/random_test_df_70_nodup.csv