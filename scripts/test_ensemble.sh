cd ../code

echo "44 features"
python ./ensemble_test.py --models ../models/classification_sklearn_models_44_cv/*_test_model.pkl --test_csv ../data/query_files9/random_test_df_44.csv

echo "70 features"
python ./ensemble_test.py --models ../models/classification_sklearn_models_70_cv/*_test_model.pkl --test_csv ../data/query_files9/random_test_df_70.csv
