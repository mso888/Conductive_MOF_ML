cd ../code

date

echo "44 features random"
python test_models.py \
--model_pkls  ../models/classification_sklearn_models_44_cv/*_test_model.pkl \
--test_csv ../data/query_files9/random_test_df_44_nodup.csv

echo "70 features random"
python test_models.py \
--model_pkls  ../models/classification_sklearn_models_70_cv/*_test_model.pkl \
--test_csv ../data/query_files9/random_test_df_70_nodup.csv

echo "44 features similarity"
python test_models.py \
--model_pkls  ../models/classification_sklearn_models_44_similarity/*_test_model.pkl \
--test_csv ../data/query_files9/similarity_test_df_44_nodup.csv

echo "70 features similarity"
python test_models.py \
--model_pkls  ../models/classification_sklearn_models_70_similarity/*_test_model.pkl \
--test_csv ../data/query_files9/similarity_test_df_70_nodup.csv

date