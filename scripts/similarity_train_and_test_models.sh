cd ../code

date

echo "44 features"
python ./train_rf.py \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --output_dir ../models/classification_sklearn_models_44_similarity/ \
    --max_evals 20 &> ../scripts/train_rf_similarity_44.out &
sleep 1
python ./train_NN.py \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --output_dir ../models/classification_sklearn_models_44_similarity/ \
    --max_evals 20 &> ../scripts/train_NN_similarity_44.out &
sleep 1
python ./train_lr.py \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --output_dir ../models/classification_sklearn_models_44_similarity/ \
    --max_evals 20 &> ../scripts/train_lr_similarity_44.out &
sleep 1
python ./train_svm.py \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv \
    --output_dir ../models/classification_sklearn_models_44_similarity/ \
    --max_evals 20 &> ../scripts/train_svm_similarity_44.out &

wait

echo "70 features"
python ./train_rf.py \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --output_dir ../models/classification_sklearn_models_70_similarity/ \
    --max_evals 20 &> ../scripts/train_rf_similarity_70.out &
sleep 1
python ./train_NN.py \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --output_dir ../models/classification_sklearn_models_70_similarity/ \
    --max_evals 20 &> ../scripts/train_NN_similarity_70.out &
sleep 1
python ./train_lr.py \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --output_dir ../models/classification_sklearn_models_70_similarity/ \
    --max_evals 20 &> ../scripts/train_lr_similarity_70.out &
sleep 1
python ./train_svm.py \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --valid_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv \
    --output_dir ../models/classification_sklearn_models_70_similarity/ \
    --max_evals 20 &> ../scripts/train_svm_similarity_70.out &

wait

echo "44 features"
python ./test_and_save_model.py rf \
    ../models/classification_sklearn_models_44_similarity/rf_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv
python ./test_and_save_model.py NN \
    ../models/classification_sklearn_models_44_similarity/NN_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv
python ./test_and_save_model.py lr \
    ../models/classification_sklearn_models_44_similarity/lr_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv
python ./test_and_save_model.py svm \
    ../models/classification_sklearn_models_44_similarity/svm_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_44_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_44_nodup.csv

echo "70 features"
python ./test_and_save_model.py rf \
    ../models/classification_sklearn_models_70_similarity/rf_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv
python ./test_and_save_model.py NN \
    ../models/classification_sklearn_models_70_similarity/NN_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv
python ./test_and_save_model.py lr \
    ../models/classification_sklearn_models_70_similarity/lr_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv
python ./test_and_save_model.py svm \
    ../models/classification_sklearn_models_70_similarity/svm_best_params.json \
    --train_csv ../data/query_files9/similarity_train_df_70_nodup.csv \
    --test_csv ../data/query_files9/similarity_test_df_70_nodup.csv

date