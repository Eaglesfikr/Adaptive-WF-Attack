dataset=TimeDrift

file_name=test_0313
python -u exp/4_coteaching.py \
    --dataset ${dataset} \
    --model Coteaching \
    --device cuda:6 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file 0313

file_name=0327
python -u exp/4_coteaching.py \
    --dataset ${dataset} \
    --model Coteaching \
    --device cuda:6 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file ${file_name}

file_name=0410
python -u exp/4_coteaching.py \
    --dataset ${dataset} \
    --model Coteaching \
    --device cuda:6 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file ${file_name}

file_name=0709
python -u exp/4_coteaching.py \
    --dataset ${dataset} \
    --model Coteaching \
    --device cuda:6 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file ${file_name}