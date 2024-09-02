dataset=TimeDrift

file_name=0709
python -u analysis/4_plot.py \
    --dataset ${dataset} \
    --model Proteus \
    --device cuda:7 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file ${file_name}