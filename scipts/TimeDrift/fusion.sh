dataset=TimeDrift

for file_name in test_0313 0327 0410 0709
do
    python -u exp/0_fusion.py \
        --dataset ${dataset} \
        --model Fusion \
        --device cuda:2 \
        --valid_file ${file_name} \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name 0313 \
        --result_file ${file_name}
done