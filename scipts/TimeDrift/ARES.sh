dataset=TimeDrift

python -u exp/train.py \
        --dataset ${dataset} \
        --model ARES \
        --device cuda:1 \
        --train_file train_0313 \
        --valid_file valid_0313 \
        --feature DIR \
        --seq_len 10000 \
        --train_epochs 30 \
        --batch_size 64 \
        --learning_rate 0.0014 \
        --optimizer Adam \
        --eval_metrics Precision Recall F1-score \
        --save_metric F1-score \
        --save_name 0313

for file_name in test_0313 0327 0410 0709
do
    python -u exp/test.py \
        --dataset ${dataset} \
        --model ARES \
        --device cuda:0 \
        --valid_file ${file_name} \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name 0313 \
        --result_file ${file_name}
done