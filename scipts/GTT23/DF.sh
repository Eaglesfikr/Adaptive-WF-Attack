dataset=GTT23

python -u exp/train.py \
        --dataset ${dataset} \
        --model DF \
        --device cuda:3 \
        --train_file week_0 \
        --valid_file week_1 \
        --feature DIR \
        --seq_len 5000 \
        --train_epochs 30 \
        --batch_size 128 \
        --learning_rate 2e-3 \
        --optimizer Adam \
        --eval_metrics Precision Recall F1-score \
        --save_metric F1-score \
        --save_name week_0

for week in {1..12}
do
        python -u exp/test.py \
                --dataset ${dataset} \
                --model DF \
                --device cuda:3 \
                --valid_file week_${week} \
                --test_file week_${week} \
                --feature DIR \
                --seq_len 5000 \
                --batch_size 256 \
                --eval_metrics Precision Recall F1-score \
                --load_name week_0 \
                --result_file week_${week}
done