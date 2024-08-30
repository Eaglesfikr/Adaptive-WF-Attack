dataset=GTT23

for week in {1..12}
do
        python -u exp/0_fusion.py \
        --dataset ${dataset} \
        --model Fusion \
        --device cuda:3 \
        --valid_file week_${week} \
        --test_file week_${week} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name week_0 \
        --result_file week_${week}
done