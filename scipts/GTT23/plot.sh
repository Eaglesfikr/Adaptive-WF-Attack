dataset=GTT23
week=12

for temp in {1..10}
do
    python -u analysis/4_plot.py \
        --dataset ${dataset} \
        --model Proteus \
        --device cuda:5 \
        --origin_file week_0 \
        --valid_file week_${week} \
        --test_file week_${week} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name week_0 \
        --temp ${temp} \
        --result_file week_${week}
done

# for week in {1..12}
# do
#     python -u analysis/4_plot.py \
#         --dataset ${dataset} \
#         --model Proteus \
#         --device cuda:5 \
#         --origin_file week_0 \
#         --valid_file week_${week} \
#         --test_file week_${week} \
#         --feature DIR \
#         --seq_len 10000 \
#         --batch_size 256 \
#         --eval_metrics Precision Recall F1-score \
#         --load_name week_0 \
#         --result_file week_${week}
# done