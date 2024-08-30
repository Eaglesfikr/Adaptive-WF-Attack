dataset=AWF

python -u exp/0_fusion.py \
        --dataset ${dataset} \
        --model Fusion \
        --device cuda:2 \
        --valid_file inferior \
        --test_file inferior \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name superior \
        --result_file superior2inferior

python -u exp/0_fusion.py \
        --dataset ${dataset} \
        --model Fusion \
        --device cuda:2 \
        --valid_file superior \
        --test_file superior \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name inferior \
        --result_file inferior2superior