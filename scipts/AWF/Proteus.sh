dataset=AWF

python -u exp/1_proteus.py \
        --dataset ${dataset} \
        --model Proteus \
        --device cuda:3 \
        --origin_file superior \
        --valid_file inferior \
        --test_file inferior \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name superior \
        --result_file superior2inferior

python -u exp/1_proteus.py \
        --dataset ${dataset} \
        --model Proteus \
        --device cuda:3 \
        --origin_file inferior \
        --valid_file superior \
        --test_file superior \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name inferior \
        --result_file inferior2superior