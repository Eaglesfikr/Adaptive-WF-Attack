dataset=AWF

for file_name in superior inferior
do
    python -u exp/train.py \
            --dataset ${dataset} \
            --model DF \
            --device cuda:0 \
            --train_file ${file_name} \
            --valid_file ${file_name} \
            --feature DIR \
            --seq_len 5000 \
            --train_epochs 30 \
            --batch_size 128 \
            --learning_rate 2e-3 \
            --optimizer Adam \
            --eval_metrics Precision Recall F1-score \
            --save_metric F1-score \
            --save_name ${file_name}
done

python -u exp/test.py \
        --dataset ${dataset} \
        --model DF \
        --device cuda:0 \
        --valid_file inferior \
        --test_file inferior \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name superior \
        --result_file superior2inferior

python -u exp/test.py \
        --dataset ${dataset} \
        --model DF \
        --device cuda:0 \
        --valid_file superior \
        --test_file superior \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name inferior \
        --result_file inferior2superior