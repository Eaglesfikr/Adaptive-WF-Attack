dataset=VersionDrift

for file_name in Tor46 Tor47 Tor48
do
    python -u exp/4_coteaching.py \
        --dataset ${dataset} \
        --model Coteaching \
        --device cuda:7 \
        --origin_file Tor45 \
        --valid_file ${file_name} \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name Tor45 \
        --result_file Tor45to${file_name}
done

for file_name in Tor45 Tor46 Tor47
do
    python -u exp/4_coteaching.py \
        --dataset ${dataset} \
        --model Coteaching \
        --device cuda:7 \
        --origin_file Tor48 \
        --valid_file ${file_name} \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 10000 \
        --batch_size 256 \
        --eval_metrics Precision Recall F1-score \
        --load_name Tor48 \
        --result_file Tor48to${file_name}
done