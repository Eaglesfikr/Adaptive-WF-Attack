dataset=VersionDrift

for version in 45 46 47 48
do
        python -u exp/train.py \
                --dataset ${dataset} \
                --model ARES \
                --device cuda:7 \
                --train_file Tor${version} \
                --valid_file Tor${version} \
                --feature DIR \
                --seq_len 10000 \
                --train_epochs 30 \
                --batch_size 64 \
                --learning_rate 0.0014 \
                --optimizer Adam \
                --eval_metrics Precision Recall F1-score \
                --save_metric F1-score \
                --save_name Tor${version}

        for file_name in Tor45 Tor46 Tor47 Tor48
        do
                python -u exp/test.py \
                --dataset ${dataset} \
                --model ARES \
                --device cuda:7 \
                --valid_file ${file_name} \
                --test_file ${file_name} \
                --feature DIR \
                --seq_len 10000 \
                --batch_size 256 \
                --eval_metrics Precision Recall F1-score \
                --load_name Tor${version} \
                --result_file Tor${version}2${file_name}
        done
done