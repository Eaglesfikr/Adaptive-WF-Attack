dataset=TimeDrift
model=RF
ablation=entropy

wait
rm -rf checkpoints/${dataset}/${model}/${ablation}.pth
cp checkpoints/${dataset}/${model}/max_f1.pth checkpoints/${dataset}/${model}/${ablation}.pth
wait

for file_name in day270
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model ${model} \
    --device cuda:5 \
    --test_file tam_${file_name} \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}

    python -u exp/ablation/8_entropy.py \
      --dataset ${dataset} \
      --model ${model} \
      --device cuda:5 \
      --train_file tam_train \
      --test_file tam_${file_name} \
      --feature TAM \
      --seq_len 1800 \
      --batch_size 128 \
      --eval_metrics Accuracy Precision Recall F1-score \
      --load_name ${ablation} \
      --model_save_name ${ablation} \
      --result_file ${ablation}_${file_name} 

done