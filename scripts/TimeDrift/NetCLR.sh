pretrian_dataset=TimeDrift
dataset=TimeDrift

python -u exp/pretrain.py \
  --dataset ${pretrian_dataset} \
  --model NetCLR \
  --device cuda:0 \
  --train_epochs 100 \
  --train_file train \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --save_name pretrain

python -u exp/train.py \
  --dataset ${dataset} \
  --model NetCLR \
  --device cuda:0 \
  --feature DIR \
  --seq_len 5000 \
  --train_file train \
  --valid_file valid \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 3e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric Accuracy \
  --load_file checkpoints/${pretrian_dataset}/NetCLR/pretrain.pth \
  --save_name max_f1

for file_name in test 240327 240410 240709 240816 241209
do
    python -u exp/test.py \
      --dataset ${dataset} \
      --model NetCLR \
      --device cuda:0 \
      --test_file ${file_name} \
      --feature DIR \
      --seq_len 5000 \
      --batch_size 256 \
      --eval_metrics Accuracy Precision Recall F1-score \
      --load_name max_f1 \
      --result_file ${file_name}

    python -u exp/proteus.py \
        --dataset ${dataset} \
        --model NetCLR \
        --device cuda:0 \
        --train_file train \
        --test_file ${file_name} \
        --feature DIR \
        --seq_len 5000 \
        --batch_size 128 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        --result_file Proteus_${file_name} 
done