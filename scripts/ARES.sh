dataset=CW

python -u exp/train.py \
  --dataset ${dataset} \
  --model ARES \
  --device cuda:7 \
  --feature DIR \
  --seq_len 10000 \
  --train_epochs 30 \
  --batch_size 512 \
  --learning_rate 2e-3 \
  --optimizer AdamW \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

python -u exp/test.py \
  --dataset ${dataset} \
  --model ARES \
  --device cuda:7 \
  --feature DIR \
  --seq_len 10000 \
  --batch_size 256 \
  --eval_metrics Accuracy Precision Recall F1-score \
  --load_name max_f1