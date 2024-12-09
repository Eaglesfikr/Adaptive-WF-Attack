dataset=TimeDrift

python -u exp/train.py \
  --dataset ${dataset} \
  --model TikTok \
  --device cuda:2 \
  --feature DT \
  --seq_len 5000 \
  --train_epochs 30 \
  --batch_size 128 \
  --learning_rate 2e-3 \
  --optimizer Adamax \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

for file_name in test 0327 0410 0709
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model TikTok \
    --device cuda:2 \
    --test_file ${file_name} \
    --feature DT \
    --seq_len 5000 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}
done