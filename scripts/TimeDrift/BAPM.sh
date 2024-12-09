dataset=TimeDrift

python -u exp/train.py \
  --dataset ${dataset} \
  --model BAPM \
  --device cuda:4 \
  --feature DIR \
  --seq_len 8500 \
  --train_epochs 30 \
  --batch_size 128 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

for file_name in test 0327 0410 0709
do
  python -u exp/test.py \
    --dataset ${dataset} \
    --model BAPM \
    --device cuda:4 \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 8500 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file ${file_name}
done