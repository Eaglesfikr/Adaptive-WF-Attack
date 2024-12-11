dataset=TimeDrift

for filename in train valid test 240327 240410 240709 240816 241209
do 
    python -u exp/dataset_process/gen_tam.py \
      --dataset ${dataset} \
      --seq_len 5000 \
      --in_file ${filename}
done

python -u exp/train.py \
  --dataset ${dataset} \
  --model RF \
  --device cuda:7 \
  --train_file tam_train \
  --valid_file tam_valid \
  --feature TAM \
  --seq_len 1800 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1

for file_name in test 240327 240410 240709 240816 241209
do
    python -u exp/test.py \
    --dataset ${dataset} \
    --model RF \
    --device cuda:7 \
    --valid_file tam_valid \
    --test_file tam_${file_name} \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file NoAdapt_${file_name}

    python -u exp/test.py \
    --dataset ${dataset} \
    --model RF \
    --device cuda:7 \
    --valid_file tam_valid \
    --test_file tam_${file_name} \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 256 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name proteus \
    --result_file preAdapt_conti_${file_name}

    python -u exp/proteus.py \
    --dataset ${dataset} \
    --model RF \
    --device cuda:7 \
    --train_file tam_train \
    --test_file tam_${file_name} \
    --feature TAM \
    --seq_len 1800 \
    --batch_size 128 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name proteus \
    --model_save_name proteus \
    --result_file afterAdapt_conti_${file_name} 
done