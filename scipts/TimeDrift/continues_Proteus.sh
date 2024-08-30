dataset=TimeDrift

file_name=test_0313
python -u exp/1_proteus.py \
    --dataset ${dataset} \
    --model Proteus \
    --device cuda:7 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name train_0313 \
    --result_file 0313

wait 
cp -r checkpoints/${dataset}/Proteus/ARES.pth checkpoints/${dataset}/ARES/0313.pth
cp -r checkpoints/${dataset}/Proteus/DF.pth   checkpoints/${dataset}/DF/0313.pth

file_name=0327
python -u exp/1_proteus.py \
    --dataset ${dataset} \
    --model Proteus \
    --device cuda:7 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name 0313 \
    --result_file ${file_name}

wait 
cp -r checkpoints/${dataset}/Proteus/ARES.pth checkpoints/${dataset}/ARES/${file_name}.pth
cp -r checkpoints/${dataset}/Proteus/DF.pth   checkpoints/${dataset}/DF/${file_name}.pth

file_name=0410
python -u exp/1_proteus.py \
    --dataset ${dataset} \
    --model Proteus \
    --device cuda:7 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name 0327 \
    --result_file ${file_name}

wait 
cp -r checkpoints/${dataset}/Proteus/ARES.pth checkpoints/${dataset}/ARES/${file_name}.pth
cp -r checkpoints/${dataset}/Proteus/DF.pth   checkpoints/${dataset}/DF/${file_name}.pth

file_name=0709
python -u exp/1_proteus.py \
    --dataset ${dataset} \
    --model Proteus \
    --device cuda:7 \
    --origin_file train_0313 \
    --valid_file ${file_name} \
    --test_file ${file_name} \
    --feature DIR \
    --seq_len 10000 \
    --batch_size 256 \
    --eval_metrics Precision Recall F1-score \
    --load_name 0410 \
    --result_file ${file_name}

wait 
cp -r checkpoints/${dataset}/Proteus/ARES.pth checkpoints/${dataset}/ARES/${file_name}.pth
cp -r checkpoints/${dataset}/Proteus/DF.pth   checkpoints/${dataset}/DF/${file_name}.pth
