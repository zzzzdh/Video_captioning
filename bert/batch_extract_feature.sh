#!/bin/bash
set -e

CONFIG=/home/cheer/Project/DailyReport/bert/multi_cased_L-12_H-768_A-12
CKPT=/home/cheer/Project/DailyReport/data/pretraining_output

for file in $(ls /home/cheer/Project/DailyReport/data/OCR_text |grep java_0);
do
python extract_features.py \
  --input_file=/home/cheer/Project/DailyReport/data/OCR_text/$file \
  --output_file=/home/cheer/Project/DailyReport/data/code_features/${file/.txt/.jsonl} \
  --vocab_file=$CONFIG/vocab.txt \
  --bert_config_file=$CONFIG/bert_config.json \
  --init_checkpoint=$CKPT/model.ckpt-100000 \
  --layers=-1 \
  --max_seq_length=64 \
  --batch_size=32
done


