#!/bin/bash
set -e

TFRECORD=/home/cheer/Project/DailyReport/data/test_bert/tf_examples.tfrecord
CONFIG=/home/cheer/Project/DailyReport/bert/multi_cased_L-12_H-768_A-12
OUTPUT=/home/cheer/Project/DailyReport/data/test_bert

python run_pretraining.py \
  --input_file=${TFRECORD} \
  --output_dir=${OUTPUT} \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=${CONFIG}/bert_config.json \
  --init_checkpoint=${CONFIG}/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=64 \
  --max_predictions_per_seq=10 \
  --num_train_steps=10000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
