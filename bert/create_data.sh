#!/bin/bash
set -e

INPUT=/home/cheer/Project/my_experiment/youtube_data/alltext.txt
OUTPUT=/home/cheer/Project/DailyReport/data/test_bert/tf_examples.tfrecord
VOCAB=/home/cheer/Project/DailyReport/bert/multi_cased_L-12_H-768_A-12/vocab.txt

python create_pretraining_data.py \
  --input_file=${INPUT} \
  --output_file=${OUTPUT} \
  --vocab_file=${VOCAB} \
  --do_lower_case=False \
  --max_seq_length=64 \
  --max_predictions_per_seq=10 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
