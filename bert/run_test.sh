export BERT_BASE_DIR=/home/cheer/Project/DailyReport/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/home/cheer/Project/DailyReport/data/GLUE
export TRAINED_CLASSIFIER=/home/cheer/Project/DailyReport/test/model.ckpt-343

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/home/cheer/Project/DailyReport/test
