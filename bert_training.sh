#!/bin/bash
#description: 启动BERT 训练

echo '正在启动 BERT training...'
cd bert_demo

export BERT_BASE_DIR=/data1/zhousanfu/multi_cased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/data1/zhousanfu


python3 run_classifier.py 
-task_name imodis 
-do_train True 
-do_lower_case False 
-do_eval True 
-data_dir $TRAINED_CLASSIFIER 
-vocab_file $TRAINED_CLASSIFIER/vocab.txt 
-bert_config_file $TRAINED_CLASSIFIER/bert_config.json 
-init_checkpoint $TRAINED_CLASSIFIER/bert_model.ckpt 
-train_batch_size 32 
-learning_rate 5e-5 
-num_train_epochs 3.0 
-max_seq_length 128 
-output_dir $TRAINED_CLASSIFIER