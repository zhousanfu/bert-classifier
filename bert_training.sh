#!/bin/bash
#description: 启动BERT 训练
export BERT_BASE_DIR=/data1/zhousanfu/multi_cased_L-12_H-768_A-12
export TRAINED_CLASSIFIER=/data1/zhousanfu
export DATA_NAME=hello_v1



echo '正在启动 BERT training...'
#echo '1 数据析分'
#python3 data_op.py 


echo '2 开始训练'
python3 run_classifier.py 
-task_name hello 
-do_train True 
-do_eval True 
-do_predict True 
-data_dir $TRAINED_CLASSIFIER/$DATA_NAME 
-vocab_file $TRAINED_CLASSIFIER/vocab.txt 
-bert_config_file $TRAINED_CLASSIFIER/bert_config.json 
-init_checkpoint $TRAINED_CLASSIFIER/bert_model.ckpt 
-train_batch_size 32 
-learning_rate 2e-5 
-num_train_epochs 5.0 
-max_seq_length 128 
-output_dir $TRAINED_CLASSIFIER/$DATA_NAME

echo '3 模型压缩'
python3 freeze_graph.py 
-bert_model_dir $BERT_BASE_DIR 
-model_dir $TRAINED_CLASSIFIER/$DATA_NAME 
-max_seq_len 128 
-num_labels 15