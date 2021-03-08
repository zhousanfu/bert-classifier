# bert单语种模型训练与部署
Tag: bert训练与部署Flask API接口
> bert训练与部署Flask API接口;bert在单语种的准确率较高，但训练与使用占用资源过大，可考虑跨语言模型：XLM

# 一、文件说明
配置文件(bert_config.json)：用于指定模型的超参数
词典文件(vocab.txt)：用于WordPiece 到 Word id的映射
Tensorflow checkpoint（bert_model.ckpt）：包含了预训练模型的权重（实际包含三个文件）

# 二、BERT结构
第一阶段：Pre-training，利用无标记的语料训练一个语言模型；
第二阶段：Fine-tuning, 利用预训练好的语言模型，完成具体的NLP下游任务。（run_classifier.py和run_squad.py）
extract_features.py-提取特征向量的



# 三、运行
## 3.1服务部署(bert-base)
```
pip install bert-base==0.0.7 -i https://pypi.python.org/simple
bert-base-serving-start \
    -model_dir "训练好的模型路径" \
    -bert_model_dir "bert预训练模型路径" \
    -model_pb_dir "classification_model.pb文件路径" \
    -mode CLASS \  # 模式, 咱们是分类所以用CLASS
    -max_seq_len 128 \  # 序列长度与上边保持一致
    -port 7006 \  # 端口号, 不要与其他程序冲突
    -port_out 7007 # 端口号

bert-base-serving-start -model_dir /data1/zhousanfu/imo_v1 -bert_model_dir /data1/zhousanfu/multi_cased_L-12_H-768_A-12 -model_pb_dir /data1/zhousanfu/imo_v1 -mode CLASS -max_seq_len 128 -port 5575 -port_out 5576
```

## 3.2模型压缩
> 运行后会在输出文件夹中多出一个 classification_model.pb 文件, 就是压缩后的模型
```
python freeze_graph.py \
    -bert_model_dir="bert预训练模型地址" \
    -model_dir="模型输出地址(和上边模型训练输出地址一样即可)" \
    -max_seq_len=128 \  # 序列长度, 需要与训练时 max_seq_length 参书相同
    -num_labels=3  # label数量

python3 /home/zhousanfu/bert_classifier/freeze_graph.py -bert_model_dir=/data1/zhousanfu/multi_cased_L-12_H-768_A-12 -model_dir=/data1/zhousanfu/imo_v1 -max_seq_len=512 -num_labels=2
```

## 3.3部署测试调用
```
from bert_base.client import BertClient
str1="我爱北京天安门"
str2 = "哈哈哈哈"
with BertClient(show_server_config=False, check_version=False, check_length=False, mode="CLASS", port=5575, port_out=5576) as bc:
    res = bc.encode([str1, str2])
print(res)
[{'pred_label': ['2', '1'], 'score': [0.9999899864196777, 0.9999299049377441]}]
```

## 3.4报错bert_base/server/http.py：
```
sudo pip install flask 
sudo pip install flask_compress
sudo pip install flask_cors
sudo pip install flask_json
```

## 3.5 bert运行：
### run_classifier.py 解说[https://blog.csdn.net/weixin_41845265/article/details/107071939]
### 最全详细：[https://blog.csdn.net/weixin_43320501/article/details/93894946?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control]
```
checkpoint 记录可用的模型信息
eval_results.txt 验证集的结果信息
eval.tf_record 记录验证集的二进制信息
events.out.tfevents.1590500393.instance-py09166k 用于tensorboard查看详细信息
graph.pbtxt 记录tensorflow的结构信息
label2id.pkl 标签信息 （额外加的）
model.ckpt-0* 这里是记录最近的三个文件
model.ckpt-2250.data 所有变量的值
model.ckpt-2250.index 可能是用于映射图和权重关系，0.11版本后引入
model.ckpt-2250.meta 记录完整的计算图结构
predict.tf_record 预测的二进制文件
test_results.tsv 使用预测后生成的预测结果
```

## 3.6训练：
```
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12#这里是存放中文模型的路径
export DATA_DIR=./data  #这里是存放数据的路径
 
python3 run_classifier.py \
--task_name=my \     #这里是processor的名字
--do_train=true \    #是否训练
--do_eval=true  \    #是否验证
--do_predict=false \  #是否预测（对应test）
--do_lower_case=false \
--data_dir=$DATA_DIR \ 
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=512 \最大文本程度，最大512
--train_batch_size=4 \
--learning_rate=2e-5 \
--num_train_epochs=15 \
--output_dir=./mymodel #输出目录

python3 /home/zhousanfu/bert_classifier/run_classifier.py --task_name=imodis --do_train=True --do_eval=True --do_lower_case=false --data_dir=/data1/zhousanfu/imo_v1 --vocab_file=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/bert_model.ckpt --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --max_seq_length=512 --output_dir=/data1/zhousanfu/imo_v1

python3 /home/zhousanfu/bert_classifier/run_classifier.py --task_name=hello --do_train=True --do_eval=True --do_lower_case=false --data_dir=/data1/zhousanfu/hello_v1 --vocab_file=/data1/zhousanfu/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/data1/zhousanfu/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data1/zhousanfu/chinese_L-12_H-768_A-12/bert_model.ckpt --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=5.0 --max_seq_length=512 --output_dir=/data1/zhousanfu/hello_v1
```

##　3.7预测（测试）：
> TRAINED_CLASSIFIER为刚刚训练的输出目录，无需在进一步指定模型模型名称，否则分类结果会不对
```
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export DATA_DIR=./mymodel
export ./mymodel
python3 run_classifier.py \
  --task_name=chi \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=512 \
  --output_dir=./mymodel

python3 /home/zhousanfu/bert_classifier/run_classifier.py --task_name=imodis --do_predict=True --do_lower_case=False --data_dir=/data1/zhousanfu/imo_v1 --vocab_file=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data1/zhousanfu/multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=/data1/zhousanfu/imo_v7
```

## 3.8 TF-serving 部署模型[https://blog.csdn.net/qq_42693848/article/details/107235688]
-(https://blog.csdn.net/JerryZhang__/article/details/85107506?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.control)
> 使用docker下载tfserving镜像
```
docker pull tensorflow/serving:2.1.0  # 这里下载的是tf2.1.0的版本，支持tensorflow1.15以上训练出来的模型。
docker run 
    -p 8501:8501 
    -p 8500:8500 
    --mount type=bind,source=/my/model/path/m,target=/models/m 
    -e MODEL_NAME=m 
    -t tensorflow/serving:2.1.0
上面的命令中：
(1)-p 8501:8501是端口映射，是将容器的8501端口映射到宿主机的8501端口，后面预测的时候使用该端口；
(2)-e MODEL_NAME=testnet 设置模型名称；
(3)--mount type=bind,source=/tmp/testnet,target=/models/testnet 是将宿主机的路径/tmp/testnet挂载到容器的/models/testnet下。/tmp/testnet是存放的是上述准备工作中保存的模型文件，‘testnet’是模型名称，包含一个.pb文件和一个variables文件夹，在/tmp/testnet下新建一个以数字命名的文件夹，如100001，并将模型文件放到该文件夹中。容器内部会根据绑定的路径读取模型文件；
(4)-t tensorflow/serving 根据名称“tensorflow/serving”运行容器；

$ docker run -p 8501:8501 --mount type=bind,source=/tmp/testnet,target=/models/testnet  -e MODEL_NAME=bert_NLP_hello_v1 -t tensorflow/serving &
```

