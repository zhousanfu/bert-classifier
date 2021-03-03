#!/usr/bin/env python3
#coding:utf-8
import argparse
import json
import logging
import os
import re
import string
import sys
import time
import flask
import numpy as np
from bert_base.client import BertClient
from flask import Flask, jsonify, request, render_template
sys.path.append('../')


# list 转成Json格式数据
def list_to_json(lst):
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    
    return str_json



# 切分句子
def cut_sent(txt):
    #先预处理去空格等
    txt = re.sub('([　 \t]+)',r" ",txt)  # blank word
    txt = txt.rstrip()       # 段尾如果有多余的\n就去掉它
    nlist = txt.split("\n") 
    nnlist = [x for x in nlist if x.strip()!='']  # 过滤掉空行
    return nnlist



#对句子进行预测识别
def class_pred(list_text):
    #文本拆分成句子
    #list_text = cut_sent(text)
    #print("total setance: %d" % (len(list_text)) )
    with BertClient(ip='localhost', port=5575, port_out=5576, show_server_config=False, check_version=False, check_length=False, timeout=10000 , mode='CLASS') as bc:
        #start_t = time.perf_counter()
        rst = bc.encode(list_text)
        #print('result:', rst)
        #print('time used:{}'.format(time.perf_counter() - start_t))
    result_txt = list_to_json(rst)

    return result_txt




def flask_server(args):
    app = Flask(__name__)
    #from app import routes

    @app.route('/')
    def index():
        return render_template("index.html", version='V 0.0.1')
  
    @app.route('/api/v0.1/query', methods=['POST'])
    def query ():
        res = {}
        txt = request.values['text']
        if not txt :
            res["result"]="error"
            return jsonify(res)
        lstseg = cut_sent(txt)

        if request.method == 'POST':
            result_txt = class_pred(lstseg)
        #print(result_txt)

        return result_txt


    app.run(
        host = args.ip,     #'0.0.0.0',
        port = args.port,   #8910,  
        debug = True 
    )


def main_cli ():
    pass
    parser = argparse.ArgumentParser(description='API demo server')
    parser.add_argument('-ip', type=str, default="0.0.0.0",
                        help='chinese google bert model serving')
    parser.add_argument('-port', type=int, default=8910,
                        help='listen port,default:8910')

    args = parser.parse_args()

    flask_server(args)

if __name__ == '__main__':
    main_cli()
