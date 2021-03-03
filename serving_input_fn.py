def serving_input_fn():
    # 保存模型为SaveModel格式
    # 采用最原始的feature方式，输入是feature Tensors。
    # 如果采用build_parsing_serving_input_receiver_fn，则输入是tf.Examples
     
    label_ids = tf.placeholder(tf.int32, [None, 3], name='label_ids')
    input_ids = tf.placeholder(tf.int32, [None, 200], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, 200], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, 200], name='segment_ids')
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': label_ids,
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
    })()
    return input_fn

