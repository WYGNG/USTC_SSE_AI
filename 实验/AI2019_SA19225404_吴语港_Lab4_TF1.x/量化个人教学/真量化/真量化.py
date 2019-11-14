import generate
import CNNmodel
import tensorflow as tf
from tensorflow.lite.python import lite
import numpy as np
from tensorflow.contrib.quantize import experimental_create_training_graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

if __name__ == '__main__':

    # 用reate_training_graph的文件名参数2替换
    per_trained_model_path = './restore/reate_training_graph/crack_capcha.model-20'


    with tf.Session(config=config) as sess:
        captcha = generate.generateCaptcha()

        x = tf.placeholder(tf.float32, [None, 60, 160, 1],name='x')
        y_ = tf.placeholder(tf.float32, [None, 52*4],name='y')
        keep_prob = tf.placeholder(tf.float32,name='prob')
        model = CNNmodel.CNNModel()
        y_conv = model.create_model(x)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))



        restore_dict = {}

        reader = tf.train.NewCheckpointReader(per_trained_model_path)

        for v in tf.global_variables():
            tensor_name = v.name.split(':')[0]
            if reader.has_tensor(tensor_name):
                restore_dict[tensor_name] = v

        tf.contrib.quantize.create_eval_graph(input_graph=sess.graph)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(restore_dict)
        saver.restore(sess, per_trained_model_path)

        const_graph = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=['y_conv'])
        writer = tf.summary.FileWriter('logs/', sess.graph)
        with tf.gfile.GFile('./models/frozen.pb', "wb") as f:
            f.write(const_graph.SerializeToString())

    #转lite
    FLAGS = tf.app.flags.FLAGS
    # build a converter
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file='./models/frozen.pb',
        input_arrays=['x'],
        output_arrays=['y_conv'],
        input_shapes={'x': [1, 60, 160,  1]})

    # set some attributes
    converter.inference_type = tf.uint8
    converter.inference_input_type = tf.uint8
    converter.post_training_quantize = False
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}
    converter.default_ranges_stats = (0, 6)

    # convert the pd model and write to tflite file
    tflite_quantized_model = converter.convert()
    with open('models/model.tflite', 'wb') as f:
        f.write(tflite_quantized_model)