import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.lite.python import lite

def freeze_graph():
    #用刚刚获得的路径参数替换下面路径
    per_trained_model_path = './restore/no dropout1/crack_capcha.model-20'

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True
    # 加载tensorflow的图
    saver = tf.train.import_meta_graph(per_trained_model_path + '.meta', clear_devices=clear_devices)
    graph = tf.get_default_graph()#获得默认的图
    input_graph_def = graph.as_graph_def()#返回一个序列化的图

    with tf.Session() as sess:
        # 这步将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
        saver.restore(sess, per_trained_model_path)
        output_node_names = 'y_conv'
        # 开始将参数从变量转为固定值，也叫固化
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, #会话
            input_graph_def,#你加载的图
            output_node_names.split(",")  # 输出端口，pb文件会按你这个输出端口，只保存从输入到输出的图路径，所以参数很小
        )
        #这步是用来tensorflow可视化的，在本目录下生成logs
        #按win+R  输入cmd ，输入tensorboard --logdir=F:\11\作业\人工智能\实验3\重构\logs --host=127.0.0.1
        #注意路径问题，双击点开权重，可以看到没有max和min，也就是没有真正量化
        writer = tf.summary.FileWriter('logs/', sess.graph)
        # 写入models文件
        with tf.gfile.GFile('./models/frozen.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())

    FLAGS = tf.app.flags.FLAGS

    # build a converter
    converter = lite.TFLiteConverter.from_frozen_graph(
        graph_def_file='./models/frozen.pb',
        input_arrays=['x'],#输入接口
        output_arrays=['y_conv'], #输出接口
        input_shapes={'x': [1, 60, 160, 1]})#输入的形状

    # set some attributes
    converter.inference_type = tf.float32 #简单压缩，运行还是32位
    converter.inference_input_type = tf.float32#简单压缩，运行还是32位
    converter.post_training_quantize = True#不完全量化，选True，只压缩参数
    input_arrays = converter.get_input_arrays()
    converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}#mean和std，如果是完全量化的话这会影响精度，得自己统计。
    converter.default_ranges_stats = (0, 6)

    # convert the pd model and write to tflite file
    tflite_quantized_model = converter.convert()
    with open('models/model.tflite', 'wb') as f:
        f.write(tflite_quantized_model)

if __name__ == '__main__':

    freeze_graph()