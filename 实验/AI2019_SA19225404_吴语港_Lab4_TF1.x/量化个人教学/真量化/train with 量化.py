import generate
import CNNmodel
import tensorflow as tf
import numpy as np
from tensorflow.contrib.quantize import experimental_create_training_graph
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# 用训练完的参数1的路径替换下面，获得带伪量化的参数2
per_trained_model_path = './restore/crack_capcha.model-20'


if __name__ == '__main__':
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

        #tf.contrib.quantize.create_eval_graph(input_graph=sess.graph)
        experimental_create_training_graph(input_graph=sess.graph,
                                           weight_bits=8,
                                           activation_bits=8)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(restore_dict)
        saver.restore(sess, per_trained_model_path)

        #这是训练时，查看插入伪量化节点是否成功的玩意
        for node in sess.graph.as_graph_def().node:
            if 'AssignMaxLast' in node.name or 'AssignMinLast' in node.name:
                print('node name: {}'.format(node.name))

        # const_graph = tf.graph_util.convert_variables_to_constants(
        #     sess=sess,
        #     input_graph_def=sess.graph.as_graph_def(),
        #     output_node_names=['y_conv'])
        # writer = tf.summary.FileWriter('logs/', sess.graph)
        #
        # with tf.gfile.GFile('./models/frozen.pb', "wb") as f:
        #     f.write(const_graph.SerializeToString())



        predict = tf.reshape(y_conv, [-1,4, 52])
        real = tf.reshape(y_,[-1,4, 52])
        correct_prediction = tf.equal(tf.argmax(predict,2), tf.argmax(real,2))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction,name='acc')

        # const_graph = tf.graph_util.convert_variables_to_constants(
        #     sess=sess,
        #     input_graph_def=sess.graph.as_graph_def(),
        #     output_node_names=['y_conv'])

        # with tf.gfile.GFile('./models/frozen.pb', "wb") as f:
        #     f.write(const_graph.SerializeToString())

        # saver = tf.train.Saver()
        # sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint('./restore/'))
        step = 1
        while True:
            batch_x, batch_y = captcha.get_imgs(64)
            _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x.reshape([-1,60,160,1]), y_: batch_y.reshape([-1,52*4]), keep_prob: 0.75})

            if step % 10 == 0:
                text, textlabels = captcha.get_imgs(100)
                acc = 0
                predict1 = sess.run(predict, feed_dict={x: text.reshape([-1, 60, 160, 1]), keep_prob: 1.})
                for index, i in enumerate(predict1):
                    stextlabel = textlabels[index, :]
                    stextlabel = captcha.vec2text(np.argmax(stextlabel.reshape(4, 52), -1))
                    stext = captcha.vec2text(np.argmax(i, -1))
                    if stextlabel == stext:
                        acc += 1
                print('step:%d,loss:%f' % (step, loss))
                # batch_x_test, batch_y_test = captcha.get_imgs(100)
                # acc = sess.run(accuracy, feed_dict={x: batch_x_test.reshape([-1,60,160,1]), y_: batch_y_test.reshape([-1,52*4]), keep_prob: 1.})
                print('step:%d,accuracy:%f' % (step, acc/100))
                if acc/100 ==1 or step== 10000:
                    saver.save(sess, "./restore/reate_training_graph/crack_capcha.model",global_step=step)
                    writer = tf.summary.FileWriter('logs/', sess.graph)
                    break
            step += 1