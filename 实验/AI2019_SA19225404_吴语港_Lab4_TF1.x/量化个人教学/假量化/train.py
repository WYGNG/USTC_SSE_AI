import generate
import CNNmodel1
import tensorflow as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

if __name__ == '__main__':
    captcha = generate.generateCaptcha()

    x = tf.placeholder(tf.float32, [None, 60, 160, 1],name='x')
    y_ = tf.placeholder(tf.float32, [None, 52*4],name='y')
    keep_prob = tf.placeholder(tf.float32,name='prob')

    model = CNNmodel1.CNNModel()
    y_conv = model.create_model(x,keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1,4, 52])
    real = tf.reshape(y_,[-1,4, 52])
    correct_prediction = tf.equal(tf.argmax(predict,2), tf.argmax(real,2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction,name='acc')

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('./restore/'))
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
                    saver.save(sess, "./crack_capcha.model",global_step=step)
                    break
            step += 1