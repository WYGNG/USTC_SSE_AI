#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import string
# 导入生成器模块
import generate_captcha
# 导入模型模块
import captcha_model

if __name__ == '__main__':
    # 实例化generateCaptcha类为captcha对象
    captcha = generate_captcha.generateCaptcha()
    # 获取参数
    width,height,char_num,characters,classes = captcha.get_parameter()

    x = tf.placeholder(tf.float32, [None, height,width,1])
    y_ = tf.placeholder(tf.float32, [None, char_num*classes])
    keep_prob = tf.placeholder(tf.float32)
    # 实例化模型类为model对象
    model = captcha_model.captchaModel(width,height,char_num,classes)
    y_conv = model.create_model(x,keep_prob)
    
	# 由于识别验证码本质上是对验证码中的信息进行分类，所以我们这里使用cross_entropy的方法来衡量损失。
	# 优化方式选择的是AdamOptimizer，学习率设置比较小，为1e-4，防止学习的太快而训练不好。
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1,char_num, classe s])
    real = tf.reshape(y_,[-1,char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict,2), tf.argmax(real,2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    # 保存模型saver对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        step = 1
        while True:
            # 每批生成64张验证码
            batch_x,batch_y = next(captcha.gen_captcha(64))
            # 损失函数
            _,loss = sess.run([train_step,cross_entropy],feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
            # 输出步数以及损失
            print ('step:%d,loss:%f' % (step,loss))
            if step % 100 == 0:
                # 每训练100批，进行一次验证，生成100张验证码进行测试
                batch_x_test,batch_y_test = next(captcha.gen_captcha(100))
                # 进行准确度的测试
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                # 输出步数以及准确度
                print ('############step:%d,accuracy:%f' % (step,acc))
                # 如果准确度大于0.9则保存模型为capcha_model.ckpt，并结束训练
                if acc > 0.9:
                    saver.save(sess,"./capcha_model.ckpt")
                    break
            # 步数加一
            step += 1
