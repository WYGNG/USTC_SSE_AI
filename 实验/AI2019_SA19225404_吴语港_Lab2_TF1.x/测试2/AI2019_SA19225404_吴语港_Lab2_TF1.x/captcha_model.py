# -*- coding: utf-8 -*
import tensorflow as tf
import math

class captchaModel():
    def __init__(self,
                 width = 160,
                 height = 60,
                 char_num = 4,
                 classes = 62):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.classes = classes
        
	# 这是2维卷积函数。x表示传入的待处理图片，W表示卷积核，strides=[1, 1, 1, 1]，其中第二个和第三个1分别表示x方向步长和y方向步长，
	# padding=’SAME’表示边界处理策略设为’SAME’，这样卷积处理完图片大小不变。
    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	# 这是2×2最大值池化函数。x表示待被池化处理的图片，ksize=[1, 2, 2, 1]，
	# 其中第二个和第三个2分别表示池化窗口高度和池化窗口宽度，strides和padding意义同上。
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
                              
	# 从截断的正态分布中输出随机值。X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，
	# 在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）
	# 看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
        
	# 生成一组全部都是0.1的常量数
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_model(self,x_images,keep_prob):
        #first layer
        w_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_images, w_conv1), b_conv1))
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_dropout1 = tf.nn.dropout(h_pool1,keep_prob)
        conv_width = math.ceil(self.width/2)
        conv_height = math.ceil(self.height/2)
        # w_conv1是卷积核，可以理解为一共有32个卷积核，每个卷积核的尺寸是(5, 5, 1)，
        # 即长度和宽度都是5，通道是1。每个卷积核对图片处理完就会产生一张特征图，32个卷积核对
        # 图片处理完后就会产生32个特征图，将这些特征图叠加排列，那么原本通道数为1的图片现在通道
        # 数变为图片的个数，也就是32。图片的尺寸变化为(?, 60, 160, 1) –> (?, 60, 160, 32)。
		# 随后又对图片进行一次池化处理，池化窗口为2×2，所以图片的长度和宽度都会变为原来的一半。
		# 图片的尺寸变化为(?, 60, 160, 32) → (?, 30, 80, 32)。
		# 随后又进行了一次dropout以防止过拟合，同时也是为了加大个别神经元的训练强度。

        #second layer
        w_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout1, w_conv2), b_conv2))
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_dropout2 = tf.nn.dropout(h_pool2,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)
		# w_conv2是第二层卷积神经网络的卷积核，共有64个，每个卷积核的尺寸是(5, 5, 32)，处理之后
		# 图片的尺寸变化为(?, 30, 80, 32) → (?, 30, 80, 64)。
		# 随后又对图片进行一次池化处理，池化窗口为2×2，所以图片的长度和宽度都会变为原来的一半。
		# 图片的尺寸变化为(?, 30, 80, 64) → (?, 15, 40, 64)。
		# 再进行一次dropout以防止过拟合，同时也是为了加大个别神经元的训练强度。
   
        #third layer
        w_conv3 = self.weight_variable([5, 5, 64, 64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout2, w_conv3), b_conv3))
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_dropout3 = tf.nn.dropout(h_pool3,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)
		# w_conv3是第三层卷积神经网络的卷积核，共有64个，每个卷积核的尺寸是(5, 5, 64)，处理之后
		# 图片的尺寸变化为(?, 15, 40, 64) → (?, 15, 40, 64)。
		# 随后又对图片进行一次池化处理，池化窗口为2×2，所以图片的长度和宽度都会变为原来的一半。
		# 图片的尺寸变化为(?, 15, 40, 64) → (?, 8, 20, 64)，这里的15 / 2 = 8，是因为边界策略为
		# SAME，那么遇到剩下还有不足4个像素的时候同样采取一次最大值池化处理。
		# 再进行一次dropout以防止过拟合，同时也是为了加大个别神经元的训练强度。

        #first fully layer
        conv_width = int(conv_width)
        conv_height = int(conv_height)
        w_fc1 = self.weight_variable([64*conv_width*conv_height,1024])
        b_fc1 = self.bias_variable([1024])
        h_dropout3_flat = tf.reshape(h_dropout3,[-1,64*conv_width*conv_height])
        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		# 这里就把刚刚卷积神经网络的输出作为传统神经网络的输入了，w_fc1(10240, 1024)和b_fc1(1024)
		# 分别是这一层神经网络的参数以及bias。上面代码第五行将卷积神经网络的输出数据由(?, 8, 20, 64)
		# 转为了(?, 64 * 20 * 8)，可以很明显感觉出来把所有的数据拉成了一条一维向量，然后经过矩阵
		# 处理，这里的数据变为了(1024, 1)的形状。
		
        #second fully layer
        w_fc2 = self.weight_variable([1024,self.char_num*self.classes])
        b_fc2 = self.bias_variable([self.char_num*self.classes])
        y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)
		# 再连接一次神经网络，这次不再需要添加激励函数了ReLu了，因为已经到达输出层，线性相加后直接
		# 输出就可以了，结果保存在y_conv变量里，最后将y_conv返回给调用函数。
		
        return y_conv
        # 这是外层函数调用model后所得到的训练结果。
