#!/usr/bin/python
# -*- coding: utf-8 -*

from captcha.image import ImageCaptcha
# PIL：Python Imaging Library，已经是Python平台事实上的图像处理标准库了。PIL功能非常强大，但API却非常简单易用。
from PIL import Image
import numpy as np
import random
import string
# 生成captcha验证码类
class generateCaptcha():
    # 类构造方法
    def __init__(self,
                 width = 160,#验证码图片的宽
                 height = 60,#验证码图片的高
                 char_num = 4,#验证码字符个数
                 characters = string.digits + string.ascii_uppercase + string.ascii_lowercase):#验证码组成，数字+大写字母+小写字母
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)

    # 生成captcha验证码生成方法
    def gen_captcha(self,batch_size = 50):
        # X：一个 mini-batch 的训练数据，其 shape 为 [ batch_size, height, width, 1 ]，batch_size 表示每批次多少个训练数据，height ,width 表示验证码图片的高和宽，1 表示这是1个通道的黑白图片。
        # Y：X 中每个训练数据属于哪一类验证码，其形状为 [ batch_size, class ] ，对验证码中每个字符进行 One-Hot 编码，所以 class 大小为 4*62
        # 零填充预置shape，通道数为1，灰度图像
        X = np.zeros([batch_size,self.height,self.width,1])
        img = np.zeros((self.height,self.width),dtype=np.uint8)
        Y = np.zeros([batch_size,self.char_num,self.classes])
        image = ImageCaptcha(width = self.width,height = self.height)
		
		# 这个总体的无限循环是一个训练集的生成器，执行此代码后，会在最后的yield语句返回训练集X和Y，
		# 然后循环结束。下次再想生成验证码训练集时，会从yield语句（最后一句）开始，回到开头再执行一次循环。
        while True:
            for i in range(batch_size):
            	# 生成一个验证码字符串的随机变量，self.characters为62位的字符串（0~9A~Za~z），self.char_num=4（生成4个字符）。
                captcha_str = ''.join(random.sample(self.characters,self.char_num))

				# 使用的是ImageCaptcha类的内置方法，将字符串变为图片。convert(‘L’)：表示生成的是灰度图片，就是通道数为1的黑白图片。
                img = image.generate_image(captcha_str).convert('L')
                
                # Returns the contents of this image as a sequence object containing pixel values. The sequence object is flattened, so that values for line one follow directly after the values of line zero, and so on.
                # 将此图像的内容作为包含像素值的序列对象返回。Sequence对象是新的，因此第一行的值直接跟随在零行的值之后，依此类推。
                img = np.array(img.getdata())
                
                # 每个像素值都要除以255，这是为了归一化处理，因为灰度的范围是0~255，这里除以255就让每个像素的值在0~1之间，目的是为了加快收敛速度。
                X[i] = np.reshape(img,[self.height,self.width,1])/255.0

				# 用以生成对应的测试集Y，j和ch用以遍历刚刚生成的随机字符串，j记录index（0~3，表示第几个字符），ch记录字符串中的字符。找到Y的第i条数据中的第j个字符，然后把62长度的向量和ch相关的那个置为1。
                for j,ch in enumerate(captcha_str):
                    Y[i,j,self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            # yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。return 的作用：如果没有 return，则默认执行至函数完毕，返回的值一般是yield的变量。
            yield X,Y

    def decode_captcha(self,y):
        y = np.reshape(y,(len(y),self.char_num,self.classes))
        return ''.join(self.characters[x] for x in np.argmax(y,axis = 2)[0,:])

    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes

    def gen_test_captcha(self):
        image = ImageCaptcha(width = self.width,height = self.height)
        captcha_str = ''.join(random.sample(self.characters,self.char_num))
        img = image.generate_image(captcha_str)
        # 生成测试数据并保存
        img.save(captcha_str + '.jpg')
        # X：其 shape 为 [ 1, height, width, 1 ]，1 表示每批次多少个训练数据，height ,width 表示验证码图片的高和宽，1 表示这是1个通道的黑白图片。
        # Y：X 中每个训练数据属于哪一类验证码，其形状为 [ 1, class ] ，对验证码中每个字符进行 One-Hot 编码，所以 class 大小为 4*62        
        X = np.zeros([1,self.height,self.width,1])
        Y = np.zeros([1,self.char_num,self.classes])
        img = img.convert('L')
        img = np.array(img.getdata())
        X[0] = np.reshape(img,[self.height,self.width,1])/255.0
        for j,ch in enumerate(captcha_str):
            Y[0,j,self.characters.find(ch)] = 1
        Y = np.reshape(Y,(1,self.char_num*self.classes))
        return X,Y
