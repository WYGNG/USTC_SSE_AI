from captcha.image import ImageCaptcha # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2
#生成字符对应的验证码
class generateCaptcha():
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    def random_captcha_text(self,char_set=alphabet, captcha_size=4):
        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text
    def gen_captcha_text_and_image(self):
        image = ImageCaptcha(width = 160,height = 60)
        captcha_text = self.random_captcha_text()
        captcha_text = ''.join(captcha_text) #连接字符串
        captcha = image.generate(captcha_text)
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def vec2text(self,char_pos):
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % 52
            if char_idx < 26:
                char_code = char_idx + ord('A')
            elif char_idx < 52:
                char_code = char_idx - 26 + ord('a')
            text.append(chr(char_code))
        return "".join(text)

    def text2vec(self,text):
        vector = np.zeros(4 * 52)
        def char2pos(c):
            k = ord(c) - 65
            if k > 25:
                k = ord(c) - 71
            return k

        for i, c in enumerate(text):
            idx = i * 52 + char2pos(c)
            vector[idx] = 1
        return vector

    def get_imgs(self,num):
        #获取图片
        train_imgs = np.zeros(num*160*60).reshape(num,160*60)
        test_labels = np.zeros(num*52*4).reshape(num,52*4)
        for i in range(num):
            text, image = self.gen_captcha_text_and_image()
            train_imgs[i,:] = np.mean(image,-1).flatten()/255
            test_labels[i,:] = self.text2vec(text)

        return train_imgs, test_labels

