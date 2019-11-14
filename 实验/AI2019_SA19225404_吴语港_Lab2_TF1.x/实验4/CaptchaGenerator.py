# CaptchaGenerator.py
from captcha.image import ImageCaptcha
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

number = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# 本例中未使用大小写字母生成随机验证码,但我们可以使用数字+大小写字母生成随机验证码,只需少许修改即可
alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
            "v", "w", "x", "y", "z"]
ALPHABET = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
            "V", "W", "X", "Y", "Z"]

# 以0-9中的数字生成随机验证码,如果要用数字+大小写字母生成随机验证码,只需char_set = number + alphabet + ALPHABET
char_set = number


# 生成验证码字符列表的函数
def random_captcha_text(ch_len, ch_set):
   # 将验证码包含的字符存在ch_text
   ch_text = []
   # 进行captcha_size个循环，每次循环将1个随机抽到的字符放入ch_text
   [ch_text.append(random.choice(ch_set)) for _ in range(ch_len)]

   return ch_text


# 获取和生成的验证码对应的字符图片
def gen_captcha_text_and_image(wid, hei, ch_len, ch_set):
   # 生成指定大小的图片
   img = ImageCaptcha(width=wid, height=hei)
   # 生成一个随机的验证码序列
   ch_text = random_captcha_text(ch_len, ch_set)
   # 将字符串序列中每个字符连接起来
   ch_text = "".join(ch_text)
   # 根据验证码序列生成对应的字符图片
   ch_text_img = img.generate(ch_text)
   ch_img = np.array(Image.open(ch_text_img))
   # 将图片转换成一个数组，这个数组有3个维度
   # 因为图片是用RGB模式表示的，将其转换成数组即图片的分辨率160X60的矩阵，矩阵每个元素是一个像素点上的RGB三个通道的值
   # 返回字符串形式的验证码和对应的图片矩阵(RGB形式)
   return ch_text, ch_img


# 把彩色图像转为灰度图像
def convert_image_to_gray(img):
   # 这是彩色图像转换为灰度图像的公式
   r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
   # 求得灰度值后/255进行0-1二值化
   gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
   # 也可以求r、g、b三个值的平均值作为灰度值
   img[:, :, 0], img[:, :, 1], img[:, :, 2] = gray, gray, gray
   gray_img = img
   # 创建一个新矩阵，这个矩阵中的元素个数与img矩阵一样。
   # 但是每个元素只是原来RGB三个通道中的值的一个（转换成灰度图片后，RGB三个通道的数值相等）
   # 把矩阵变成一维数组，数组元素按矩阵行顺序排列
   gray_img_array = np.array(img[:, :, 0]).flatten()
   # 获取灰度图像矩阵(RGB形式,但三通道值相同)和拉平的灰度图片一维数组(同mnist数据集的图片数据形式)
   return gray_img, gray_img_array


# 生成图像对应的标签,one_hot编码形式,但不同字符之间全部连在一起
def generate_captcha_text_label(ch_text, ch_set):
   ch_text_label = np.zeros(len(ch_text) * len(ch_set))
   for i in range(len(ch_text)):
      char = ch_text[i]
      if char.isdigit():
         char = ord(char) - 48
      elif char.islower():
         char = ord(char) - 97 + 10
      elif char.isupper():
         char = ord(char) - 65 + 10 + 26
      # 请注意选择char_set时按number、alphabet、ALPHABET的顺序
      ch_text_label[i * len(ch_set) + char] = 1

   return ch_text_label


# 把生成的标签转换回字符序列
def text_label_turn_to_char_list(ch_text_label, ch_len, ch_set):
   ch_list = []
   for i in range(ch_len):
      for j in range(len(ch_set)):
         if ch_text_label[i * len(ch_set) + j] == 1.0:
            ch_list.append(ch_set[j])
   ch_list = "".join(ch_list)

   return ch_list


# 把预测得到的标签(经过tf.argmax函数处理后的数组)转换回字符序列
def pred_label_turn_to_char_list(pred_label, ch_len, ch_set):
   ch_list = []
   [ch_list.append(ch_set[pred_label[i]]) for i in range(ch_len)]
   ch_list = "".join(ch_list)

   return ch_list


# 测试
if __name__ == "__main__":
   width = 160
   height = 60
   char_size = 4
   text, image = gen_captcha_text_and_image(width, height, char_size, char_set)
   print("生成的验证码字符序列为：", text)
   print("生成的验证码图片数据的维度：", image.shape)
   plt.figure()
   plt.title(text)
   plt.imshow(image)
   plt.show()
   gray_image, gray_image_array = convert_image_to_gray(image)
   print("生成的灰度图片对应的一维数组的：", gray_image_array)
   print("生成的灰度图片对应的一维数组的维度大小：", gray_image_array.shape)
   label = generate_captcha_text_label(text, char_set)
   print("图片的标签数组为：\n", label)
   print("标签数组的维度大小为：", label.shape)
   char_list = text_label_turn_to_char_list(label, char_size, char_set)
   print("由标签数组生成对应的字符序列：", char_list)
