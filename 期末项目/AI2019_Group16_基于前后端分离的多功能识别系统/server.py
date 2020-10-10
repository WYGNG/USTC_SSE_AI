import sys
import re
import requests
import json
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pylab as plt
import threading

from tkinter import *
from socket import *
from time import ctime
from pylab import mpl
from os import path
from PIL import Image
from collections import Counter
url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
url_body = 'https://api-cn.faceplusplus.com/humanbodypp/v1/detect'
url_body_skeleton = 'https://api-cn.faceplusplus.com/humanbodypp/v1/skeleton'
PORT = [19225]
count=range(len(PORT))
model_dir='tf/model/'
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class NodeLookup(object):
  def __init__(self,label_lookup_path=None,uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string
    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def ser(PORT,text):
  # 使用本地环回ip进行测试，正式工作使用服务端的IP地址
  HOST = '127.0.0.1'
  BUFSIZ = 1024
  ADDR = (HOST, PORT)
  tcpSerSock = socket(AF_INET, SOCK_STREAM)
  tcpSerSock.bind(ADDR)
  tcpSerSock.listen(5)
  while True:
   # 等待连接
   print('port ',PORT,' is waiting for connection...')
   tcpCliSock, addr = tcpSerSock.accept()
   # 传入工作模式信息
   print('port ',PORT,' connected from:', addr)
   data = tcpCliSock.recv(BUFSIZ).decode()
   if data=='发送图片mode':
     text.insert(INSERT,'接收到客户端发送的图片\n')
     # 发送OK字符
     tcpCliSock.send('OK'.encode())
     # 获取客户端发来的尺寸信息
     size= tcpCliSock.recv(BUFSIZ).decode()
     # 发送OK字符
     tcpCliSock.send('OK'.encode())
     # 将接收到的尺寸信息字符串转换为整型
     size=int(size)
     # 接受图片信息，尺寸信息上文已经给出
     data = tcpCliSock.recv(size)
     # 将接受到的图片写入c.jpg
     myfile = open('c.jpg', 'wb')
     myfile.write(data)
     myfile.close()
     # 发送OK字符
     tcpCliSock.send('OK'.encode())

   elif data=='人脸识别mode':
     text.insert(INSERT,'对客户端发送的图片进行人脸检测\n')
     # 字典方式对打开本地文件操作录入
     files = {'image_file':open('c.jpg', 'rb')}
     # 调用api所需要的参数
     payload = {'api_key': 'aiyUZ7KnALYTZgI8ekWI-jnpxCCI2s5z',
           'api_secret': 'jxdfCWIcFP8fZQmVQRhI-fnSFalO_iwp',
           'return_landmark': 0,
           'return_attributes':'gender,age,glass,beauty'}
     # 通过本地文件调用接口
     r = requests.post(url,files=files,data=payload)
     # 将返回的json文件转化为python的字典
     data=json.loads(r.text)
     # 输出r.text供调试使用
     print (r.text)
     # 新建字符串sdata
     sdata = ""
     # 如果至少有一张脸
     if data["faces"]:
      # 对所有的脸迭代一次
      for i in range(len(data["faces"])):
        # 获取脸的性别信息
        gender=data['faces'][i]['attributes']['gender']['value']
        # 获取脸的年龄信息
        age=data['faces'][i]['attributes']['age']['value']
        # 脸的评分通过对男女评分的平均值表示
        score=(data['faces'][i]['attributes']['beauty']['female_score']+data['faces'][0]['attributes']['beauty']['male_score'])/2
        # 保留小数点后两位
        score='%.2f'%score
        # 脸的矩形框信息分别是矩形框左上角的坐标(l,t)，矩形框的高h和宽w
        width= data['faces'][i]['face_rectangle']['width']
        top= data['faces'][i]['face_rectangle']['top']
        height= data['faces'][i]['face_rectangle']['height']
        left= data['faces'][i]['face_rectangle']['left']
        # 将上述信息通过','隔开成字符串，存入sdata
        sdata=sdata+",".join([str(top),str(left),str(width),str(height),gender,str(age),score])+','
      # 通过tcp发送出去
      tcpCliSock.send(sdata.encode())
     else: tcpCliSock.send('No face'.encode())

   elif data=='人体识别mode':
     # 字典方式对打开本地文件操作录入
     files = {'image_file':open('c.jpg', 'rb')}
     # 调用api所需要的参数
     payload = {'api_key': 'aiyUZ7KnALYTZgI8ekWI-jnpxCCI2s5z',
           'api_secret': 'jxdfCWIcFP8fZQmVQRhI-fnSFalO_iwp',
           'return_attributes':'gender,upper_body_cloth,lower_body_cloth'}
     # 通过本地文件调用接口
     r = requests.post(url_body,files=files,data=payload)
     # 将返回的json文件转化为python的字典
     data=json.loads(r.text)
     # 输出r.text供调试使用
     print (r.text)
     # 如果至少识别出一个人体
     if data["humanbodies"]:
      # 性别信息
      gender=data['humanbodies'][0]['attributes']['gender']['male']
      # 上衣颜色
      upper_body_cloth=data['humanbodies'][0]['attributes']['upper_body_cloth']['upper_body_cloth_color']
      # 下衣颜色
      lower_body_cloth=data['humanbodies'][0]['attributes']['lower_body_cloth']['lower_body_cloth_color']
      # 人体的矩形框信息分别是矩形框左上角的坐标(l,t)，矩形框的高h和宽w
      width= data['humanbodies'][0]['humanbody_rectangle']['width']
      top= data['humanbodies'][0]['humanbody_rectangle']['top']
      height= data['humanbodies'][0]['humanbody_rectangle']['height']
      left= data['humanbodies'][0]['humanbody_rectangle']['left']
      # 将上述信息通过','隔开成字符串，存入sdata
      data=",".join([str(gender),str(upper_body_cloth),str(lower_body_cloth),str(width),str(top),str(height),str(left)])
      # 通过tcp发送出去
      tcpCliSock.send(data.encode())
     else: tcpCliSock.send('No'.encode())

   elif data=='人体关键点识别mode':
     # 字典方式对打开本地文件操作录入
     files = {'image_file':open('c.jpg', 'rb')}
     # 调用api所需要的参数
     payload = {'api_key': 'aiyUZ7KnALYTZgI8ekWI-jnpxCCI2s5z',
           'api_secret': 'jxdfCWIcFP8fZQmVQRhI-fnSFalO_iwp',}
     # 通过本地文件调用接口
     r = requests.post(url_body_skeleton,files=files,data=payload)
     # 将返回的json文件转化为python的字典
     data=json.loads(r.text)
     # 新建字符串sdata
     sdata = ""
     # 输出r.text供调试使用
     print (r.text)
     # 输出人体数目供测试使用
     print (len(data["skeletons"]))
     # 人体关键点列表
     skeletonName = ['head','neck','left_shoulder','left_elbow','left_hand','right_shoulder','right_elbow','right_hand','left_buttocks','left_knee','left_foot','right_buttocks','right_knee','right_foot']
     if data["skeletons"]:
        # 遍历所有的人体
        for i in range(len(data["skeletons"])):
          # 人体的矩形框信息分别是矩形框左上角的坐标(l,t)，矩形框的高h和宽w
          width= data['skeletons'][i]['body_rectangle']['width']
          top= data['skeletons'][i]['body_rectangle']['top']
          height= data['skeletons'][i]['body_rectangle']['height']
          left= data['skeletons'][i]['body_rectangle']['left']
          # 将上述信息通过','隔开成字符串，存入sdata
          sdata =sdata + ",".join([str(top),str(left),str(width),str(height)]) + ','
          # 遍历一个人体的所有关键点
          for j in skeletonName:
            # 关键点的x坐标
            tempx = data['skeletons'][i]['landmark'][j]['x']
            # 关键点的y坐标
            tempy = data['skeletons'][i]['landmark'][j]['y']
            # 将上述信息通过','隔开成字符串，存入sdata
            sdata = sdata + ",".join([str(tempx),str(tempy)]) + ','
        # 将信息发送到客户端
        tcpCliSock.send(sdata.encode())
     else: tcpCliSock.send('No'.encode())

   elif data=='目标识别mode':
    # 确认端口号
    if PORT==19225:
     text.insert(INSERT,'对客户端发送的图片进行目标识别\n')
     image_data = tf.gfile.FastGFile('c.jpg','rb').read()
     # 创建图
     with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')
    #启动会话
     sess=tf.Session()
    #Inception-v3模型的最后一层softmax的输出
     softmax_tensor= sess.graph.get_tensor_by_name('softmax:0')
    #输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
     predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
     predictions = np.squeeze(predictions)
     node_lookup = NodeLookup()
    #取出前5个概率最大的值（top-5)
     top_5 = predictions.argsort()[-5:][::-1]
     list=[]
     for node_id in top_5:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      score ='%.5f'%score
      list.append(human_string+'!'+score+'!')
     sess.close()
     # 列表转字符串通过空格隔开
     data=''.join(list)[:-1]
     tcpCliSock.send(data.encode())
    tcpCliSock.close()

   elif data=='场景识别mode':
    # 确认端口号
    if PORT==19225:
      text.insert(INSERT,'对客户端发送的图片进行场景识别\n')
      # 获取标签名称，按行读入
      lines = tf.gfile.GFile('output_labels.txt').readlines()
      uid_to_human = {}
      # 一行一行读取数据
      for uid,line in enumerate(lines) :
        #去掉换行符
        line=line.strip('\n')
        uid_to_human[uid] = line
      # 分类编号变成描述
      def id_to_string(node_id):
        if node_id not in uid_to_human:
          return ''
        return uid_to_human[node_id]
      # 创建一个图来存放训练好的模型
      with tf.gfile.GFile('tf/model/output_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
      with tf.Session() as sess:
        # final_result为输出tensor的名字
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        # 载入图片
        image_data = tf.gfile.GFile('c.jpg', 'rb').read()
        # 把图像数据传入模型获得模型输出结果
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
        # 把结果转为1维数据
        predictions = np.squeeze(predictions)
        # 排序
        top_k = predictions.argsort()[::-1]
        list = []
        for node_id in top_k:     
          # 获取分类名称
          human_string = id_to_string(node_id)
          # 获取该分类的置信度
          score = predictions[node_id]
          # 通过列表存储置信度信息
          list+=(human_string, str(int(100*score))+'%')
        sess.close()
      # 列表转字符串
      data='!'.join(list)
      # 发送字符串
      tcpCliSock.send(data.encode())
    # 关闭连接
    tcpCliSock.close()

class Ser:
 def __init__(self):
   # 新建窗口
  window=Tk()
  window.title("人工智能-期末项目-服务端")
  # 新建文本框
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  frame1=Frame(window)
  frame1.pack()
  # 按钮
  Button(frame1,text="服务端初始化",command=self.processInit).grid(row=1,column=1)
  Button(frame1,text="清空屏幕",command=self.processClear).grid(row=1,column=2)
  window.mainloop()

 def processInit(self):
  self.text.insert(INSERT,'服务端初始化完毕\n')
  # python多线程，根据客户端的预设数量启动线程数
  # 这里只有一个客户端，所以只有启动一个线程
  self.threads=[]
  for i in count:
   t=threading.Thread(target=ser,args=(PORT[i],self.text))
   self.threads.append(t) 
  # 启动线程
  for i in count:
   self.threads[i].start()

 def processClear(self):
  # 对文本框进行清屏操作
  self.text.delete('1.0','end')

if __name__ == '__main__':
  Ser()