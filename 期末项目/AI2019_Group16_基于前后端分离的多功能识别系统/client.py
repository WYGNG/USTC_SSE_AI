# coding=gbk
from tkinter import *
from socket import *
from tkinter.filedialog import askopenfilename
import os
import re
import cv2
import numpy as np
from tkinter import messagebox
HOST = '127.0.0.1'
PORT = 19225
BUFSIZ = 1024
ADDR = (HOST, PORT)


class FaceDetect:
 def __init__(self):
  window=Tk()
  window.title("FaceDetect")
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  self.Var=StringVar()
  frame1=Frame(window)
  frame1.pack()
  Label(frame1,text="Select a picture:").grid(row=1,column=1,sticky=W)
  e=Entry(frame1,textvariable=self.Var,width=40)
  e.grid(row=1,column=2)
  Button(frame1,text="浏览",command=lambda:self.processBrowse(e)).grid(row=1,column=3)
  Button(frame1,text="确认发送",command=lambda:self.processDelivery(e)).grid(row=1,column=4)
  Button(frame1,text="检测结果",command=self.processResult).grid(row=1,column=5)
  window.mainloop()
 
 def processBrowse(self,e):
  e.delete(0,END)
  m=askopenfilename()
  e.insert(0,m)

 def processDelivery(self,e):
  self.filename=e.get()
  if os.path.exists(self.filename)==0: 
   messagebox.showinfo("Tip","The file is not exist!")
   return 0
  elif re.search('\.jpg',self.filename) is None:
   messagebox.showinfo("Tip","The file is not picture.jpg!")
   return 
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('发送图片mode'.encode())
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'准备发送图片\n')
  myfile = open(self.filename, 'rb')
  data = myfile.read()
  size = str(len(data))
  tcpCliSock.send(size.encode())
  rec = tcpCliSock.recv(BUFSIZ).decode()
  tcpCliSock.send(data)
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'传输图片完成\n')
  tcpCliSock.close()

 def processResult(self):
  # 发送工作模式给服务端
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('人脸识别mode'.encode())
  # 接受到服务端发来的信息
  rec = tcpCliSock.recv(BUFSIZ).decode()
  # 判断没有脸的情况
  if rec=='No face':
   tcpCliSock.close()
   messagebox.showinfo("Tip","No face in the picture!")
  # 有脸的情况
  else:
   # 去除','形成列表
   list=rec.split(',')
   tcpCliSock.close()
   # 使用opencv解码图片
   img = cv2.imdecode(np.fromfile(self.filename,dtype=np.uint8),-1)
   vis = img.copy()
   # 对于每张脸有7个参数，分别是矩形框左上角的坐标(l,t)，矩形框的高h和宽w，脸所对应的性别、年龄、颜值得分
   # for循环可以自动根据len(list)/7确定脸数
   for i in range(int(len(list)/7)):
    # 获取人脸矩形框参数
    t,l,w,h=int(list[0+7*i]),int(list[1+7*i]),int(list[2+7*i]),int(list[3+7*i])
    # 输入矩形框左上角以及右下角的坐标参数
    cv2.rectangle(vis, (l, t), (l+w, t+h),(255, 0, 0), 2)
    # 在每一张脸矩形框的左上角标注性别年龄颜值打分信息
    cv2.putText(vis, 'gender:'+list[4+7*i]+' '+'age:'+list[5+7*i]+' '+'score:'+list[6+7*i], (l, t), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
   # 显示处理过的图片
   cv2.imshow("Image", vis)
   cv2.waitKey (0)

class Recognize:
 def __init__(self):
  window=Tk()
  window.title("Recognize")
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  self.Var=StringVar()
  frame1=Frame(window)
  frame1.pack()
  Label(frame1,text="Select a picture:").grid(row=1,column=1,sticky=W)
  e=Entry(frame1,textvariable=self.Var,width=40)
  e.grid(row=1,column=2)
  Button(frame1,text="浏览",command=lambda:self.processBrowse(e)).grid(row=1,column=3)
  Button(frame1,text="确认发送",command=lambda:self.processDelivery(e)).grid(row=1,column=4)
  Button(frame1,text="识别结果",command=self.processResult).grid(row=1,column=5)
  window.mainloop()

 def processBrowse(self,e):
  e.delete(0,END)
  m=askopenfilename()
  e.insert(0,m)

 def processDelivery(self,e):
  self.filename=e.get()
  if os.path.exists(self.filename)==0: 
   messagebox.showinfo("Tip","The file is not exist!")
   return 0
  elif re.search('\.jpg',self.filename) is None:
   messagebox.showinfo("Tip","The file is not picture.jpg!")
   return 
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('发送图片mode'.encode())
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'准备发送图片\n')
  myfile = open(self.filename, 'rb')
  data = myfile.read()
  size = str(len(data))
  tcpCliSock.send(size.encode())
  rec = tcpCliSock.recv(BUFSIZ).decode()
  tcpCliSock.send(data)
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'传输图片完成\n')
  tcpCliSock.close()

 def processResult(self):
  # 发送工作模式给服务端
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('目标识别mode'.encode())
  # 接受到服务端发来的信息
  rec=tcpCliSock.recv(BUFSIZ).decode()
  tcpCliSock.close()
  # 服务端通过'!'区分隔开为字符串，客户端通过'!'去除隔开，转换为列表
  list=rec.split('!')
  # 使用opencv打开图片
  img = cv2.imdecode(np.fromfile(self.filename,dtype=np.uint8),-1)
  window=Tk()
  window.title("Information")
  canvas=Canvas(window,width=1400,height=700,bg='white')
  canvas.pack()
  # 输出列表的信息
  for i in range(0,len(list),2):
   canvas.create_text(700,100+60*i,text=list[i]+':'+list[i+1],font=('Times',40))
  # 显示图片
  cv2.imshow("Image", img)
  cv2.waitKey (0)

class PlaceDetect:
 def __init__(self):
  # 建立一个tkinter窗口
  window=Tk()
  window.title("场景识别")
  # 在窗口上建立文本框
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  self.Var=StringVar()
  frame1=Frame(window)
  frame1.pack()
  # 标签提示
  Label(frame1,text="Select a picture:").grid(row=1,column=1,sticky=W)
  # 建立了一个输入框
  e=Entry(frame1,textvariable=self.Var,width=40)
  e.grid(row=1,column=2)
  # 生成三个按钮 在frame1上生成 点击时调用对应函数 生成位置
  Button(frame1,text="浏览",command=lambda:self.processBrowse(e)).grid(row=1,column=3)
  Button(frame1,text="确认发送",command=lambda:self.processDelivery(e)).grid(row=1,column=4)
  Button(frame1,text="识别结果",command=self.processResult).grid(row=1,column=5)
  # 窗口循环
  window.mainloop()

 def processBrowse(self,e):
  # 清空输入框
  e.delete(0,END)
  # 调用打开文件窗口
  m=askopenfilename()
  # 将文件名插入输入框
  e.insert(0,m)

 def processDelivery(self,e):
  # 获取文件输入框的文件路径
  self.filename=e.get()
  # 判断文件路径是否存在
  if os.path.exists(self.filename)==0: 
   messagebox.showinfo("Tip","The file is not exist!")
   return 0
  # 判断是否有jpg图片文件
  elif re.search('\.jpg',self.filename) is None:
   messagebox.showinfo("Tip","The file is not picture.jpg!")
   return 
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  # 连接到服务器地址
  tcpCliSock.connect(ADDR)
  # 发送字符串'发送图片mode'让服务端知道工作模式
  tcpCliSock.send('发送图片mode'.encode())
  # 如果接受到服务端的确认就在文本框显示准备发送图片
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'准备发送图片\n')
  myfile = open(self.filename, 'rb')
  # 将文件读入data中
  data = myfile.read()
  # 获取文件字符串长度
  size = str(len(data))
  tcpCliSock.send(size.encode())
  rec = tcpCliSock.recv(BUFSIZ).decode()
  # 发送图片
  tcpCliSock.send(data)
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'传输图片完成\n')
  # 关闭连接
  tcpCliSock.close()

 def processResult(self):
  # 发送工作模式给服务端
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('场景识别mode'.encode())
  # 接受到服务端发来的信息
  rec=tcpCliSock.recv(BUFSIZ).decode()
  # 关闭连接
  tcpCliSock.close()
  # 服务端通过'!'区分隔开为字符串，客户端通过'!'去除隔开，转换为列表
  list=rec.split('!')
  # 查看列表信息，用来调试程序
  print(list)
  # 使用opencv打开图片
  img = cv2.imdecode(np.fromfile(self.filename,dtype=np.uint8),-1)
  # 新建tkinter窗口
  window=Tk()
  window.title("识别信息")
  canvas=Canvas(window,width=250,height=600,bg='white')
  canvas.pack()
  # 输出列表的信息
  for i in range(0,len(list),2):
   canvas.create_text(125,30+30*i,text=list[i]+':'+list[i+1],font=('Times',20))
  # 显示图片
  cv2.imshow("Image", img)
  cv2.waitKey (0)

class BodyDetect:
 def __init__(self):
  window=Tk()
  window.title("人体识别")
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  self.Var=StringVar()
  frame1=Frame(window)
  frame1.pack()
  Label(frame1,text="选择图片:").grid(row=1,column=1,sticky=W)
  e=Entry(frame1,textvariable=self.Var,width=40)
  e.grid(row=1,column=2)
  Button(frame1,text="浏览文件",command=lambda:self.processBrowse(e)).grid(row=1,column=3)
  Button(frame1,text="确认发送",command=lambda:self.processDelivery(e)).grid(row=1,column=4)
  Button(frame1,text="分析结果",command=self.processResult).grid(row=1,column=5)
  window.mainloop()

 def processBrowse(self,e):
  e.delete(0,END)
  m=askopenfilename()
  e.insert(0,m)

 def processDelivery(self,e):
  self.filename=e.get()
  if os.path.exists(self.filename)==0: 
   messagebox.showinfo("Tip","The file is not exist!")
   return 0
  elif re.search('\.jpg',self.filename) is None:
   messagebox.showinfo("Tip","The file is not picture.jpg!")
   return 
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('发送图片mode'.encode())
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'准备发送图片\n')
  myfile = open(self.filename, 'rb')
  data = myfile.read()
  size = str(len(data))
  tcpCliSock.send(size.encode())
  rec = tcpCliSock.recv(BUFSIZ).decode()
  tcpCliSock.send(data)
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'传输图片完成\n')
  tcpCliSock.close()

 def processResult(self):
  # 发送工作模式给服务端
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('人体识别mode'.encode())
  # 接受到服务端发来的信息
  rec = tcpCliSock.recv(BUFSIZ).decode()
  # 判断没有脸的情况
  if rec=='No':
   tcpCliSock.close()
   messagebox.showinfo("Tip","No face in the picture!")
  else:
   # 去除','形成列表
   list=rec.split(',')
   tcpCliSock.close()
   print(list)
   # 使用opencv解码图片
   img = cv2.imdecode(np.fromfile(self.filename,dtype=np.uint8),-1)
   vis = img.copy()
   # 新建tk窗口
   window=Tk()
   window.title("Information")
   # 建一个画布
   canvas=Canvas(window,width=300,height=300,bg='white')
   canvas.pack()
   # 输出三条信息，分别是男性概率，上衣颜色，下衣颜色
   canvas.create_text(150,50,text=''.join(['男性概率：',list[0]]),font=('Times',20))
   canvas.create_text(150,150,text=''.join(['上衣颜色：',list[1]]),font=('Times',20))
   canvas.create_text(150,250,text=''.join(['下衣颜色：',list[2]]),font=('Times',20))
   # 输入矩形框左上角以及右下角的坐标参数
   cv2.rectangle(vis, (int(list[6]), int(list[4])), (int(list[6])+int(list[3]), int(list[4])+int(list[5])),(255, 0, 0), 2)
   # 标注性别概率信息
   cv2.putText(vis, 'male rate：'+list[0], (int(list[6]), int(list[4])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
   # 显示处理过的图片
   cv2.imshow("Image", vis)
   cv2.waitKey (0)

class BodySkeleton:
 def __init__(self):
  window=Tk()
  window.title("人体关键点识别")
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  self.Var=StringVar()
  frame1=Frame(window)
  frame1.pack()
  Label(frame1,text="选择图片:").grid(row=1,column=1,sticky=W)
  e=Entry(frame1,textvariable=self.Var,width=40)
  e.grid(row=1,column=2)
  Button(frame1,text="浏览文件",command=lambda:self.processBrowse(e)).grid(row=1,column=3)
  Button(frame1,text="确认发送",command=lambda:self.processDelivery(e)).grid(row=1,column=4)
  Button(frame1,text="分析结果",command=self.processResult).grid(row=1,column=5)
  window.mainloop()

 def processBrowse(self,e):
  e.delete(0,END)
  m=askopenfilename()
  e.insert(0,m)

 def processDelivery(self,e):
  self.filename=e.get()
  if os.path.exists(self.filename)==0: 
   messagebox.showinfo("Tip","The file is not exist!")
   return 0
  elif re.search('\.jpg',self.filename) is None:
   messagebox.showinfo("Tip","The file is not picture.jpg!")
   return 
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('发送图片mode'.encode())
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'准备发送图片\n')
  myfile = open(self.filename, 'rb')
  data = myfile.read()
  size = str(len(data))
  tcpCliSock.send(size.encode())
  rec = tcpCliSock.recv(BUFSIZ).decode()
  tcpCliSock.send(data)
  if tcpCliSock.recv(BUFSIZ).decode()=='OK':
   self.text.insert(INSERT,'传输图片完成\n')
  tcpCliSock.close()

 def processResult(self):
  # 发送工作模式给服务端
  tcpCliSock = socket(AF_INET, SOCK_STREAM)
  tcpCliSock.connect(ADDR)
  tcpCliSock.send('人体关键点识别mode'.encode())
  # 接受到服务端发来的信息
  rec = tcpCliSock.recv(BUFSIZ).decode()
  # 判断没有人体的情况
  if rec=='No':
   tcpCliSock.close()
   messagebox.showinfo("Tip","No!")
  # 有人体的情况
  else:
   # 去除','形成列表
   list=rec.split(',')
   # 关闭tcp连接
   tcpCliSock.close()
   # 输出list供调试使用
   print(list)
   # 输出list长度
   print(len(list))
   # 使用opencv解码图片
   img = cv2.imdecode(np.fromfile(self.filename,dtype=np.uint8),-1)
   vis = img.copy()
   # 关键点名称集合（列表）
   skeletonName = ['head','neck','left_shoulder','left_elbow','left_hand','right_shoulder','right_elbow','right_hand','left_buttocks','left_knee','left_foot','right_buttocks','right_knee','right_foot']
   # 对于每个人体有32个参数，分别是矩形框左上角的坐标(l,t)，矩形框的高h和宽w
   # 以及十四个关键点所对应的x，y坐标 32=4+2*14
   # 这里的坐标是相对于矩形框的坐标
   # for循环可以自动根据len(list)/32确定人体数
   for i in range(int(len(list)/32)):
    # 获取人体矩形框参数
    t,l,w,h=int(list[0+32*i]),int(list[1+32*i]),int(list[2+32*i]),int(list[3+32*i])
    # 输入矩形框左上角以及右下角的坐标参数
    cv2.rectangle(vis, (l, t), (l+w, t+h),(255, 0, 0), 2)
    # 通过循环来标注14个关键点的位置以及名称
    for j in range(14):
      # 通过关键点的相对矩形框坐标生成相当于图片的坐标
      pos=(l+int(list[4+2*j+32*i]),t+int(list[5+2*j+32*i]))
      # 用小圈标注出关键点的位置
      cv2.circle(vis, pos, 5, color=(0, 255, 0))
      # 标注名称信息
      cv2.putText(vis, skeletonName[j], pos, cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
   # 显示标注完的图片
   cv2.imshow("Image", vis)
   cv2.waitKey (0)

class Cli:
 def __init__(self):
  # 新建窗口（客户端的基础窗口）
  window=Tk()
  window.title("人工智能-期末项目-客户端")
  # 新建文本框
  self.text=Text(window,width=100,height=20)
  self.text.pack()
  # 按钮的一排
  frame1=Frame(window)
  frame1.pack()
  # 5个按钮
  Button(frame1,text="人脸识别",command=self.processFaceDetect).grid(row=1,column=1)
  Button(frame1,text="目标识别",command=self.processRecognize).grid(row=1,column=3)
  Button(frame1,text="人体识别",command=self.processBodyDetect).grid(row=1,column=5)
  Button(frame1,text="人体关键点识别",command=self.processBodySkeleton).grid(row=1,column=7)
  Button(frame1,text="场景识别",command=self.processPlaceDetect).grid(row=1,column=9)  
  window.mainloop()

 def processFaceDetect(self):
  self.text.insert(INSERT,'人脸识别开启\n')
  # FaceDetect对象实例化
  FaceDetect()

 def processRecognize(self):
  self.text.insert(INSERT,'目标识别开启\n')
  # Recognize对象实例化
  Recognize()

 def processBodyDetect(self):
  self.text.insert(INSERT,'人体识别开启\n')
  # BodyDetect对象实例化
  BodyDetect()
  
 def processBodySkeleton(self):
  self.text.insert(INSERT,'人体关键点识别开启\n')
  # BodySkeleton对象实例化
  BodySkeleton()

 def processPlaceDetect(self):
  self.text.insert(INSERT,'场景识别开启\n')
  # PlaceDetect对象实例化
  PlaceDetect()

# 如果运行本文件，而不是当包导入，那么就执行下面的程序
if __name__ == '__main__':
  Cli()