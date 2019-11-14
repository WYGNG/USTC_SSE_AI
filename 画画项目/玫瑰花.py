# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:12:36 2019

@author: WYG
"""



from turtle import *
import time

setup(1000,800,0,0)
speed(0)
penup()
seth(90)
fd(340)
seth(0)
pendown()

speed(5)
begin_fill()
fillcolor('red')
circle(50,30)

for i in range(10):
    fd(1)
    left(10)

circle(40,40)

for i in range(6):
    fd(1)
    left(3)

circle(80,40)

for i in range(20):
    fd(0.5)
    left(5)

circle(80,45)

for i in range(10):
    fd(2)
    left(1)

circle(80,25)

for i in range(20):
    fd(1)
    left(4)

circle(50,50)

time.sleep(0.1)

circle(120,55)

speed(0)

seth(-90)
fd(70)

right(150)
fd(20)

left(140)
circle(140,90)

left(30)
circle(160,100)

left(130)
fd(25)

penup()
right(150)
circle(40,80)
pendown()

left(115)
fd(60)

penup()
left(180)
fd(60)
pendown()

end_fill()

right(120)
circle(-50,50)
circle(-20,90)

speed(1)
fd(75)

speed(0)
circle(90,110)

penup()
left(162)
fd(185)
left(170)
pendown()
circle(200,10)
circle(100,40)
circle(-52,115)
left(20)
circle(100,20)
circle(300,20)
speed(1)
fd(250)

penup()
speed(0)
left(180)
fd(250)
circle(-300,7)
right(80)
circle(200,5)
pendown()

left(60)
begin_fill()
fillcolor('green')
circle(-80,100)
right(90)
fd(10)
left(20)
circle(-63,127)
end_fill()

penup()
left(50)
fd(20)
left(180)

pendown()
circle(200,25)

penup()
right(150)

fd(180)

right(40)
pendown()
begin_fill()
fillcolor('green')
circle(-100,80)
right(150)
fd(10)
left(60)
circle(-80,98)
end_fill()

penup()
left(60)
fd(13)
left(180)

pendown()
speed(1)
circle(-200,23)



exitonclick()
