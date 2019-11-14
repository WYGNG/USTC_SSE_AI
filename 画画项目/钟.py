# -*- coding:utf-8 –*-
# 用turtlr画时钟
# 以自定义shape的方式实现
import turtle as t
import datetime as d
def skip(step):  # 抬笔，跳到一个地方
    t.penup()
    t.forward(step)
    t.pendown()
def drawClock(radius):  # 画表盘
    t.speed(0)
    t.mode("logo")  # 以Logo坐标、角度方式
    t.hideturtle()
    t.pensize(7)
    t.home()  # 回到圆点
    for j in range(60):
        skip(radius)
        if (j % 5 == 0):
            t.forward(20)
            skip(-radius - 20)
        else:
            t.dot(5)
            skip(-radius)
        t.right(6)
def makePoint(pointName, len):  # 钟的指针，时针、分针、秒针
    t.penup()
    t.home()
    t.begin_poly()
    t.back(0.1 * len)
    t.forward(len * 1.1)
    t.end_poly()
    poly = t.get_poly()
    t.register_shape(pointName, poly)  # 注册为一个shape
def drawPoint():  # 画指针
    global hourPoint, minPoint, secPoint, fontWriter
    makePoint("hourPoint", 100)
    makePoint("minPoint", 120)
    makePoint("secPoint", 140)
    hourPoint = t.Pen()  # 每个指针是一只新turtle
    hourPoint.shape("hourPoint")
    hourPoint.shapesize(1, 1, 6)
    minPoint = t.Pen()
    minPoint.shape("minPoint")
    minPoint.shapesize(1, 1, 4)
    secPoint = t.Pen()
    secPoint.shape("secPoint")
    secPoint.pencolor('red')
    fontWriter = t.Pen()
    fontWriter.pencolor('gray')
    fontWriter.hideturtle()
def getWeekName(weekday):
    weekName = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']
    return weekName[weekday]
def getDate(year, month, day):
    return "%s-%s-%s" % (year, month, day)
def realTime():
    curr = d.datetime.now()
    curr_year = curr.year
    curr_month = curr.month
    curr_day = curr.day
    curr_hour = curr.hour
    curr_minute = curr.minute
    curr_second = curr.second
    curr_weekday = curr.weekday()
    t.tracer(False)
    secPoint.setheading(360 / 60 * curr_second)
    minPoint.setheading(360 / 60 * curr_minute)
    hourPoint.setheading(360 / 12 * curr_hour + 30 / 60 * curr_minute)
    fontWriter.clear()
    fontWriter.home()
    fontWriter.penup()
    fontWriter.forward(80)
    # 用turtle写文字
    fontWriter.write(getWeekName(curr_weekday), align="center", font=("Courier", 14, "bold"))
    fontWriter.forward(-160)
    fontWriter.write(getDate(curr_year, curr_month, curr_day), align="center", font=("Courier", 14, "bold"))
    t.tracer(True)
    print(curr_second)
    t.ontimer(realTime, 100)  # 每隔100毫秒调用一次realTime()
def main():
    t.tracer(False)
    drawClock(160)
    drawPoint()
    realTime()
    t.tracer(True)
    t.mainloop()
if __name__ == '__main__':
    main()
