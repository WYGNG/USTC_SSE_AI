from turtle import *
from random import *
from math import *
class Tree:
    def __init__(self):
        setup(1000, 700)
        bgcolor(1, 1, 1)  # 背景色
        # ht()  # 隐藏turtle
        speed(10)  # 速度 1-10渐进，0 最快
        # tracer(1, 100)    # 设置绘图屏幕刷新频率，参数1设置在正常刷新频次的第参数1次刷新，参数2设置每次刷新的时延
        tracer(0, 0)
        pu()  # 抬笔
        backward(100)
        # 保证笔触箭头方向始终不向下，此处使其左转90度，而不是右转
        left(90)  # 左转90度
        backward(300)  # 后退300
    def tree(self, n, l):
        pd()  # 下笔
        # 阴影效果
        t = cos(radians(heading() + 45)) / 8 + 0.25
        pencolor(t, t, t)
        pensize(n / 1.2)
        forward(l)  # 画树枝
        if n > 0:
            b = random() * 15 + 10  # 右分支偏转角度
            c = random() * 15 + 10  # 左分支偏转角度
            d = l * (random() * 0.25 + 0.7)  # 下一个分支的长度
            # 右转一定角度,画右分支
            right(b)
            self.tree(n - 1, d)
            # 左转一定角度，画左分支
            left(b + c)
            self.tree(n - 1, d)
            # 转回来
            right(c)
        else:
            # 画叶子
            right(90)
            n = cos(radians(heading() - 45)) / 4 + 0.5
            pencolor(n, n * 0.8, n * 0.8)
            fillcolor(n, n * 0.8, n * 0.8)
            begin_fill()
            circle(3)
            left(90)
            end_fill()
            # 添加0.3倍的飘落叶子
            if random() > 0.7:
                pu()
                # 飘落
                t = heading()
                an = -40 + random() * 40
                setheading(an)
                dis = int(800 * random() * 0.5 + 400 * random() * 0.3 + 200 * random() * 0.2)
                forward(dis)
                setheading(t)
                # 画叶子
                pd()
                right(90)
                n = cos(radians(heading() - 45)) / 4 + 0.5
                pencolor(n * 0.5 + 0.5, 0.4 + n * 0.4, 0.4 + n * 0.4)
                fillcolor(n, n * 0.8, n * 0.8)
                begin_fill()
                circle(2)
                left(90)
                end_fill()
                pu()
                # 返回
                t = heading()
                setheading(an)
                backward(dis)
                setheading(t)
                # pass
        pu()
        backward(l)  # 退回
def main():
    tree = Tree()
    tree.tree(12, 100)  # 递归7层
    done()
if __name__ == '__main__':
    main()
