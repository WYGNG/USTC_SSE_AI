#SquareSpiral1.py
import turtle as t
# t = turtle.Pen()
t.bgcolor("black")
sides=eval(input("输入要绘制的边的数目，请输入2-6的数字！"))
colors=["red","yellow","green","blue","orange","purple"]
for x in range(150):
    t.pencolor(colors[x%sides])
    t.forward(x*3/sides+x)
    t.left(360/sides+1)
    t.width(x*sides/200)
t.exitonclick()
print("####结束####")
