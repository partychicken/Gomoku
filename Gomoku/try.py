from math import floor
from board import Board
import tkinter as tk

piece_sz = 36
board_st = (46, 46)
board_ed = (board_st[0] + 14*piece_sz, board_st[1] + 14*piece_sz)

root  = tk.Tk()
root.title("Gomoku")
root.geometry("760x580")

def click_event(event):
    x = floor((event.x-board_st[0]+piece_sz/2) / piece_sz)
    y = floor((event.y-board_st[1]+piece_sz/2) / piece_sz)
    pass

canvas = tk.Canvas(root, width = board_ed[0]+40, height = board_ed[1]+40)
canvas.bind("<Button-1>", click_event)  #对鼠标进行事件绑定，方便获取点击位置的坐标，下篇会用到
canvas.grid(row = 0, column = 0, rowspan = 6)

for i in range(15):
    canvas.create_line(board_st[0], (board_st[1] + piece_sz*i), \
        board_ed[0], (board_st[1] + piece_sz*i))
    canvas.create_line((board_st[0] + piece_sz*i), board_st[1], \
        (board_st[0] + piece_sz*i), board_ed[1])
#点
point_x = [3, 3, 11, 11, 7]
point_y = [3, 11, 3, 11, 7]
for i in range(5):
    canvas.create_oval(board_st[0]+piece_sz*point_x[i]-4, board_st[1]+piece_sz*point_y[i]-4, 
                       board_st[0]+piece_sz*point_x[i]+4, board_st[1]+piece_sz*point_y[i]+4,\
                       fill = "black")
#数字坐标
for i in range(15):
    label = tk.Label(canvas, text = str(i + 1), fg = "black",
                     width = 2, anchor = 'e')
    label.place(x = board_st[0]-piece_sz/2-2, y = board_st[1]+piece_sz*i, anchor = 'e')
#字母坐标
count = 0
for i in range(65, 81):
    if chr(i) == 'I':  continue;
    label = tk.Label(canvas, text = chr(i), fg = "black")
    label.place(x = board_st[0]+piece_sz*count, y = board_st[1]-piece_sz/2-2, anchor = 's')
    count += 1

click_x, click_y = 0, 0

#事件监听处理
def coorBack(event):  #return coordinates of cursor 返回光标坐标
    global click_x, click_y
    click_x = event.x
    click_y = event.y
    coorJudge()

#落子
def putPiece(piece_color):
    global coor_black, coor_white
    canvas.create_oval(click_x - piece_sz, click_y - piece_sz,
                       click_x + piece_sz, click_y + piece_sz, 
                       fill = piece_color, tags = ("piece"))

#找出离鼠标点击位置最近的棋盘线交点，调用putPiece落子
def coorJudge():
    global click_x, click_y
    coor = coor_black + coor_white
    global person_flag, show_piece
    #print("x = %s, y = %s" % (click_x, click_y))
    item = canvas.find_closest(click_x, click_y)
    tags_tuple = canvas.gettags(item)
    if len(tags_tuple) > 1:
        tags_list = list(tags_tuple)
        coor_list = tags_list[:2]
        try:
            for i in range(len(coor_list)):
                coor_list[i] = int(coor_list[i])
        except ValueError:
            pass
        else:
            coor_tuple = tuple(coor_list)
            (click_x, click_y) = coor_tuple
            #print("tags = ", tags_tuple, "coors = ", coor_tuple)
            if ( (click_x, click_y) not in coor )and( click_x in pieces_x )and( click_y in pieces_y ):
                #print("True")
                if person_flag != 0:
                    if person_flag == 1:
                        putPiece("black")
                        showChange("white")
                        var.set("执白棋")
                    elif person_flag == -1:
                        putPiece("white")
                        showChange("black")
                        var.set("执黑棋")
                    person_flag *= -1
            #else:
                #print("False")

root.mainloop()