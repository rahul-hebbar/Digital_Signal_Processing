import numpy as np
import tkinter
import time

def draw_dwt(dwttb):
    win = tkinter.Tk()
    win.title("Dynamic Time Wraping Table")
    for j in range(-1,l1):
        for i in range(-1,l2):
            if j == -1 and i > -1:
                tkinter.Label(win, text = "%.2f"%y[i], font = ("",15), width = 7).grid(column = j+1, row = i+1)
            elif j != -1 and i > -1:
                tkinter.Label(win,text="%.2f"%dwttb[j][i],font=("",15),borderwidth=4,relief='solid',width=7).grid(column=j+1,row=i+1)
            elif j != -1 and i == -1:
                tkinter.Label(win, text = "%.2f"%x[j], font = ("",15), width = 7).grid(column = j+1, row = i+1)
    win.mainloop()

def draw_wrp(dwttb,ind):
    win = tkinter.Tk()
    win.title("Warping table")
    for j in range(-1,l1):
        for i in range(-1,l2):
            if j == -1 and i > -1:
                tkinter.Label(win, text = "%.2f"%y[i], font = ("",15), width = 7).grid(column = j+1, row = i+1)
            elif j != -1 and i > -1:
                if (i,j) not in ind:
                    tkinter.Label(win,text="%.2f"%dwttb[j][i],font=("",15),borderwidth=4,relief='solid',width=7).grid(column=j+1,row=i+1)
                else:
                     tkinter.Label(win,text="%.2f"%dwttb[j][i],font=("",15),borderwidth=4,relief='solid',width=7,bg="red").grid(column=j+1,row=i+1)
            elif j != -1 and i == -1:
                tkinter.Label(win, text = "%.2f"%x[j], font = ("",15), width = 7).grid(column = j+1, row = i+1)
    win.mainloop()

def DwtTable(newl,news):
    global l1,l2,x,y
    x = newl
    y = news
    l1 = len(x)
    l2 = len(y)
    dwt_table = np.zeros((l1,l2))
    for j in range(l1):
        for i in range(l2):
            cost = (y[i]-x[j])**2
            # if i == 0 and j == 0:
            #     n,m,k,l = 0,0,0,0
            # elif j == 0 and i != 0:
            #     n,m,k,l = i-1,0,0,i-1
            # elif i == 0 and j != 0:
            #     n,m,k,l = 0,j-1,j-1,0
            # else:
            #     n,m,k,l = i,j,j-1,i-1
            # dwt_table[j][i] = cost + min(dwt_table[k][l],dwt_table[k][n],dwt_table[m][l])
            dwt_table[j][i] = cost
    return dwt_table

def WarpPath(dwttb):
    dis = dwttb[l1-1][l2-1]
    ind = [(l2-1,l1-1)]
    (i,j) = ind[0]
    while i != 0 and j != 0:
        if i == 0 and j != 0:
            n,m,k,l = 0,j-1,j-1,0
        elif j == 0 and i != 0:
            n,m,k,l = i-1,0,0,i-1
        else:
            n,m,k,l = i,j,j-1,i-1
        if dwttb[k][l] <= dwttb[k][n] and  dwttb[k][l] <= dwttb[m][l]:
            d =  dwttb[k][l]
            i,j = l,k
        elif dwttb[k][n] <= dwttb[k][l] and dwttb[k][n] <= dwttb[m][l]:
            d = dwttb[k][n]
            i,j = n,k
        else:
            d = dwttb[m][l]
            i,j = l,m
        dis += d
        ind.append((i,j))
    return dis,ind