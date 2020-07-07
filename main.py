import numpy as np
from enum import Enum
from random import randint
import tkinter as tk  # 使用Tkinter前需要先导入
from astar import *
from ga import *
from tkinter import ttk, StringVar

global gen_num
global fit_num


def main():

    window = tk.Tk()
    gen_num = StringVar()
    fit_num = StringVar()
    open_length = StringVar()
    closed_length = StringVar()
    var00 = StringVar()
    var01 = StringVar()
    var02 = StringVar()
    var10 = StringVar()
    var11 = StringVar()
    var12 = StringVar()
    var20 = StringVar()
    var21 = StringVar()
    var22 = StringVar()

    def astar():
        gen_num.set('0')
        fit_num.set('∞')
        # start_state = [[1, 3, 2], [8, 9, 4], [5, 6, 7]]
        # start_state = [[6,7,8], [5, 9 , 1], [4, 3, 2]]
        begin_state = [[4, 5, 8], [3, 7, 2], [1, 6, 9]]
        # state2 = [[4, 8, 5], [3, 7, 2], [1, 6, 0]]
        goal_state = [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
        board = Board(begin_state, 0)
        info = board.a_star()
        info_path = info['path']
        len1 = info['len']
        len2 = info['whole']
        for i in info_path:
            for j in range(3):
                for k in range(3):
                    var00.set(str(i[0][0]))
                    var01.set(str(i[0][1]))
                    var02.set(str(i[0][2]))
                    var10.set(str(i[1][0]))
                    var11.set(str(i[1][1]))
                    var12.set(str(i[1][2]))
                    var20.set(str(i[2][0]))
                    var21.set(str(i[2][1]))
                    var22.set(str(i[2][2]))
                    open_length.set(str(len1))
                    closed_length.set(str(len2))
                    window.update()

    def ga():
        open_length.set('0')
        closed_length.set('0')
        global CHROMOSOME_LENGTH
        generation, numOfIncrement, bestmdis = 0, 0, 36
        bestSelection = []

        population = initialize_population()
        while generation < MAX_GENERATION:
            generation += 1

            for item in (population):
                mutation(item)

            slct = selection(population)
            current_puzzle = slct[0][1].puzzle
            mdis = slct[0][1].fitness()
            population = [item[0] for item in slct]

            if (mdis < bestmdis):
                bestmdis = mdis

                bestSelection = slct[0]

            if (generation // ADDED_INTERVAL_GEN > numOfIncrement):
                numOfIncrement += 1
                CHROMOSOME_LENGTH += ADDED_LENGTH
                update(CHROMOSOME_LENGTH)
            gen_num.set(generation)
            fit_num.set(mdis)
            var00.set(str(current_puzzle[0, 0]))
            var01.set(str(current_puzzle[0, 1]))
            var02.set(str(current_puzzle[0, 2]))
            var10.set(str(current_puzzle[1, 0]))
            var11.set(str(current_puzzle[1, 1]))
            var12.set(str(current_puzzle[1, 2]))
            var20.set(str(current_puzzle[2, 0]))
            var21.set(str(current_puzzle[2, 1]))
            var22.set(str(current_puzzle[2, 2]))
            window.update()

            if (mdis == 0):
                break

            crossover(population)

    window.title('8-puzzle')
    window.geometry('810x640')  # 这里的乘是小x
    # window.iconbitmap('./logo.ico')
    window.resizable(False, False)
    frame1 = tk.Frame(window, bg='white', width='145', height='600', relief='groove', bd=3)
    frame1.pack(side='left', fill = 'both')
    frame2 = tk.Frame(window, bg='white', width='490', height='300', relief='solid')
    frame2.pack(side='top')
    frame3 = tk.Frame(window, width = '490', height = '300', relief='groove')
    frame3.pack()
    label_for_title = tk.Label(frame1, text = '8-puzzle-question', width=13, height=1, fg="green", font=("Arial", 20))
    # label_for_title.pack()
    label_for_title.grid(row = 0, column = 0, columnspan = 3)
    method = tk.StringVar()
    gen_num.set('0')
    fit_num.set('∞')

    '''def ga_show():
        gen_num.set('0')
        fit_num.set('∞')
        lab_for_gen = tk.Label(frame3, text='代数', width=7, height=1, font=("Arial", 20))
        lab_for_fit = tk.Label(frame3, text='适应值', width=7, height=1, font=("Arial", 20))
        val_for_gen = tk.Label(frame3, textvariable=gen_num, width=7, height=1, font=("Arial", 20))
        val_for_fit = tk.Label(frame3, textvariable=fit_num, width=7, height=1, font=("Arial", 20))
        lab_for_gen.grid(row=2, column=0, sticky="w")
        val_for_gen.grid(row=2, column=1, sticky='w')
        lab_for_fit.grid(row=1, column=0, sticky='w')
        val_for_fit.grid(row=1, column=1, sticky='w')'''

    def start():
        # print(method.get())
        if method.get() == "A*算法":
            astar()

        else:
            ga()
            # ga_show()

    algorithm = ttk.Combobox(frame1, state='readonly', textvariable = method)
    algorithm.grid(row = 2, column = 0, pady = 40)
    algorithm['value'] = ('A*算法', '遗传算法')
    algorithm.current(0)
    button1 = tk.Button(frame1, text='开始', command=start, fg='green', width=18)
    button1.grid(row = 7, column = 0, padx = 3)

    button3 = tk.Button(frame1, text = "退出", command = window.quit, fg = 'red', width = 18)
    button3.grid(row = 8, column = 0, pady = 20)


    var00.set("1")
    var01.set("3")
    var02.set("2")
    var10.set("8")
    var11.set("9")
    var12.set("4")
    var20.set("5")
    var21.set("6")
    var22.set("7")

    ab00 = tk.Label(frame2, textvariable=var00,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=0, column=0, padx=0, pady=0)
    ab01 = tk.Label(frame2, textvariable=var01,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=0, column=1, padx=0, pady=0)
    ab02 = tk.Label(frame2, textvariable=var02,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=0, column=2, padx=0, pady=0)
    ab10 = tk.Label(frame2, textvariable=var10,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=1, column=0, padx=0, pady=0)
    ab11 = tk.Label(frame2, textvariable=var11,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=1, column=1, padx=0, pady=0)
    ab12 = tk.Label(frame2, textvariable=var12,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=1, column=2, padx=0, pady=0)
    ab20 = tk.Label(frame2, textvariable=var20,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=2, column=0, padx=0, pady=0)
    ab21 = tk.Label(frame2, textvariable=var21,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=2, column=1, padx=0, pady=0)
    ab22 = tk.Label(frame2, textvariable=var22,  fg='green', height = 1, width = 2, font=("Arial", 100)).grid(row=2, column=2, padx=0, pady=0)
    #

    gen_num.set('0')
    fit_num.set('∞')
    lab_for_gen = tk.Label(frame3, text = '代数', width=10, height=1, font=("Arial", 20))
    lab_for_fit = tk.Label(frame3, text = '适应值', width=10, height=1, font=("Arial", 20))
    val_for_gen = tk.Label(frame3, textvariable = gen_num, width=4, height=1, font=("Arial", 20))
    val_for_fit = tk.Label(frame3, textvariable = fit_num, width=4, height=1, font=("Arial", 20))
    lab_for_gen.grid(row = 2, column = 0,  sticky = "w")
    val_for_gen.grid(row = 2, column = 1,  sticky = 'w')
    lab_for_fit.grid(row = 1, column = 0,  sticky = 'w')
    val_for_fit.grid(row = 1, column = 1,  sticky = 'w')

    # open_length = StringVar()
    # closed_length = StringVar()
    #
    interval_line = tk.Canvas(frame3, height = 5, width = 162, background = 'green').grid()
    # interval_line2 = tk.Canvas(frame3, height=5, background='blue').grid(row = 3, column = 1)
    open_length.set('0')
    closed_length.set('0')
    lab_for_open = tk.Label(frame3, text = 'open表长度', width = 10, height = 1, font = ("Arial", 20))
    val_for_open = tk.Label(frame3, textvariable = open_length, width = 4, height = 1, font=("Arial", 20))
    lab_for_closed = tk.Label(frame3, text='closed表长度', width=10, height=1, font=("Arial", 20))
    val_for_closed = tk.Label(frame3, textvariable=closed_length, width=4, height=1, font=("Arial", 20))
    lab_for_open.grid(row=4, column=0, sticky="w")
    val_for_open.grid(row=4, column=1, sticky='w')
    lab_for_closed.grid(row=5, column=0, sticky='w')
    val_for_closed.grid(row=5, column=1, sticky='w')
    window.mainloop()


if __name__ == "__main__":
    main()
# TODO:
#  1. 将astar算法整合至图形界面
#  2. astar算法重要的参数同时及时显示出来
#  3. 后期美化，如果有时间，而且代价小的话
