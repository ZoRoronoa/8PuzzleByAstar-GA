# Last Edit 2019-11-19
# This program will solve the 8 puzzle "game" by using A* search.
# Made by Jason wu

import math
import random
import copy
import numpy as np


class Board():
    def __init__(self, begin_state, g, self_parent=None):
        self.state = begin_state
        self.end_state = [[1, 2, 3], [8, 9, 4], [7, 6, 5]]
        if self_parent is None:
            self.parent = None
        else:
            self.parent = self_parent

        self.g = g
        self.h = self.cal_h()

    def move_left(self, x, y, ):
        new_state = copy.deepcopy(self.state)
        tmp = new_state[x][y]
        new_state[x][y] = new_state[x][y - 1]
        new_state[x][y - 1] = tmp
        return new_state

    def move_right(self, x, y):
        new_state = copy.deepcopy(self.state)
        tmp = new_state[x][y]
        new_state[x][y] = new_state[x][y + 1]
        new_state[x][y + 1] = tmp
        return new_state

    def move_up(self, x, y):
        new_state = copy.deepcopy(self.state)
        tmp = new_state[x][y]
        new_state[x][y] = new_state[x - 1][y]
        new_state[x - 1][y] = tmp
        return new_state

    def move_down(self, x, y):
        new_state = copy.deepcopy(self.state)
        tmp = new_state[x][y]
        new_state[x][y] = new_state[x + 1][y]
        new_state[x + 1][y] = tmp
        return new_state

    def is_end_state(self):
        return is_same(self.state, self.end_state)

    # 每走一步，节点的g值应当自增1
    def get_sub_node(self):
        sub_node = []
        x, y = find_where(self.state, 9)
        if x < 2:
            sub_node.append(Board(self.move_down(x, y), self.g + 1, self))
        if x > 0:
            sub_node.append(Board(self.move_up(x, y), self.g + 1, self))
        if y > 0:
            sub_node.append(Board(self.move_left(x, y), self.g + 1, self))
        if y < 2:
            sub_node.append(Board(self.move_right(x, y), self.g + 1, self))
        if self.parent is None:
            return sub_node
        else:
            for obj in sub_node:
                if is_same(obj.parent.parent.state, obj.state):
                    sub_node.pop(find_node_in_list(obj, sub_node))
            return sub_node

    def get_path(self):
        path_list = [self.state]
        pointer = self
        while pointer.parent:
            pointer = pointer.parent
            path_list.insert(0, pointer.state)
        return path_list

    # h值 是当前棋盘和目标棋盘上各点距离之和
    def cal_h(self):
        h = 0
        for x in range(len(self.state)):
            for y in range(len(self.state[0])):

                if self.state[x][y] != self.end_state[x][y]:
                    if self.state[x][y] != 9:
                        x_end, y_end = find_where(self.end_state, self.state[x][y])
                        h += abs(x_end - x) + abs(y_end - y)
        return h

    def a_star(self):
        # 分别是open表和closed表
        open = []
        closed = []
        self_parent = self
        g = self.g
        h = self.h
        end_state = self.end_state
        # count = 0

        open.append(self)
        # 终止条件之一是open表为空
        while open:
            current_board = get_node_in_open(open)
            open.pop(find_node_in_list(current_board, open))
            closed.append(current_board)
            # print(len(open), end=',')
            # print(len(closed))
            if current_board.is_end_state():
                # TODO： 使用正确的方式，输出，这里直接选择输出列表不是很好康
                info = {}
                temp = current_board.get_path()
                # print(temp)
                length_of_open = len(open)
                len_of_closed = len(closed)
                info['path'] = temp
                info['len'] = length_of_open
                info['whole'] = len_of_closed
                return info
            child_node = current_board.get_sub_node()
            for obj in child_node:
                if obj in closed:
                    # continue
                    for ex in closed:
                        if obj == ex:
                            if obj.g + obj.h < ex.ht + ex.g:
                                ex.g = obj.g
                                ex.h = obj.h
                                ex.parent = obj.parent
                                open.insert(0, ex)
                                closed.pop(find_node_in_list(ex, closed))
                    return
                elif obj in open:
                    for tmp in open:
                        if obj == tmp:
                            if obj.g + obj.h < tmp.h + tmp.g:
                                tmp.g = obj.g
                                tmp.h = obj.h
                                tmp.parent = obj.parent

                else:
                    open.append(obj)


# 功能类似于np.where()
def find_where(state, index):
    for x in range(len(state)):
        if index in state[x]:
            for y in range(len(state[x])):
                if index == state[x][y]:
                    return x, y  # return the coordinates for the index.
    return -1, -1  # If no index is found, return -1 and -1


# 在open表中找到目标节点
def get_node_in_open(open):
    if len(open) == 1:
        return open[0]

    board = open[0]

    for i in range(len(open)):
        if board.h + board.g >= open[i].h + open[i].g:
            board = open[i]

    return board


def find_node_in_list(node, List):
    if len(List) == 1:
        return 0

    for x in range(len(List)):
        if node == List[x]:
            return x


# 判断两节点内容是否一致
def is_same(state1, state2):
    for x in range(len(state1)):
        for y in range(len(state1)):
            if state1[x][y] != state2[x][y]:
                return False
    return True


'''def main():
    # begin_state = [[1, 3, 2], [8, 9, 4], [5, 6, 7]]
    # start_state = [[1, 3, 2], [8, 9, 4], [5, 6, 7]]
    # state2 = [[4, 8, 5], [3, 7, 2], [1, 6, 0]]
    start_state = [[4, 5, 8], [3, 7, 2], [1, 6, 9]]
    end_state = [[1, 2, 3], [8, 9, 4], [7, 6, 5]]

    board = Board(start_state, 0)
    board.a_star()

main()'''