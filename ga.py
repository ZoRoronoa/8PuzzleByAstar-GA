import numpy as np
from enum import Enum
from random import randint
import tkinter as tk

MAX_GENERATION = 1000
POPULATION_SIZE = 34
CHROMOSOME_LENGTH = 20
# 每一代个体最后只会选择适应度最高的3个个体
# NUMBER_OF_SELECTED_CHROMOSOME = 3
SELECTED_INDIVIDUALS = 10
# 每隔50代，染色体的长度会增加一次
ADDED_INTERVAL_GEN = 50
# 每次染色体增加的长度是5
ADDED_LENGTH = 5

# initial = np.array([[1, 3, 2], [8, 9, 4], [5, 6, 7]])
begin_state = np.array([[4, 5, 8], [3, 7, 2], [1, 6, 9]])
# [[4, 5, 8], [3, 7, 2], [1, 6, 9]]
end_state = np.array([[1, 2, 3], [8, 9, 4], [7, 6, 5]])
# goal = np.array([[1, 2, 3], [8, 9, 4], [7, 6, 5]])
# astar算法的初始化

class Move(Enum):
    U = 1
    R = 2
    D = 3
    L = 4

    # def isEqual(self, move):
    #     return self == move

    def is_same(self, move):
        return self == move

    def is_opposite(self, move):
        return abs(self.value - move.value) == 2

    def get_opposite(self):
        return Move(self.value - 2) if self.value > 2 else Move(self.value + 2)

    def change_another(self):
        enums = list(Move)
        enums.remove(self)
        return enums[randint(0, 2)]

    def change_another_exc_opp(self):
        enums = list(Move)
        enums.remove(self)
        enums.remove(self.get_opposite())
        return enums[randint(0, 1)]


class Puzzle:

    def __init__(self):
        self.puzzle = np.array(begin_state)

    def move(self, move):

        # 矩阵内置方法，可以获得9所在位置的坐标并且赋值给x, y
        x, y = np.where(self.puzzle == 9)

        if move == Move.U:
            self.swap([x, y], [x-1, y])
        elif move == Move.R:
            # if y == 2:
            #     raise IndexError(
            #         "the y coordinate exceeds the range of the puzzle.")
            self.swap([x, y], [x, y+1])
        elif move == Move.D:
            # if x == 2:
            #     raise IndexError(
            #         "the x coordinate exceeds the range of the puzzle.")
            self.swap([x, y], [x+1, y])
        elif move == Move.L:
            # if y == 0:
            #     raise IndexError("the y coordinate cannot be a negative value")
            self.swap([x, y], [x, y-1])
#
    # def __swap(self, coordinate1, coordinate2):
    #     tmp = self.puzzle[coordinate1[0], coordinate1[1]]
    #     self.puzzle[coordinate1[0], coordinate1[1]
    #                 ] = self.puzzle[coordinate2[0], coordinate2[1]]
    #     self.puzzle[coordinate2[0], coordinate2[1]] = tmp

    def swap(self, status1, status2):
        tmp = self.puzzle[status1[0], status1[1]]
        self.puzzle[status1[0], status1[1]
                    ] = self.puzzle[status2[0], status2[1]]
        self.puzzle[status2[0], status2[1]] = tmp

    # 适应度函数是当前棋盘每个位置与目标位置的距离差(包括x和y坐标)之和
    def fitness(self):
        sum_distance = 0
        for i in range(3):
            for j in range(3):
                if end_state[i, j] == 9:
                    continue
                x, y = np.where(self.puzzle == end_state[i, j])
                sum_distance += abs(x[0]-i) + abs(y[0]-j)
        return sum_distance

    # def __str__(self):
    #     return str(self.puzzle)


# def createChromosome(length=CHROMOSOME_LENGTH):
#     global CHROMOSOME_LENGTH
#     chromosome = []
#     enums = list(Move)
#     [chromosome.append(enums[randint(0, 3)]) for i in range(length)]
#     return chromosome


def initial_individual(length=CHROMOSOME_LENGTH):
    global CHROMOSOME_LENGTH
    chromosome = []
    enums = list(Move)
    [chromosome.append(enums[randint(0, 3)]) for i in range(length)]
    return chromosome

def initialize_population():
    global CHROMOSOME_LENGTH
    population = []
    [population.append(initial_individual(CHROMOSOME_LENGTH))
     for i in range(POPULATION_SIZE)]
    return population
# <chromosome>'a (yani List<Direction>) düzeltme uygular
# - 3x3 lük puzzle da peş peşe 3 kere aynı yöne hareket yapılamaz
# - Peş peşe zıt hareketler yapmak anlamsızdır/gereksizdir

# 变异函数
# 变异率如何计算，考虑到没隔50代的个体长度会发生改变，这里的主要变化来自于个体的长度变化和交叉
# 考虑不适用长度呢？
# TODO: CHROMOSOME_LENGTH变量的值必须改变，随着主界面函数中内容改变而改变


def mutation(chromosome):
    global CHROMOSOME_LENGTH
    length = len(chromosome)
    if length < 2:
        return chromosome
    # 如果长度不够就继续加上去, 染色体长度不够可能导致在一定的代数之内无法找到合适的变化次序:)
    if length < CHROMOSOME_LENGTH:
        chromosome += initial_individual(CHROMOSOME_LENGTH-length)
    # is_opposite() 意味着上/下， 或者左/右
    if chromosome[0].is_opposite(chromosome[1]):
        chromosome[1] = chromosome[1].change_another()

    for i in range(2, length):
        if chromosome[i].is_same(chromosome[i-2]) and chromosome[i].is_same(chromosome[i-1]):
            chromosome[i] = chromosome[i-1].change_another_exc_opp()

        elif chromosome[i].is_opposite(chromosome[i-1]):
            chromosome[i] = chromosome[i-1].change_another()

# 根据染色体的变化找到其对应的棋盘，并且依据此来计算出来适应度，为以后的筛选做好准备
# def applyChromosomeToPuzzle(chromosome):
#     puzzle = Puzzle()
#     i = 0
#     while i < len(chromosome):
#         try:
#             if (puzzle.fitness() == 0):
#                 return [chromosome[:i], puzzle]
#             puzzle.move(chromosome[i])
#             i += 1
#         except IndexError:
#             chromosome[i] = chromosome[i].change_another_exc_opp()
#     return [chromosome, puzzle]


def puzzle_from_chromosome(chromosome):
    puzzle = Puzzle()
    i = 0
    while i < len(chromosome):
        try:
            if puzzle.fitness() == 0:
                return [chromosome[:i], puzzle]
            puzzle.move(chromosome[i])
            i += 1
            # 防止列表越界
        except IndexError:
            chromosome[i] = chromosome[i].change_another_exc_opp()
    return [chromosome, puzzle]


# 递归，即将得到的适应度最高的3个个体分别交叉，得到遗传的后代
# 每次交叉诞生10个后代，所以下一个共有33个子代
# 10个体中选择8个进行交叉繁衍，交叉率为0.8
# 选择的个体每队生成6个个体，种群数量维持在34（10 + 3 * 8）
def crossover(chromosomes):
    i1 = randint(0, SELECTED_INDIVIDUALS - 1)
    j1 = (i1 + 1) % SELECTED_INDIVIDUALS
    i2 = (i1 + 2) % SELECTED_INDIVIDUALS
    j2 = (i2 + 1) % SELECTED_INDIVIDUALS
    i3 = (i1 + 3) % SELECTED_INDIVIDUALS
    j3 = (i3 + 1) % SELECTED_INDIVIDUALS
    i4 = (i1 + 4) % SELECTED_INDIVIDUALS
    j4 = (i4 + 1) % SELECTED_INDIVIDUALS
    chromosomes += crossing(chromosomes[i1], chromosomes[j1])
    chromosomes += crossing(chromosomes[i2], chromosomes[j2])
    chromosomes += crossing(chromosomes[i3], chromosomes[j3])
    chromosomes += crossing(chromosomes[i4], chromosomes[j4])
    # if SELECTED_INDIVIDUALS == index+1:
    #     return
    # for i in range(index+1, SELECTED_INDIVIDUALS):
    #     chromosomes += (crossing(chromosomes[index], chromosomes[i]))
    # crossover(chromosomes, index+1)


# 交叉算子采用多点交叉法
def crossing(chromosome1, chromosome2):
    i = randint(0, CHROMOSOME_LENGTH//2-1)
    j = randint(CHROMOSOME_LENGTH//2, CHROMOSOME_LENGTH)

    # c1 = chromosome1[:i] + chromosome2[i:]
    # c2 = chromosome2[:i] + chromosome1[i:]

    # c3 = chromosome1[:j] + chromosome2[j:]
    # c4 = chromosome2[:j] + chromosome1[j:]

    c5 = chromosome1[:i] + chromosome2[i:j] + chromosome1[j:]
    c6 = chromosome2[:i] + chromosome1[i:j] + chromosome2[j:]

    c7 = chromosome1[j:] + chromosome1[:i] + chromosome2[i:j]
    c8 = chromosome2[j:] + chromosome2[:i] + chromosome1[i:j]

    c9 = chromosome2[i:j] + chromosome1[:i] + chromosome1[j:]
    c10 = chromosome1[i:j] + chromosome2[:i] + chromosome2[j:]

    # return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
    return [c5, c6, c7, c8, c9, c10]


# 选择最高的3个个体【染色体】，同时返回对应的棋盘Puzzle
# 选择操作采用基于
# “适应值比例”
# 的选择
def selection(chromosomes):
    res = []
    for chromosome in chromosomes:
        tmp = puzzle_from_chromosome(chromosome)
        res.append([tmp[0], tmp[1]])
    # 按照适应度大小排序
    res.sort(key=lambda x: x[1].fitness())
    return res[:SELECTED_INDIVIDUALS]


# def getStrOfChromosome(chromosome):
#     txt = ', '.join([x.name for x in chromosome])
#     return f"{txt}"

def update(integer):
    global CHROMOSOME_LENGTH
    CHROMOSOME_LENGTH = integer


'''def solutions():
    global CHROMOSOME_LENGTH
    generation, numOfIncrement, bestmdis = 0, 0, 36
    bestSelection = []
    best_puzzle = np.array(begin_state)
    population = initialize_population()
    while generation < MAX_GENERATION:
        generation += 1
        print("population当前的长度是:")
        print(len(population))
        # mutasyon
        for item in (population):
            # print("population中的item是：\n")
            # print(item)
            mutation(item)

        # seçilim
        slct = selection(population)
        # 挑选出适应度最高的3个个体
        print("挑选出来适应度最高的三个个体是之后")
        # for item in slct:
        #     print(item[0], end=',')
        current_puzzle = slct[0][1].puzzle
        mdis = slct[0][1].fitness()
        population = [item[0] for item in slct]
        # population现在只有3个个体
        print("现在的population长度是：")
        print(len(population))
        print("每个个体的染色体的长度是：")
        print(len(population[0]))
        # en iyi seçim

        if (mdis < bestmdis):
            bestmdis = mdis
            bestSelection = slct[0]
            best_puzzle = slct[0][1].puzzle

        # kromozom uzunluğunu arttırma

        if (generation//ADDED_INTERVAL_GEN > numOfIncrement):
            numOfIncrement += 1
            CHROMOSOME_LENGTH += ADDED_LENGTH

        # fit_num = StringVar()
        # gen_num = StringVar()
        # gen_num.set(generation)
        # fit_num.set(mdis)
        print(f"generation: {generation} | fitness: {mdis}\n")
        # print(f"Current population:\n {population}")
        # info = {}
        # info['gen_num'] = generation
        # info['fit_num'] = mdis
        # info['puz'] = current_puzzle
        # info_all.append(info)
        # generation_num.append(generation)
        # fitness_num.append(mdis)
        # current_puzzle_status.append(current_puzzle)

        # nfo['gen_num'] = generation_num
        # nfo['fit_num'] = fitness_num
        # nfo['puz'] = current_puzzle_status
        #  typeof:current_puzzle
        # <class 'numpy.ndarray'>
        print(f'puzzle:{best_puzzle}')
        # for i in range(3):
        #     for j in range(3):
        #         print(current_puzzle[i, j], end=',')
        # Sonuç bulundu
        if (mdis == 0):
            break
        # 染色体交叉之后将种群数量从3变成了33
        # 查看交叉操作的具体代码
        crossover(population)
        print("染色体交叉之后的population长度是: " + str(len(population)))

    print("---------------------------")
    print("begin_state")
    print(begin_state)
    print()
    print("end_state")
    print(end_state)
    print("---------------------------")
    print(f"fitness: {bestSelection[1].fitness()}")
    print(f"best chromosome\n{getStrOfChromosome(bestSelection[0])}")
    print(f"final status\n{bestSelection[1]}")'''


# if __name__ == "__main__":
#     solutions()
#
# TODO:
#  1. 将astar算法整合至图形界面
#  2. astar算法重要的参数同时及时显示出来
#  3. 后期美化，如果有时间，而且代价小的话
