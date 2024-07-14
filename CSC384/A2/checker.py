import argparse
from copy import deepcopy
import sys
import time
import numpy
import math

cache = {} # you can use this to implement state caching!
explored = set()

class State:
    # This class is used to represent a state.
    # board : a list of lists that represents the 8*8 board
    def __init__(self, board, is_red):

        self.board = board
        self.width = 8
        self.height = 8
        self.is_red = is_red

    def display(self):
        for i in self.board:
            for j in i:
                print(j, end="")
            print("")
        print("")

    def move_down_left(self, y, x):
        if y < 7 and x > 0 and self.board[y + 1][x - 1] == '.':
            if self.is_red:
                if self.board[y][x] == 'R':
                    copy = deepcopy(self.board)
                    copy[y + 1][x - 1] = 'R'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
            else:
                if self.board[y][x] == 'B':
                    copy = deepcopy(self.board)
                    copy[y + 1][x - 1] = 'B'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
                if self.board[y][x] == 'b':
                    copy = deepcopy(self.board)
                    if y == 6:
                        copy[y + 1][x - 1] = 'B'
                    else:
                        copy[y + 1][x - 1] = 'b'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
        return None

    def move_down_right(self, y, x):
        if y < 7 and x < 7 and self.board[y + 1][x + 1] == '.':
            if self.is_red:
                if self.board[y][x] == 'R':
                    copy = deepcopy(self.board)
                    copy[y + 1][x + 1] = 'R'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
            else:
                if self.board[y][x] == 'B':
                    copy = deepcopy(self.board)
                    copy[y + 1][x + 1] = 'B'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
                if self.board[y][x] == 'b':
                    copy = deepcopy(self.board)
                    if y == 6:
                        copy[y + 1][x + 1] = 'B'
                    else:
                        copy[y + 1][x + 1] = 'b'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
        return None

    def move_up_left(self, y, x):
        if y > 0 and x > 0 and self.board[y - 1][x - 1] == '.':
            if self.is_red:
                if self.board[y][x] == 'r':
                    copy = deepcopy(self.board)
                    if y == 6:
                        copy[y - 1][x - 1] = 'R'
                    else:
                        copy[y - 1][x - 1] = 'r'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
                if self.board[y][x] == 'R':
                    copy = deepcopy(self.board)
                    copy[y - 1][x - 1] = 'R'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
            else:
                if self.board[y][x] == 'b':
                    copy = deepcopy(self.board)
                    copy[y - 1][x - 1] = 'b'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
        return None

    def move_up_right(self, y, x):
        if y > 0 and x < 7 and self.board[y - 1][x + 1] == '.':
            if self.is_red:
                if self.board[y][x] == 'r':
                    copy = deepcopy(self.board)
                    if y == 1:
                        copy[y - 1][x + 1] = 'R'
                    else:
                        copy[y - 1][x + 1] = 'r'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
                if self.board[y][x] == 'R':
                    copy = deepcopy(self.board)
                    copy[y - 1][x + 1] = 'R'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
            else:
                if self.board[y][x] == 'b':
                    copy = deepcopy(self.board)
                    copy[y - 1][x + 1] = 'b'
                    copy[y][x] = '.'
                    return State(copy, not self.is_red)
        return None

    def jump_down_left(self, y, x):
        if y < 6 and x > 1 and self.board[y + 2][x - 2] == '.':
            if self.is_red:
                if self.board[y][x] == 'R' and self.board[y + 1][x - 1] in 'bB':
                    copy = deepcopy(self.board)
                    copy[y + 2][x - 2] = 'R'
                    copy[y][x] = '.'
                    copy[y + 1][x - 1] = '.'
                    return State(copy, self.is_red)
            else:
                if self.board[y][x] == 'B' and self.board[y + 1][x - 1] in 'rR':
                    copy = deepcopy(self.board)
                    copy[y + 2][x - 2] = 'B'
                    copy[y][x] = '.'
                    copy[y + 1][x - 1] = '.'
                    return State(copy, self.is_red)
                if self.board[y][x] == 'b' and self.board[y + 1][x - 1] in 'rR':
                    copy = deepcopy(self.board)
                    if y == 5:
                        copy[y + 2][x - 2] = 'B'
                    else:
                        copy[y + 2][x - 2] = 'b'
                    copy[y][x] = '.'
                    copy[y + 1][x - 1] = '.'
                    return State(copy, self.is_red)
        return None

    def jump_down_right(self, y, x):
        if y < 6 and x < 6 and self.board[y + 2][x + 2] == '.':
            if self.is_red:
                if self.board[y][x] == 'R' and self.board[y + 1][x + 1] in 'bB':
                    copy = deepcopy(self.board)
                    copy[y + 2][x + 2] = 'R'
                    copy[y][x] = '.'
                    copy[y + 1][x + 1] = '.'
                    return State(copy, self.is_red)
            else:
                if self.board[y][x] == 'B' and self.board[y + 1][x + 1] in 'rR':
                    copy = deepcopy(self.board)
                    copy[y + 2][x + 2] = 'B'
                    copy[y][x] = '.'
                    copy[y + 1][x + 1] = '.'
                    return State(copy, self.is_red)
                if self.board[y][x] == 'b' and self.board[y + 1][x + 1] in 'rR':
                    copy = deepcopy(self.board)
                    if y == 5:
                        copy[y + 2][x + 2] = 'B'
                    else:
                        copy[y + 2][x + 2] = 'b'
                    copy[y][x] = '.'
                    copy[y + 1][x + 1] = '.'
                    return State(copy, self.is_red)
        return None

    def jump_up_left(self, y, x):
        if y > 1 and x > 1 and self.board[y - 2][x - 2] == '.':
            if self.is_red:
                if self.board[y][x] == 'r' and self.board[y - 1][x - 1] in 'bB':
                    copy = deepcopy(self.board)
                    if y == 2:
                        copy[y - 2][x - 2] = 'R'
                    else:
                        copy[y - 2][x - 2] = 'r'
                    copy[y][x] = '.'
                    copy[y - 1][x - 1] = '.'
                    return State(copy, self.is_red)
            else:
                if self.board[y][x] == 'b' and self.board[y - 1][x - 1] in 'rR':
                    copy = deepcopy(self.board)
                    copy[y - 2][x - 2] = 'b'
                    copy[y][x] = '.'
                    copy[y - 1][x - 1] = '.'
                    return State(copy, self.is_red)
        return None

    def jump_up_right(self, y, x):
        if y > 1 and x < 6 and self.board[y - 2][x + 2] == '.':
            if self.is_red:
                if self.board[y][x] == 'r' and self.board[y - 1][x + 1] in 'bB':
                    copy = deepcopy(self.board)
                    if y == 2:
                        copy[y - 2][x + 2] = 'R'
                    else:
                        copy[y - 2][x + 2] = 'r'
                    copy[y][x] = '.'
                    copy[y - 1][x + 1] = '.'
                    return State(copy, self.is_red)
            else:
                if self.board[y][x] == 'b' and self.board[y - 1][x + 1] in 'rR':
                    copy = deepcopy(self.board)
                    copy[y - 2][x + 2] = 'b'
                    copy[y][x] = '.'
                    copy[y - 1][x + 1] = '.'
                    return State(copy, self.is_red)
        return None

    def can_cap(self):
        for y in range(8):
            for x in range(8):
                if self.board[y][x] in 'rR':
                    if self.jump_down_left(y, x) is not None:
                        return True
                    if self.jump_down_right(y, x) is not None:
                        return True
                if self.board[y][x] in 'bB':
                    if self.jump_up_left(y, x) is not None:
                        return True
                    if self.jump_up_right(y, x) is not None:
                        return True
        return False

    def find_cap(self, y, x, cap):
        move = []
        if not self.can_cap() and cap == True:
            move.append(self)
        m1 = self.jump_down_left(y, x)
        if m1 is not None:
            move.append(m1)
            if m1.can_cap():
                move.extend(m1.find_cap(y + 2, x - 2, True))
        m2 = self.jump_down_right(y, x)
        if m2 is not None:
            move.append(m2)
            if m2.can_cap():
                move.extend(m2.find_cap(y + 2, x + 2, True))
        m3 = self.jump_up_left(y, x)
        if m3 is not None:
            move.append(m3)
            if m3.can_cap():
                move.extend(m3.find_cap(y - 2, x - 2, True))
        m4 = self.jump_up_right(y, x)
        if m4 is not None:
            move.append(m4)
            if m4.can_cap():
                move.extend(m4.find_cap(y - 2, x + 2, True))
        res = []
        for m in move:
            res.append(State(m.board, not self.is_red))
        return res

    def find_move(self, y, x):
        move = []
        m1 = self.move_down_left(y, x)
        if m1 is not None:
            move.append(m1)
        m2 = self.move_down_right(y, x)
        if m2 is not None:
            move.append(m2)
        m3 = self.move_up_left(y, x)
        if m3 is not None:
            move.append(m3)
        m4 = self.move_up_right(y, x)
        if m4 is not None:
            move.append(m4)
        return move

    def find_all_moves(self):
        jump = []
        move = []
        for y in range(8):
            for x in range(8):
                jump.extend(self.find_cap(y, x))
                move.extend(self.find_move(y, x))
        if len(jump) > 0:
            return jump
        else:
            return move

    def heuristic(self):
        h = 0
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == 'r':
                    h += 1
                if self.board[y][x] == 'b':
                    h -= 1
                if self.board[y][x] == 'R':
                    h += 10
                if self.board[y][x] == 'B':
                    h -= 10
        return h

    def alphabeta(self, alpha=-math.inf, beta=math.inf, depth=0, limit=10,sort_heuristic = True):
        best_suc = None
        global explored
        if (self.board, self.is_red) in explored:
            if self.is_red:
                return None, -math.inf
            if self.find_all_moves() != []:
                return None, self.heuristic()
            else:
                return None, math.inf
        explored.add((self.board, self.is_red))
        if self.is_red:
            best = -math.inf
            move = self.find_all_moves()
            if sort_heuristic:
                move.sort(key=lambda x: x.heuristic())
                move.reverse()
        else:
            best = math.inf
            move = self.find_all_moves()
            if sort_heuristic:
                move.sort(key=lambda x: x.heuristic())
        if depth == limit and move == []:
            return best_suc, best
        if depth == limit and move != []:
            return best_suc, self.heuristic()
        for suc in move:
            next_suc, next_val = suc.alphabeta(alpha, beta, depth + 1, limit, sort_heuristic)
            if self.is_red:
                if best < next_val:
                    best = next_val
                    best_suc = suc
                if best >= beta:
                    return best_suc, best
                alpha = max(alpha, best)
            else:
                if best > next_val:
                    best = next_val
                    best_suc = suc
                if best <= alpha:
                    return best_suc, best
                beta = min(beta, best)
        return best_suc, best



def get_opp_char(player):
    if player in ['b', 'B']:
        return ['r', 'R']
    else:
        return ['b', 'B']

def get_next_turn(curr_turn):
    if curr_turn == 'r':
        return 'b'
    else:
        return 'r'


def read_from_file(filename):

    f = open(filename)
    lines = f.readlines()
    board = tuple([[str(x) for x in l.rstrip()] for l in lines])
    f.close()

    return board

def write_to_file(state,output_file):
    with open(output_file, 'w') as f:
        for i in range(7):
            r = ''
            for j in state.board[i]:
                r += str(j)
            f.write(r + '\n')
        r=''
        for j in state.board[7]:
            r += str(j)
        f.write(r)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board, True)
    turn = 'r'
    ctr = 0
    best_suc, best_value = state.alphabeta(-math.inf, math.inf,0,10,True)
    if best_suc is not None:
        write_to_file(best_suc, args.outputfile)
    else:
        write_to_file(state, args.outputfile)

    sys.stdout = open(args.outputfile, 'w')

    sys.stdout = sys.__stdout__

