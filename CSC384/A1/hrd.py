from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
from queue import PriorityQueue
import sys
sys.setrecursionlimit(10000)

#====================================================================================

char_goal = '1'
char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = 5

        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()


    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()

    def write(self):
        """
        Print out the current board.

        """
        board = ''
        for i, line in enumerate(self.grid):
            for ch in line:
                board = board + ch
            board += '\n'

        return board
        

class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.
        self.cost = 0
        # check cycles

    def __lt__(self, successors):
        return self.depth > successors.depth

    def __repr__(self):
        return '{}'.format(self.board.grid)

    def __gt__(self, other):
        return self.cost > other.cost



def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    g_found = False

    for line in puzzle_file:

        for x, ch in enumerate(line):

            if ch == '^': # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v'))
            elif ch == '<': # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h'))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None))
                    g_found = True
        line_index += 1

    puzzle_file.close()

    board = Board(pieces)
    
    return board

def goal_state(state):
    grid = state.board.grid
    if grid[3][2] == char_goal and grid[4][1] == char_goal:
        return True
    return False

# def cost(state, explored):

def find_successors(state):
    empty_board = []
    suc = []
    grid = state.board.grid
    double = False
    count = [False]
    orient = 'hor'

    for y in range(5):
        for x in range(4):
            if grid[y][x] == '.':
                empty_board.append([y, x])

    if empty_board[0][0] == empty_board[1][0] and abs(empty_board[0][1] - empty_board[1][1]) == 1:
        double = True
    elif empty_board[0][1] == empty_board[1][1] and abs(empty_board[0][0] - empty_board[1][0]) == 1:
        double = True
        orient = 'ver'

    def move(x, y, xdir, ydir):
        if grid[y][x] == char_single:
            pieces = deepcopy(state.board.pieces)
            for i in pieces:
                if i.is_single and i.coord_x == x and i.coord_y == y:
                    i.coord_x += xdir
                    i.coord_y += ydir
                    to_add = State(Board(pieces), state.f, state.depth + 1, state)
                    suc.append(to_add)
                    break

        elif grid[y][x] == '<':
            pieces = deepcopy(state.board.pieces)
            if not ydir:
                for i in pieces:
                    if i.coord_x == x and i.coord_y == y:
                        i.coord_x += xdir
                        i.coord_y += ydir
                        to_add = State(Board(pieces), state.f, state.depth + 1, state)
                        suc.append(to_add)
                        break
            else:
                if grid[y + ydir][x + 1] == '.':
                    for i in pieces:
                        if i.coord_x == x and i.coord_y == y:
                            i.coord_x += xdir
                            i.coord_y += ydir
                            to_add = State(Board(pieces), state.f, state.depth + 1, state)
                            suc.append(to_add)
                            break

        elif grid[y][x] == '>':
            pieces = deepcopy(state.board.pieces)
            if not ydir:
                for i in pieces:
                    if i.coord_x == x - 1 and i.coord_y == y:
                        i.coord_x += xdir
                        i.coord_y += ydir
                        to_add = State(Board(pieces), state.f, state.depth + 1, state)
                        suc.append(to_add)
                        break

        elif grid[y][x] == '^':
            pieces = deepcopy(state.board.pieces)
            if not xdir:
                for i in pieces:
                    if i.coord_x == x and i.coord_y == y:
                        i.coord_x += xdir
                        i.coord_y += ydir
                        to_add = State(Board(pieces), state.f, state.depth + 1, state)
                        suc.append(to_add)
                        break
            else:
                if grid[y + 1][x + xdir] == '.':
                    for i in pieces:
                        if i.coord_x == x and i.coord_y == y:
                            i.coord_x += xdir
                            i.coord_y += ydir
                            to_add = State(Board(pieces), state.f, state.depth + 1, state)
                            suc.append(to_add)
                            break

        elif grid[y][x] == 'v':
            pieces = deepcopy(state.board.pieces)
            if not xdir:
                for i in pieces:
                    if i.coord_x == x and i.coord_y == y - 1:
                        i.coord_x += xdir
                        i.coord_y += ydir
                        to_add = State(Board(pieces), state.f, state.depth + 1, state)
                        suc.append(to_add)
                        break

        elif grid[y][x] == char_goal:
            pieces = deepcopy(state.board.pieces)
            if double and not count[0]:
                if orient == 'ver':
                    if (y != 0 and grid[y + 1][x + xdir] == '.' and grid[y + 1][x] == char_goal) or (
                            y != 4 and grid[y - 1][x + xdir] == '.' and grid[y - 1][x] == char_goal):
                        for i in pieces:
                            if i.is_goal:
                                i.coord_x += xdir
                                i.coord_y += ydir
                                count[0] = True
                                to_add = State(Board(pieces), state.f, state.depth + 1, state)
                                suc.append(to_add)
                                break
                elif orient == 'hor':
                    if (x != 0 and grid[y + ydir][x + 1] == '.' and grid[y][x + 1] == char_goal) or (
                            x != 3 and grid[y + ydir][x - 1] == '.' and grid[y][x - 1] == char_goal):
                        for i in pieces:
                            if i.is_goal:
                                i.coord_x += xdir
                                i.coord_y += ydir
                                count[0] = True
                                to_add = State(Board(pieces), state.f, state.depth + 1, state)
                                suc.append(to_add)
                                break

    for i in empty_board:
        yc = i[0]
        xc = i[1]
        if yc != 0:
            move(xc, yc - 1, 0, +1)
        if yc != 4:
            move(xc, yc + 1, 0, -1)
        if xc != 0:
            move(xc - 1, yc, +1, 0)
        if xc != 3:
            move(xc + 1, yc, -1, 0)
    return suc


def dfs(initial_state):
    frontier = [initial_state]
    visited = set()
    depth = 0

    while frontier:
        current = heappop(frontier)
        depth += 1
        if goal_state(current):
            print('find solution')
            return get_sol(current)
        else:
            successors = find_successors(current)
            for successor in successors:
                if str(successor.board.grid) not in visited:
                    heappush(frontier, successor)
        visited.add(str(current.board.grid))

def get_sol(state):
    f = open(args.outputfile, 'w')
    path = ''
    while state.parent != None:
        path = str(state.board.write()) + "\n" + path
        state = state.parent
    path = str(state.board.write()) + "\n" + path
    path = path.rstrip()
    f.write(path)

def heuristic_func(state):
    index = []
    for y, line in enumerate(state.board.grid):
        for x, element in enumerate(line):
            if element == char_goal:
                index.append((y, x))
                break

    square = index[0]
    return abs(square[0] - 3) + abs(square[1] - 1)

# def cost(explored):
#     c = len(explored)
#     return c


def a_star(state):
    # frontier = [(state.cost+(heuristic_func(state)), state)]
    # visited = set()
    # depth = 0
    # while frontier:
    #     (cost, current) = heappop(frontier)
    #     depth += 1
    #     if goal_state(current):
    #         print('find solution')
    #         return get_sol(current)
    #     else:
    #         for successor in find_successors(current):
    #             if str(successor.board.grid) not in visited:
    #                 heappush(frontier, (heuristic_func(successor)+1+state.cost, successor))
    #     visited.add(str(current.board.grid))
    #     current.board.display()

    visited = set()
    frontier = PriorityQueue()
    state.cost = 0
    frontier.put((heuristic_func(state)+state.cost, state))
    while not frontier.empty():
        current = frontier.get()[1]
        if goal_state(current):
            print('find!')
            return get_sol(current)
        else:
            for successor in find_successors(current):
                if str(successor.board.grid) not in visited:
                    state.cost += 1
                    visited.add(str(successor))
                    frontier.put((heuristic_func(successor)+state.cost, successor))
                    print(heuristic_func(successor)+state.cost)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()

    # read the board from the file
    board = read_from_file(args.inputfile)
    initial_state = State(board, 0, 0, None)
    # print(initial_state.board.grid)
    # print(board.grid)
    # print(goal_state(initial_state))
    # print(find_successors(initial_state))
    if args.algo == 'dfs':
        dfs(initial_state)
    else:
        a_star(initial_state)
    # print(initial_state.board)
    # print(len(initial_state.board.grid[0]))



