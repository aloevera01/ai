"""
Tic Tac Toe Player
"""
from copy import deepcopy
from math import inf

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    num_X = 0
    num_O = 0

    for row in board:
        for el in row:
            if el == X:
                num_X += 1
            if el == O:
                num_O += 1

    if num_X > num_O:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = set()

    for i, row in enumerate(board):
        for j, el in enumerate(row):
            if el == EMPTY:
                possible_moves.add((i, j))
    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY or action[0] > 2 or action[1] > 2:
        raise ValueError
    new_board = deepcopy(board)
    p = player(new_board)
    new_board[action[0]][action[1]] = p
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for ind in range(len(board)):
        if board[ind][0] == board[ind][1] and board[ind][0] == board[ind][2] and board[ind][0] != EMPTY:
            return board[ind][0]
        if board[0][ind] == board[1][ind] and board[2][ind] == board[0][ind] and board[0][ind] != EMPTY:
            return board[0][ind]

    if ((board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0])) \
            and board[1][1] != EMPTY:
        return board[1][1]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if all(el != EMPTY for row in board for el in row):
        return True
    if winner(board) is not None:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def find_score(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return utility(board)

    possible_moves = actions(board)
    scores = []
    for move in possible_moves:
        new_board = result(board, move)
        scores.append(find_score(new_board))

    if player(board) == X:
        return max(scores)
    elif player(board) == O:
        return min(scores)


def minimax(board):
    if terminal(board):
        return None

    best_score = -inf
    best_move = None
    if player(board) == O:
        best_score = inf
        best_move = None

    possible_moves = actions(board)
    for move in possible_moves:
        new_board = result(board, move)
        score = find_score(new_board)
        if (best_score < score and player(board) == X) or (best_score > score and player(board) == O):
            best_score = score
            best_move = move

    return best_move