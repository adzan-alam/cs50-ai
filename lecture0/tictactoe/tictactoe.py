"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                x_count += 1
            if board[i][j] == O:
                o_count += 1

    if x_count == o_count:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    user = player(board)
    new_board = initial_state()

    # make a copy of current board
    for i in range(3):
        for j in range(3):
            new_board[i][j] = board[i][j]

    # now perform the action
    a, b = action
    if (
        (a == 0 or a == 1 or a == 2)
        and (b == 0 or b == 1 or b == 2)
        and board[a][b] == EMPTY
    ):
        new_board[a][b] = user
    else:
        raise ValueError("action co-ordinates not in bound of board")
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    win_configs = [
        # Horizontal Rows
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # Vertical Columns
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # Diagonals
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]

    for config in win_configs:
        (r1, c1), (r2, c2), (r3, c3) = config
        if (
            board[r1][c1] == board[r2][c2] == board[r3][c3]
            and board[r1][c1] is not None
        ):
            return board[r1][c1]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return winner(board) != None or len(actions(board)) == 0


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)
    best_action = None

    if current_player == X:
        # X wants to maximize the score
        best_score = -math.inf

        for action in actions(board):
            score = min_value(result(board, action))
            if score > best_score:
                best_score = score
                best_action = action

    else:
        # O wants to minimize the score
        best_score = math.inf
        for action in actions(board):
            score = max_value(result(board, action))
            if score < best_score:
                best_score = score
                best_action = action

    return best_action


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -math.inf
    for action in actions(board):
        transition = result(board, action)
        v = max(v, min_value(transition))
        if v == 1:
            return v

    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = math.inf
    for action in actions(board):
        transition = result(board, action)
        v = min(v, max_value(transition))
        if v == -1:
            return v

    return v
