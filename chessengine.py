import chess
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import OrderedDict
from operator import itemgetter
board = chess.Board()
print(board)

# global variable
white_player = False #False = bot play, True = human play
black_player = False
counter = 0 # dùng để tìm số lần gọi negamax để đánh giá hiệu năng
checkmate = 9999
DEPTH = 3 #Độ sâu max
#Piece square table
#bàn cờ chạy từ dưới lên trên còn table chạy từ trên xuống dưới
PT = [
 0,  0,  0,  0,  0,  0,  0,  0,
 5, 10, 10,-20,-20, 10, 10,  5,
 5, -5,-10,  0,  0,-10, -5,  5,
 0,  0,  0, 20, 20,  0,  0,  0,
 5,  5, 10, 25, 25, 10,  5,  5,
10, 10, 20, 30, 30, 20, 10, 10,
50, 50, 50, 50, 50, 50, 50, 50,
 0,  0,  0,  0,  0,  0,  0,  0  ]
NT = [
-50,-40,-30,-30,-30,-30,-40,-50,
-40,-20,  0,  5,  5,  0,-20,-40,
-30,  5, 10, 15, 15, 10,  5,-30,
-30,  0, 15, 20, 20, 15,  0,-30,
-30,  5, 15, 20, 20, 15,  5,-30,
-30,  0, 10, 15, 15, 10,  0,-30,
-40,-20,  0,  0,  0,  0,-20,-40,
-50,-40,-30,-30,-30,-30,-40,-50 ]
BT = [
-20,-10,-10,-10,-10,-10,-10,-20,
-10,  5,  0,  0,  0,  0,  5,-10,
-10, 10, 10, 10, 10, 10, 10,-10,
-10,  0, 10, 10, 10, 10,  0,-10,
-10,  5,  5, 10, 10,  5,  5,-10,
-10,  0,  5, 10, 10,  5,  0,-10,
-10,  0,  0,  0,  0,  0,  0,-10,
-20,-10,-10,-10,-10,-10,-10,-20  ]
RT =  [
 0, 0, 0, 5, 5, 0, 0, 0,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
-5, 0, 0, 0, 0, 0, 0, -5,
 5, 10, 10, 10, 10, 10, 10, 5,
 0, 0, 0, 0, 0, 0, 0, 0  ]
QT = [
-20,-10,-10, -5, -5,-10,-10,-20,
-10,  0,  0,  0,  0,  0,  0,-10,
-10,  5,  5,  5,  5,  5,  0,-10,
  0,  0,  5,  5,  5,  5,  0, -5,
 -5,  0,  5,  5,  5,  5,  0, -5,
-10,  0,  5,  5,  5,  5,  0,-10,
-10,  0,  0,  0,  0,  0,  0,-10,
-20,-10,-10, -5, -5,-10,-10,-20  ]
KT = [
20, 30, 10,  0,  0, 10, 30, 20,
 20, 20,  0,  0,  0,  0, 20, 20,
-10,-20,-20,-20,-20,-20,-20,-10,
-20,-30,-30,-40,-40,-30,-30,-20,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30  ]
# victim           P    N    B    R    Q    K
# attacker    P   105, 205, 305, 405, 505, 605
#             N   104, 204, 304, 404, 504, 604
#             B   103, 203, 303, 403, 503, 603
#             R   102, 202, 302, 402, 502, 602
#             Q   101, 201, 301, 401, 501, 601
#             K   100, 200, 300, 400, 500, 600
MVVLVA = [
105, 205, 305, 405, 505, 605,
104, 204, 304, 404, 504, 604,
103, 203, 303, 403, 503, 603,
102, 202, 302, 402, 502, 602,
101, 201, 301, 401, 501, 601,
100, 200, 300, 400, 500, 600  ]

model_path = r'C:\Users\Admin\PycharmProjects\chess\venv\estimator\1638985597'
model = tf.saved_model.load(model_path)


def predict(df_eval, imported_model):
    """
    df_eval -- pd.DataFrame
    imported_model -- tf.saved_model
    """
    col_names = df_eval.columns
    dtypes = df_eval.dtypes
    predictions = []
    for row in df_eval.iterrows():
        example = tf.train.Example()
        for i in range(len(col_names)):
            dtype = dtypes[i]
            col_name = col_names[i]
            value = row[1][col_name]
            if dtype == 'object':
                value = bytes(value, 'utf-8')
                example.features.feature[col_name].bytes_list.value.extend([value])
            elif dtype == 'float':
                example.features.feature[col_name].float_list.value.extend([value])
            elif dtype == 'int':
                example.features.feature[col_name].int64_list.value.extend([value])
        predictions.append(imported_model.signatures['predict'](examples=tf.constant([example.SerializeToString()])))
    return predictions


def get_board_features(board):
    """
    board -- chess.Board()
    """
    board_features = []
    for square in chess.SQUARES:
        board_features.append(str(board.piece_at(square)))
    return board_features


def get_move_features(move):
    """
    move -- chess.Move
    """
    from_ = np.zeros(64)
    to_ = np.zeros(64)
    from_[move.from_square] = 1
    to_[move.to_square] = 1
    return from_, to_


def get_possible_moves_data(current_board):
    """
    current_board -- chess.Board()
    """
    data = []
    moves = list(current_board.legal_moves)
    for move in moves:
        from_square, to_square = get_move_features(move)
        row = np.concatenate((get_board_features(current_board), from_square, to_square))
        data.append(row)

    board_feature_names = chess.SQUARE_NAMES
    move_from_feature_names = ['from_' + square for square in chess.SQUARE_NAMES]
    move_to_feature_names = ['to_' + square for square in chess.SQUARE_NAMES]

    columns = board_feature_names + move_from_feature_names + move_to_feature_names

    df = pd.DataFrame(data=data, columns=columns)

    for column in move_from_feature_names:
        df[column] = df[column].astype(float)
    for column in move_to_feature_names:
        df[column] = df[column].astype(float)
    return df


def find_best_moves(current_board, model, proportion):
    """
    current_board -- chess.Board()
    model -- tf.saved_model
    proportion -- proportion of best moves returned
    """
    moves = list(current_board.legal_moves)
    df_eval = get_possible_moves_data(current_board)
    predictions = predict(df_eval, model)
    good_move_probas = []

    for prediction in predictions:
        proto_tensor = tf.make_tensor_proto(prediction['probabilities'])
        proba = tf.make_ndarray(proto_tensor)[0][1]
        good_move_probas.append(proba)

    dict_ = dict(zip(moves, good_move_probas))
    dict_ = OrderedDict(sorted(dict_.items(), key=itemgetter(1), reverse=True))

    best_moves = list(dict_.keys())

    return best_moves[0:int(len(best_moves) * proportion)]

# Đánh giá bàn cờ
def evaluate_board():

    # Kiểm tra xem có checkmate để trả điểm cao, stalemate trả lại 0 suy ra tránh stalemate
    if board.is_checkmate():
        if board.turn:
            return -checkmate
        else:
            return checkmate
    elif board.is_stalemate():
        return 0

    # In hoa là trắng không in hoa là đen
    # gán số lượng quân cờ
    P = len(board.pieces(chess.PAWN, chess.WHITE))
    p = len(board.pieces(chess.PAWN, chess.BLACK))
    N = len(board.pieces(chess.KNIGHT, chess.WHITE))
    n = len(board.pieces(chess.KNIGHT, chess.BLACK))
    B = len(board.pieces(chess.BISHOP, chess.WHITE))
    b = len(board.pieces(chess.BISHOP, chess.BLACK))
    R = len(board.pieces(chess.ROOK, chess.WHITE))
    r = len(board.pieces(chess.ROOK, chess.BLACK))
    Q = len(board.pieces(chess.QUEEN, chess.WHITE))
    q = len(board.pieces(chess.QUEEN, chess.BLACK))

    # material = (white_pieces - black_pieces) * (giá trị của quân cờ)
    material = (P - p) * 100 + (N - n) * 320 + (B - b) * 330 + (R - r) * 500 + (Q - q) * 900

    Pawnpos = sum([PT[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    Pawnpos = Pawnpos + sum([-PT[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])

    Knightpos = sum([NT[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    Knightpos = Knightpos + sum([-NT[chess.square_mirror(i)] for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    Bishoppos = sum([BT[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    Bishoppos = Bishoppos + sum([-BT[chess.square_mirror(i)] for i in board.pieces(chess.BISHOP, chess.BLACK)])

    Rookpos = sum([RT[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    Rookpos = Rookpos + sum([-RT[chess.square_mirror(i)] for i in board.pieces(chess.ROOK, chess.BLACK)])

    Queenpos = sum([QT[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    Queenpos = Queenpos + sum([-QT[chess.square_mirror(i)] for i in board.pieces(chess.QUEEN, chess.BLACK)])

    Kingpos = sum([KT[i] for i in board.pieces(chess.KING, chess.WHITE)])
    Kingpos = Kingpos + sum([-KT[chess.square_mirror(i)] for i in board.pieces(chess.KING, chess.BLACK)])

    evalutaion = material + Pawnpos + Knightpos + Bishoppos + Rookpos + Queenpos + Kingpos
    return evalutaion


def addmove():
    list = []
    bestscore = -checkmate
    move_value = {}
    legal_moves = find_best_moves(board, model, 0.5)
    for moves in legal_moves:
        if board.piece_type_at(moves.to_square) != None:
            movescore = MVVLVA[(board.piece_type_at(moves.from_square) - 1) * 6 + board.piece_type_at(moves.to_square) - 1]
            move_value[moves] = movescore
        else:
            move_value[moves] = 0
    sort_orders = sorted(move_value.items(), key=lambda x: x[1], reverse=True)
    for i in sort_orders:
        list.append(i[0])
    return list


def Negamax(depth, turn_multiplier, alpha, beta):
    global themove
    global counter
    counter = counter + 1
    if depth == 0:
        return turn_multiplier * evaluate_board()
    else:
        lim_score = -checkmate
        legal_moves = addmove()
        for moves in legal_moves:

            board.push(moves)
            score = -Negamax(depth -1, -turn_multiplier, -beta, -alpha)
            if score > lim_score:
                lim_score = score
                if depth == DEPTH:
                    themove = moves
            if lim_score > alpha:
                alpha = lim_score
            board.pop()
            if alpha >= beta:
                break
        return lim_score


def Bestmove():
    global themove
    themove = None
    Negamax(DEPTH, 1 if board.turn else -1, -checkmate, checkmate)
    return themove

while board.is_stalemate() == 0 and board.is_checkmate() == 0 : #check for stale mate or check mate
    if board.turn:
        if white_player == True:
            try:
                legalmoves = str(board.legal_moves) # cast legal moves to string
                print('Legal moves: ',legalmoves[33:])
                print("input your move :")
                move = input()
                if move == 'retry': #type retry to unmake move
                    board.pop()
                    board.pop()
                    print(board)
                elif move == 'quit': #quit game
                    break
                else: # make your move
                    board.push_san(move)
                    print(board)
            except:
                    print('Illegal move, try input again')
                    continue
        else:
            #findmoveminmax(DEPTH)
            White_AImove = Bestmove()
            print(counter)
            legalmoves = str(board.legal_moves)  # cast legal moves to string
            print(legalmoves[33:])
            board.push(White_AImove)
            print(board)
    else:
        if black_player == True:
            try:
                legalmoves = str(board.legal_moves) # cast legal moves to string
                print(legalmoves[33:])
                print("input your move :")
                move = input()
                if move == 'retry': #type retry to unmake move
                    board.pop()
                    board.pop()
                    print(board)
                elif move == 'quit': #quit game
                    break
                else: # make your move
                    board.push_san(move)
                    print(board)
            except:
                    print('Illegal move, try input again')
                    continue
        else:
            BAImove = Bestmove()
            print(counter)
            legalmoves = str(board.legal_moves)  # cast legal moves to string
            print(legalmoves[33:])
            board.push(BAImove)
            print(board)
outcome = board.outcome() # display reason for end the game and winner: false = black, true = white
print(outcome)



