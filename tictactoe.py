# X is the NN
# o is random

import numpy as np
import random
import copy
# from icecream import ic
# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with mean 0
        self.input_size = input_size # 18
        self.hidden_size = hidden_size # 30
        self.output_size = output_size # 1

        # Initialize weights and biases for hidden and output layers
        self.hidden_weights = np.random.uniform(size=(input_size, hidden_size)) - 0.5
        self.hidden_bias = np.random.uniform(size=(1, hidden_size)) - 0.5
        self.output_weights = np.random.uniform(size=(hidden_size, output_size)) - 0.5
        self.output_bias = np.random.uniform(size=(1, output_size)) - 0.5

# GET BOARD AND RETURN BOARD SCORE
    def forward(self, inputs):
        # Forward Propagation
        self.hidden_layer_activation = np.dot([inputs], self.hidden_weights)
        # ic(self.hidden_layer_activation)
        self.hidden_layer_activation += self.hidden_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.output_weights)
        self.output_layer_activation += self.output_bias
        predicted_output = sigmoid(self.output_layer_activation)

        return predicted_output

    def backward(self, inputs, win, predicted_output, learning_rate):
        # Backpropagation
        error = win - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)

        # Updating Weights and Biases
        self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        self.hidden_weights += np.array([inputs]).T.dot(d_hidden_layer) * learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# TicTacToe Game Class
class TicTacToe:
    def __init__(self):
        self.board = [['.' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

    # Make a random move in the game
    def make_move(self, row, col):
        if self.board[row][col] == '.': #is empty
            self.board[row][col] = self.current_player # make a move
            if self.check_winner():
                # Update wins count and neural network
                global ww, ll
                if self.current_player == 'X':
                    ww += 1
                else:
                    ll += 1
                print(f"Player {self.current_player} wins! Xwins:Owins = {ww}:{ll} ")
                self.update_nn()
                return True
            elif self.is_draw():
                print("It's a draw!")
                return True
            else:
                self.switch_player()
        return False

    # Switch player turns
    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    # Check for a winner
    def check_winner(self):
        # Check rows and columns for a win
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '.':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '.':
                return True

        # Check diagonals for a win
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '.' or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] != '.':
            return True
        return False

    # Check if it's a draw
    def is_draw(self):
        return all(cell != '.' for row in self.board for cell in row)

    # Convert board state to an array for the neural network
    def brd2arr(self, brd):
        arr = []
        for col in range(3):
            for row in range(3):
                c = brd[row][col]
                if c == 'X':
                    arr += [1, 0]
                elif c == 'O':
                    arr += [0, 1]
                else:
                    arr += [0, 0]
        return arr

    # Get the best move for the neural network
    def   nn_move(self):
        best_score = -999
        best_row = 0
        best_col = 0
        best_s = []
        for col in range(3):
            for row in range(3):
                if self.board[row][col] == '.':
                    s = copy.deepcopy(self.board)
                    s[row][col] = 'X'
                    arr = self.brd2arr(s)
                    score = nn.forward(arr)# CHECK IF BEST MOVE IN nn
                    # check for the best move
                    if score > best_score:
                        best_score = score
                        best_row = row
                        best_col = col
                        best_s = s.copy()

        self.history.append(best_s)
        return best_row, best_col

    # Update the neural network based on game history
    def update_nn(self):
        win = 0.0
        if self.current_player == 'X':
            win = 1.0
        learning_rate = 0.1
        self.history = reversed(self.history)
        for scenario in self.history:
            arr = self.brd2arr(scenario)
            score = nn.forward(arr)
            nn.backward(arr, win, score, learning_rate)
            learning_rate *= 0.7

    # Play a game between the neural network and a random player
    def play_random_game(self):
        self.history = []
        while True:
            if self.current_player == 'X':
                row, col = self.nn_move()
            else:
                row, col = random.randint(0, 2), random.randint(0, 2)
            if self.make_move(row, col):
                self.print_board()
                break
            elif self.is_draw():
                print("It's a draw!")
                self.print_board()
                break

# Create a neural network
nn = NeuralNetwork(18, 30, 1)

# Initialize variables to track wins for X and O players
ww = 0
ll = 0

# Run 1000 games between the neural network and a random player
for i in range(1000):
    print(f'game {i}')
    game = TicTacToe()
    game.play_random_game()