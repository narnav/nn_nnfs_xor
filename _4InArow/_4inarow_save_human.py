import numpy as np
import random
import copy
import json

LOAD_SAVED_MODEL = True
HUMAN_MODE = True

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.hidden_weights = np.random.uniform(size=(input_size, hidden_size)) - 0.5
        self.hidden_bias = np.random.uniform(size=(1, hidden_size)) - 0.5
        self.output_weights = np.random.uniform(size=(hidden_size, output_size)) - 0.5
        self.output_bias = np.random.uniform(size=(1, output_size)) - 0.5

    def forward(self, inputs):
        self.hidden_layer_activation = np.dot([inputs], self.hidden_weights)
        self.hidden_layer_activation += self.hidden_bias
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.output_weights)
        self.output_layer_activation += self.output_bias
        predicted_output = sigmoid(self.output_layer_activation)

        return predicted_output

    def backward(self, inputs, win, predicted_output, learning_rate):
        error = win - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        self.hidden_weights += np.array([inputs]).T.dot(d_hidden_layer) * learning_rate
        self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    def save_model(self, file_name):
        model = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'hidden_weights': self.hidden_weights.tolist(),
            'hidden_bias': self.hidden_bias.tolist(),
            'output_weights': self.output_weights.tolist(),
            'output_bias': self.output_bias.tolist()
        }
        with open(file_name, 'w') as json_file:
            json.dump(model, json_file)

    def load_model(self, file_name):
        with open(file_name, 'r') as json_file:
            model = json.load(json_file)
            self.input_size = model['input_size']
            self.hidden_size = model['hidden_size']
            self.output_size = model['output_size']
            self.hidden_weights = np.array(model['hidden_weights'])
            self.hidden_bias = np.array(model['hidden_bias'])
            self.output_weights = np.array(model['output_weights'])
            self.output_bias = np.array(model['output_bias'])

# 4 in a Row Game Class
class FourInARow:
    def __init__(self):
        self.board = [['.' for _ in range(7)] for _ in range(6)]
        self.current_player = 'X'

    def print_board(self):
        for row in self.board:
            print(' '.join([str(x) for x in row]))
        print()

    def make_move(self, col):
        for row in range(5, -1, -1):
            if self.board[row][col] == '.':
                self.board[row][col] = self.current_player
                if self.check_winner():
                    global ww, ll
                    if self.current_player == 'X':
                        ww += 1
                    else:
                        ll += 1
                    print(f"Player {self.current_player} wins! X wins:O wins = {ww}:{ll}")
                    self.update_nn()
                    return True
                elif self.is_draw():
                    print("It's a draw!")
                    return True
                else:
                    self.switch_player()
                return False

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        # Check rows for a win
        for row in range(6):
            for col in range(4):
                if self.board[row][col] == self.board[row][col + 1] == self.board[row][col + 2] == self.board[row][col + 3] != '.':
                    return True

        # Check columns for a win
        for col in range(7):
            for row in range(3):
                if self.board[row][col] == self.board[row + 1][col] == self.board[row + 2][col] == self.board[row + 3][col] != '.':
                    return True

        # Check positively sloped diagonals
        for row in range(3):
            for col in range(4):
                if self.board[row][col] == self.board[row + 1][col + 1] == self.board[row + 2][col + 2] == self.board[row + 3][col + 3] != '.':
                    return True

        # Check negatively sloped diagonals
        for row in range(3, 6):
            for col in range(4):
                if self.board[row][col] == self.board[row - 1][col + 1] == self.board[row - 2][col + 2] == self.board[row - 3][col + 3] != '.':
                    return True

        return False

    def is_draw(self):
        return all(cell != '.' for row in self.board for cell in row)

    def brd2arr(self, brd):
        arr = []
        for row in range(6):
            for col in range(7):
                c = brd[row][col]
                if c == 'X':
                    arr += [1, 0]
                elif c == 'O':
                    arr += [0, 1]
                else:
                    arr += [0, 0]
        return arr

    def nn_move(self):
        best_score = -999
        best_row = 0
        best_col = 0
        best_s = []
        for col in range(7):
            for row in range(6):
                if self.board[row][col] == '.':
                    s = copy.deepcopy(self.board)
                    s[row][col] = 'X'
                    arr = self.brd2arr(s)
                    score = nn.forward(arr)
                    if score > best_score:
                        best_score = score
                        best_row = row
                        best_col = col
                        best_s = s.copy()
        self.history.append(best_s)
        return best_row, best_col

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

    def play_random_game(self):
        self.history = []
        while True:
            if self.current_player == 'X':
                row, col = self.nn_move()
            else:
                col = random.randint(0, 6)
                for row in range(5, -1, -1):
                    if self.board[row][col] == '.':
                        break
            if self.make_move(col):
                self.print_board()
                break
            elif self.is_draw():
                print("It's a draw!")
                self.print_board()
                break

    def play_human_game(self):
        self.history = []
        while True:
            self.print_board()
            if self.current_player == 'X':
                row, col = self.nn_move()
            else:
                while True:
                    try:
                        col = int(input("Enter the column (0-6): "))
                        if col < 0 or col > 6 or self.board[0][col] != '.':
                            raise ValueError
                        break
                    except ValueError:
                        print("Invalid column, please try again.")
                for row in range(5, -1, -1):
                    if self.board[row][col] == '.':
                        break
            if self.make_move(col):
                self.print_board()
                break
            elif self.is_draw():
                print("It's a draw!")
                self.print_board()
                break

# Create a neural network
nn = NeuralNetwork(84, 30, 1)

# Initialize variables to track wins for X and O players
ww = 0
ll = 0

if not LOAD_SAVED_MODEL:
    # Run 1000 games between the neural network and a random player
    for i in range(1000):
        print(f'game {i}')
        game = FourInARow()
        game.play_random_game()

    # Save the model
    nn.save_model('4_in_a_row_model.json')
else:
    # To load the model in the future
    nn.load_model

if HUMAN_MODE :
    game = FourInARow()
    game.play_human_game()