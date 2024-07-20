import numpy as np
import random
import copy
import json
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout

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
                    self.update_nn()
                    return True
                elif self.is_draw():
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
                break
            elif self.is_draw():
                break

    def play_human_game(self):
        self.history = []
        while True:
            if self.current_player == 'X':
                row, col = self.nn_move()
            else:
                col = int(input("Enter the column (0-6): "))
                for row in range(5, -1, -1):
                    if self.board[row][col] == '.':
                        break
            if self.make_move(col):
                break
            elif self.is_draw():
                break

# Kivy GUI Class
class FourInARowApp(App):
    def build(self):
        self.game = FourInARow()
        self.layout = GridLayout(cols=7, rows=7)
        self.buttons = [[Button(font_size=24) for _ in range(7)] for _ in range(6)]
        self.col_buttons = [Button(text=str(i), font_size=24) for i in range(7)]

        for col in range(7):
            button = self.col_buttons[col]
            button.bind(on_press=lambda btn, y=col: self.column_pressed(y))
            self.layout.add_widget(button)

        for row in range(6):
            for col in range(7):
                button = self.buttons[row][col]
                button.disabled = True
                self.layout.add_widget(button)

        return self.layout

    def column_pressed(self, col):
        if self.game.current_player == 'O':
            for row in range(5, -1, -1):
                if self.game.board[row][col] == '.':
                    if self.game.make_move(col):
                        self.update_buttons()
                        if self.game.check_winner() or self.game.is_draw():
                            self.show_popup()
                        else:
                            self.ai_move()
                    break

    def ai_move(self):
        row, col = self.game.nn_move()
        self.game.make_move(col)
        self.update_buttons()
        if self.game.check_winner() or self.game.is_draw():
            self.show_popup()

    def update_buttons(self):
        for row in range(6):
            for col in range(7):
                button = self.buttons[row][col]
                if self.game.board[row][col] == 'X':
                    button.text = 'X'
                    button.background_color = [1, 0, 0, 1]  # Red for AI
                elif self.game.board[row][col] == 'O':
                    button.text = 'O'
                    button.background_color = [0, 0, 1, 1]  # Blue for Human
                else:
                    button.text = ''
                    button.background_color = [1, 1, 1, 1]  # White for empty

    def show_popup(self):
        layout = BoxLayout(orientation='vertical')
        if self.game.check_winner():
            winner_text = f"Player {self.game.current_player} wins!"
        else:
            winner_text = "It's a draw!"
        label = Label(text=winner_text)
        button = Button(text="Close", size_hint=(1, 0.2))
        layout.add_widget(label)
        layout.add_widget(button)
        popup = Popup(title='Game Over', content=layout, size_hint=(0.5, 0.5))
        button.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    # Load saved model if available
    nn = NeuralNetwork(84, 30, 1)
    if LOAD_SAVED_MODEL:
        nn.load_model('4_in_a_row_model.json')

    FourInARowApp().run()
