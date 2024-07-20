import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        return self.board

    def available_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, row, col, player):
        if self.board[row, col] == 0:
            self.board[row, col] = player
            return True
        return False

    def check_winner(self):
        for player in [1, -1]:
            for i in range(3):
                if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                    return player
            if self.board.trace() == player * 3 or np.fliplr(self.board).trace() == player * 3:
                return player
        if not self.available_moves():
            return 0  # Draw
        return None  # Game ongoing

    def render(self):
        for row in self.board:
            print(' '.join([str(x) for x in row]))
        print()

# Initialize the game
game = TicTacToe()
game.reset()
game.render()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.w1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y):
        m = x.shape[1]
        dz2 = self.a2 - y
        dw2 = (1 / m) * np.dot(dz2, self.a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.w2.T, dz2) * self.sigmoid_derivative(self.z1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            self.forward(x)
            self.backward(x, y)
            if epoch % 100 == 0:
                loss = np.mean((self.a2 - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

# Initialize the neural network
input_size = 9  # 3x3 board flattened
hidden_size = 18
output_size = 9  # One output for each cell in the board
lr = 0.01
nn = NeuralNetwork(input_size, hidden_size, output_size, lr)

# Generate training data
def generate_training_data(n_samples):
    X = []
    y = []
    for _ in range(n_samples):
        game.reset()
        board = game.board.flatten()
        player = 1
        while True:
            moves = game.available_moves()
            if not moves:
                break
            move = moves[np.random.randint(len(moves))]
            game.make_move(move[0], move[1], player)
            board = game.board.flatten()
            target = np.zeros(9)
            if game.check_winner() == player:
                target[move[0] * 3 + move[1]] = 1
            elif game.check_winner() is None:
                target[move[0] * 3 + move[1]] = 0.5
            else:
                target[move[0] * 3 + move[1]] = 0
            X.append(board * player)
            y.append(target)
            player = -player
    return np.array(X).T, np.array(y).T

# Train the neural network
x_train, y_train = generate_training_data(10000)
nn.train(x_train, y_train, epochs=100)

def get_move_from_nn(board, player):
    input_board = board.flatten() * player
    output = nn.forward(input_board.reshape(-1, 1))
    move = np.argmax(output)
    return divmod(move, 3)

# Play a game with the neural network
game.reset()
game.render()
current_player = 1

while True:
    if current_player == 1:
        move = get_move_from_nn(game.board, current_player)
    else:
        move = get_move_from_nn(game.board, current_player)
    
    game.make_move(move[0], move[1], current_player)
    game.render()
    
    winner = game.check_winner()
    if winner is not None:
        if winner == 0:
            print("It's a draw!")
        else:
            print(f"Player {winner} wins!")
        break
    
    current_player = -current_player
