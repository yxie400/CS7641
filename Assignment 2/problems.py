import random
import math
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

class OptimizationProblem:
    def __init__(self, maximize=False):
        self.maximize = int(maximize) * 2 - 1

    def score(self, state):
        """Return the score of a state. It should be maximized."""
        raise NotImplementedError

class Queens(OptimizationProblem):
    def __init__(self, n, maximize=False):
        super().__init__(maximize)
        self.n = n
        self.state = self.init_state()

    def init_state(self):
        """Generate a random state."""
        return random.sample(range(self.n), self.n)

    def random_neighbor(self, state, size):
        """Generate random neighbors by swapping 2 queens."""
        neighbors = []
        for _ in range(size):
            neighbor = state.copy()
            p1, p2 = random.sample(range(self.n), 2)
            neighbor[p1], neighbor[p2] = neighbor[p2], neighbor[p1]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, state):
        """Evaluate the fitness of a state."""
        row_counts = {}
        diag_up_counts = {}
        diag_down_counts = {}

        fitness = 0

        for i in range(self.n):
            row = state[i]
            diag_up = state[i] - i
            diag_down = state[i] + i

            if row in row_counts:
                row_counts[row] += 1
            else:
                row_counts[row] = 1

            if diag_up in diag_up_counts:
                diag_up_counts[diag_up] += 1
            else:
                diag_up_counts[diag_up] = 1

            if diag_down in diag_down_counts:
                diag_down_counts[diag_down] += 1
            else:
                diag_down_counts[diag_down] = 1

        for count in row_counts.values():
            if count > 1:
                fitness += count - 1

        for count in diag_up_counts.values():
            if count > 1:
                fitness += count - 1

        for count in diag_down_counts.values():
            if count > 1:
                fitness += count - 1

        return fitness

    def score(self, state):
        return self.maximize * self.evaluate(state)



class TSP(OptimizationProblem):
    def __init__(self, n, maximize=False):
        super().__init__(maximize)
        self.n = n
        self.coords_list = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
        self.state = self.init_state()

    def init_state(self):
        """Generate a random state."""
        return list(range(self.n))

    def random_neighbor(self, state, size):
        """Generate random neighbors by swapping 2 cities."""
        neighbors = []
        for _ in range(size):
            neighbor = state.copy()
            p1, p2 = random.sample(range(self.n), 2)
            neighbor[p1], neighbor[p2] = neighbor[p2], neighbor[p1]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, state):
        """Calculate the total distance of the tour."""
        total_distance = 0
        for i in range(self.n):
            x1, y1 = self.coords_list[state[i]]
            x2, y2 = self.coords_list[state[(i + 1) % self.n]]
            total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total_distance

    def score(self, state):
        """Return the evaluation score."""
        return self.maximize * self.evaluate(state)
    

class NeuralNetworkOptimization(OptimizationProblem):
    def __init__(self, model, X, y, evaluation_function, maximize=False):
        super().__init__(maximize)
        self.model = model
        self.X = X
        self.y = y
        self.evaluation_function = evaluation_function
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.state = self.init_state()

    def init_state(self):
        """Initialize the neural network and set random weights."""
        self.model.fit(self.X_train, self.y_train)  # Fit once to initialize weights
        return self.get_weights()

    def get_weights(self):
        """Get the current weights of the neural network."""
        weights = []
        for coef, intercept in zip(self.model.coefs_, self.model.intercepts_):
            weights.extend(coef.flatten())
            weights.extend(intercept)
        return np.array(weights)

    def set_weights(self, weights):
        """Set the weights of the neural network."""
        weights = np.array(weights)  # Ensure weights are a numpy array
        start = 0
        for i in range(len(self.model.coefs_)):
            end = start + self.model.coefs_[i].size
            self.model.coefs_[i] = weights[start:end].reshape(self.model.coefs_[i].shape)
            start = end
            end = start + self.model.intercepts_[i].size
            self.model.intercepts_[i] = weights[start:end]
            start = end

    def random_neighbor(self, state, size):
        """Generate random neighbors by slightly modifying the weights."""
        neighbors = []
        for _ in range(size):
            neighbor = state.copy()
            idx = random.randint(0, len(state) - 1)
            neighbor[idx] += random.uniform(-0.1, 0.1)  # Small perturbation
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, state):
        """Evaluate the performance of the neural network using the custom evaluation function."""
        self.set_weights(state)
        y_pred = self.model.predict(self.X_val)
        evaluation_score = self.evaluation_function(self.y_val, y_pred)
        print(f"Evaluation Score: {evaluation_score}")
        return evaluation_score

    def score(self, state):
        """Return the evaluation score (negative for minimization)."""
        score = -self.evaluate(state)  # Negate the evaluation function for minimization
        print(f"Score: {score}")
        return score