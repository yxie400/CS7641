#from problems import Queens, NeuralNetworkOptimization
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationResult:
    state: list
    trajectory: list
    trajectory_score: list

class OptimizationAlgorithm:
    def __init__(self, problem):
        self.problem = problem

    def solve(self):
        raise NotImplementedError

class RandomHillClimb(OptimizationAlgorithm):
    def __init__(self, problem, max_iter=1000, n_neighbors=10, restart_limit=10):
        super().__init__(problem)
        self.max_iter = max_iter
        self.n_neighbors = n_neighbors
        self.restart_limit = restart_limit

    def solve(self):
        current = self.problem.init_state()
        current_score = self.problem.score(current)
        trajectory = [current_score]
        trajectory_score = [current_score]

        no_improvement_count = 0

        for _ in range(self.max_iter):
            neighbors = self.problem.random_neighbor(current, size=self.n_neighbors)
            neighbor_scores = [self.problem.score(neighbor) for neighbor in neighbors]

            best_neighbor_idx = neighbor_scores.index(max(neighbor_scores))
            best_neighbor = neighbors[best_neighbor_idx]
            best_neighbor_score = neighbor_scores[best_neighbor_idx]

            if best_neighbor_score > current_score:
                current = best_neighbor
                current_score = best_neighbor_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.restart_limit:
                current = self.problem.init_state()
                current_score = self.problem.score(current)
                no_improvement_count = 0

            trajectory.append(current_score)
            trajectory_score.append(current_score)

        return OptimizationResult(state=current, trajectory=trajectory, trajectory_score=trajectory_score)



class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self, problem, max_iter=1000, T=1.0, alpha=0.99, restart_limit=100):
        super().__init__(problem)
        self.max_iter = max_iter
        self.T = T
        self.alpha = alpha
        self.restart_limit = restart_limit

    def accept_prob(self, current_score, neighbor_score):
        if neighbor_score > current_score:
            return 1
        else:
            return np.exp((neighbor_score - current_score) / self.T)

    def solve(self):
        current = self.problem.state
        current_score = self.problem.score(current)
        best = current
        best_score = current_score

        trajectory = [current]
        trajectory_score = [current_score]
        
        no_improvement_count = 0

        for iter_num in range(self.max_iter):
            neighbor = self.problem.random_neighbor(current, 1)[0]
            neighbor_score = self.problem.score(neighbor)
            print(no_improvement_count)
            print(f"Iteration {iter_num}, Current Score: {current_score}, Neighbor Score: {neighbor_score}")

            if neighbor_score > current_score:
                current = neighbor
                current_score = neighbor_score
                no_improvement_count = 0  # Reset the no improvement counter

                # Update the best state found so far
                if neighbor_score > best_score:
                    best = neighbor
                    best_score = neighbor_score

            else:
                no_improvement_count += 1

            # Perform random restart if no improvement for a certain number of iterations
            if no_improvement_count >= self.restart_limit:
                print(f"Random restart at iteration {iter_num}")
                current = self.problem.init_state()
                current_score = self.problem.score(current)
                no_improvement_count = 0  # Reset the no improvement counter

            self.T *= self.alpha
            trajectory.append(current)
            trajectory_score.append(current_score)

        return OptimizationResult(state=best, trajectory=trajectory, trajectory_score=trajectory_score)
    


class GeneticAlgorithm(OptimizationAlgorithm):
    def __init__(self, problem, max_iter=1000, pop_size=100, elite_size=10, mutation_rate=0.01):
        super().__init__(problem)
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def solve(self):
        population = [self.problem.init_state() for _ in range(self.pop_size)]
        best_score = float('-inf') if self.problem.maximize else float('inf')
        best_solution = None
        
        trajectory = []
        trajectory_score = []

        for _ in range(self.max_iter):
            population.sort(key=lambda x: self.problem.score(x), reverse=self.problem.maximize)
            
            if (self.problem.maximize and self.problem.score(population[0]) > best_score) or \
               (not self.problem.maximize and self.problem.score(population[0]) < best_score):
                best_score = self.problem.score(population[0])
                best_solution = population[0]

            new_population = population[:self.elite_size]

            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(population[:self.elite_size], 2)
                child = self.crossover(parent1, parent2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)

            population = new_population
            trajectory.append(best_solution)
            trajectory_score.append(best_score)

        return OptimizationResult(state=best_solution, trajectory=trajectory, trajectory_score=trajectory_score)

    def crossover(self, parent1, parent2):
        # Uniform crossover
        child = np.where(np.random.rand(len(parent1)) < 0.5, parent1, parent2)
        return child

    def mutate(self, individual):
        idx = random.randint(0, len(individual) - 1)
        individual[idx] += random.uniform(-0.1, 0.1)
        return individual






# if __name__ == '__main__':
#     n = 8
#     problem = Queens(n)
#     print(random_hill_climb(problem))
#     print(simulated_annealing(problem))
#     print(genetic_algorithm(problem))

# n = 8
# problem = Queens(n)
# result = RandomHillClimb(problem, max_iter=10000).solve()
# print(result)


# result = SimulatedAnnealing(problem, max_iter=100000, T=5.0, alpha=0.9999).solve()
# print(result)

# result = GeneticAlgorithm(problem).solve()
# print(result)

# # Output the results
# print("Final state:", result.state)
# print("Trajectory:", result.trajectory)
# print("Trajectory scores:", result.trajectory_score)


# n = 8
# problem = TSP(n)
# result = RandomHillClimb(problem, max_iter=10000).solve()
# print(result)




# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# import openml
# def download_and_preprocess_data(dataset_id, encoding_method='OneHotEncoder'):
#     dataset = openml.datasets.get_dataset(dataset_id)
#     df, _, _, _ = dataset.get_data(dataset_format="dataframe")
#     X = df.iloc[:, :-1]
#     y = df.iloc[:, -1]
#     if dataset_id == 31:
#         y = y.replace({'good': 1, 'bad': 0})
#     else:
#         y = [int(i) for i in y.values]
#         if max(y) > 1:
#             y = [i-1 for i in y]
#     numerical_cols = X.select_dtypes(include=['number']).columns
#     categorical_cols = X.select_dtypes(include=['category']).columns

#     if encoding_method == 'OneHotEncoder':
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', StandardScaler(), numerical_cols),
#                 ('cat', OneHotEncoder(), categorical_cols)
#             ])
#     elif encoding_method == 'LabelEncoder':
#         label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}
#         for col in categorical_cols:
#             X[col] = label_encoders[col].transform(X[col])
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', StandardScaler(), numerical_cols)
#             ], remainder='passthrough')
#     else:
#         raise ValueError("Encoding method must be either 'OneHotEncoder' or 'LabelEncoder'")
    
#     X = preprocessor.fit_transform(X)
#     return X, y



# data_id = 31
# X, y = download_and_preprocess_data(data_id, encoding_method='OneHotEncoder')

# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, recall_score, make_scorer, confusion_matrix
# model = MLPClassifier(hidden_layer_sizes=(128, 32, 32), max_iter=1, warm_start=True, random_state=42)

# def cost_function(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     cost_matrix = np.array([[0, 1], [5, 0]])  # a for false positive, b for false negative
#     total_cost = np.sum(cm * cost_matrix)*-1
#     return total_cost

# nn_opt = NeuralNetworkOptimization(model, X, y, cost_function, maximize=False)


# result = RandomHillClimb(nn_opt, max_iter=1000).solve()
# result = SimulatedAnnealing(nn_opt, max_iter=1000, T=5.0, alpha=0.999).solve()
# result = GeneticAlgorithm(nn_opt).solve()
# min(result.trajectory_score)