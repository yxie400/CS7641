import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from algorithms import RandomHillClimb, SimulatedAnnealing, GeneticAlgorithm
from problems import Queens, TSP, NeuralNetworkOptimization

# Define a function to run and evaluate the algorithms
def run_experiment(problem, algorithm, params):
    if algorithm == "rhc":
        solver = RandomHillClimb(problem, **params)
    elif algorithm == "sa":
        solver = SimulatedAnnealing(problem, **params)
    elif algorithm == "ga":
        solver = GeneticAlgorithm(problem, **params)
    else:
        raise ValueError("Unknown algorithm")

    result = solver.solve()
    return result

# Define a function to run and collect results for different hyperparameters
def run_experiments(problem, algorithm, params_list):
    results = []
    for params in params_list:
        result = run_experiment(problem, algorithm, params)
        results.append((params, result))
    return results

# Define a function to compute the running max or min
def compute_running_best(scores, mode="max"):
    running_best = []
    best_so_far = scores[0]
    for score in scores:
        if mode == "max":
            best_so_far = max(best_so_far, score)
        elif mode == "min":
            best_so_far = min(best_so_far, score)
        running_best.append(best_so_far)
    return running_best

# Define a function to save the plots with trajectory scores
def save_plot(results, title, x_label, y_label, directory, filename, score_label):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(10, 6))
    for params, result in results:
        x = range(len(result.trajectory_score))
        if y_label == "Trajectory Score":
            y = compute_running_best(result.trajectory_score, mode="max" if score_label == "Max Score" else "min")
        else:
            y = result.trajectory_score
        plt.plot(x, y, label=f"{params}", alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(score_label)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(directory, filename))
    plt.close()



def nn_cost_function(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cost_matrix = np.array([[0, 1], [5, 0]])  # a for false positive, b for false negative
    total_cost = np.sum(cm * cost_matrix)
    return total_cost


# Function to run n-Queens experiments
def run_n_queens_experiments():
    random.seed(42)
    n = 10
    problem = Queens(n)
    
    rhc_n_neighbors = [3, 5, 10]
    rhc_restart_limits = [0, 10, 50]
    sa_initial_temps = [1, 10, 100]
    sa_alphas = [0.99, 0.999]
    ga_mutation_rates = [0.1, 0.2]
    ga_elite_sizes = [10, 20, 30]

    directory = "n_queens_results"

    # Random Hill Climbing
    for n_neighbors in rhc_n_neighbors:
        rhc_params = [{"max_iter": 5000, "n_neighbors": n_neighbors, "restart_limit": rl} for rl in rhc_restart_limits]
        rhc_results = run_experiments(problem, "rhc", rhc_params)
        save_plot(rhc_results, f"Random Hill Climbing Performance (n-Queens, n_neighbors={n_neighbors})", 
                  "Iteration", "Trajectory Score", directory, f"rhc_n_neighbors_{n_neighbors}.png", "Max Score")

    # Simulated Annealing
    for alpha in sa_alphas:
        sa_params = [{"max_iter": 5000, "T": t, "alpha": alpha} for t in sa_initial_temps]
        sa_results = run_experiments(problem, "sa", sa_params)
        save_plot(sa_results, f"Simulated Annealing Performance (n-Queens, alpha={alpha})", 
                  "Iteration", "Trajectory Score", directory, f"sa_alpha_{alpha}.png", "Max Score")

    # Genetic Algorithm
    for mutation_rate in ga_mutation_rates:
        ga_params = [{"max_iter": 5000, "pop_size": 200, "elite_size": es, "mutation_rate": mutation_rate} for es in ga_elite_sizes]
        ga_results = run_experiments(problem, "ga", ga_params)
        save_plot(ga_results, f"Genetic Algorithm Performance (n-Queens, mutation_rate={mutation_rate})", 
                  "Iteration", "Trajectory Score", directory, f"ga_mutation_rate_{mutation_rate}.png", "Max Score")

# Function to run TSP experiments
def run_tsp_experiments():
    random.seed(123)
    n = 8
    problem = TSP(n)

    rhc_n_neighbors = [3, 5, 10]
    rhc_restart_limits = [0, 10, 50]
    sa_initial_temps = [1, 10, 100]
    sa_alphas = [0.99, 0.999]
    ga_mutation_rates = [0.1, 0.2]
    ga_elite_sizes = [10, 20, 30]

    directory = "tsp_results"

    # Random Hill Climbing
    for n_neighbors in rhc_n_neighbors:
        rhc_params = [{"max_iter": 5000, "n_neighbors": n_neighbors, "restart_limit": rl} for rl in rhc_restart_limits]
        rhc_results = run_experiments(problem, "rhc", rhc_params)
        save_plot(rhc_results, f"Random Hill Climbing Performance (TSP, n_neighbors={n_neighbors})", 
                  "Iteration", "Trajectory Score", directory, f"rhc_n_neighbors_{n_neighbors}.png", "Max Score")

    # Simulated Annealing
    for alpha in sa_alphas:
        sa_params = [{"max_iter": 5000, "T": t, "alpha": alpha} for t in sa_initial_temps]
        sa_results = run_experiments(problem, "sa", sa_params)
        save_plot(sa_results, f"Simulated Annealing Performance (TSP, alpha={alpha})", 
                  "Iteration", "Trajectory Score", directory, f"sa_alpha_{alpha}.png", "Max Score")

    # Genetic Algorithm
    for mutation_rate in ga_mutation_rates:
        ga_params = [{"max_iter": 5000, "pop_size": 200, "elite_size": es, "mutation_rate": mutation_rate} for es in ga_elite_sizes]
        ga_results = run_experiments(problem, "ga", ga_params)
        save_plot(ga_results, f"Genetic Algorithm Performance (TSP, mutation_rate={mutation_rate})", 
                  "Iteration", "Trajectory Score", directory, f"ga_mutation_rate_{mutation_rate}.png", "Max Score")

# Function to download and preprocess data for neural network optimization
def download_and_preprocess_data(dataset_id, encoding_method='OneHotEncoder'):
    import openml
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    dataset = openml.datasets.get_dataset(dataset_id)
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if dataset_id == 31:
        y = y.replace({'good': 1, 'bad': 0})
    else:
        y = [int(i) for i in y.values]
        if max(y) > 1:
            y = [i-1 for i in y]
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['category']).columns

    if encoding_method == 'OneHotEncoder':
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(), categorical_cols)
            ])
    elif encoding_method == 'LabelEncoder':
        label_encoders = {col: LabelEncoder().fit(X[col]) for col in categorical_cols}
        for col in categorical_cols:
            X[col] = label_encoders[col].transform(X[col])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols)
            ], remainder='passthrough')
    else:
        raise ValueError("Encoding method must be either 'OneHotEncoder' or 'LabelEncoder'")
    
    X = preprocessor.fit_transform(X)
    return X, y

# Function to run neural network weight optimization experiments
def run_nn_experiments():
    random.seed(42)
    data_id = 31
    X, y = download_and_preprocess_data(data_id, encoding_method='OneHotEncoder')

    model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1, warm_start=True, random_state=42)

    nn_opt = NeuralNetworkOptimization(model, X, y, nn_cost_function, maximize=False)

    rhc_n_neighbors = [3, 5, 10]
    rhc_restart_limits = [0, 10, 50]
    sa_initial_temps = [1, 10, 100]
    sa_alphas = [0.995, 0.999, 0.9999]
    ga_mutation_rates = [0.1, 0.2]
    ga_elite_sizes = [10, 20, 30]

    directory = "nn_results"


    # Random Hill Climbing
    for n_neighbors in rhc_n_neighbors:
        rhc_params = [{"max_iter": 1000, "n_neighbors": n_neighbors, "restart_limit": rl} for rl in rhc_restart_limits]
        rhc_results = run_experiments(nn_opt, "rhc", rhc_params)
        save_plot(rhc_results, f"Random Hill Climbing Performance (NN Optimization, n_neighbors={n_neighbors})", 
                  "Iteration", "Trajectory Score", directory, f"rhc_n_neighbors_{n_neighbors}.png", "Max Score")

    for alpha in sa_alphas:
        sa_params = [{"max_iter": 1000, "T": t, "alpha": alpha} for t in sa_initial_temps]
        sa_results = run_experiments(nn_opt, "sa", sa_params)
        save_plot(sa_results, f"Simulated Annealing Performance (NN Optimization, alpha={alpha})", 
                  "Iteration", "Trajectory Score", directory, f"sa_alpha_{alpha}.png", "Max Score")

    # Genetic Algorithm
    for mutation_rate in ga_mutation_rates:
        ga_params = [{"max_iter": 1000, "pop_size": 100, "elite_size": es, "mutation_rate": mutation_rate} for es in ga_elite_sizes]
        ga_results = run_experiments(nn_opt, "ga", ga_params)
        save_plot(ga_results, f"Genetic Algorithm Performance (NN Optimization, mutation_rate={mutation_rate})", 
                  "Iteration", "Trajectory Score", directory, f"ga_mutation_rate_{mutation_rate}.png", "Max Score")

# Run all experiments
if __name__ == '__main__':
    print("Running n-Queens experiments...")
    #run_n_queens_experiments()
    
    print("Running TSP experiments...")
    #run_tsp_experiments()

    print("Running Neural Network Optimization experiments...")
    run_nn_experiments()
