import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, make_scorer, confusion_matrix
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve, validation_curve, RandomizedSearchCV
import matplotlib.pyplot as plt
import numpy as np
import os
import openml
from scipy.stats import randint, uniform
import matplotlib

matplotlib.use('Agg')

RANDOM_SEED = 12345

def initialize_dist():
    dist = {
        'KNN': {
            'n_neighbors': np.arange(1, 31, 2),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        },
        'MLP': {
            'hidden_layer_sizes': [(128,), (32, 32, 32, 32, 32, 32, 32), (1024, 64), (256, 128, 64, 32, 8, 8, )],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant','adaptive'],
            'max_iter': [50, 100, 200, 300, 400, 500]
        },
        'SVM':{
            'C': np.logspace(-4, 1, 6),
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'degree': np.arange(1, 5), # Used only for 'poly' kernel
            'gamma': ['scale', 'auto']  # Used for 'rbf', 'poly', and 'sigmoid' kernels
        },
        'XGBoost': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(1, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.5, 1),
            'colsample_bytree': uniform(0.5, 1),
            'gamma': uniform(0, 0.5),
        },
    }
    return dist

def load_data(data_id):
    dataset = openml.datasets.get_dataset(data_id)
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")
    return df

def download_and_preprocess_data(dataset_id, encoding_method='OneHotEncoder'):
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

def initialize_models():
    models = {
        'KNN': KNeighborsClassifier(),
        'MLP': MLPClassifier(),
        'SVM': SVC(),
        'XGBoost': XGBClassifier()
    }
    return models

def initialize_hyperparameters():
    hyperparameters = {
        'KNN': {"param_name":"n_neighbors", "param_range" : np.arange(1, 31, 2)},
        'MLP': {"param_name":"max_iter", "param_range" : [50, 100, 200, 300, 400, 500]},
        'SVM':{"param_name":"C", "param_range" : np.logspace(-4, 1, 6)},
        'XGBoost': {"param_name":"max_depth", "param_range" : np.arange(1, 10, 1)},
    }
    return hyperparameters

def plot_learning_curves(directory, name, model, X, y, scoring, encoding_method):
    train_sizes = np.linspace(.1, 1.0, 5)
    _, axes = plt.subplots(1, 1, figsize=(10, 6))
        
    axes.set_title(f'Learning Curve ({name})_{encoding_method}')
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    axes.grid()
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(os.path.join(directory, f'learning_curve_{type(model).__name__}_{encoding_method}.png'))
    plt.close()


def plot_validation_curves(directory, model, X, y, param_name, param_range, scoring, encoding_method):
    if param_name == 'C':
        train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=5,  scoring=scoring)
    else:
        train_scores, test_scores = validation_curve(model, X, y, param_name=param_name, param_range=param_range, cv=5, n_jobs=-1, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.plot(param_range, train_scores_mean, label='Training score')
    plt.plot(param_range, test_scores_mean, label='Cross-validation score')
    plt.title(f'Validation Curve for {type(model).__name__}_{encoding_method} ({param_name})')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.savefig(os.path.join(directory, f'validation_curve_{type(model).__name__}_{encoding_method}_{param_name}.png'))
    plt.close()

def randomized_cv_search(model, param_distributions, X, y, scoring):
    random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=100, cv=5, scoring=scoring, n_jobs=-1)
    random_search.fit(X, y)
    return random_search.best_estimator_, random_search.best_score_


def save_results(directory, filename, best_model, best_score, X_test, y_test, scoring):
    test_score = scoring(y_true = y_test, estimator = best_model, X = X_test)
    
    results = {
        "parameters": best_model.get_params(),
        "train_score": best_score,
        "test_score": test_score
    }
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(os.path.join(directory, filename), 'w') as f:
        f.write(str(results))


def cost_function(y_true, y_pred, a, b):
    cm = confusion_matrix(y_true, y_pred)
    cost_matrix = np.array([[0, a], [b, 0]])  # a for false positive, b for false negative
    total_cost = np.sum(cm * cost_matrix)*-1
    return total_cost

def custom_cost_scorer(y_true, y_pred, a, b):
    return cost_function(y_true, y_pred, a, b)

def run_exp(dataset_id, encoding_method='OneHotEncoder', scoring='accuracy'):

    if isinstance(scoring, str):
        if scoring == 'accuracy':
            scoring = make_scorer(accuracy_score)
        elif scoring == 'recall':
            scoring = make_scorer(recall_score)
    elif isinstance(scoring, list):
        scoring = make_scorer(custom_cost_scorer, greater_is_better=False, a=scoring[0], b=scoring[1])
    else:
        raise ValueError("Scoring must be either 'accuracy', 'recall', or a list of two integers")

    X, y = download_and_preprocess_data(dataset_id, encoding_method)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

    # Directory to save results and plots
    results_dir = str(dataset_id)

    # Step 2: Initialize and compare models
    models = initialize_models()
    for name, model in models.items():
        plot_learning_curves(results_dir, name, model, X_train, y_train, scoring=scoring, encoding_method=encoding_method)

    # Step 3: Generate validation curves
    hyperparameters = initialize_hyperparameters()
    for name, model in models.items():
        param_name = hyperparameters[name]['param_name']
        param_range = hyperparameters[name]['param_range']
        plot_validation_curves(results_dir, model, X_train, y_train, param_name=param_name, param_range=param_range, scoring=scoring, encoding_method=encoding_method)

    # Step 4: Randomized CV search for the best model
    dist = initialize_dist()
    for name, model in models.items():
        best_model, best_score = randomized_cv_search(model, dist[name], X_train, y_train, scoring)
        save_results(results_dir, f'best_{name}_{encoding_method}.txt', best_model, best_score, X_test, y_test, scoring)

if __name__ == '__main__':
    
    for encoding_method in ['OneHotEncoder', 'LabelEncoder']:
        run_exp(31, encoding_method=encoding_method, scoring=[1,5])
        run_exp(1461, encoding_method=encoding_method, scoring='recall')