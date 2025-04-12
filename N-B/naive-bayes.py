import pandas as pd
import numpy as np

# Load the wine dataset
red_wine = pd.read_csv('winequality-red.csv', sep=';')
white_wine = pd.read_csv('winequality-white.csv', sep=';')

# Combine the two datasets
wine = pd.concat([red_wine, white_wine], ignore_index=True)

# Bin quality scores into three classes
wine['quality_group'] = pd.cut(wine['quality'], bins=[0, 5, 7, 10], labels=['low', 'medium', 'high'])

# Features and target variable
X = wine.drop(columns=['quality', 'quality_group']).values
Y = wine['quality_group'].values

# Manual standardization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Split into train (80%) and test (20%)
np.random.seed()
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))
train_idx = indices[:train_size]
test_idx = indices[train_size:]
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
Y_train, Y_test = Y[train_idx], Y[test_idx]

# Gaussian Naive Bayes implementation
def gaussian_naive_bayes_predict(X_train, Y_train, X_test):
    unique_labels = np.unique(Y_train)
    n_features = X_train.shape[1]
    predictions = []
    
    class_params = {}
    class_priors = {}
    for label in unique_labels:
        X_class = X_train[Y_train == label]
        n_class = X_class.shape[0]
        class_priors[label] = n_class / len(Y_train)
        means = np.mean(X_class, axis=0)
        variances = np.var(X_class, axis=0, ddof=1)
        class_params[label] = (means, variances)
    
    for test_point in X_test:
        posteriors = {}
        for label in unique_labels:
            means, variances = class_params[label]
            prior = class_priors[label]
            log_likelihood = 0
            for i in range(n_features):
                a_i = test_point[i]
                mu_i = means[i]
                log_likelihood += (
                    -0.5 * np.log(2 * np.pi * variances[i])
                    - ((a_i - mu_i) ** 2) / (2 * variances[i])
                )
            log_posterior = np.log(prior + 1e-10) + log_likelihood
            posteriors[label] = log_posterior
        pred_label = max(posteriors, key=posteriors.get)
        predictions.append(pred_label)
    
    return np.array(predictions)

# Run Naive Bayes 10 times and save results
nb_accuracies = []
output_file = 'nb_accuracies.txt'

with open(output_file, 'w') as f:
    f.write("Naive Bayes Accuracies\n")
    for run in range(10):
        indices = np.random.permutation(len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        Y_pred = gaussian_naive_bayes_predict(X_train, Y_train, X_test)
        acc = np.sum(Y_test == Y_pred) / len(Y_test)
        nb_accuracies.append(acc)
        
        # Write accuracy to file
        f.write(f"Run {run + 1}: {acc:.4f}\n")
    
    # Write average accuracy
    avg_acc = np.mean(nb_accuracies)
    f.write(f"Average Accuracy: {avg_acc:.4f}\n")

# Single run for printing
Y_pred = gaussian_naive_bayes_predict(X_train, Y_train, X_test)
acc = np.sum(Y_test == Y_pred) / len(Y_test)
print(f"Naive Bayes Accuracy (single run): {acc:.4f}")