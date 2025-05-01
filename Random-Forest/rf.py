import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import math

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

# Custom Random Forest implementation
class CustomRandomForest:
    def __init__(self, n_estimators=100, max_features='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Set max_features
        if self.max_features == 'auto':
            self.max_features = int(math.log2(n_features) + 1)
        
        # Train each tree on a bootstrap sample
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Initialize a decision tree
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                class_weight='balanced',
                random_state=np.random.randint(0, 10000)
            )
            
            # Train the tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from each tree
        predictions = np.zeros((X.shape[0], self.n_estimators), dtype=object)
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        
        # Majority voting
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[i, :]
            unique_classes, counts = np.unique(votes, return_counts=True)
            majority_class = unique_classes[np.argmax(counts)]
            final_predictions.append(majority_class)
        
        return np.array(final_predictions)

# Run Random Forest 10 times and save results
rf_accuracies = []
output_file = 'rf_accuracies.txt'

with open(output_file, 'w') as f:
    f.write("Random Forest Accuracies\n")
    for run in range(10):
        indices = np.random.permutation(len(X))
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Train Custom Random Forest model
        rf = CustomRandomForest(n_estimators=100, max_features='auto', random_state=run)
        rf.fit(X_train, Y_train)
        Y_pred = rf.predict(X_test)
        acc = np.sum(Y_test == Y_pred) / len(Y_test)
        rf_accuracies.append(acc)
        
        # Write accuracy to file
        f.write(f"Run {run + 1}: {acc:.4f}\n")
    
    # Write average accuracy
    avg_acc = np.mean(rf_accuracies)
    f.write(f"Average Accuracy: {avg_acc:.4f}\n")

# Single run for printing
rf = CustomRandomForest(n_estimators=100, max_features='auto', random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
acc = np.mean(rf_accuracies)
print(f"Random Forest Average Accuracy (10 runs): {acc:.4f}")