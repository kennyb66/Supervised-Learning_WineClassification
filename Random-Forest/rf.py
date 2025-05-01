import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, Y_train)
        Y_pred = rf_model.predict(X_test)
        acc = np.sum(Y_test == Y_pred) / len(Y_test)
        rf_accuracies.append(acc)
        
        # Write accuracy to file
        f.write(f"Run {run + 1}: {acc:.4f}\n")
    
    # Write average accuracy
    avg_acc = np.mean(rf_accuracies)
    f.write(f"Average Accuracy: {avg_acc:.4f}\n")

# Single run for printing
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)
acc = np.sum(Y_test == Y_pred) / len(Y_test)
print(f"Random Forest Average Accuracy (10 runs): {acc:.4f}")