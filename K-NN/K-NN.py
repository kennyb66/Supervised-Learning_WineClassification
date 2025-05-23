import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Function to calculate Euclidean distance (2-norm) between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point2 - point1) ** 2))

# Weighted k-Nearest Neighbors algorithm
def KNN_predict_weighted(X_train, Y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Compute distances to all training points
        distances = np.array([euclidean_distance(test_point, train_point) for train_point in X_train])
        
        # Find the k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_distances = distances[k_indices]
        k_labels = Y_train[k_indices]
        
        # Compute weights as inverse distances (1/distance)
        weights = 1 / (k_distances + 1e-10)  # Add small constant to avoid division by zero
        weights /= weights.sum()  # Normalize weights to sum to 1 
        
        # Weighted voting: sum weights for each class
        unique_labels = np.unique(k_labels)
        weighted_votes = {label: 0 for label in unique_labels}
        for label, weight in zip(k_labels, weights):
            weighted_votes[label] += weight
        
        # Predict the class with the highest total weight
        pred_label = max(weighted_votes, key=weighted_votes.get)
        predictions.append(pred_label)
    
    return np.array(predictions)

# Predictions for k = 7
Y_pred = KNN_predict_weighted(X_train, Y_train, X_test, k=7)

# Calculate accuracy
acc = np.sum(Y_test == Y_pred) / len(Y_test)
print(f"Accuracy: {acc:.4f}")

# Data from the CSV
weighted = [0.7685, 0.7923, 0.8008, 0.7954, 0.7892, 0.7877, 0.7862, 0.7846, 0.7823, 0.7731]
unweighted = [0.7215, 0.7092, 0.7038, 0.7277, 0.7169, 0.7223, 0.7269, 0.7369, 0.6985, 0.7095]

# Paired bar plot
x = np.arange(len(weighted))  # 0 to 9 for 10 runs
width = 0.35  # Width of the bars

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, weighted, width, label='Weighted 7-NN', color='#ff9999')
plt.bar(x + width/2, unweighted, width, label='Unweighted 7-NN', color='#66b3ff')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.title('Weighted vs. Unweighted 7-NN Accuracy Across 10 Runs')
plt.xticks(x, [f'Run {i+1}' for i in range(len(weighted))])
plt.legend()
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()