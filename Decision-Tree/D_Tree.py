import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        unique_labels = np.unique(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(unique_labels) == 1):
            return {'leaf': True, 'class': self._most_common_label(y)}
        
        # Find the best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:  # No valid split found
            return {'leaf': True, 'class': self._most_common_label(y)}
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively grow left and right subtrees
        left_tree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        parent_entropy = self._entropy(y)
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            # Sample some thresholds if too many
            if len(thresholds) > 100:
                thresholds = np.percentile(X[:, feature], np.linspace(0, 100, 100))
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Calculate information gain
                n_left = len(left_y)
                n_right = len(right_y)
                child_entropy = (n_left / n_samples) * self._entropy(left_y) + \
                              (n_right / n_samples) * self._entropy(right_y)
                gain = parent_entropy - child_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _most_common_label(self, y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def _predict_single(self, x, tree):
        if tree['leaf']:
            return tree['class']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    # Compute confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)
        cm[true_idx, pred_idx] += 1
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

# Example usage with the wine dataset
if __name__ == "__main__":
    # Load and prepare data
    red_wine = pd.read_csv('winequality-red.csv', sep=';')
    white_wine = pd.read_csv('winequality-white.csv', sep=';')
    wine = pd.concat([red_wine, white_wine], ignore_index=True)
    wine['quality_group'] = pd.cut(wine['quality'], bins=[0, 5, 7, 10], labels=['low', 'medium', 'high'])
    
    X = wine.drop(columns=['quality', 'quality_group']).values
    Y = wine['quality_group'].values
    
    # Standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Train-validation-test split (70/15/15)
    np.random.seed()
    indices = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    X_train, X_val, X_test = X_scaled[train_idx], X_scaled[val_idx], X_scaled[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]
    
    # Hyperparameter tuning: try different max_depth values
    max_depths = [3, 5, 7, 10]
    val_accuracies = []
    best_depth = None
    best_val_acc = 0
    best_model = None
    
    for depth in max_depths:
        dt = DecisionTree(max_depth=depth, min_samples_split=2)
        dt.fit(X_train, Y_train)
        Y_val_pred = dt.predict(X_val)
        val_acc = np.sum(Y_val == Y_val_pred) / len(Y_val)
        val_accuracies.append(val_acc)
        print(f"Max Depth {depth} Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_depth = depth
            best_model = dt
    
    # Evaluate the best model on the test set
    Y_test_pred = best_model.predict(X_test)
    test_acc = np.sum(Y_test == Y_test_pred) / len(Y_test)
    print(f"Best Max Depth: {best_depth}")
    print(f"Test Accuracy with Best Model: {test_acc:.4f}")
    
    # Plot confusion matrix for test set
    classes = ['low', 'medium', 'high']
    plot_confusion_matrix(Y_test, Y_test_pred, classes, title='Confusion Matrix (Decision Tree, Test Set)')
    
    # Plotting validation and test accuracies
    plt.figure(figsize=(8, 6))
    plt.bar(np.array(max_depths) - 0.2, val_accuracies, width=0.4, label='Validation Accuracy', color='#99ff99')
    plt.bar([best_depth + 0.2], [test_acc], width=0.4, label='Test Accuracy (Best Depth)', color='#ff9999')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Validation and Test Accuracy')
    plt.xticks(max_depths)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('decision_tree_validation_test_accuracy.png')