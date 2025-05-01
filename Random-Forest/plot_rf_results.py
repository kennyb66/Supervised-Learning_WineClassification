import numpy as np
import matplotlib.pyplot as plt

# Read Random Forest accuracies from file
rf_accuracies = []
with open('rf_accuracies.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Run'):
            acc = float(line.split(': ')[1].strip())
            rf_accuracies.append(acc)

# Plotting
x = np.arange(len(rf_accuracies))
width = 0.5

plt.figure(figsize=(10, 6))
plt.bar(x, rf_accuracies, width, label='Random Forest', color='#66b3ff')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy Across 10 Runs')
plt.xticks(x, [f'Run {i+1}' for i in range(len(rf_accuracies))])
plt.legend()
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('rf_accuracy_plot.png')
plt.show()