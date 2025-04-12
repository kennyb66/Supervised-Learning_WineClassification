import numpy as np
import matplotlib.pyplot as plt

# Read Naive Bayes accuracies from file
nb_accuracies = []
with open('nb_accuracies.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Run'):
            acc = float(line.split(': ')[1].strip())
            nb_accuracies.append(acc)

# Plotting
x = np.arange(len(nb_accuracies))
width = 0.5

plt.figure(figsize=(10, 6))
plt.bar(x, nb_accuracies, width, label='Gaussian Naive Bayes', color='#66b3ff')
plt.xlabel('Run')
plt.ylabel('Accuracy')
plt.title('Gaussian Naive Bayes Accuracy Across 10 Runs')
plt.xticks(x, [f'Run {i+1}' for i in range(len(nb_accuracies))])
plt.legend()
plt.ylim(0, 1)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('nb_accuracy_plot.png')
plt.show()