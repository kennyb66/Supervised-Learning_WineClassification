#!/bin/bash

# autorun.sh
# Script to run the Naive Bayes Python script and plot results

# Run the Naive Bayes script
echo "Running Naive Bayes script..."
python naive-bayes.py

# Run the plotting script
echo "Generating plot..."
python plot_nb_results.py