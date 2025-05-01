#!/bin/bash

# autorun.sh
# Script to run the Random Forest Python script and plot results

# Run the Random Forest script
echo "Running Random Forest script..."
python rf.py

# Run the plotting script
echo "Generating plot..."
python plot_rf_results.py