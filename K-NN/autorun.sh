#!/bin/bash
# This script runs the Weighted K-NN algorithm 10 times and saves the output to a file.

for i in {1..10}
do
    # Run the Python script and save the output to a file
    python K-NN.py >> knn_output.txt 
done