#!/bin/bash
# This script runs the Weighted K-NN algorithm 10 times and saves the output to a file.

for i in {1..10}
do
    # Run weighted 7-NN and save the output to a file
    python K-NN.py >> knn_output.txt 

    # Run weighted 7-NN (with new data) and save the output to a file
    python K-NN_v2.py >> knn_output.txt
done