#!/bin/bash

dt=0.01
min_freq=0.1
max_freq=10
input_path=$1/data.csv
output_path=$1/inversed.csv

echo "Inversing..."

#harminv -t $dt 0.1-10 -v < data/data.csv > data/inversed.csv
harminv -t $dt $min_freq-$max_freq < $input_path > $output_path

echo "Done."
echo "Inversed data saved to $output_path"