#!/bin/bash

dt=0.01
min_freq=0.1
max_freq=10
input_file=$1
output_file=$2/inversion.csv
echo "Inversing..."

#harminv -t $dt 0.1-10 -v < data/data.csv > data/inversed.csv
harminv -t $dt $min_freq-$max_freq < $input_file > $output_file

echo "Done."
echo "Inversed data saved to $output_file"