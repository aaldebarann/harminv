#!/bin/bash

dt=0.01
min_freq=0.1
max_freq=10
echo "Inversing..."

#harminv -t $dt 0.1-10 -v < data/data.csv > data/inversed.csv
harminv -t $dt $min_freq-$max_freq < $1 > $2/inversion.csv

echo "Done."
echo "Inversed data saved to ./data/inversed.csv"