#!/bin/bash

dt=0.01

echo "Inversing..."

#harminv -t $dt 0.1-10 -v < data/data.csv > data/inversed.csv
harminv -t $dt 0.1-10 < $1 > $2

echo "Done."
echo "Inversed data saved to ./data/inversed.csv"