#!/bin/bash

dt=0.01

#harminv -t 0.01 1-10 -v < data/data.csv > data/inversed.csv
harminv -t 0.01 1-10 < data/data.csv> data/inversed.csv

echo "Done."