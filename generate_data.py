import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import subprocess
import os

def save(signal, path):
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(signal)

    # Print the DataFrame to a CSV file
    df.to_csv(path, index=False, header=False)

    with open(path, 'r') as file:
        data = file.read()

    data = data.replace('j', 'i').replace('(', '').replace(')', '')

    with open(path, 'w') as file:
        file.write(data)

    print('Data saved to {0}'.format(path))

def exp(A, freq, phase, decay, time):
    return A * np.exp( -1j * (2 * np.pi * freq * time - phase) - decay * time)

def cos(A, freq, phase, decay, time):
    return A * np.exp(-decay * time) * np.cos(2 * np.pi * freq * time - phase)

def sample_1(output_path):
    time = np.linspace(0, 1, 100)
    signal = exp(A=1, freq=1, phase=2, decay=0.1, time=time)
    save(signal, path=output_path)

def sample_2(output_path):
    time = np.linspace(0, 1, 100)
    signal = exp(A=1, freq=1, phase=2, decay=0.1, time=time) + exp(A=2, freq=0.5, phase=0, decay=0.1, time=time)
    save(signal, path=output_path)

def sample_3(output_path):
    time = np.linspace(0, 1, 100)
    signal = exp(A=1, freq=1, phase=2, decay=0.1, time=time) + exp(A=2, freq=0.5, phase=0, decay=0.1, time=time)
    save(signal, path=output_path)

def inversion(signal, dir):
    save(signal, os.path.join('data', dir, 'data.csv'))
    subprocess.run(['./harminv.sh', dir])

def get_inversed(time, dir):
    subprocess.run(['./harminv__.sh', dir])
    df = pd.read_csv(os.path.join(dir, 'inversed.csv'), sep=',\s+')
    print(os.path.join(dir, 'inversed.csv'))
    
    signal = np.zeros(len(time), dtype='complex128')
    print(len(time))
    print(df.head())
    print(df.columns)
    for index, row in df.iterrows():
        #print(row['frequency'])
        signal += exp(A=row['amplitude'], freq=row['frequency'], phase=row['phase'], decay=row['decay constant'], time=time)
    signal = signal.astype('float64')
    return signal

def main():
    time = np.linspace(0, 1, 100)
    signal = cos(A=1, freq=1, phase=2, decay=0.1, time=time) 
    signal += cos(A=0.25, freq=3, phase=2, decay=0.2, time=time)
    signal += cos(A=0.2, freq=7, phase=1, decay=0.2, time=time)

    sigma = 1

    noised = gaussian_filter(signal, sigma=sigma, radius=5)

    name = r'noised_{0}'.format(sigma)
    dir = os.path.join('data', name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    save(noised, os.path.join(dir, 'data.csv'))


if __name__ == '__main__':
    sys.exit(main() or 0)