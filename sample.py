import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import subprocess
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",
                        help="Directory to save output files",
                        required=True,)
    parser.add_argument("--sigma",
                        help="sigma",
                        required=True,
                        type=float)
    return parser.parse_args()

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
    print('{0}, {1}, {2}, {3}'.format(freq, decay, A, phase))
    return A * np.exp( -1j * (2 * np.pi * freq * time - phase) - decay * time)

def cos(A, freq, phase, decay, time):
    print('{0}, {1}, {2}, {3}'.format(freq, decay, A/2, phase))
    print('{0}, {1}, {2}, {3}'.format(-freq, decay, A/2, phase))
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

def get_inversed(time, dir):
    input_data = os.path.join(dir, 'data.csv')
    subprocess.run(['./harminv.sh', input_data, dir])
    df = pd.read_csv(os.path.join(dir, 'inversion.csv'), engine='python', sep=', ')
    
    signal = np.zeros(len(time), dtype='complex128')
    print('Inversed signal')
    print('frequency, decay, amplitude, phase')
    for index, row in df.iterrows():
        signal += exp(A=row['amplitude'], freq=row['frequency'], phase=row['phase'], decay=row['decay constant'], time=time)
    signal = signal.astype('float64')
    return signal

def plot(time, signal, inversed, output_dir, sigma):
    plt.title(r'Исходный и восстановленный сигнал при зашумлении с $\sigma$={0}'.format(sigma))
    plt.plot(time, signal, 'b', label='Исходный сигнал')
    plt.plot(time, inversed, 'r', label='Восстановленный сигнал')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'original_and_inverse_signal.png'))
    plt.show()

def main():
    args = parse_arguments()
    sigma = args.sigma
    dir = args.dir

    time = np.linspace(0, 1, 100)
    print('Original signal')
    print('frequency, decay, amplitude, phase')
    signal = cos(A=1, freq=1, phase=2, decay=0.1, time=time) 
    signal += cos(A=0.25, freq=3, phase=2, decay=0.2, time=time)
    signal += cos(A=0.2, freq=7, phase=1, decay=0.2, time=time)

    noised = gaussian_filter(signal, sigma=sigma, radius=5)

    if not os.path.exists(dir):
        os.mkdir(dir)
    save(noised, os.path.join(dir, 'data.csv'))
    
    inversed = get_inversed(time, dir)

    plot(time, signal, inversed, dir, sigma)


if __name__ == '__main__':
    sys.exit(main() or 0)