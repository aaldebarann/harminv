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
    parser.add_argument("-in", "--input_path", help="Path to source .csv file", required=True)
    parser.add_argument("-out", "--output_path", help="Output .csv file path", required=True)
    parser.add_argument("-v", "--verbose", help="Output .csv file path", required=False)
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

def run_inversion(input):
    subprocess.run(['./harminv.sh', input])
    df = pd.read_csv('./data/inversed.csv', engine='python', sep=', ')
    return df

def build_signal(harmonics, time):
    signal = np.zeros(len(time), dtype='complex128')
    print('Inversed signal')
    print('frequency, decay, amplitude, phase')
    for index, row in harmonics.iterrows():
        signal += exp(A=row['amplitude'], freq=row['frequency'], phase=row['phase'], decay=row['decay constant'], time=time)
    signal = signal.astype('float64')
    return signal

def plot(time, signal, inversed, output_dir):
    plt.title(r'Восстановленный сигнал при зашумлении')

    print(time.shape, signal.shape, inversed.shape)
    plt.plot(time, signal, 'bo', label='Данные эксперимента')
    plt.plot(time, inversed, 'r', label='Восстановленный сигнал')
        
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'original_and_inverse_signal.png'))
    plt.show()

def load_signal(input):
    df = pd.read_csv(input, engine='python', headers=None)
    data = np.asarray(df)
    print(df.head())
    data = data[0]
    return data

def main():
    args = parse_arguments()
    
    signal = load_signal(args.input_path)
    time = np.linspace(0, len(signal), 1)
    harmonics = run_inversion(args.input_path)    
    inversed_signal = build_signal(harmonics, time)

    plot(time, signal, inversed_signal, dir)


if __name__ == '__main__':
    sys.exit(main() or 0)