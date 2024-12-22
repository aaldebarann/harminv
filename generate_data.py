import numpy as np
import pandas as pd

time = np.linspace(0, 1, 100)

#signal = np.exp(-time)

signal = np.exp((-1+2j)*time)

# Convert the numpy array to a DataFrame
df = pd.DataFrame(signal)

# Print the DataFrame to a CSV file
df.to_csv('./data/data.csv', index=False, header=False)

with open('./data/data.csv', 'r') as file:
    data = file.read()

data = data.replace('j', 'i').replace('(', '').replace(')', '')

with open('./data/data.csv', 'w') as file:
    file.write(data)

