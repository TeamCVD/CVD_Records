import numpy as np
import pandas as pd
import ast

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

# Importing the dataset
dataset = pd.read_csv('../Tokenized_Outputs/Encoded_code.csv')

# select first 100 dataset and visualize it
dataset_list = dataset['token'].tolist()

smoothenedArray = []

window = 10
for i in range(len(dataset_list)):
    y = ast.literal_eval(dataset_list[i])
    smoothed_y = np.convolve(y, np.ones(window)/window, mode='same')
    fft_y = np.fft.fft(smoothed_y)
    fft_y = np.abs(fft_y)/max(np.abs(fft_y))
    smoothed_y2 = np.convolve(fft_y, np.ones(window)/window, mode='same')
    smoothenedArray.append(smoothed_y)


smoothened_code_array = np.array(smoothenedArray)

# scalar = MaxAbsScaler()
# smoothened_code_array = scalar.fit_transform(smoothened_code_array)
# scaled_smoothened_code_array = scalar.transform(smoothened_code_array)


label = dataset['label']
# print(label.shape)

print("---------------------CONVERTING INTO DATAFRAME---------------")
df = pd.DataFrame({'token': smoothened_code_array.tolist(), 'label': label})

# Converting to csv and saving it
df.to_csv("smoothened.csv", index=False)
df.head()
