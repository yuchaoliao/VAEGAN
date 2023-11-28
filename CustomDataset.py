import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, filenames, batch_size, bit_size = 32):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        self.filenames= filenames
        self.batch_size = batch_size
        self.bit_size = bit_size
    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))   # Number of chunks.
    def __getitem__(self, idx): #idx means index of the chunk.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]   # This extracts one batch of file names from the list `filenames`.
        data = []
        temp = pd.read_csv(open(self.filenames[0],'r'))
        number_of_variable = float(len(temp)) / self.bit_size  # calculate number of variables may include directives
        number_of_variable = int(number_of_variable)  
        
        for file in batch_x:
            temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
            
            data.append(temp.values.reshape(number_of_variable, self.bit_size, 1)) # Convert column data to matrix like data with one channel
        data = np.asarray(data).reshape(-1, 1, number_of_variable, self.bit_size) 
        
        if idx == self.__len__():  
            raise IndexError

        return data