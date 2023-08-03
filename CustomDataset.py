import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, filenames, batch_size):
        # `filenames` is a list of strings that contains all file names.
        # `batch_size` determines the number of files that we want to read in a chunk.
        self.filenames= filenames
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))   # Number of chunks.
    def __getitem__(self, idx): #idx means index of the chunk.
        # In this method, we do all the preprocessing.
        # First read data from files in a chunk. Preprocess it. Extract labels. Then return data and labels.
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]   # This extracts one batch of file names from the list `filenames`.
        data = []
        for file in batch_x:
            temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
            # for k in range(7*32):
            #     temp.loc[-1] = [0.]  # adding a row
            #     temp.index = temp.index + 1  # shifting index
            #     temp.sort_index(inplace=True)
            # for k in range(7*32):
            #     temp.loc[len(temp.index)] = [0.]
            data.append(temp.values.reshape(20,32,1)) # Convert column data to matrix like data with one channel
        data = np.asarray(data).reshape(-1,1,20,32) # Because of Pytorch's channel first convention
        
        # The following condition is actually needed in Pytorch. Otherwise, for our particular example, the iterator will be an infinite loop.
        # Readers can verify this by removing this condition.
        if idx == self.__len__():  
            raise IndexError

        return data