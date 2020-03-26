
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self,datasetfilepath,indexfilepath):
        data=np.array(pd.read_csv(datasetfilepath,sep=','))
        indices=self.extract_indices(indexfilepath)
        data=data[indices]
        self.X=np.array(data)[:,:-1]
        self.Y=np.array(data)[:,-1]
    
    def extract_indices(self,indexfilepath):
	    trainindices=pd.read_csv(indexfilepath)
	    trainindices=np.array(trainindices,dtype=int).reshape(-1,)
	    return trainindices

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        return self.X[idx,:],self.Y[idx]
        