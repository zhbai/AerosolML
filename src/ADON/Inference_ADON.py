## Inference using new trained ML model for the app of aerosol emulation 
## by Z.Bai, zhebai@lbl.gov

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import xarray as xr
from time import perf_counter
from utils import *
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
# torch.get_num_threads()

# device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cpu")
fea_data = xr.open_dataset("saved_data.nc", engine="netcdf4")
lon, lat, lev = fea_data['longitude'], fea_data['latitude'], fea_data['level']
mean_X, std_X, mean_y, std_y = fea_data['mean_X'].values, fea_data['std_X'].values, fea_data['mean_y'].values, fea_data['std_y'].values
cldfr_idx = fea_data['cldfr_idx'].values
X_test = fea_data['X_test'].values
ymean = fea_data['ymean'].values
v = fea_data['basis'].values

def testslice(X_test, cldfr_idx, i):
    X_test_loc = np.empty((cldfr_idx.shape[1],3), dtype='float64')    
    X_test_loc[:,0] = lev[cldfr_idx[0]]
    ncol = cldfr_idx[1]
    X_test_loc[:,1] = lat[ncol]
    X_test_loc[:,2] = lon[ncol]
    X_test_t = np.reshape(i*np.ones(X_test_loc.shape[0]),(-1,1))#.astype('float32')
    X_test = np.hstack((X_test, np.hstack((X_test_loc, X_test_t))))
    return X_test
    
class InferenceModel:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize the InferenceModel class.
        
        Parameters:
            model_path (str): path to the trained model
            device(str): device to run the inference on ('cpu' or 'cuda')
        """
        
        self.device = torch.device(device)
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval() # evaluation mode
        self.tensor = transforms.Compose([toTensor()])
        # define preprocessing transformation
        self.transform = transforms.Compose([
            # transforms.ToTensor(),
            # toTensor(),
            # powTransform(1.0/3),
            normalize(mean_X, std_X)
        ])
            
    def load_model(self, model_path):
        """
        Load the trained model 
        """
        branch_net = BranchNet(39, 20+1)
        trunk_net = TrunkNet(4, 20, torch.tensor(v, dtype=torch.float64))
        model = MyNet_ADON(branch_net, trunk_net, torch.tensor(ymean, dtype=torch.float64)).double()
        model.load_state_dict(torch.load(model_path, map_location = self.device))
        for param in model.parameters():
            param.requires_grad = False
        return model

    def preprocess(self, sample):
        sample = self.tensor(sample)
        sample[:,:-4] = odd_pow(sample[:,:-4], 1.0/3)
        sample = self.transform(sample)
        return sample

    def postprocess(self, sample):
        sample = transform_y(sample, mean_y, std_y)
        sample = odd_pow(sample,3)
        return sample

    def predict(self, sample):
        sample = self.preprocess(sample)
        sample = sample.to(self.device)
        with torch.no_grad():
            output = self.model(sample[:,:-4],sample[:,-4:])
        output = self.postprocess(output)
        return output

# Example
if __name__ == "__main__":
    model_path = './model_Gelu_L1_500epoch_cbrt_DON53_PODloc_Ens1_4season_8days_43_20_sbatch.pth'
    X_aug = testslice(X_test, cldfr_idx, 72)
    t0_start = perf_counter()
    inference_model = InferenceModel(model_path, device=device)
    t0_stop = perf_counter()
    print("Elapsed time for loading the inference model in seconds:", t0_stop-t0_start)
    t1_start = perf_counter()
    output = inference_model.predict(X_aug)
    t1_stop = perf_counter()
    # print("Elapsed time for predicting output in seconds:", t1_stop-t1_start)
    print(output)