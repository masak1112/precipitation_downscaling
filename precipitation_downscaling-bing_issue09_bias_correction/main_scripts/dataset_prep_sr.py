# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-13"
import pandas as pd

import xarray as xr
import torch
import numpy as np
import pathlib
import math
import os
import json
import gc
import torchvision
import dask
dask.config.set({'array.slicing.split_large_chunks': True})
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class PrecipDatasetSR(torch.utils.data.IterableDataset):
    """
    This is the class used for generate dataset generator for precipitation downscaling
    """

    def __init__(self, file_path: str = None, 
                 batch_size     : int = 16, 
                 vars_in        : list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "u700_in","v700_in"],
                 vars_out       : list = ["yw_hourly_tar"], 
                 seed           : int = 1234, 
                 k              : float = 0.1, 
                 mode           : str = "train",
                 stat_path      : str = None,):
        """
        file_path : the path to the directory of .nc files
        vars_in   : the list contains the input variable names
        var_out   : the list contains the output variable name
        batch_size: the number of samples per iteration
        seed      : specify a seed so that we can generate the same random index for shuffle function
        stat_path : the path to the directory of training .nc files
        mode      : "train", "val" or "test"
        k.        : the value for transofmration 
        """

        super(PrecipDatasetSR).__init__()

        self.file_path = file_path
        self.vars_in = vars_in
        self.var_out = vars_out
        self.batch_size = batch_size
        self.seed = seed
        self.k = k 
        self.mode = mode
        self.stat_path = stat_path

        '''
        '''
        # Search for files
        p = pathlib.Path(self.file_path)
        print("Going to open the following files:",self.file_path)

        files = sorted(p.rglob('*.nc'))
        print("Going to open the following files:", files)
        if len(files) < 1:
            raise RuntimeError('No files found.')
        print("Going to open the following files:", files)
        self.in_data,self.out_data,self.top,self.lats,self.lons,self.time = self.get_input_target(files)
        if self.mode == "train":
            self.idx_perm = self.shuffle()
            self.save_stats()
        else:
            self.idx_perm = np.arange(0, self.n_samples)
        print(f"all samples is {self.n_samples}")
        with open(os.path.join(self.stat_path, "statistics.json"), 'r') as file:
            self.stats = json.load(file)

    def save_stats(self):
        output_file = os.path.join(self.stat_path, "statistics.json")
        in_min = self.in_data.min().item()
        in_max = self.in_data.max().item()
        in_std = self.in_data.std().item()
        in_avg = self.in_data.mean().item()
        out_min = self.out_data.min().item()
        out_max = self.out_data.max().item()
        out_std = self.out_data.std().item()
        out_avg = self.out_data.mean().item()
        stats = {'in_min':in_min,'in_max':in_max,'in_std':in_std,'in_avg':in_avg,
                 'out_min':out_min,'out_max':out_max,'out_std':out_std,'out_avg':out_avg,}
        with open(output_file,'w') as f:
            json.dump(stats, f)
        
    def get_input_target(self,files):
        dt = xr.open_mfdataset(files, combine='nested', concat_dim='time').compute()
        hr_orig = dt['hr_orig'].values
        fcst = dt['fcst'].values
        # del nan value
        no_nan_idx = []
        for i in range(hr_orig.shape[0]):
            #remove Nan values and no rain images, or nan values in the input data
            if (not np.isnan(hr_orig[i]).any()) and np.max(hr_orig[i])>=0.1 and np.min(fcst[i])>=0:
                no_nan_idx.append(i) 
        print(f'nan len {len(no_nan_idx)}')
        dt = dt.isel(time = no_nan_idx)
        fcst = dt['fcst'].values
        hr_orig = dt['hr_orig'].values
        top  = dt['tops'].values
        lats = dt['lats'].values
        lons = dt['lons'].values
        #time = torch.from_numpy(dt['time'].values.astype('datetime64[s]').astype('int'))
        time = pd.to_datetime(dt['time'].values)
        time = np.stack(
            (time.year.values, time.month.values, time.day.values,time.hour.values), axis=1)
        self.n_samples = fcst.shape[0]
        assert fcst.shape[0] == hr_orig.shape[0]
        # top,lat,lon,t
        
        print(f"fcst shape is {fcst.shape},hr_orig shape is { hr_orig.shape}")
        top = (top-312.71216) / 442.65375

        # y = log(x + k) - log(k) xr.ufuncs.log(da)
        fcst  = np.log(fcst + self.k) - np.log(self.k)
        hr_orig  = np.log(hr_orig+ self.k) - np.log(self.k)
        return fcst,hr_orig,top,lats,lons,time


    def __iter__(self):
         #min-max score
        def normalize(x, x_min,x_max):
            return ((x - x_min)/(x_max-x_min))

        def normalize(x, avg,std):
            return (x-avg)/std
        iter_start, iter_end = 0, int(len(self.idx_perm)/self.batch_size)
        for bidx in range(iter_start, iter_end):
            idx_list = self.idx_perm[range(bidx * self.batch_size, (bidx + 1) * self.batch_size )]
            x = torch.from_numpy(self.in_data[idx_list].astype(np.float32))
            y = torch.from_numpy(self.out_data[idx_list].astype(np.float32))
            lons = torch.from_numpy(self.lons[idx_list].astype(np.float32))
            lats = torch.from_numpy(self.lats[idx_list].astype(np.float32))
            x_top = torch.from_numpy(self.top[idx_list].astype(np.float32))
            t = torch.from_numpy(self.time[idx_list])
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            x_top = x_top.unsqueeze(1)
            # x = normalize(x,self.stats['in_min'],self.stats['in_max'])
            # y = normalize(y,self.stats['out_min'],self.stats['out_max'])
            # x = normalize(x,self.stats['in_min'],self.stats['in_max'])
            # y = normalize(y,self.stats['out_min'],self.stats['out_max'])
            yield  {'L': x, 'H': y, "idx": idx_list, "T":t, "lons":lons, "lats":lats, "top":x_top,}

    def shuffle(self):
        """
        shuffle the index 
        """
        print("Shuffling the index ....")
        np.random.seed(self.seed)
        idx_perm = np.random.permutation(np.arange(self.n_samples)) 
        return idx_perm

    
   

def run():
    data_loader = PrecipDatasetSR(
        file_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/deviation_correction/train/",
        vars_in = ["tp"],
        vars_out = ["yw_hourly_tar"],
        mode= 'train',
        k = 0.005,
        stat_path= '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/deviation_correction/train'
        )
    print("created data_loader")
    x = 0
    for batch_idx, train_data in enumerate(data_loader):
        inputs = train_data["L"]
        m = inputs.min()
        m1 = inputs.max()
        print("inputs shape:", inputs.shape) # （ 32,5,16,16）
        target = train_data["H"]
        m3 = target.min()
        m4 = target.max()
        print("target shape:", target.shape) # （ 32,5,16,16）
        x += inputs.shape[0]
    print(x)
      

if __name__ == "__main__":
    run()


'''
nohup pyth
on precipitation_downscaling-bing_issue09_bias_correction/main_scripts/main_train.py --train_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/deviation_correction/train  --val_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/downscaling_precipitation/precip_dataset_new/val/2017-01 --save_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/zx_results/ex16_all_val_norm_weight_maeloss_80_end1 --model_type unet --epochs 350 --k 0.005  > output.log 2>&1 &
'''









