# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-13"


import xarray as xr
import torch
import numpy as np
import pathlib
import math
import torchvision
import os
import json
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')



class PrecipCorrDatasetInter(torch.utils.data.IterableDataset):
    """
    This is the class used for generate dataset generator for precipitation downscaling
    """

    def __init__(self, file_path: str = None, batch_size: int = 32,
                 seed: int = 1234, k: float = 0.5, stat_path: str = None):
        """
        "This is the iteration for precipitgaiton correction (after diffusion)"
        file_path : the path to the directory of .nc files
        batch_size: the number of samples per iteration
        patch_size: the patch size for low-resolution image,
                    the corresponding high-resolution patch size should be muliply by scale factor (sf)
        sf        : the scaling factor from low-resolution to high-resolution
        seed      : specify a seed so that we can generate the same random index for shuffle function
        stat_dir  : the path to the directory of training .nc files
        """

        super(PrecipCorrDatasetInter).__init__()

        self.file_path = file_path
        self.batch_size = batch_size
        self.seed = seed
        self.k = k 
        self.stat_path = stat_path



        # Search for files
        p = pathlib.Path(self.file_path)
        print("Going to open the following files:",self.file_path)

        files = sorted(p.rglob('*.nc'))
        print("Going to open the following files:", files)
        if len(files) < 1:
            raise RuntimeError('No files found.')
        print("Going to open the following files:", files)
       
        self.vars_in_patches_list, self.vars_out_patches_list, self.times_patches_list, self.lats, self.lons, self.tops  = self.process_netcdf(files)
        print('self.times_patches_list: {}'.format(self.times_patches_list))


        stat_file = os.path.join(stat_path, "statistics.json")
        with open(stat_file,'r') as f:
            stat_data = json.load(f)
        
        print("stat_data",stat_data)
        self.vars_out_patches_mean = stat_data["yw_hourly_tar_avg"]
        self.vars_out_patches_std = stat_data["yw_hourly_tar_std"]

        print("The total number of samples after filtering NaN values:", len(self.vars_in_patches_list))
        
        self.n_samples = len(self.vars_in_patches_list)
   
        self.idx_perm = np.arange(0, self.n_samples)
        #print("var_out size",self.vars_out_patches_list)
 

    def process_netcdf(self, filenames: int = None):
        """
        process netcdf files: filter the Nan Values, split to patches
        """
        print("Loading data from the file:", filenames)
        dt = xr.open_mfdataset(filenames,   concat_dim="time", combine="nested")
        print("dt",dt)
        # get input variables, and select the regions
        inputs = dt["fcst"].values
        outputs = dt["hr"].values
        tops = dt["tops"].values
        
        self.idx = inputs.shape[0]
        n_lat = dt["lats"].values.shape[0]
        n_lon = dt["lons"].values.shape[0]


        # log-transform -> log(x+k)-log(k)
        inputs = np.log(inputs+self.k)-np.log(self.k)
        outputs = np.log(outputs+self.k)-np.log(self.k)

        da_in = torch.from_numpy(inputs)
        da_out = torch.from_numpy(outputs)
      
        times = dt["time"].values  # get the timestamps
        times = np.transpose(np.stack([dt["time"].dt.year,dt["time"].dt.month,dt["time"].dt.day,dt["time"].dt.hour]))  
        times = torch.from_numpy(times)      

        lats = dt["lats"].values
        lons = dt["lons"].values

    
        return da_in, da_out, times, lats,lons, tops




    def __iter__(self):
 
        
        iter_start, iter_end = 0, int(self.idx/self.batch_size)-1 
 
        self.idx = 0
        lon = torch.from_numpy(self.lons)
        lat = torch.from_numpy(self.lats)
        top = torch.from_numpy(self.tops)
        #min-max score
        def normalize(x, x_min,x_max):
            return ((x - x_min)/(x_max-x_min))


        for bidx in range(iter_start, iter_end):

            #initialise x, y for each batch
            # x  stores the low resolution images, y for high resolution,
            # t is the corresponding timestamps, cidx is the index
            x = torch.zeros(self.batch_size, 1, 160, 160)
            y = torch.zeros(self.batch_size, 160, 160)
            x_top = torch.zeros(self.batch_size, 1, 160, 160)
            t = torch.zeros(self.batch_size, 4, dtype = torch.int)
            cidx = torch.zeros(self.batch_size, 1, dtype = torch.int) #store the index
            lats = torch.zeros(self.batch_size, 160)
            lons = torch.zeros(self.batch_size, 160)

            for jj in range(self.batch_size):
                print("idx",self.idx)
                cid = self.idx_perm[self.idx]
                x[jj] = (self.vars_in_patches_list[cid] - self.vars_out_patches_mean) / self.vars_out_patches_std
                y[jj] = (self.vars_out_patches_list[cid] - self.vars_out_patches_mean) / self.vars_out_patches_std
                t[jj] = self.times_patches_list[cid]
                lats[jj] = lat[cid]
                lons[jj] = lon[cid]
                cidx[jj] = torch.tensor(cid, dtype=torch.int)
                x_top[jj] = normalize(top[cid], -182, 3846)

                self.idx += 1

            yield  {'L': x, 'H': y, "idx": cidx, "T":t, "lons":lons, "lats":lats, "top":x_top}

def run():
    data_loader = PrecipCorrDatasetInter(file_path="/p/home/jusers/gong1/juwels/bing/precipitation_downscaling/results/exp_20231203_diffusion_t4_newscale2_only_rain_t02_yan_zscore_05_top_test_train_correction"
    ,stat_path="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train")
    print("created data_loader")
    for batch_idx, train_data in enumerate(data_loader):
        inputs = train_data["L"]
        target = train_data["H"]
        idx = train_data["idx"]
        times = train_data["T"]
        print("inputs", inputs.size())
        print("target", target.size())
        print("idx", idx)
        print("batch_idx", batch_idx)
        print("timestamps,", times)

if __name__ == "__main__":
    run()











