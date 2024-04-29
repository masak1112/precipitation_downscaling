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
import os
import json
import gc
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"

cuda = torch.cuda.is_available()
if cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class PrecipDatasetInter(torch.utils.data.IterableDataset):
    """
    This is the class used for generate dataset generator for precipitation downscaling
    """

    def __init__(self, file_path: str = None, 
                 batch_size     : int = 32, 
                 patch_size     : int = 16,
                 vars_in        : list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "u700_in","v700_in"],
                 vars_out       : list = ["yw_hourly_tar"], 
                 sf             : int = 10,
                 seed           : int = 1234, 
                 k              : float = 0.5, 
                 mode           : str = "train",
                 stat_path      : str = None,
                 local          : bool = False):
        """
        file_path : the path to the directory of .nc files
        vars_in   : the list contains the input variable names
        var_out   : the list contains the output variable name
        batch_size: the number of samples per iteration
        patch_size: the patch size for low-resolution image,
                    the corresponding high-resolution patch size should be muliply by scale factor (sf)
        sf        : the scaling factor from low-resolution to high-resolution
        seed      : specify a seed so that we can generate the same random index for shuffle function
        stat_path : the path to the directory of training .nc files
        mode      : "train", "val" or "test"
        k.        : the value for transofmration 
        local.    : True: local sampling, False: Global sampling
        """

        super(PrecipDatasetInter).__init__()

        self.file_path = file_path
        self.patch_size = patch_size
        self.sf = sf  # scaling factor
        self.vars_in = vars_in
        self.var_out = vars_out
        self.batch_size = batch_size
        self.seed = seed
        self.k = k 
        self.mode = mode
        self.stat_path = stat_path
        
        #initialise the list to store the inputs and outputs
        self.vars_in_patches_list = []
        self.vars_out_patches_list = []
        self.times_patches_list = []
        
        _prcpids = ['yw','cp','lsp','tp']
        self._prcp_indexes = []
        i = 0

        while i < len(vars_in):
            for j in range(len(_prcpids)):
                if _prcpids[j] in vars_in[i]:
                    self._prcp_indexes.append(i)
            i += 1
        print('self.prcp_indexes: {}'.format(self._prcp_indexes))

        # Search for files
        p = pathlib.Path(self.file_path)
        print("Going to open the following files:",self.file_path)

        files = sorted(p.rglob('preproc_ifs_radklim_*.nc'))
        print("Going to open the following files:", files)
        if len(files) < 1:
            raise RuntimeError('No files found.')
        print("Going to open the following files:", files)

        if local:
            self.vars_in_patches_list, self.vars_out_patches_list, self.times_patches_list  = self.process_netcdf_local(files)
        else:
            self.vars_in_patches_list, self.vars_out_patches_list, self.times_patches_list = self.process_netcdf(files)

        print('self.times_patches_list: {}'.format(self.times_patches_list))
        stat_file = os.path.join(stat_path, "statistics.json")


        #get the topography dataset
        self.dt_top = xr.open_dataset("/p/project/deepacf/maelstrom/data/ap5/downscaling_ifs2radklim/srtm_data/topography_srtm_ifs2radklim.nc")
    

        if self.mode == "train":
            self.vars_in_patches_min = [] 
            self.vars_in_patches_max = [] 
            self.vars_in_patches_avg = []
            self.vars_in_patches_std = [] 
            for i in range(self.vars_in_patches_list.size()[1]):
                self.vars_in_patches_list[:,i,:,:]= torch.nan_to_num(self.vars_in_patches_list[:,i,:,:], nan=0)
                self.vars_in_patches_min.append(torch.min(self.vars_in_patches_list[:,i,:,:]))
                self.vars_in_patches_max.append(torch.max(self.vars_in_patches_list[:,i,:,:]))
                self.vars_in_patches_avg.append(torch.mean(self.vars_in_patches_list[:,i,:,:]))
                self.vars_in_patches_std.append(torch.std(self.vars_in_patches_list[:,i,:,:]))

            self.vars_out_patches_min = torch.min(self.vars_out_patches_list)
            self.vars_out_patches_max = torch.max(self.vars_out_patches_list)
            self.vars_out_patches_avg = torch.mean(self.vars_out_patches_list)
            self.vars_out_patches_std = torch.std(self.vars_out_patches_list)
            self.save_stats()
        else:
            with open(stat_file,'r') as f:
                stat_data = json.load(f)
                print("Loading the stats file:", stat_file)
            self.vars_in_patches_min = []
            self.vars_in_patches_max = []
            self.vars_in_patches_avg = []
            self.vars_in_patches_std = [] 
            for i in range(len(self.vars_in)):
                self.vars_in_patches_min.append(stat_data[self.vars_in[i]+'_min'])
                self.vars_in_patches_max.append(stat_data[self.vars_in[i]+'_max'])
                self.vars_in_patches_avg.append(stat_data[self.vars_in[i]+'_avg'])
                self.vars_in_patches_std.append(stat_data[self.vars_in[i]+'_std'])
            self.vars_out_patches_min = stat_data[self.var_out[0]+'_min']
            self.vars_out_patches_max = stat_data[self.var_out[0]+'_max']
            self.vars_out_patches_avg = stat_data[self.var_out[0]+'_avg']
            self.vars_out_patches_std = stat_data[self.var_out[0]+'_std']
        
        print("The total number of samples after filtering NaN and zeros values:", len(self.vars_in_patches_list))
        
        self.n_samples = len(self.vars_in_patches_list)
        #print("var_out size",self.vars_out_patches_list)
        
        if self.mode == "train":
            self.idx_perm = self.shuffle() 
        else:
            self.idx_perm = np.arange(1, self.n_samples)
        
    def save_stats(self):
        output_file = os.path.join(self.stat_path, "statistics.json")
        stats = {}
        for i in range(len(self.vars_in)):
            print("save stats")
            print("i",self.vars_in_patches_min[i])
            key = self.vars_in[i]+'_min'
            stats.update({key:float(self.vars_in_patches_min[i])})
            key = self.vars_in[i]+'_max'
            stats.update({key:float(self.vars_in_patches_max[i])}) 
            key = self.vars_in[i]+'_avg'
            stats.update({key:float(self.vars_in_patches_avg[i])}) 
            key = self.vars_in[i]+'_std'
            stats.update({key:float(self.vars_in_patches_std[i])}) 
        
        key = self.var_out[0]+'_min'
        stats.update({key:float(self.vars_out_patches_min)})
        key = self.var_out[0]+'_max'
        stats.update({key:float(self.vars_out_patches_max)})

        key = self.var_out[0]+'_avg'
        stats.update({key:float(self.vars_out_patches_avg)})
        key = self.var_out[0]+'_std'
        stats.update({key:float(self.vars_out_patches_std)})
        #save to output directory
        with open(output_file,'w') as f:
            json.dump(stats, f)
        print("The statistic has been stored to the json file: ", output_file)



    def process_netcdf(self, filenames: int = None, local=False):
        """
        process netcdf files: filter the Nan Values, split to patches
        """
        print("Loading data from the file:", filenames)
        dt = xr.open_mfdataset(filenames)
        
        # get input variables, and select the regions
        inputs = dt[self.vars_in[:-1]].isel(lon = slice(2, 114)).sel(lat = slice(47.5, 60))
        #Add new variables tp in to the datasets
        tp =  [((inputs["cp_in"][i] +inputs["lsp_in"][i])*1000-(inputs["cp_in"][i-1] +inputs["lsp_in"][i-1])*1000).values 
               for i in range(len(inputs.time))]
        
        #replace nan with zero values
        inputs["tp"] = (['time', 'lat', 'lon'], tp)
        #print("inputs",inputs)
    
 
        #inputs = dt[self.vars_in].sel(lon = slice(10, 12)).sel(lat = slice(50, 52))
        lats = inputs["lat"].values
        lons = inputs["lon"].values
        self.dx = lons[1] - lons[0]

        lon_sl , lat_sl = slice(lons[0]-self.dx/2, lons[-1]+self.dx/2), slice(lats[0]-self.dx, lats[-1]+self.dx)
        #output = dt[self.var_out].sel({"lon_tar": lon_sl, "lat_tar":lat_sl})
        output = dt[self.var_out].sel({"lon_tar": lon_sl, "lat_tar":lat_sl}).isel(lon_tar = slice(0, 1120)).isel(lat_tar = slice(0, 840))
        
        # Get lons and lats from the output dataset
        lats_tar = output["lat_tar"].values
        lons_tar = output["lon_tar"].values

        n_lat = output["lat_tar"].values.shape[0]
        n_lon = output["lon_tar"].values.shape[0]

        self.n_patches_x = int(n_lon/(self.patch_size * self.sf))
        self.n_patches_y = int(n_lat/(self.patch_size * self.sf))
        self.num_patches_img = self.n_patches_x * self.n_patches_y
        print("num_patches_images are", self.n_patches_x , self.n_patches_y)

        inputs_nparray = inputs.to_array(dim = "variables").squeeze().values.astype(np.float32)
        outputs_nparray = output.to_array(dim = "variables").squeeze().values.astype(np.float32)
        # log-transform -> log(x+k)-log(k)
        
        inputs_nparray = np.nan_to_num(inputs_nparray, nan=0)

        da_in = torch.from_numpy(inputs_nparray)
        da_out = torch.from_numpy(outputs_nparray)
        

        self.n_samples = da_out.shape[0]

        del inputs_nparray, outputs_nparray
        gc.collect()

        times = inputs["time"].values  # get the timestamps
        times = np.transpose(np.stack([dt["time"].dt.year,dt["time"].dt.month,dt["time"].dt.day,dt["time"].dt.hour]))        

        # split into small patches, the return dim are [vars, samples,n_patch_x, n_patch_y, patch_size, patch_size]
        vars_in_patches = da_in.unfold(2, self.patch_size, 
                                       self.patch_size).unfold(3, 
                                                               self.patch_size, self.patch_size)
        vars_in_patches_shape = list(vars_in_patches.shape)

        #sanity check to make sure the number of patches is as we expected
        assert self.n_patches_x * self.n_patches_y == int(vars_in_patches_shape[2] * vars_in_patches_shape[3])
        
        vars_in_patches = torch.reshape(vars_in_patches, [vars_in_patches_shape[0],
                                                          vars_in_patches_shape[1] * vars_in_patches_shape[2] *
                                                          vars_in_patches_shape[3],
                                                          vars_in_patches_shape[4], vars_in_patches_shape[5]])

        
        
        vars_in_patches = torch.transpose(vars_in_patches, 0, 1) #[samples,vars,n_lats,m_lons]
         
        ## Replicate times for patches in the same image
        times_patches = torch.from_numpy(np.array([ x for x in times for _ in range(self.num_patches_img)]))
        # lons_patches = torch.from_numpy(np.array([ x for x in lons for _ in range(self.num_patches_img)]))
        # lats_patches = torch.from_numpy(np.array([ x for x in lats for _ in range(self.num_patches_img)]))
     
        
        ## sanity check 
        assert len(times_patches) ==  vars_in_patches_shape[1] * vars_in_patches_shape[2] * vars_in_patches_shape[3]

        vars_out_patches = da_out.unfold(1, self.patch_size * self.sf,
                                         self.patch_size * self.sf).unfold(2,
                                                                       self.patch_size * self.sf,
                                                                       self.patch_size * self.sf)
                                                         
        vars_out_patches_shape = list(vars_out_patches.shape)
        vars_out_patches = torch.reshape(vars_out_patches,
                                         [vars_out_patches_shape[0] * vars_out_patches_shape[1] *
                                          vars_out_patches_shape[2],
                                          vars_out_patches_shape[3], vars_out_patches_shape[4]])
        
        del da_in, da_out 
        gc.collect()

        #get lats and lons for each patch
        lats_in = torch.from_numpy(lats_tar)
        lons_in = torch.from_numpy(lons_tar)
        self.lats_in_patches = lats_in.unfold(0, self.patch_size*self.sf, 
                                              self.patch_size*self.sf)
        self.lons_in_patches = lons_in.unfold(0, self.patch_size*self.sf, 
                                              self.patch_size*self.sf)
    
        no_nan_idx = []
        
        #clean the datasets
        # get the indx if there any nan in the sample，
        # Follow the paper Harris, Each sub-image was scored on “how rainy” 
        # it was in that image and categorized into one of four bins,
        # depending on what fraction of pixels contained rainfall (>0.1 mm/hr) – 0%–25%, 25%–50%, 50%–75%, or 75%–100%.



        # threshold = 0
        # for i in range(vars_out_patches.shape[0]):
        #     #remove Nan values and no rain images

        #     if (not torch.isnan(vars_out_patches[i]).any() and torch.max(vars_out_patches[i])>=torch.tensor(0.1).to(device)):

        #         # #Calculate the fraction of rain pixels
        #         # rain_pix = torch.numel(vars_out_patches[i][vars_out_patches[i].to(device) >= torch.log10(torch.tensor(0.1)).to(device)])
        #         # print("Index {}: Rain_pixels: {}".format(i,rain_pix))
        #         # if rain_pix/((self.patch_size*self.sf)**2) > threshold:
        #         no_nan_idx.append(i) 
        #         # else:
        #         #     pass
                
    
        for i in range(vars_out_patches.shape[0]):
            #remove Nan values and no rain images, or nan values in the input data
            if (not torch.isnan(vars_out_patches[i]).any()) and torch.min(vars_in_patches[i][-1])>0 and torch.max(vars_out_patches[i])>=torch.tensor(0.1).to(device):
                no_nan_idx.append(i) 


        #Yan's method
        # log-transform -> log(x+k)-log(k)
        # print("vars_in_patches[self._prcp_indexes] ", vars_in_patches[self._prcp_indexes] )
        # print("torch.log(torch.tensor(self.k).to(device)",torch.log(torch.tensor(self.k).to(device)))
        print("pre indexes are",self._prcp_indexes)
        
        print("var_in_patches",vars_in_patches[:,6,:,:])
        vars_in_patches[:,self._prcp_indexes,:,:] = torch.log((vars_in_patches[:,self._prcp_indexes,:,:]) +  
                                                              torch.tensor(self.k).to("cpu")) - torch.log(torch.tensor(self.k).to("cpu"))
        

        vars_out_patches= torch.log(vars_out_patches+
                                    torch.tensor(self.k).to("cpu"))-torch.log(torch.tensor(self.k).to("cpu"))

        # vars_in_patches[:,self._prcp_indexes,:,:] = torch.log10((vars_in_patches[:,self._prcp_indexes,:,:]) + torch.log(torch.tensor(self.k).to("cpu")))
        # vars_out_patches= torch.log10(vars_out_patches+ torch.log(torch.tensor(self.k).to("cpu")))


        ## The data processing appraoch based on Leinnon' 2023 paper  
        # vars_in_patches[:,self._prcp_indexes,:,:][vars_in_patches[:,self._prcp_indexes,:,:]<0.1]  = torch.log10(torch.tensor(0.02).to("cpu"))
        # vars_in_patches[:,self._prcp_indexes,:,:][vars_in_patches[:,self._prcp_indexes,:,:]>=0.1] = torch.log10(vars_in_patches[:,self._prcp_indexes,:,:][vars_in_patches[:,self._prcp_indexes,:,:]>=0.1])
        # vars_out_patches[vars_out_patches<0.1] = torch.log10(torch.tensor(0.02).to("cpu"))
        # vars_out_patches[vars_out_patches>=0.1] = torch.log10(vars_out_patches[vars_out_patches>=0.1])

    
        ## Harris paper data preprocessing method
        # inputs_nparray[self._prcp_indexes] = np.log10(inputs_nparray[self._prcp_indexes]+1)
        # outputs_nparray = np.log10(outputs_nparray+1)
        # print('inputs_nparray shape: {}'.format(inputs_nparray.shape))
        # print('inputs_nparray[self._prcp_indexes] shape: {}'.format(inputs_nparray[self._prcp_indexes].shape))

        #[no_nan_idx.append(i) for i in range(vars_out_patches.shape[0]) if (not torch.isnan(vars_out_patches[i]).any())]
       
        # Get the index if the zero values
        #[no_zeros_idx.append(i) for i in range(vars_out_patches.shape[0]) if torch.count_nonzero(vars_out_patches[i]) > 10]  
        # no_nan_idx = list(set(no_nan_idx+no_zeros_idx))
        print("no nan idx", len(no_nan_idx))
        #print("no_zeros_idx", no_zeros_idx)
        print("There are No. {} patches out of {} without Nan Values ".format(len(no_nan_idx), len(vars_out_patches)))
        
        # change the index from List to LongTensor type
        no_nan_idx = torch.LongTensor(no_nan_idx)

        # Only get the patch that without NaN values
        vars_out_pathes = torch.index_select(vars_out_patches, 0, no_nan_idx)
      
        vars_in_patches = torch.index_select(vars_in_patches, 0, no_nan_idx)
       
        
        times_patches= torch.index_select(times_patches, 0, no_nan_idx)
        # lons_no_nan = torch.index_select(lons_patches,0,no_nan_idx)
        # lats_no_nan = torch.index_select(lats_patches,0, no_nan_idx)
        assert len(vars_out_pathes) == len(vars_in_patches)
    
        return vars_in_patches, vars_out_pathes, times_patches


    def process_netcdf_local(self, filenames: int = None):
        """
        process netcdf files: filter the Nan Values, split to patches
        """
        print("Loading data from the file:", filenames)
        dt = xr.open_mfdataset(filenames)
       

        # get input variables, and select the regions
        inputs = dt[self.vars_in].isel(lon = slice(2, 18)).sel(lat = slice(47.5, 60)).isel(long = slice(0,16))
        output = dt[self.var_out].isel(lon_tar = slice(16, 160+16)).sel(lat_tar = slice(47.41, 60)).isel(long = slice(0,160))


        n_lat = inputs["lat"].values.shape[0]
        n_lon = inputs["lon"].values.shape[0]

        assert inputs.dims["time"] == output.dims["time"]
        assert inputs.dims["lat"] * self.sf == output.dims["lat_tar"]

        inputs_nparray = inputs.to_array(dim = "variables").squeeze().values.astype(np.float32)
        outputs_nparray = output.to_array(dim = "variables").squeeze().values.astype(np.float32)

        
        # log-transform -> log(x+k)-log(k)
        inputs_nparray[self._prcp_indexes] = np.log(inputs_nparray[self._prcp_indexes]+self.k)-np.log(self.k)
        outputs_nparray = np.log(outputs_nparray+self.k)-np.log(self.k)

        
        print('inputs_nparray shape: {}'.format(inputs_nparray.shape))
        print('inputs_nparray[self._prcp_indexes] shape: {}'.format(inputs_nparray[self._prcp_indexes].shape))

        da_in = torch.from_numpy(inputs_nparray)
        da_out = torch.from_numpy(outputs_nparray)
        del inputs_nparray, outputs_nparray
        gc.collect()
        times = inputs["time"].values  # get the timestamps
        times = np.transpose(np.stack([dt["time"].dt.year,dt["time"].dt.month,dt["time"].dt.day,dt["time"].dt.hour]))        

        print("Original input shape:", da_in.shape)

        no_nan_idx = []

        # get the indx if there any nan in the sample
        [no_nan_idx.append(i) for i in range(da_in.shape[0]) if not torch.isnan(da_in[i]).any()]

        print("There are No. {} patches out of {} without Nan Values ".format(len(no_nan_idx), len(da_out)))

        # change the index from List to LongTensor type
        no_nan_idx = torch.LongTensor(no_nan_idx)

        # Only get the patch that without NaN values
        vars_out_pathes_no_nan = torch.index_select(da_in, 0, no_nan_idx)
        vars_in_patches_no_nan = torch.index_select(da_out, 0, no_nan_idx)
        times_no_nan = torch.index_select(times, 0, no_nan_idx)
        assert len(vars_out_pathes_no_nan) == len(vars_in_patches_no_nan)

        return vars_in_patches_no_nan, vars_out_pathes_no_nan, times_no_nan   
    
    def shuffle(self):
        """
        shuffle the index 
        """
        print("Shuffling the index ....")
        multiformer_np_rng = np.random.default_rng(self.seed)
        idx_perm = multiformer_np_rng.permutation(self.n_samples)

        # restrict to multiples of batch size
        idx = int(math.floor(self.n_samples/self.batch_size)) * self.batch_size

        idx_perm = idx_perm[:idx]
        print("idx_perm",idx_perm)
        return idx_perm

    
    def __iter__(self):

        iter_start, iter_end = 0, int(len(self.idx_perm)/self.batch_size)-1 
        self.idx = 0

        #min-max score
        def normalize(x, x_min,x_max):
            return ((x - x_min)/(x_max-x_min))

        # def normalize(x, avg,std):
        #     return (x-avg)/std

        
        # def normalize(x, x_min,x_max):
        #     return (x - x_min)/(x_max-x_min)
        # This is previous approach, which perfrms bad for precipitation for UNET AND GAN
        # transform_x = torchvision.transforms.Normalize(self.vars_in_patches_avg, self.vars_in_patches_std)

        for bidx in range(iter_start, iter_end):

            #initialise x, y for each batch
            # x  stores the low resolution images, y for high resolution,
            # t is the corresponding timestamps, cidx is the index
            x = torch.zeros(self.batch_size, len(self.vars_in), self.patch_size, self.patch_size)
            y = torch.zeros(self.batch_size, self.patch_size * self.sf, self.patch_size * self.sf)
            x_top = torch.zeros(self.batch_size, 1, self.patch_size * self.sf, self.patch_size * self.sf)
            t = torch.zeros(self.batch_size, 4, dtype = torch.int)
            cidx = torch.zeros(self.batch_size, 1, dtype = torch.int) #store the index
            lons = torch.zeros(self.batch_size, self.patch_size*self.sf)
            lats = torch.zeros(self.batch_size, self.patch_size*self.sf)

            for jj in range(self.batch_size):
                
                cid = self.idx_perm[self.idx]
                for i in range(len(self.vars_in_patches_min)):
                    #x[jj][i] = normalize(self.vars_in_patches_list[cid][i],self.vars_in_patches_avg[i],self.vars_in_patches_std[i])
                     x[jj][i] = normalize(self.vars_in_patches_list[cid][i],self.vars_in_patches_min[i],self.vars_in_patches_max[i])

                # for i in range(len(self.vars_in_patches_min)):
                #     if i not in self._prcp_indexes:
                #          x[jj][i] = normalize(self.vars_in_patches_list[cid][i],self.vars_in_patches_min[i],self.vars_in_patches_max[i])

                
                # data transformation based on leinnon 2023 paperf
                y[jj] = ((self.vars_out_patches_list[cid] - self.vars_out_patches_min) / (self.vars_out_patches_max- self.vars_out_patches_min)) 
                #y[jj] = (self.vars_out_patches_list[cid] - self.vars_out_patches_avg) / (self.vars_out_patches_std) 
                # y[jj] = self.vars_out_patches_list[cid]

                t[jj] = self.times_patches_list[cid]
                lats_lons_cid = cid%self.num_patches_img 
                lons_cid = int(lats_lons_cid%self.n_patches_x)
                lats_cid = int(lats_lons_cid/self.n_patches_x)
                lats[jj] = self.lats_in_patches[lats_cid]
                lons[jj] = self.lons_in_patches[lons_cid]
                #get topography
                
                tops = self.dt_top.sel(lon_tar=slice(lons[jj].cpu().numpy()[0]-self.dx/20,
                                                     lons[jj].cpu().numpy()[-1]+self.dx/20)).sel(lat_tar=slice(lats[jj].cpu().numpy()[0]-self.dx/20,
                                                                                                    lats[jj].cpu().numpy()[-1]+self.dx/20))["surface_elevation"].values
                
                
          
                tops = torch.from_numpy(np.expand_dims(np.transpose(tops,(1,0)),0))

                x_top[jj] = normalize(tops, -182, 3846)
                cidx[jj] = torch.tensor(cid, dtype=torch.int)

                self.idx += 1
            yield  {'L': x, 'H': y, "idx": cidx, "T":t, "lons":lons, "lats":lats, "top":x_top}

def run():
    data_loader = PrecipDatasetInter(file_path="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train_small/")
    print("created data_loader")
    for batch_idx, train_data in enumerate(data_loader):
        inputs = train_data["L"]
        target = train_data["H"]
        idx = train_data["idx"]
        times = train_data["T"]
        lats = train_data["lats"]
        top = train_data["top"]

        print("top", top)
        # print("target max", torch.max(target))
        #print("target", target.size())
        #print("idx", idx)
        #print("batch_idx", batch_idx)
        #print("timestamps,", times)
        #print("len of lats",len(lats))

if __name__ == "__main__":
    run()












