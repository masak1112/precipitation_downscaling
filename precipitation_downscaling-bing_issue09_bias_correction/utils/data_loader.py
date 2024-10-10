
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

from torch.utils.data import DataLoader
from dataset_prep import PrecipDatasetInter
from dataset_prep_sr import PrecipDatasetSR
from main_scripts.dataset_temp import CustomTemperatureDataset

def create_loader(file_path: str = None,
                  batch_size: int = 32,
                  patch_size: int = 16,
                  sf: int = 10,
                  seed: int = 1234,
                  k:float = 0.02,
                  mode:str = "train",
                  stat_path:str = None,
                  dataset_type: str = "precipitation",
                  verbose: int = 0):

    """
    file_path       : the path to the directory of .nc files
    vars_in         : the list contains the input variable namsaes
    var_out         : the list contains the output variable name
    batch_size      : the number of samples per iteration
    patch_size      : the patch size for low-resolution image,
                        the corresponding high-resolution patch size should be muliply by scale factor (sf)
    sf              : the scaling factor from low-resolution to high-resolution
    seed            : specify a seed so that we can generate the same random index for shuffle function
    dataset_type    : specify which dataset type we want to load
    """
    if dataset_type == "precipitation":
        '''
        vars_in = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in","u700_in","v700_in","tp"]
        #vars_in = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in","u700_in","v700_in","tp"]
        #vars_in = ["tclw_in","tcwv_in","lsp_in", "cp_in", "tp"] # "tclw_in","tcwv_in",
        vars_out = ["yw_hourly_tar"]
        dataset = PrecipDatasetInter(file_path,
                                     batch_size,
                                     patch_size,
                                     vars_in,
                                     vars_out,
                                     sf,
                                     seed,
                                     k,
                                     mode,
                                     stat_path)
        dataloader = DataLoader(dataset, batch_size=None)
        '''
        dataset = PrecipDatasetSR(
                    file_path = file_path,
                    vars_in = ["tp"],
                    vars_out = ["yw_hourly_tar"],
                    mode = mode,
                    k = k,
                    seed = seed,
                    batch_size = batch_size,
                    stat_path = stat_path,
                    )
        dataloader = DataLoader(dataset, batch_size=None)
        
    elif dataset_type == "temperature":
        dataset = CustomTemperatureDataset(file_path=file_path, verbose=verbose)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        NotImplementedError

    return dataloader

'''
cape_in: 对流有效位能（Convective Available Potential Energy）
tclw_in: 总云水量（Total Cloud Liquid Water）
sp_in: 海平面气压（Sea Level Pressure）
tcwv_in: 总 column 水汽（Total Column Water Vapor）
lsp_in: 液态降水量（Liquid Surface Precipitation）
cp_in: 比湿（Specific Humidity）或其他特定气象参数
tisr_in: 太阳辐射总量（Total Incoming Solar Radiation）
tp: 总降水量（Total Precipitation）

'''