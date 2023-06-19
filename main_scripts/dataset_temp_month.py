# SPDX-FileCopyrightText: 2022 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Driver-script to perform inference on trained downscaling models.
"""

__author__ = "Maxim Bragilovski, Bing Gong"
__email__ = "maximbr@post.bgu.ac.il"
__date__ = "2022-12-17"
__update__ = "2022-12-17"

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import time
import json
import sys

sys.path.append('../')
from handle_data.handle_data_temp import HandleUnetData
from torch.utils.data import DataLoader


class CustomTemperatureDatasetByMonth(Dataset):

    def __init__(self, file_path: str = None, batch_size: int = 32, verbose: int = 0, seed: int = 1234):
        self.ds_ins = None
        self.ds_tar = None
        self.ds_in = None
        self.file_path = file_path
        self.verbose = verbose
        self.ds = xr.open_dataset(file_path)
        # self.ds.to_netcdf("saved_on_disk.nc")
        times = self.ds['time']
        self.years = []
        # for item in times:
        #     b = item.values
        #     year = b.astype(str).split('-')[0]
        #     if year not in self.years:
        #
        #
        # self.byTime = []

        self.ds = self.ds.sel(time=slice("2006-01-01", "2008-01-01"))
        self.log = self.ds.sizes['rlon']
        self.lat = self.ds.sizes['rlat']
        self.times = np.transpose(np.stack(
            [self.ds["time"].dt.year, self.ds["time"].dt.month, self.ds["time"].dt.day, self.ds["time"].dt.hour]))

        self.process_era5_netcdf()

    def process_era5_netcdf(self):
        """
        process netcdf files: normalization,
        """

        def reshape_ds(ds):
            da = ds.to_array(dim="variables")
            da = da.transpose(..., "variables")
            return da

        # if you want to select months, then revise here, just pass the month
        # ds_train = self.ds.sel(time=slice("2006-01-01", "2006-01-01"))

        start = time.time()
        da_train = reshape_ds(self.ds)
        end = time.time()
        # print(f'Reshaping took {(end - start) / 60} minutes')

        self.n_samples = da_train.sizes['time']
        # print(da_train.sizes)

        norm_dims = ["time", "rlat", "rlon"]
        if self.verbose == 0:
            start = time.time()
            da_norm, mu, std = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
            end = time.time()
            # print(f'Normalization took {(end - start) / 60} minutes')
            for save in [(mu, 'mu'), (std, 'std')]:
                self.save_stats(save[0], save[1])
        if self.verbose == 1:
            mu_train = self.load_stats('mu')
            std_train = self.load_stats('std')
            da_norm = HandleUnetData.z_norm_data(da_train, mu=mu_train, std=std_train)

        da_norm = da_norm.astype(np.float32)
        start = time.time()
        da_norm.load()
        end = time.time()

        divided_by_years = self.split_by_month_xarray(da_norm)
        print(divided_by_years['01']['time'][divided_by_years['01']['time'].shape[0] - 1].values.astype(str).split('-')[0])
        # print(f'Loading took {(end - start) / 60} minutes')
        def gen(darr_in, darr_tar):
            ds_train_in = []
            ds_train_tar = []
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                x = np.squeeze(darr_in.isel({"time": t}).to_array().values, axis=0)
                ds_train_in.append(torch.from_numpy(x).permute(2, 0, 1))
                y = np.squeeze(darr_tar.isel({"time": t}).to_array().values, axis=0)
                # vector_tar = torch.from_numpy(darr_tar.isel({"time": t}).values[:, :, 0][..., np.newaxis])
                ds_train_tar.append(torch.from_numpy(y).permute(2, 0, 1))  # [:, :, 1]
            if ds_train_in:
                a = torch.stack(ds_train_in)
                b = torch.stack(ds_train_tar)
            else:
                a = torch.from_numpy(np.array([]))
                b = torch.from_numpy(np.array([]))
            return a, b

        start = time.time()
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.ds_in, self.ds_tar, self.time = [], [], []
        for ds in divided_by_years:
            da_in, da_tar, times = split_in_tar(divided_by_years[ds])
            self.ds_in.append(da_in)
            self.ds_tar.append(da_tar)
            self.time.append(times)

        self.time = np.concatenate(self.time, axis=0)
        # da_in, da_tar, times = split_in_tar(da_norm)
        end = time.time()
        # print(f'splitting took {(end - start) / 60} minutes')
        start = time.time()
        self.ds_ins, self.ds_tars = [], []
        for i in range(len(self.ds_in)):

           ds_in, ds_tar = gen(self.ds_in[i], self.ds_tar[i])
           self.ds_ins.append(ds_in)
           self.ds_tars.append(ds_tar)

        end = time.time()
        # print(f'generation took {(end - start) / 60} minutes')
        # print(self.ds_ins[3].shape[0] == 0)
        self.ds_ins = [tensor for tensor in self.ds_ins if tensor.shape[0] != 0]
        self.ds_tars = [tensor for tensor in self.ds_tars if tensor.shape[0] != 0]

        self.ds_ins = np.concatenate(self.ds_ins, axis=0)
        self.ds_tars = np.concatenate(self.ds_tars, axis=0)


    def save_stats(self, to_save, name):
        """
        Saving the statistics of the train data set to a json file
        """
        dict_to_save = to_save.to_dict()
        # json_object = json.dump(dict_to_save)
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path, 'w') as f:
            json.dump(dict_to_save, f)

    def load_stats(self, name):
        """
        Loading the statistics of the train data of normalization a validation/test dataset
        """
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path) as json_file:
            data = json.load(json_file)

        for key in data.keys():
            print(key)

        return xr.DataArray.from_dict(data)

    def split_by_month_xarray(self, ds):
        output = {}
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        for month in months:
            output[month] = []
        times = ds['time']
        print(times.shape)

        # while int(min_year) <= int(max_year):

        for month in months:

            min_year = times[0].values.astype(str).split('-')[0]
            max_year = times[times.shape[0] - 1].values.astype(str).split('-')[0]
            print(min_year, max_year, month,'---------------------')
            while int(min_year) <= int(max_year):

                start_date = min_year + '-' + month + '-' + '01'
                end_date = min_year + '-' + month + '-' + '28'
                output[month].append(ds.sel(time=slice(start_date, end_date)))
                print(start_date, end_date)
                min_year_int = int(min_year) + 1
                min_year = str(min_year_int)

        for month in months:
            # x = output[month][0].to_dataset(name='a')
            # y = output[month][1].to_dataset(name='a')
            # merge = xr.merge([output[month][0].to_dataset(name='a'), output[month][1].to_dataset(name='a')])
            xarray_by_month = xr.merge([x.to_dataset(name='a') for x in output[month]])
            output[month] = xarray_by_month

        return output

    def __len__(self):
        return len(self.ds_ins)

    def __getitem__(self, idx):
        output = {'L': self.ds_ins[idx], 'H': self.ds_tars[idx], "T": self.times[idx]}
        return output


def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
    """
    Split data array with variables-dimension into input and target data for downscaling.
    :param da: The unsplitted data array.
    :param target_var: Name of target variable which should consttute the first channel
    :return: The splitted data array.
    """
    invars = [var for var in da["variables"].values if var.endswith("_in")]
    tarvars = [var for var in da["variables"].values if var.endswith("_tar")]
    # darr_tar.sel({"variables": "t_2m_tar"}).isel( < ... >)

    # ensure that ds_tar has a channel coordinate even in case of single target variable
    roll = False
    if len(tarvars) == 1:
        sl_tarvars = tarvars
    else:
        sl_tarvars = slice(*tarvars)
        if tarvars[0] != target_var:  # ensure that target variable appears as first channel
            roll = True

    da_in, da_tar = da.sel({"variables": invars}), da.sel(variables=sl_tarvars)
    if roll: da_tar = da_tar.roll(variables=1, roll_coords=True)

    # get the time
    times = da_in["time"].values
    return da_in, da_tar, times


def run():
    # datasets_by_month = split_by_month_xarray(file_path="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc")
    # //p//scratch//deepacf//maelstrom//maelstrom_data//ap5//downscaling_benchmark_dataset//preprocessed_era5_crea6//t2m//all_files//downscaling_tier2_train.nc
    datasets_by_month = CustomTemperatureDatasetByMonth(
        # file_path="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc")
    file_path = "//p//scratch//deepacf//maelstrom//maelstrom_data//ap5//downscaling_benchmark_dataset//preprocessed_era5_crea6//t2m//all_files//downscaling_tier2_train.nc")
    train_dataloader = DataLoader(datasets_by_month, batch_size=32, shuffle=False)

    for i, train_data in enumerate(train_dataloader):
        print(i, train_data["T"])


    # p.step()
    # data_loader = CustomTemperatureDataset(
    #     file_path="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc")
    # train_dataloader = DataLoader(data_loader, batch_size=32, shuffle=False, num_workers=8)
    #
    # def trace_handler(prof):
    #     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")
    #
    # with torch.profiler.profile(
    #         activities=[ torch.profiler.ProfilerActivity.CPU,
    #             torch.profiler.ProfilerActivity.CUDA,],
    #         schedule=torch.profiler.schedule(
    #             wait=1,
    #             warmup=1,
    #             active=2),
    #         #on_trace_ready=trace_handler,
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log1')
    #         # used when outputting for tensorboard
    #         ) as p:
    #     for i, train_data in enumerate(train_dataloader):
    #         print(i, train_data["L"].cuda())
    #         p.step()


if __name__ == "__main__":
    run()
