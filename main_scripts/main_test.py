
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-08-22"

import argparse
import sys
import torch
from get_model import get_model
from get_dataset import get_data_info
import numpy as np
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion import UNet_diff
from models.diffusion_utils import GaussianDiffusion 
from utils.data_loader import create_loader
from main_scripts.main_train import BuildModel
#System packages
import os
import json
from datetime import datetime
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type = str, required = True,
                        default = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test",
                        help = "The directory where test data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The output directory")
    parser.add_argument("--dataset_type", type=str, default="precipitation", help="The dataset type: temperature, precipitation")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--k", type = int, default = 0.01, help = "The parameter for log-transform")
    parser.add_argument("--stat_dir", type = str, required = True,
                        default = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
                        help = "The directory where the statistics json file of training data is stored")    
    parser.add_argument("--checkpoint", type = str, required = True, help = "Please provide the checkpoint file")

    args = parser.parse_args()


    #some parameters for diffusion models
    if args.model_type == "diffusion":
        diffusion = True
        conditional = False
        if not conditional:
            print("This is a un-conditional diffusion models!")
    else:
        diffusion = False
        conditional = False

    n_channels, upscale, img_size = get_data_info(args.dataset_type, patch_size=16)

    print("The model {} is selected for training".format(args.model_type))

    netG, _ = get_model(args.model_type, args.dataset_type, img_size, n_channels, upscale)


    model = BuildModel(netG, diffusion=diffusion, conditional=conditional)

    test_loader = create_loader(file_path=args.test_dir,
                                mode="test",
                                stat_path=args.stat_dir)
    
    stat_file = os.path.join(args.stat_dir, "statistics.json")
    print("The statsitics json files is opened from", stat_file)
    
    with open(stat_file,'r') as f:
        stat_data = json.load(f)

    vars_in_patches_mean  = stat_data['yw_hourly_in_mean']
    vars_in_patches_std   = stat_data['yw_hourly_in_std']
    vars_out_patches_mean = stat_data['yw_hourly_tar_mean']
    vars_out_patches_std  = stat_data['yw_hourly_tar_std']

    with torch.no_grad():
        model.netG.load_state_dict(torch.load(args.checkpoint))#['model_state_dict']
        idx = 0
        input_list = []
        pred_list = []
        ref_list = []
        cidx_list = []
        times_list = []
        noise_pred_list = []
        all_sample_list = [] #this is ony for difussion model inference
        hr_list = []
        lats_list = []
        lons_list = []
        for i, test_data in enumerate(test_loader):
            idx += 1
            batch_size = test_data["L"].shape[0]
            cidx_temp = test_data["idx"]
            times_temp = test_data["T"]
            lats = test_data["lats"].cpu().numpy()
            lons = test_data["lons"].cpu().numpy()
            cidx_list.append(cidx_temp.cpu().numpy())
            times_list.append(times_temp.cpu().numpy())
            model.feed_data(test_data)
            #we must place calculate the shape of input here, due to for diffussion model, 
            #The L is upsampling to higher resolution before feed into the model through 'feed_data' function
            image_size = model.L.shape[2]

            #Get the low resolution inputs
            input_vars = test_data["L"]
            input_temp = input_vars.cpu().numpy()
            #input_temp = np.squeeze(input_vars[:,-1,:,:])*vars_in_patches_std+vars_in_patches_mean
            #input_temp = np.exp(input_temp.cpu().numpy()+np.log(args.k))-args.k

            input_list.append(input_temp)
            lats_list.append(lats)
            lons_list.append(lons)
     
            
            model.netG_forward()
            
            if args.model_type == "diffusion":
                gd = GaussianDiffusion(conditional=conditional, timesteps=200, model=model.netG)
                #now, we only use the unconditional difussion model, meaning the inputs are only noise.
                #This is the first test, later, we will figure out how to use conditioanl difussion model.
                print("Start reverse process")
                if conditional:
                    x_in = model.L
                else:
                    x_in = None
                samples = gd.sample(image_size=image_size, batch_size=batch_size, 
                                    channels=n_channels+1, x_in=x_in)
                #chose the last channle and last varialbe (precipitation)
                sample_last = samples[-1] * vars_out_patches_std+vars_out_patches_mean
                # we can make some plot here
                #all_sample_list = all_sample_list.append(sample_last)
                preds = np.exp(sample_last.cpu().numpy()+np.log(args.k))-args.k
                #pred_temp = np.exp(pred_temp.cpu().numpy()+np.log(args.k))-args.k
                ref = model.H.cpu().numpy() #this is the true noise
                noise_pred = model.E.cpu().numpy() #predict the noise
                noise_pred_list.append(noise_pred)
                hr = np.exp(model.hr.cpu().numpy()+np.log(args.k))-args.k
            else:
                #Get the prediction values
                # print("the shape of the output",model.E.cpu().numpy().shape)
                # print("max values", np.max(model.E.cpu().numpy()))
                # print("min values", np.min(model.E.cpu().numpy()))
                #preds = model.E.cpu().numpy()
                preds = model.E.cpu().numpy() * vars_out_patches_std + vars_out_patches_mean
                preds = np.exp(preds+np.log(args.k))-args.k
                #Get the groud truth values
                #ref = model.H.cpu().numpy()
                ref = model.H.cpu().numpy() * vars_out_patches_std + vars_out_patches_mean
                ref = np.exp(ref+np.log(args.k))-args.k

            ref_list.append(ref)
            pred_list.append(preds)   
            if args.model_type == "diffusion":
                hr_list.append(hr)

        
        cidx = np.squeeze(np.concatenate(cidx_list,0))
        times = np.concatenate(times_list,0)
        pred = np.concatenate(pred_list,0)
        ref = np.concatenate(ref_list,0)
        intL = np.concatenate(input_list,0)
        lats_hr = np.concatenate(lats_list, 0)
        lons_hr = np.concatenate(lons_list, 0)
        print("lats_list shape", lons_hr.shape)
  \
  

        if args.model_type == "diffusion":
            hr_list = np.concatenate(hr_list,0)
            print("len of hr_list",len(hr_list))
        
        datetimes = []
        for i in range(times.shape[0]):
            times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
            datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

        if len(pred.shape) == 4:
            pred = pred[:, 0 , : ,:]
        if len(ref.shape) == 4:
            ref = ref[:, 0,: ,:]
        if len(intL.shape) == 4:
            intL = intL[:, 0,: ,:]
        print("shape of ref", ref.shape)

        if args.model_type == "diffusion":
            noiseP = np.concatenate(noise_pred_list,0)
            if len(noiseP.shape) == 4:
                noiseP = noiseP[:, 0, :, :]
            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
                    refe = (["time", "lat", "lon"], ref),
                    noiseP = (["time", "lat", "lon"], noiseP),
                    hr = (["time", "lat", "lon"], hr_list),
                    lats = (["time", "lat"], lats_hr),
                    lons = (["time", "lon"], lons_hr),
                    ),
                coords = dict(
                    time = datetimes,
                    pitch_idx = cidx,
                    ),
                attrs = dict(description = "Precipitation downscaling data."),
                )
        else:
            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
                    refe = (["time", "lat", "lon"], ref),
                    lats = (["time", "lat"], lats_hr),
                    lons = (["time", "lon"], lons_hr),
                ),
                coords = dict(
                    time = datetimes,
                    pitch_idx = cidx,
                    ),
                attrs = dict(description = "Precipitation downscaling data."),
                )

        os.makedirs(args.save_dir,exist_ok=True)

        # ds.to_netcdf(os.path.join(args.save_dir,'prcp_downs_'+args.model_type+'.nc'))
        months, datasets = zip(*ds.groupby("time.month"))
        save_paths = [os.path.join(args.save_dir,'prcp_downs_'+args.model_type+f'_{y}.nc') for y in months]
        print('save_paths: {}'.format(save_paths))
        xr.save_mfdataset(datasets, save_paths)
        
if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor') 
    main()
