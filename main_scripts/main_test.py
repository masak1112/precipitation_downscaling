
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
    else:
        diffusion = False

    n_channels, upscale, img_size = get_data_info(args.dataset_type, patch_size=16)

    print("The model {} is selected for training".format(args.model_type))

    netG, _ = get_model(args.model_type, args.dataset_type, img_size, n_channels, upscale)

    #default parameters
    hparams =  {"G_lossfn_type": "l2",
              "G_optimizer_type": "adam",
               "G_optimizer_lr": 5.e-04,
                "G_optimizer_betas":[0.9, 0.999],
                "G_optimizer_wd": 5.e-04,
                "timesteps":200,
                "conditional": True,
                "diffusion":diffusion}

    model = BuildModel(netG, diffusion=diffusion, conditional=True, hparams=hparams)

    test_loader = create_loader(file_path=args.test_dir,
                                mode="test",
                                stat_path=args.stat_dir,
                                batch_size=8)
    
    #Get and load the statistics information from the training directory for denormalisation
    stat_file = os.path.join(args.stat_dir, "statistics.json")
    print("The statsitics json files is opened from", stat_file)
    
    with open(stat_file,'r') as f:
        stat_data = json.load(f)

    vars_in_patches_min = stat_data['yw_hourly_in_min']
    vars_in_patches_max  = stat_data['yw_hourly_in_max']
    vars_out_patches_min = stat_data['yw_hourly_tar_min']
    vars_out_patches_max  = stat_data['yw_hourly_tar_max']


    #Diffusion model

    if args.model_type == "diffusion":
            
        with torch.no_grad():
            model.netG.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
            idx = 0
            input_list = []      #low-resolution inputs
            pred_list = []       # prediction high-resolution results
            ref_list = []        # true noise
            cidx_list = []       # image index
            times_list = []      #timestamps
            noise_pred_list = [] # predicted noise
            hr_list = []         # ground truth images
            lats_list = []       #lats
            lons_list = []       #lons
            pred_first_list = []
            pred_100_list = []
            pred_50_list = []
            pred_150_list = []
            pred_last_list = []
            for i, test_data in enumerate(test_loader):
                idx += 1
                batch_size = test_data["L"].shape[0]
                print("batdh_size",batch_size)
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
                #input_temp = input_vars[:,-1,:,:].cpu().numpy()
                input_temp = ((np.squeeze(input_vars[:,-1,:,:]) )* (vars_in_patches_max- vars_in_patches_min)+ vars_in_patches_min).cpu().numpy()
                input_temp = np.exp(input_temp+np.log(args.k))-args.k
 

                with torch.no_grad():
                    model.netG_forward(i)
                
                gd = GaussianDiffusion(conditional=True, timesteps=200, model=model.netG)
                #now, we only use the unconditional difussion model, meaning the inputs are only noise.
                #This is the first test, later, we will figure out how to use conditioanl difussion model.
                print("Start reverse process")

                x_in = model.L
                samples = gd.sample(image_size=image_size, 
                                    batch_size=batch_size, 
                                    x_in=x_in)
                
                print("len of of samples,", len(samples))
                #chose the last channle and last varialbe (precipitation)
                sample_last = samples[-1].cpu().numpy()  #
                preds = samples[-1].cpu().numpy() 
                preds[preds<0] = 0
                # preds[preds<-2] = 0
                # preds[preds>=-2] = 10**preds[preds>=-2]
                #sample_last_clip = (sample_last + 1)/2
                #preds = preds * (vars_out_patches_max - vars_out_patches_min) + vars_out_patches_min 
                #log-transform -> log(x+k)-log(k)
                #preds =np.exp(preds+np.log(args.k))-args.k

                sample_first = samples[0].cpu().numpy()

                sample_50 = samples[50].cpu().numpy()


                sample_100 = samples[100].cpu().numpy()


                sample_150 = samples[150].cpu().numpy()
                # we can make some plot here
                #all_sample_list = all_sample_list.append(sample_last)
                #preds = sample_last.cpu().numpy()
    
    
                #pred_temp = np.exp(pred_temp.cpu().numpy()+np.log(args.k))-args.k
                ref = model.H.cpu().numpy() #this is the true noise
                noise_pred = model.E.cpu().numpy() #predict the noise
                
                hr = model.hr.cpu().numpy()

                #hr = (model.hr.cpu().numpy()) * (vars_out_patches_max - vars_out_patches_min) + vars_out_patches_min 
                #hr = np.exp(hr+np.log(args.k))-args.k
                
            
                input_list.append(input_temp) #ground truth images
                lats_list.append(lats)
                lons_list.append(lons)
                ref_list.append(ref)  #true noise
                noise_pred_list.append(noise_pred) # predicted noise
                pred_list.append(preds)  #predicted high-resolution images
                pred_last_list.append(sample_last)
                pred_first_list.append(sample_first)
                pred_100_list.append(sample_100)
                pred_50_list.append(sample_50)
                pred_150_list.append(sample_150)
                hr_list.append(hr) #grount truth
        
        cidx = np.squeeze(np.concatenate(cidx_list,0))
        times = np.concatenate(times_list,0)
        pred = np.concatenate(pred_list,0)
        pred_last = np.concatenate(pred_last_list,0)
        pred_first = np.concatenate(pred_first_list,0)
        pred_50 = np.concatenate(pred_50_list,0)
        pred_100 = np.concatenate(pred_100_list,0)
        pred_150 = np.concatenate(pred_150_list,0)
        pred = np.concatenate(pred_list,0)
        ref = np.concatenate(ref_list,0)
        intL = np.concatenate(input_list,0)
        lats_hr = np.concatenate(lats_list, 0)
        lons_hr = np.concatenate(lons_list, 0)
        hr_list = np.concatenate(hr_list,0)

        print("pred_first shape", pred_first.shape)
                
        datetimes = []
        for i in range(times.shape[0]):
            times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
            datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

        if len(pred.shape) == 4:
            pred = pred[:, 0 , : ,:]
            pred_50 = pred_50[:, 0 , : ,:]
            pred_100 = pred_100[:, 0 , : ,:]
            pred_150 = pred_150[:, 0 , : ,:]
            pred_first = pred_first[:, 0 , : ,:]
            pred_last = pred_last[:, 0 , : ,:]
        if len(ref.shape) == 4:
            ref = ref[:, 0,: ,:]
        if len(intL.shape) == 4:
            intL = intL[:, 0,: ,:]
        print("shape of ref", ref.shape)
        if len(hr_list.shape) == 4:
            hr_list = hr_list[:, 0,: ,:]
        print("shape of hr_list",hr_list.shape)


        noiseP = np.concatenate(noise_pred_list,0)

        if len(noiseP.shape) == 4:
            noiseP = noiseP[:, 0, :, :]
            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
                    fcst_first = (["time", "lat", "lon"], np.squeeze(pred_first)),
                    fcast_last=(["time", "lat", "lon"], np.squeeze(pred_last)),
                    fcst_50 = (["time", "lat", "lon"], np.squeeze(pred_50)),
                    fcst_100 = (["time", "lat", "lon"], np.squeeze(pred_100)),
                    fcst_150 = (["time", "lat", "lon"], np.squeeze(pred_150)),
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
            with torch.no_grad():
                model.netG.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
                idx = 0
                input_list = []  #low-resolution inputs
                pred_list = []   # prediction high-resolution results
                cidx_list = []   # image index
                times_list = []  #timestamps
                hr_list = []  # ground truth images
                lats_list = [] #lats
                lons_list = [] #lons
                for i, test_data in enumerate(test_loader):
                    idx += 1
                    batch_size = test_data["L"].shape[0]
                    cidx_temp = test_data["idx"]
                    times_temp = test_data["T"]
                    lats = test_data["lats"].cpu().numpy()
                    lons = test_data["lons"].cpu().numpy()
      
                    model.feed_data(test_data)

                    #Get the low resolution inputs
                    input_vars = test_data["L"]
                    input_temp = input_vars.cpu().numpy()
                    input_temp = np.squeeze(input_vars[:,-1,:,:])* (vars_in_patches_max- vars_in_patches_min )+ vars_in_patches_min 
                    input_temp = np.exp(input_temp.cpu().numpy()+np.log(args.k))-args.k
 
                    model.netG_forward()
                    #Get the prediction values
                    preds = model.E.cpu().numpy() * (vars_out_patches_max -vars_out_patches_min) + vars_out_patches_min 
                    preds = np.exp(preds+np.log(args.k))-args.k

                    #Get the groud truth values
                    hr = test_data["H"].cpu().numpy() * (vars_out_patches_max -vars_out_patches_min) + vars_out_patches_min 
                    hr = np.exp(hr+np.log(args.k))-args.k

                    
                    lats_list.append(lats)
                    lons_list.append(lons)
                    cidx_list.append(cidx_temp.cpu().numpy())
                    times_list.append(times_temp.cpu().numpy())
                    input_list.append(input_temp) #ground truth images
                    hr_list.append(hr) #grount truth
                    pred_list.append(preds)  #predicted high-resolution images
                
                cidx = np.squeeze(np.concatenate(cidx_list,0))
                times = np.concatenate(times_list,0)
                pred = np.concatenate(pred_list,0)
                intL = np.concatenate(input_list,0)
                lats_hr = np.concatenate(lats_list, 0)
                lons_hr = np.concatenate(lons_list, 0)
                hr_list = np.concatenate(hr_list,0)
               
  
                datetimes = []
                for i in range(times.shape[0]):
                    times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
                    datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

            if len(pred.shape) == 4:
                pred = pred[:, 0 , : ,:]
            if len(intL.shape) == 4:
                intL = intL[:, 0,: ,:]
            if len(hr_list.shape) == 4:
                hr_list = hr_list[:, 0,: ,:]


            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
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
