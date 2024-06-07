
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-22"


import argparse
from selectors import EpollSelector
import sys
import os
import json
from get_dataset import get_data_info
import torch
from get_model import get_model
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion  import UNet_diff
from models.network_unet2 import UNetModel
from models.network_unet import Upsampling
from utils.data_loader import create_loader
from models.diffusion_utils import GaussianDiffusion
from models.network_critic import Discriminator as critic
from utils.data_loader import create_loader
from train_scripts.wgan_train import BuildWGANModel
from train_scripts.train import BuildModel
#from flopth import flopth
from ptflops import get_model_complexity_info
###Weights and Bias
import wandb
os.environ["WANDB_MODE"]="offline"
##os.environ["WANDB_API_KEY"] = key
device = "cuda" if torch.cuda.is_available() else "cpu"

available_models = ["unet", "wgan", "diffusion", "swinIR","swinUnet","diffusion2"]

def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
        val_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/val",
        save_dir: str = "../results",
        checkpoint_save: int = 200,
        epochs: int = 2,
        type_net: str = "unet",
        dataset_type: str = "temperature",
        batch_size: int = 2,
        patch_size: int = 16,
        wandb_id: str = None,
        checkpoint = None,
        **kwargs):

    """
    :param train_dir       : the directory that contains the training dataset NetCDF files
    :param val_dir        : the directory that contains the testing dataset NetCDF files
    :param checkpoint_save : how many steps to save checkpoint
    :param n_channels      : the number of input variables/channels
    :param save_dir        : the directory where the checkpoint results are save
    :param epochs          : the number of epochs
    :param type_net        : the type of the models
    """
    
    #Initial Wandb
    id = wandb.util.generate_id() + type_net
    # Set default hyper-parameters for all the models and used for wandb log
    config = {"epochs":epochs, "batch_size": batch_size, 
              "train_dir": train_dir, "val_dir":val_dir, 
              "save_dir":save_dir, "type_net":type_net,
              "patch_size": patch_size, "checkpoint": checkpoint
              }

    if type_net not in available_models:
        raise("type net should be one of the following models:",available_models)

    type_net = type_net.lower() 
    
    #some parameters for diffusion models
    if type_net == "diffusion" or type_net == "diffusion2":
        diffusion = True
    else:
        diffusion = False

    
    # This is the model hparameters should be tailored to each model afterwards
    hparams = {}
    
    # get the dataset information based on the data type
    n_channels, upscale, img_size = get_data_info(dataset_type, patch_size=patch_size)

    # Load the datasets
    train_loader = create_loader(file_path = train_dir, 
                                 batch_size = batch_size,
                                 patch_size = patch_size, 
                                 dataset_type = dataset_type, 
                                 stat_path=train_dir,
                                 mode="train")
                                 
    val_loader = create_loader(file_path=val_dir,
                               mode="test",
                               batch_size = batch_size,
                               stat_path=train_dir,
                               patch_size=patch_size,
                               dataset_type=dataset_type)
                               
    print("The model {} is selected for training".format(type_net))

    netG, netC = get_model(type_net, dataset_type, img_size, n_channels, upscale)

    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)

    if type_net == "wgan":
        netC_params = sum(p.numel() for p in netC.parameters() if p.requires_grad)
        print("Total trainalbe parameters of the generator:", netG_params)
        print("Total trainalbe parameters of the critic:", netC_params)
        #flops, params = get_model_complexity_info(netC,  (1, 160, 160),as_strings=True)
        #print("flops for critic network, params for critic network", flops, params)
    elif type_net!="diffusion":
        #calculate the model size
        #flops, params = get_model_complexity_info(netG,  (n_channels, img_size[0], img_size[1]),as_strings=True)
        netG_params = None
        #print("flops, params", flops, params)
        #calculate the trainable parameters
        print("Total trainalbe parameters of the network:", netG_params)
    else:
        print("Flops calculation is not implemented yet for diffusion model")

    # Build models for wgan and other models with one nerual network
    if type_net == "wgan":
        hparams = {"critic_iterations": 5, "lr_gn": 1.e-05, 
                  "lr_gn_end":1.e-06, "lr_critic": 1.e-06,
                  "decay_start":25, "decay_end": 50,
                  "lambada_gp":10, "recon_weight":1000}
        
        config.update(hparams)

        model = BuildWGANModel(generator=netG,
                               save_dir=save_dir,
                               critic=netC,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               hparams=config,
                               dataset_type=dataset_type)
    else:
        #default parameters for other models (except WGAN)
        hparams = {"G_lossfn_type": "l2",
                   "G_optimizer_type": "adam",
                   "G_optimizer_lr": 5.e-04,
                  "G_optimizer_betas":[0.9, 0.999],
                  "G_optimizer_wd": 5.e-04,
                   "diffusion": diffusion}
        
        config.update(hparams)
        model = BuildModel(netG,
                           save_dir = save_dir,
                           hparams = config,
                           train_loader = train_loader,
                           val_loader = val_loader,
                           checkpoint = checkpoint)
        
    #config.update(hparams)
    wandb.init(project="Precip_downscaling",reinit=True, id=id, dir=save_dir, config=config)
    wandb.run.name = id
    wandb.config.update(hparams)
    wandb.config.update({"lr":model.G_optimizer_lr})
    model.fit()
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type = str, required = True,
                        help = "The directory where training data (.nc files) are stored")
    parser.add_argument("--val_dir", type = str, required = True,
                        help = "The directory where validation data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type = int, default = 2, help = "The checkpoint directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--dataset_type", type=str, default="precipitation", help="The dataset type: temperature, precipitation")
    parser.add_argument("--wandb_id", type=str, default="None",help="Please provided the wandb id")
    parser.add_argument("--checkpoint", type=str, default="None",help="Please provided the checkpoint path")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #save the args to the checkpoint directory
    with open(os.path.join(args.save_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys = True, indent = 4))

    run(train_dir = args.train_dir,
        val_dir = args.val_dir,
        save_dir = args.save_dir,
        epochs = args.epochs,
        type_net = args.model_type,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        patch_size=16,
        wandb_id=args.wandb_id,
        checkpoint=args.checkpoint)

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()

