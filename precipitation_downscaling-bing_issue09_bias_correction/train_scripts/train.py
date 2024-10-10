
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-12-08"


import time
from collections import OrderedDict
from torch.optim import Adam
import torch
import torch.nn as nn
import os
from torch.optim import lr_scheduler
import wandb
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from models.diffusion_utils import GaussianDiffusion
from models.network_unet import Upsampling
from utils.data_loader import create_loader
from utils.other_utils import dotdict
import pickle

#SBATCH --mail-ty
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
pname = "./logs/profile"

class Weight_Loss(nn.Module):
    def __init__(self):
        super(Weight_Loss, self).__init__()
    def init_w(self,y_true):
        weights = torch.tensor(y_true,requires_grad=False) # 
        MIN = torch.tensor(np.log(1 + 0.1),dtype = weights.dtype,requires_grad=False)
        MAX = torch.tensor(np.log(1 + 80),dtype = weights.dtype,requires_grad=False)
        weights[y_true < MIN] = MIN 
        weights[y_true >= MAX] = MAX
        return weights.to('cuda') 
    def forward(self, pred, target):
        error = torch.abs(pred - target)  # L1
        #error = torch.pow(error, 2)  # L2
        w = self.init_w(target)
        return torch.mean(w * error)

class BuildModel:
    def __init__(self, netG,
                 save_dir    : str = "../results",
                 train_loader: object = None,
                 val_loader  : object = None,
                 dataset_type: str = 'precipitation',
                 save_freq   : int = 8000,
                 checkpoint  : str = None,
                 hparams     : dict = None,
                 **kwargs):

        """f
        :param netG            : the network 
        :param save_dir        : the save model path   
        :param kwargs: 
            conditional: bool type,  if diffusion enabled
        """

        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = netG

        self.netG.to(device)

        #Get paramers
        self.hparams = dotdict(hparams)
        self.G_lossfn_type = self.hparams.G_lossfn_type
        self.G_optimizer_type = self.hparams.G_optimizer_type
        self.G_optimizer_lr = self.hparams.G_optimizer_lr
        self.G_optimizer_betas = self.hparams.G_optimizer_betas
        self.G_optimizer_wd = self.hparams.G_optimizer_wd
        self.epochs = self.hparams.epochs
        self.diffusion = self.hparams.diffusion #: if enable diffusion, the "conditional"must be defined

        self.save_dir = save_dir
        self.dataset_type=dataset_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_freq = save_freq
        self.checkpoint = checkpoint
        self.schedulers = []
        self.loss_history = []
        self.loss_baseline_history = []
        self.log_interval = 180
        self.iteration = 0 
        if self.diffusion:
            self.conditional = True


    def init_train(self):
        wandb.watch(self.netG, log_freq=100)
        if os.path.exists(self.checkpoint):
            print("Loaing the following checkpoint", self.checkpoint)
            self.load_model()
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        if self.G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss()
        elif self.G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss()
        elif self.G_lossfn_type == "huber":
            self.G_lossfn = nn.SmoothL1Loss() ##need to check if this works or not
        elif self.G_lossfn_type == "weight_MAE":
            self.G_lossfn = Weight_Loss()
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(self.G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, 
                                lr = self.G_optimizer_lr,
                                betas = self.G_optimizer_betas,
                                weight_decay = self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        milestones = [2500, 5000,7000, 8000,10000, 12000,15000, 20000, 25000,30000, 35000,40000],
                                                        gamma = 0.8))

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        """
        iter_label: current step
        """
        self.save_network('G', iter_label)

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, network_label,iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = self.netG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save({"iteration": iter_label,
                    "model_state_dict":state_dict}, 
                    save_path)


    def load_model(self):
        """
        Retrieve the trained model
         torch.save({"iteration": iter_label,
                    "model_state_dict":state_dict}, 
                    save_path)
        """
        print("The following checkpoint is loaded", self.checkpoint)
        ck = torch.load(self.checkpoint)
        self.netG.load_state_dict(ck["model_state_dict"])
        self.iteration =  ck["iteration"]
        print("The following checkpoint step is ", self.iteration )
        

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):

        self.L = data['L'].cuda()
        self.top = data["top"].cuda()
        #print("self.top in feed data",self.top.shape)
        upsampling = Upsampling(in_channels = 1) # 8
        self.L_inter = upsampling(self.L)
        if self.diffusion:
            upsampling = Upsampling(in_channels = 8) # 8
            self.L = upsampling(self.L)
        self.H = data['H'].cuda()

    def count_flops(self,data):
        # Count the number of FLOPs
        c_ops = count_ops(self.netG,data)
        print("The number of FLOPS is:",c_ops)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self,idx=None):

        if not self.diffusion:
            self.E = self.netG(self.L, self.top) #[:,0,:,:]
            #print("The prediction shape (E):", self.E.cpu().numpy().shape)
            #print("The min of prediction", np.min(self.E.detach().cpu().numpy()))
            #print("The max of prediction", np.max(self.E.detach().cpu().numpy()))
        else:
            
            if len(self.H.shape) == 3:
                self.H = torch.unsqueeze(self.H, dim = 1)
            
            self.hr = self.H
            h_shape = self.H.shape

            t = torch.randint(0, 200, (h_shape[0],), device = device).long()
   
            noise = torch.randn_like(self.hr)
          
            gd = GaussianDiffusion(model = self.netG, timesteps = 200, conditional=self.conditional)
            x_noisy = gd.q_sample(x_start = self.hr, t = t, noise=noise)
            print("x_nosey shape", x_noisy.shape) #[16,1,160,160][batch_size,chanel,img,img]

            #save noise images
            if idx < 3:
                examples = [self.hr.detach().cpu().numpy()]
                with open('example_5132_idx_{}_t_0.pkl'.format(idx),'wb') as f:
                    pickle.dump(examples, f)

                for i in [1, 50, 100, 150, 199]:
                    j = [i] * h_shape[0]
                    #i = torch.range(1, 16*10, step=10, device = device).long()
                    noise_image = gd.q_sample(x_start = self.hr, t = torch.from_numpy(np.array(j)),noise=noise).detach().cpu().numpy()
                    #dtype=torch.int, device=device
                    #examples.append(noise_image)
                    with open('example_5132_idx_{}_t_{}.pkl'.format(idx,i),'wb') as f:
                        pickle.dump(noise_image, f)
                            
            self.E = self.netG(torch.cat([self.L, x_noisy], dim = 1), t, self.top)

            self.H = noise #if using difussion, the output is not the prediction values, but the predicted noise

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self,idx,current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward(idx)
        if len(self.E.shape) == 3:
            self.E = torch.unsqueeze(self.E, axis=1)
        if len(self.H.shape) ==3:
            self.H = torch.unsqueeze(self.H, axis=1)
        if not len(self.E.shape) == len(self.H.shape):
            raise ("The shape of generated data and ground truth are not the same as above")
        self.G_loss = self.G_lossfn(self.E, self.H)  # pred / target
        self.G_loss_base = self.G_lossfn(self.L_inter,self.H)  # input[:-1:] / target
        #print('input / target',self.L[:,-1,:,:].unsqueeze(1).shape, self.H.shape)(-1,h,16,16) (-1,1,16,16)
        if current_step % self.log_interval == 0 or current_step == 1:
            self.loss_history.append((current_step, self.G_loss.item()))
            self.loss_baseline_history.append((current_step, self.G_loss_base.item()))
        self.G_loss.backward()
        self.G_optimizer.step()

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float()
        out_dict['E'] = self.E.detach()[0].float()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float()
        return out_dict

    #get learning rate
    def get_lr(self):
        for param_group in self.G_optimizer.param_groups:
            return param_group['lr']

    def save_loss_plot(self,window_size = 2):
        steps, losses = zip(*self.loss_history)
        _, losses_base = zip(*self.loss_baseline_history)
        # avg
        losses = np.array(losses)
        indices = np.arange(0, len(steps), window_size)
        losses = [losses[i:i + window_size].mean() for i in indices]

        losses_base = np.array(losses_base)
        losses_base = [losses_base[i:i + window_size].mean() for i in indices]
        steps = range(len(losses))
        plt.figure(figsize=(10, 5))
        plt.plot(steps, losses, label='Training Loss', color='blue')  # Plotting training loss
        plt.plot(steps, losses_base, label='Baseline Loss(interpolation)', color='orange')  # Plotting baseline loss
        plt.title(f'Training Loss Over Steps (window_size = {window_size})')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_loss_curve.png'))  
        plt.close()

    #train model
    def fit(self):
        self.init_train()
        current_step = self.iteration 
        for epoch in range(self.epochs):
            for i, train_data in enumerate(self.train_loader):
                st = time.time()
                current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                self.update_learning_rate(current_step)

                lr = self.get_lr()  # get learning rate

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                self.feed_data(train_data)
            
                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                self.optimize_parameters(i,current_step)
                print("Model Loss {} after step {}".format(self.G_loss, current_step))
                #print("E data",self.E.shape)
                # -------------------------------
                # 4) Save model
                # -------------------------------
                if (current_step % self.save_freq) == 0 and current_step>15000:
                    self.save(current_step)
                    print("Model saved")
                    print("learnign rate",lr)
                    print("Time per step:", time.time() - st)
                wandb.log({"loss": self.G_loss, "lr": lr})
            
        self.save_loss_plot()
        self.save(current_step)

            # with torch.no_grad():
            #     val_loss = 0
            #     counter = 0
            #     for j, val_data in enumerate(self.val_loader):
            #         counter = counter + 1
            #         self.feed_data(val_data)
            #         self.netG_forward(j)
            #         val_loss = val_loss + self.G_lossfn(self.E, self.H).detach()
            #     val_loss = val_loss / counter
            #     print("training loss:", self.G_loss.item())
            #     print("validation loss:", val_loss.item())
            #     print("lr", lr)

            #self.schedulers[0].step(val_loss.item())


