import torch
import torch.nn as nn


# input: (-1,5,16,16)
# top: (-1,1,160,160)
# target: (-1,16,16)
class FCnet(nn.Module):
    def __init__(self,n_channels):
        super(FCnet, self).__init__()
        self.n_channels = n_channels
        self.fc1 = nn.Linear(self.n_channels*16*16, 1024)
        self.fc_top = nn.Linear(160*160,1024) 

        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        
    
    def forward(self, x,top):
        # sigm # remove top unet
        x = x.reshape(-1,self.n_channels*16*16) # flatten
        top = top.reshape(-1,160*160)
        top = self.fc_top(top)
        x = torch.relu(self.fc1(x)+ top) 
        x = self.fc2(x)
        x = self.fc3(x)
        #x = torch.sigmoid(x)
        x = x.reshape(-1,16,16)    
        return x
    
# x = torch.rand((3,5,16,16))
# net = FCnet()

# Y = net(x)
# print(Y)