import torch 
import toch.nn as nn 
import torch.nn.functional as F
import testlaoder, testset from dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #LAYERS
        self.conv1=nn.Conv2d(1,6,5)
        self.conv2=nn.Conv2d()
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        ##Feed Forward Functions
        x = F.max_pool2d(F.relu(self.canv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.canv2(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=5
        return num_features
    
    