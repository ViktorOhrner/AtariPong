import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelCNN(nn.Module):
    def __init__(self, output_size=6):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        # Output is size 64x6x6 = 2304

        # Action Layer
        self.Allinear1 = nn.Linear(in_features=1024, out_features=128)
        self.Alrelu = nn.LeakyReLU()
        self.Allinear2 = nn.Linear(in_features=128, out_features=output_size)

        # Value Layer
        self.Vllinear1 = nn.Linear(in_features=1024, out_features=128)
        self.Vlrelu = nn.LeakyReLU()
        self.Vllinear2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # CNN block
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        
        # Flatten every batch
        x = x.view(x.size(0), -1)

        # Action layer
        #print(x.shape)
        Ax = self.Alrelu(self.Allinear1(x))
        Ax = self.Allinear2(Ax) # No activation on last layer

        # Value layer
        Vx = self.Vlrelu(self.Vllinear1(x))
        Vx = self.Vllinear2(Vx) # No activation on last layer

        q = Vx + (Ax - Ax.mean(dim=1, keepdim=True))

        return q