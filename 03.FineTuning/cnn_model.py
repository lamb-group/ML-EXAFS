import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, output_layer, p = 0.2):
        super(CNNModel,self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05, inplace=True))
    
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.05, inplace=True),
            nn.MaxPool1d(2))
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.05, inplace=True),
            nn.MaxPool1d(2))
        
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.05, inplace=True))
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            # High-capacity projection layer
            nn.Linear(256 * 55, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p),
        
            # Bottleneck compression layer
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Dropout(p),
        
            # Final regression layer
            nn.Linear(512, output_layer),
            nn.Softplus())
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.fc_block(x)
        return x
    
 