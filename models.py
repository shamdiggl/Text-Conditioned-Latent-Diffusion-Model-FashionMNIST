import torch
import torch.nn as nn

# Define the convolutional classifier model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResnetBlock(16,16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), # Output: [batch_size, 16, 14, 14]
            nn.ReLU(),
            ResnetBlock(16,32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 7, 7]
            nn.ReLU(),
            ResnetBlock(32,32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),  # Output: [batch_size, 32, 4, 4]
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 10),
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet_AutoEncoder(nn.Module):
    def __init__(self):
        super(ResNet_AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, 
                      kernel_size=3, stride=1, padding=1), 
            # Output: [batch_size, 16, 28, 28]
            nn.ReLU(),
            ResnetBlock(16,32),
            nn.ReLU(),
            ResnetBlock(32,64),
            nn.ReLU(),
            ResnetBlock(64,64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, 
                      kernel_size=3, stride=2, padding=1), 
            # Output: [batch_size, 1, 14, 14]
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=64, 
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            ResnetBlock(64,64),
            nn.ReLU(),
            ResnetBlock(64,32),
            nn.ReLU(),
            ResnetBlock(32,16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, 
                      kernel_size=3, stride=1, padding=1), # Output: [batch_size, 1, 28, 28]
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of channels in the input
        :param out_channels: is the number of channels in the output
        """
        super().__init__()
        
        # First conv and norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        # Second conv and norm      
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.projection = nn.Identity()

    
    def forward(self, x: torch.Tensor):
        h = x

        # First normalization and convolution layer
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.relu(h)

        # Second normalization and convolution layer
        h = self.conv2(h)
        h = self.norm2(h)
      
        # Map and add residual
        return self.projection(x) + h
    
