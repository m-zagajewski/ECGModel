"""
Module for the ECG encoder model.

This module provides a CNN-based model for encoding ECG signals into a latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGEncoder(nn.Module):
    """CNN-based encoder for ECG signals."""
    
    def __init__(self, in_channels=12, output_dim=128, kernel_sizes=[5, 3, 3, 3, 3], 
                 strides=[1, 2, 2, 2, 2], channels=[32, 64, 128, 256, 512]):
        """
        Initialize the ECG encoder.
        
        Args:
            in_channels (int, optional): Number of input channels (ECG leads)
            output_dim (int, optional): Dimension of the output latent representation
            kernel_sizes (list, optional): Kernel sizes for each convolutional layer
            strides (list, optional): Strides for each convolutional layer
            channels (list, optional): Number of channels for each convolutional layer
        """
        super(ECGEncoder, self).__init__()
        
        assert len(kernel_sizes) == len(strides) == len(channels), \
            "kernel_sizes, strides, and channels must have the same length"
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(
            nn.Conv1d(in_channels, channels[0], kernel_size=kernel_sizes[0], 
                     stride=strides[0], padding=kernel_sizes[0]//2)
        )
        
        # Remaining convolutional layers
        for i in range(1, len(channels)):
            self.conv_layers.append(
                nn.Conv1d(channels[i-1], channels[i], kernel_size=kernel_sizes[i], 
                         stride=strides[i], padding=kernel_sizes[i]//2)
            )
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(channels[i]) for i in range(len(channels))
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to get the final latent representation
        self.fc = nn.Linear(channels[-1], output_dim)
        
    def forward(self, x):
        """
        Forward pass through the ECG encoder.
        
        Args:
            x (torch.Tensor): Input ECG signals of shape [batch_size, in_channels, time_steps]
            
        Returns:
            torch.Tensor: Latent representation of shape [batch_size, output_dim]
        """
        # Apply convolutional layers with batch normalization and ReLU
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = F.relu(bn(conv(x)))
        
        # Apply global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Apply fully connected layer
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for the ECG encoder."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """
        Initialize the residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size for the convolutional layers
            stride (int, optional): Stride for the first convolutional layer
        """
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetECGEncoder(nn.Module):
    """ResNet-based encoder for ECG signals."""
    
    def __init__(self, in_channels=12, output_dim=128, base_channels=64, 
                 blocks=[2, 2, 2, 2]):
        """
        Initialize the ResNet ECG encoder.
        
        Args:
            in_channels (int, optional): Number of input channels (ECG leads)
            output_dim (int, optional): Dimension of the output latent representation
            base_channels (int, optional): Base number of channels
            blocks (list, optional): Number of residual blocks in each layer
        """
        super(ResNetECGEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.output_dim = output_dim
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, base_channels, blocks[0], stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels*2, blocks[1], stride=2)
        self.layer3 = self._make_layer(base_channels*2, base_channels*4, blocks[2], stride=2)
        self.layer4 = self._make_layer(base_channels*4, base_channels*8, blocks[3], stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer to get the final latent representation
        self.fc = nn.Linear(base_channels*8, output_dim)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a layer of residual blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of residual blocks
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Layer of residual blocks
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the ResNet ECG encoder.
        
        Args:
            x (torch.Tensor): Input ECG signals of shape [batch_size, in_channels, time_steps]
            
        Returns:
            torch.Tensor: Latent representation of shape [batch_size, output_dim]
        """
        # Initial convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x