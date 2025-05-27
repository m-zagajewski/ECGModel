"""
Module for the tabular data model.

This module provides an MLP-based model for processing tabular clinical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """MLP-based model for tabular clinical data."""
    
    def __init__(self, input_dim, output_dim=128, hidden_dims=[256, 128], 
                dropout_rate=0.3, batch_norm=True):
        """
        Initialize the tabular MLP model.
        
        Args:
            input_dim (int): Dimension of the input features
            output_dim (int, optional): Dimension of the output latent representation
            hidden_dims (list, optional): Dimensions of the hidden layers
            dropout_rate (float, optional): Dropout rate for regularization
            batch_norm (bool, optional): Whether to use batch normalization
        """
        super(TabularMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Create layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the tabular MLP model.
        
        Args:
            x (torch.Tensor): Input tabular features of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Latent representation of shape [batch_size, output_dim]
        """
        return self.model(x)


class TabularFeatureExtractor(nn.Module):
    """Feature extractor for tabular clinical data with feature importance."""
    
    def __init__(self, input_dim, output_dim=128, hidden_dims=[256, 128], 
                dropout_rate=0.3, batch_norm=True):
        """
        Initialize the tabular feature extractor.
        
        Args:
            input_dim (int): Dimension of the input features
            output_dim (int, optional): Dimension of the output latent representation
            hidden_dims (list, optional): Dimensions of the hidden layers
            dropout_rate (float, optional): Dropout rate for regularization
            batch_norm (bool, optional): Whether to use batch normalization
        """
        super(TabularFeatureExtractor, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0]) if batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_bns.append(nn.BatchNorm1d(hidden_dims[i+1]) if batch_norm else nn.Identity())
            self.hidden_dropouts.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Feature importance layer (attention mechanism)
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        """
        Forward pass through the tabular feature extractor.
        
        Args:
            x (torch.Tensor): Input tabular features of shape [batch_size, input_dim]
            
        Returns:
            tuple: (output, attention_weights)
                - output (torch.Tensor): Latent representation of shape [batch_size, output_dim]
                - attention_weights (torch.Tensor): Feature importance weights of shape [batch_size, input_dim]
        """
        # Calculate feature importance weights
        attention_weights = torch.sigmoid(self.attention(x)).squeeze(-1)
        
        # Apply attention weights to input
        weighted_input = x * attention_weights.unsqueeze(1)
        
        # Input layer
        out = self.input_layer(weighted_input)
        out = self.input_bn(out)
        out = F.relu(out)
        out = self.input_dropout(out)
        
        # Hidden layers
        for layer, bn, dropout in zip(self.hidden_layers, self.hidden_bns, self.hidden_dropouts):
            out = layer(out)
            out = bn(out)
            out = F.relu(out)
            out = dropout(out)
        
        # Output layer
        out = self.output_layer(out)
        
        return out, attention_weights


class TabularClassifier(nn.Module):
    """Standalone classifier for tabular clinical data."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.3, batch_norm=True):
        """
        Initialize the tabular classifier.
        
        Args:
            input_dim (int): Dimension of the input features
            hidden_dims (list, optional): Dimensions of the hidden layers
            dropout_rate (float, optional): Dropout rate for regularization
            batch_norm (bool, optional): Whether to use batch normalization
        """
        super(TabularClassifier, self).__init__()
        
        # Feature extractor
        self.feature_extractor = TabularFeatureExtractor(
            input_dim=input_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        """
        Forward pass through the tabular classifier.
        
        Args:
            x (torch.Tensor): Input tabular features of shape [batch_size, input_dim]
            
        Returns:
            tuple: (logits, attention_weights)
                - logits (torch.Tensor): Classification logits of shape [batch_size, 1]
                - attention_weights (torch.Tensor): Feature importance weights of shape [batch_size, input_dim]
        """
        # Extract features
        features, attention_weights = self.feature_extractor(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, attention_weights
    
    def predict_proba(self, x):
        """
        Predict probability of positive class.
        
        Args:
            x (torch.Tensor): Input tabular features of shape [batch_size, input_dim]
            
        Returns:
            tuple: (probabilities, attention_weights)
                - probabilities (torch.Tensor): Probabilities of positive class of shape [batch_size, 1]
                - attention_weights (torch.Tensor): Feature importance weights of shape [batch_size, input_dim]
        """
        logits, attention_weights = self.forward(x)
        probabilities = torch.sigmoid(logits)
        return probabilities, attention_weights