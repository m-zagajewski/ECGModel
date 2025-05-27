"""
Module for the fusion model architecture.

This module provides a multimodal model that combines ECG and tabular data
for predicting 30-day mortality in ICU patients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ecg_encoder import ECGEncoder, ResNetECGEncoder
from src.models.tabular_model import TabularMLP, TabularFeatureExtractor


class ConcatenationFusion(nn.Module):
    """Simple fusion model that concatenates ECG and tabular features."""
    
    def __init__(self, ecg_encoder, tabular_model, fusion_dim=256, dropout_rate=0.3):
        """
        Initialize the concatenation fusion model.
        
        Args:
            ecg_encoder (nn.Module): Encoder for ECG signals
            tabular_model (nn.Module): Model for tabular data
            fusion_dim (int, optional): Dimension of the fusion layer
            dropout_rate (float, optional): Dropout rate for regularization
        """
        super(ConcatenationFusion, self).__init__()
        
        self.ecg_encoder = ecg_encoder
        self.tabular_model = tabular_model
        
        # Determine the combined dimension
        self.combined_dim = ecg_encoder.output_dim + tabular_model.output_dim
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 1)
        )
    
    def forward(self, ecg, tabular):
        """
        Forward pass through the fusion model.
        
        Args:
            ecg (torch.Tensor): ECG signals of shape [batch_size, channels, time_steps]
            tabular (torch.Tensor): Tabular features of shape [batch_size, features]
            
        Returns:
            torch.Tensor: Predicted logits for 30-day mortality
        """
        # Get ECG features
        ecg_features = self.ecg_encoder(ecg)
        
        # Get tabular features
        tabular_features = self.tabular_model(tabular)
        
        # Concatenate features
        combined_features = torch.cat([ecg_features, tabular_features], dim=1)
        
        # Apply fusion layers
        logits = self.fusion_layer(combined_features)
        
        return logits


class AttentionFusion(nn.Module):
    """Fusion model that uses attention to combine ECG and tabular features."""
    
    def __init__(self, ecg_encoder, tabular_feature_extractor, fusion_dim=256, dropout_rate=0.3):
        """
        Initialize the attention fusion model.
        
        Args:
            ecg_encoder (nn.Module): Encoder for ECG signals
            tabular_feature_extractor (TabularFeatureExtractor): Feature extractor for tabular data
            fusion_dim (int, optional): Dimension of the fusion layer
            dropout_rate (float, optional): Dropout rate for regularization
        """
        super(AttentionFusion, self).__init__()
        
        self.ecg_encoder = ecg_encoder
        self.tabular_feature_extractor = tabular_feature_extractor
        
        # Determine the combined dimension
        self.ecg_dim = ecg_encoder.output_dim
        self.tabular_dim = tabular_feature_extractor.output_dim
        self.combined_dim = self.ecg_dim + self.tabular_dim
        
        # Cross-modal attention
        self.ecg_attention = nn.Linear(self.ecg_dim, 1)
        self.tabular_attention = nn.Linear(self.tabular_dim, 1)
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 1)
        )
    
    def forward(self, ecg, tabular):
        """
        Forward pass through the attention fusion model.
        
        Args:
            ecg (torch.Tensor): ECG signals of shape [batch_size, channels, time_steps]
            tabular (torch.Tensor): Tabular features of shape [batch_size, features]
            
        Returns:
            tuple: (logits, feature_importance)
                - logits (torch.Tensor): Predicted logits for 30-day mortality
                - feature_importance (dict): Dictionary containing feature importance weights
        """
        # Get ECG features
        ecg_features = self.ecg_encoder(ecg)
        
        # Get tabular features and importance weights
        tabular_features, tabular_importance = self.tabular_feature_extractor(tabular)
        
        # Calculate cross-modal attention weights
        ecg_weight = torch.sigmoid(self.ecg_attention(ecg_features))
        tabular_weight = torch.sigmoid(self.tabular_attention(tabular_features))
        
        # Normalize weights
        total_weight = ecg_weight + tabular_weight
        ecg_weight = ecg_weight / total_weight
        tabular_weight = tabular_weight / total_weight
        
        # Apply attention weights
        weighted_ecg = ecg_features * ecg_weight
        weighted_tabular = tabular_features * tabular_weight
        
        # Concatenate weighted features
        combined_features = torch.cat([weighted_ecg, weighted_tabular], dim=1)
        
        # Apply fusion layers
        logits = self.fusion_layer(combined_features)
        
        # Store feature importance
        feature_importance = {
            'ecg_weight': ecg_weight.detach(),
            'tabular_weight': tabular_weight.detach(),
            'tabular_feature_importance': tabular_importance.detach()
        }
        
        return logits, feature_importance


class MultimodalMortalityPredictor(nn.Module):
    """Complete multimodal model for predicting 30-day mortality."""
    
    def __init__(self, ecg_config=None, tabular_config=None, fusion_config=None):
        """
        Initialize the multimodal mortality predictor.
        
        Args:
            ecg_config (dict, optional): Configuration for the ECG encoder
            tabular_config (dict, optional): Configuration for the tabular model
            fusion_config (dict, optional): Configuration for the fusion model
        """
        super(MultimodalMortalityPredictor, self).__init__()
        
        # Default configurations
        if ecg_config is None:
            ecg_config = {
                'model_type': 'cnn',  # 'cnn' or 'resnet'
                'in_channels': 12,
                'output_dim': 128
            }
        
        if tabular_config is None:
            tabular_config = {
                'model_type': 'mlp',  # 'mlp' or 'feature_extractor'
                'input_dim': 20,  # This should be set based on the actual data
                'output_dim': 128,
                'hidden_dims': [256, 128],
                'dropout_rate': 0.3,
                'batch_norm': True
            }
        
        if fusion_config is None:
            fusion_config = {
                'model_type': 'concatenation',  # 'concatenation' or 'attention'
                'fusion_dim': 256,
                'dropout_rate': 0.3
            }
        
        # Create ECG encoder
        if ecg_config['model_type'] == 'cnn':
            self.ecg_encoder = ECGEncoder(
                in_channels=ecg_config['in_channels'],
                output_dim=ecg_config['output_dim']
            )
        elif ecg_config['model_type'] == 'resnet':
            self.ecg_encoder = ResNetECGEncoder(
                in_channels=ecg_config['in_channels'],
                output_dim=ecg_config['output_dim']
            )
        else:
            raise ValueError(f"Unknown ECG model type: {ecg_config['model_type']}")
        
        # Create tabular model
        if tabular_config['model_type'] == 'mlp':
            self.tabular_model = TabularMLP(
                input_dim=tabular_config['input_dim'],
                output_dim=tabular_config['output_dim'],
                hidden_dims=tabular_config['hidden_dims'],
                dropout_rate=tabular_config['dropout_rate'],
                batch_norm=tabular_config['batch_norm']
            )
            self.has_feature_importance = False
        elif tabular_config['model_type'] == 'feature_extractor':
            self.tabular_model = TabularFeatureExtractor(
                input_dim=tabular_config['input_dim'],
                output_dim=tabular_config['output_dim'],
                hidden_dims=tabular_config['hidden_dims'],
                dropout_rate=tabular_config['dropout_rate'],
                batch_norm=tabular_config['batch_norm']
            )
            self.has_feature_importance = True
        else:
            raise ValueError(f"Unknown tabular model type: {tabular_config['model_type']}")
        
        # Create fusion model
        if fusion_config['model_type'] == 'concatenation':
            self.fusion_model = ConcatenationFusion(
                ecg_encoder=self.ecg_encoder,
                tabular_model=self.tabular_model,
                fusion_dim=fusion_config['fusion_dim'],
                dropout_rate=fusion_config['dropout_rate']
            )
        elif fusion_config['model_type'] == 'attention':
            if not self.has_feature_importance:
                raise ValueError("Attention fusion requires a TabularFeatureExtractor")
            self.fusion_model = AttentionFusion(
                ecg_encoder=self.ecg_encoder,
                tabular_feature_extractor=self.tabular_model,
                fusion_dim=fusion_config['fusion_dim'],
                dropout_rate=fusion_config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown fusion model type: {fusion_config['model_type']}")
        
        self.fusion_type = fusion_config['model_type']
    
    def forward(self, ecg, tabular):
        """
        Forward pass through the multimodal mortality predictor.
        
        Args:
            ecg (torch.Tensor): ECG signals of shape [batch_size, channels, time_steps]
            tabular (torch.Tensor): Tabular features of shape [batch_size, features]
            
        Returns:
            torch.Tensor or tuple: Predicted logits for 30-day mortality, and optionally feature importance
        """
        return self.fusion_model(ecg, tabular)
    
    def predict_proba(self, ecg, tabular):
        """
        Predict probability of 30-day mortality.
        
        Args:
            ecg (torch.Tensor): ECG signals of shape [batch_size, channels, time_steps]
            tabular (torch.Tensor): Tabular features of shape [batch_size, features]
            
        Returns:
            torch.Tensor or tuple: Predicted probabilities for 30-day mortality, and optionally feature importance
        """
        output = self.forward(ecg, tabular)
        
        if self.fusion_type == 'attention':
            logits, feature_importance = output
            probabilities = torch.sigmoid(logits)
            return probabilities, feature_importance
        else:
            logits = output
            probabilities = torch.sigmoid(logits)
            return probabilities